import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import erfinv

from models.augmentators import SemiSupervisedStrongAugmentator, SemiSupervisedWeakAugmentator
from optimizers.focal_loss import FocalLossBinary


class SemiSupervisedLearner1d(nn.Module):
    def __init__(
        self,
        model,
        semisupervised=True,
        wu=1,
        threshold=0.95,
        use_weak_labels=False,
        enforce_weak_consistency=False,
        enforce_sparse_loc=False,
        enforce_sparse_attn=False,
        sparsity_modulator=1e-4,
        augmentator_mz_bins=6,
        augmentator_scale=[0.875, 1.125],
        augmentator_mean=0,
        augmentator_std=1,
        augmentator_F=1,
        augmentator_m_F=1,
        augmentator_T=5,
        augmentator_m_T=1,
        regularizer_mode='none',
        regularizer_sigma_min=4,
        regularizer_sigma_max=16,
        regularizer_p_min=0.5,
        regularizer_p_max=0.5,
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        model_device='cpu',
        debug=False
    ):
        super(SemiSupervisedLearner1d, self).__init__()
        self.model = model
        self.semisupervised = semisupervised
        self.wu = wu
        self.threshold = threshold
        self.use_weak_labels = use_weak_labels
        self.enforce_weak_consistency = enforce_weak_consistency
        self.enforce_sparse_loc = enforce_sparse_loc
        self.enforce_sparse_attn = enforce_sparse_attn
        self.sparsity_modulator = sparsity_modulator
        self.device = model_device

        self.weak_augmentator = SemiSupervisedWeakAugmentator(
            mz_bins=augmentator_mz_bins,
            scale=augmentator_scale,
            device=self.device
        )

        self.strong_augmentator = SemiSupervisedStrongAugmentator(
            mz_bins=augmentator_mz_bins,
            device=self.device,
            scale=augmentator_scale,
            mean=augmentator_mean,
            std=augmentator_std,
            num_F=augmentator_F,
            m_F=augmentator_m_F,
            T=augmentator_T,
            m_T=augmentator_m_T
        )

        self.regularizer_mode = regularizer_mode
        self.regularizer_sigma_min = regularizer_sigma_min
        self.regularizer_sigma_max = regularizer_sigma_max
        self.regularizer_p_min = regularizer_p_min
        self.regularizer_p_max = regularizer_p_max

        self.loss = FocalLossBinary(
            alpha=loss_alpha,
            gamma=loss_gamma,
            logits=loss_logits,
            reduction=loss_reduction
        )

        self.debug = debug

    def get_model(self):
        return self.model

    def generate_zebra_mask(
        self,
        length,
        sigma_min=4,
        sigma_max=16,
        p_min=0.5,
        p_max=0.5
    ):
        sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))
        p = np.random.uniform(p_min, p_max)
        noise_image = np.random.normal(size=length)
        noise_image_smoothed = gaussian_filter1d(noise_image, sigma)
        threshold = (
            erfinv((p * 2) - 1) * (2 ** 0.5) * noise_image_smoothed.std()
            + noise_image_smoothed.mean())

        return (noise_image_smoothed > threshold).astype(float)

    def forward(self, unlabeled_batch, labeled_batch=None, labels=None):
        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'

            b_ul, c_ul, l_ul = unlabeled_batch.size()

            if self.regularizer_mode == 'cutmix':
                if b_ul % 2 != 0:
                    unlabeled_batch = torch.cat(
                        [
                            unlabeled_batch,
                            torch.zeros(1, c_ul, l_ul).to(self.device)
                        ],
                        dim=0
                    )
                    b_ul += 1

            orig_setting = self.model.output_mode

            self.model.output_mode = 'all'
            out_dict = self.model(labeled_batch)
            weak_output, strong_output, attn = (
                out_dict['cla'],
                out_dict['loc'],
                out_dict['attn']
            )

            if self.use_weak_labels:
                labeled_loss = torch.mean(
                    self.loss(weak_output, labels)
                )
            else:
                weak_labeled_loss = 0

                if self.enforce_weak_consistency:
                    weak_labeled_loss = torch.mean(
                        self.loss(
                            weak_output,
                            torch.max(labels, dim=1, keepdim=True)[0]
                        )
                    )

                labeled_loss = (
                    torch.mean(self.loss(strong_output, labels))
                    + weak_labeled_loss
                )

            if self.enforce_sparse_loc:
                labeled_loss = labeled_loss + torch.mean(
                    torch.norm(strong_output, p=1, dim=1)
                    * self.sparsity_modulator)

            if self.enforce_sparse_attn:
                labeled_loss = labeled_loss + torch.mean(
                    torch.norm(attn, p=1, dim=1) * self.sparsity_modulator)

            if self.semisupervised:
                strongly_augmented = self.strong_augmentator(unlabeled_batch)
                weakly_augmented = self.weak_augmentator(unlabeled_batch)

                if self.enforce_weak_consistency:
                    self.model.output_mode = 'all'
                    out_dict = self.model(weakly_augmented)
                    weak_output, strong_output = (
                        out_dict['cla'],
                        out_dict['loc']
                    )

                    weak_pseudo_labels = (weak_output >= 0.5).float()
                    weak_quality_modulator = (
                        (weak_output >= self.threshold).float()
                        + (weak_output <= (1 - self.threshold))
                    ).reshape(1, -1).squeeze()

                    # Variable required for cutmix
                    lam = None
                else:
                    self.model.output_mode = 'loc'
                    strong_output = self.model(strongly_augmented)

                strong_pseudo_labels = (strong_output >= 0.5).float()
                strong_quality_modulator = (
                    (strong_output >= self.threshold).float()
                    + (strong_output <= (1 - self.threshold))
                )

                # Variable required for cutmix
                b_ul_half = b_ul // 2

                if self.regularizer_mode != 'none':
                    regularizer_mask = torch.from_numpy(
                        self.generate_zebra_mask(
                            l_ul,
                            sigma_min=self.regularizer_sigma_min,
                            sigma_max=self.regularizer_sigma_max,
                            p_min=self.regularizer_p_min,
                            p_max=self.regularizer_p_max
                        )
                    ).float().to(self.device)

                    if self.regularizer_mode == 'cutout':
                        strongly_augmented = (
                            strongly_augmented * regularizer_mask)
                        strong_quality_modulator = (
                            strong_quality_modulator * regularizer_mask)
                    elif self.regularizer_mode == 'cutmix':
                        if self.enforce_weak_consistency:
                            lam = (
                                torch.sum(regularizer_mask)
                                / regularizer_mask.nelement()
                            )

                        strongly_augmented = (
                            (
                                strongly_augmented[0:b_ul_half]
                                * regularizer_mask)
                            + (strongly_augmented[b_ul_half:]
                                * (1 - regularizer_mask)))
                        strong_pseudo_labels = (
                            (
                                strong_pseudo_labels[0:b_ul_half]
                                * regularizer_mask)
                            + (strong_pseudo_labels[b_ul_half:]
                                * (1 - regularizer_mask)))
                        strong_quality_modulator = (
                            (strong_quality_modulator[0:b_ul_half]
                                * regularizer_mask)
                            + (strong_quality_modulator[b_ul_half:]
                                * (1 - regularizer_mask)))

                strong_quality_modulator = torch.mean(
                    strong_quality_modulator.reshape(1, -1).squeeze())

                self.model.output_mode = 'all'
                out_dict = self.model(strongly_augmented)
                weak_output, strong_output, attn = (
                    out_dict['cla'],
                    out_dict['loc'],
                    out_dict['attn']
                )

                if (
                    self.enforce_weak_consistency
                    and self.regularizer_mode == 'cutmix'
                ):
                    weak_unlabeled_loss_a = lam * torch.mean(
                        self.loss(
                            weak_output,
                            weak_pseudo_labels[:b_ul_half]
                        )[weak_quality_modulator[:b_ul_half].bool()]
                    )

                    if torch.isnan(weak_unlabeled_loss_a):
                        weak_unlabeled_loss_a = 0.0

                    weak_unlabeled_loss_b = (1 - lam) * torch.mean(
                        self.loss(
                            weak_output,
                            weak_pseudo_labels[b_ul_half:]
                        )[weak_quality_modulator[b_ul_half:].bool()]
                    )

                    if torch.isnan(weak_unlabeled_loss_b):
                        weak_unlabeled_loss_b = 0.0

                    weak_unlabeled_loss = (
                        weak_unlabeled_loss_a + weak_unlabeled_loss_b
                    )
                elif self.enforce_weak_consistency:
                    weak_unlabeled_loss = torch.mean(
                        self.loss(
                            weak_output,
                            weak_pseudo_labels
                        )[weak_quality_modulator.bool()]
                    )

                    if torch.isnan(weak_unlabeled_loss):
                        weak_unlabeled_loss = 0.0

                strong_unlabeled_loss = strong_quality_modulator * torch.mean(
                    self.loss(strong_output, strong_pseudo_labels)
                )

                self.model.output_mode = orig_setting

                if self.enforce_weak_consistency:
                    unlabeled_loss = (
                        weak_unlabeled_loss + strong_unlabeled_loss)
                else:
                    unlabeled_loss = strong_unlabeled_loss

                if self.enforce_sparse_loc:
                    unlabeled_loss = (
                        unlabeled_loss
                        + strong_quality_modulator * torch.mean(
                            torch.norm(strong_output, p=1, dim=1)
                            * self.sparsity_modulator))

                if self.enforce_sparse_attn:
                    unlabeled_loss = (
                        unlabeled_loss
                        + strong_quality_modulator * torch.mean(
                            torch.norm(attn, p=1, dim=1)
                            * self.sparsity_modulator))
            else:
                unlabeled_loss = 0

            if self.debug:
                if self.semisupervised:
                    if self.use_weak_labels:
                        num_positive = int(torch.sum(labels).item())
                    else:
                        num_positive = 'n/a'

                    if self.enforce_weak_consistency:
                        if isinstance(weak_unlabeled_loss, float):
                            weak_unlabeled_loss_debug = weak_unlabeled_loss
                        else:
                            weak_unlabeled_loss_debug = (
                                weak_unlabeled_loss.item())

                        weak_unlabeled_loss_debug = (
                            f'{weak_unlabeled_loss_debug:.8f}')
                        weak_quality_modulator_debug = torch.mean(
                            weak_quality_modulator).item()
                        weak_quality_modulator_debug = (
                            f'{weak_quality_modulator_debug:.8f}')
                    else:
                        weak_unlabeled_loss_debug = 'n/a'
                        weak_quality_modulator_debug = 'n/a'

                    print(
                        f'L Loss: {labeled_loss.item():.8f}, '
                        f'# Positive: {num_positive}, '
                        f'UL Loss: {unlabeled_loss.item():.8f}, '
                        'Weak Quality Modulator u: '
                        f'{weak_quality_modulator_debug}, '
                        f'Weak UL Loss: {weak_unlabeled_loss_debug}, '
                        'Strong Quality Modulator u: '
                        f'{strong_quality_modulator.item():.8f}, '
                        f'Strong UL Loss: {strong_unlabeled_loss.item():.8f}, '
                        'Weighted UL Loss: '
                        f'{self.wu * unlabeled_loss.item():.8f}'
                    )
                else:
                    print(f'L Loss: {labeled_loss.item():.8f}')

            return labeled_loss + self.wu * unlabeled_loss
        else:
            return self.model(unlabeled_batch)


# TODO: Update forward to match parent structure
class SemiSupervisedAlignmentLearner1d(SemiSupervisedLearner1d):
    def __init__(
        self,
        model,
        semisupervised=True,
        wu=1,
        threshold=0.85,
        use_weak_labels=False,
        enforce_weak_consistency=False,
        enforce_sparse_loc=False,
        enforce_sparse_attn=False,
        sparsity_modulator=1e-4,
        augmentator_mz_bins=6,
        augmentator_scale=[0.875, 1.125],
        augmentator_mean=0,
        augmentator_std=1,
        augmentator_F=1,
        augmentator_m_F=1,
        augmentator_T=5,
        augmentator_m_T=1,
        regularizer_mode='none',
        regularizer_sigma_min=4,
        regularizer_sigma_max=16,
        regularizer_p_min=0.5,
        regularizer_p_max=0.5,
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        model_device='cpu',
        debug=False
    ):
        super(SemiSupervisedAlignmentLearner1d, self).__init__(
            model,
            semisupervised=semisupervised,
            wu=wu,
            threshold=threshold,
            use_weak_labels=use_weak_labels,
            enforce_weak_consistency=enforce_weak_consistency,
            enforce_sparse_loc=enforce_sparse_loc,
            enforce_sparse_attn=enforce_sparse_attn,
            sparsity_modulator=sparsity_modulator,
            augmentator_mz_bins=augmentator_mz_bins,
            augmentator_scale=augmentator_scale,
            augmentator_mean=augmentator_mean,
            augmentator_std=augmentator_std,
            augmentator_F=augmentator_F,
            augmentator_m_F=augmentator_m_F,
            augmentator_T=augmentator_T,
            augmentator_m_T=augmentator_m_T,
            regularizer_mode=regularizer_mode,
            regularizer_sigma_min=regularizer_sigma_min,
            regularizer_sigma_max=regularizer_sigma_max,
            regularizer_p_min=regularizer_p_min,
            regularizer_p_max=regularizer_p_max,
            loss_alpha=loss_alpha,
            loss_gamma=loss_gamma,
            loss_logits=loss_logits,
            loss_reduction=loss_reduction,
            model_device=model_device,
            debug=debug
        )

    def forward(
        self,
        unlabeled_batch,
        templates,
        template_labels,
        labeled_batch=None,
        labels=None
    ):
        b_ul, c_ul, l_ul = unlabeled_batch.size()

        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'

            if self.regularizer_mode == 'cutmix':
                if b_ul % 2 != 0:
                    unlabeled_batch = unlabeled_batch[0:b_ul - 1]

            self.model.aggregate_output = self.use_weak_labels

            labeled_loss = torch.mean(
                self.loss(
                    self.model(labeled_batch, templates, template_labels),
                    labels
                )
            )

            self.model.aggregate_output = False

            strongly_augmented = self.strong_augmentator(unlabeled_batch)
            weakly_augmented = self.weak_augmentator(unlabeled_batch)
            weak_output = self.model(
                weakly_augmented, templates, template_labels)
            pseudo_labels = (weak_output >= 0.5).float()
            quality_modulator = (
                (weak_output >= self.threshold).float()
                + (weak_output <= (1 - self.threshold))
            )

            if self.regularizer_mode != 'none':
                regularizer_mask = torch.from_numpy(
                    self.generate_zebra_mask(
                        l_ul,
                        sigma_min=self.regularizer_sigma_min,
                        sigma_max=self.regularizer_sigma_max,
                        p_min=self.regularizer_p_min,
                        p_max=self.regularizer_p_max
                    )
                ).float().to(self.device)

                if self.regularizer_mode == 'cutout':
                    strongly_augmented = strongly_augmented * regularizer_mask
                    quality_modulator = quality_modulator * regularizer_mask
                elif self.regularizer_mode == 'cutmix':
                    b_ul_half = b_ul // 2
                    strongly_augmented = (
                        (
                            strongly_augmented[0:b_ul_half].to(self.device) *
                            regularizer_mask
                        ) +
                        (
                            strongly_augmented[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )
                    pseudo_labels = (
                        (
                            pseudo_labels[0:b_ul_half].to(self.device) *
                            regularizer_mask
                        ) +
                        (
                            pseudo_labels[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )
                    quality_modulator = (
                        (
                            quality_modulator[0:b_ul_half].to(self.device) *
                            regularizer_mask
                        ) +
                        (
                            quality_modulator[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )

            quality_modulator = quality_modulator.reshape(1, -1).squeeze()

            quality_modulator = torch.mean(quality_modulator)

            unlabeled_loss = torch.mean(
                quality_modulator *
                self.loss(
                    self.model(
                        strongly_augmented, templates, template_labels),
                    pseudo_labels
                )
            )

            if self.debug:
                print(
                    f'Labeled Loss: {labeled_loss.item()}, '
                    f'Unlabeled Loss: {unlabeled_loss.item()}, '
                    f'Weighted UL Loss: {self.wu * unlabeled_loss.item()}'
                )

            return labeled_loss + self.wu * unlabeled_loss
        else:
            return self.model(unlabeled_batch, templates, template_labels)
