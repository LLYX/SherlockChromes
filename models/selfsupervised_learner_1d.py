import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.augmentators import (
    SelfSupervisedGlobalAugmentatorOne,
    SelfSupervisedGlobalAugmentatorTwo,
    SelfSupervisedLocalAugmentator)
from models.modelzoo1d.dain import DAIN_Layer


class LinearFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        activation='relu'
    ):
        super(LinearFeedForwardNetwork, self).__init__()
        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise NotImplementedError('invalid activation')

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))

                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise NotImplementedError('invalid activation')

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x


class MultiCropWrapper1d(nn.Module):
    def __init__(self, model, head, normalize=True, normalization_mode='full'):
        super(MultiCropWrapper1d, self).__init__()
        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=model.in_channels)
        else:
            self.normalization_layer = nn.Identity()

        self.model = model
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True)[1], 0)

        orig_setting = self.model.output_mode
        self.model.output_mode = 'tkn'
        start_idx = 0

        for end_idx in idx_crops:
            _out = self.model(torch.cat(x[start_idx: end_idx]))

            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))

            start_idx = end_idx

        self.model.output_mode = orig_setting

        return self.head(output)


class SelfSupervisedLoss1d(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, max_epochs, student_temp=0.1,
                 center_momentum=0.9):
        super(SelfSupervisedLoss1d, self).__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer('center', torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max_epochs - warmup_teacher_temp_epochs) * teacher_temp))

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss


class SelfSupervisedLearner1d(nn.Module):
    def __init__(self, model, device='cpu', num_local_crops=8, max_epochs=300):
        super(SelfSupervisedLearner1d, self).__init__()
        self.global_augmentator_1 = SelfSupervisedGlobalAugmentatorOne(
            scale=(0.6, 1),
            size=175,
            mode='linear',
            mz_bins=420,
            device=device,
            mean=0,
            std=1)
        self.global_augmentator_2 = SelfSupervisedGlobalAugmentatorTwo(
            scale=(0.6, 1),
            size=175,
            mode='linear',
            mz_bins=420,
            num_F=2,
            m_F=42,
            T=7,
            m_T=5)
        self.local_augmentator = SelfSupervisedLocalAugmentator(
            scale=(0.2, 0.6),
            size=105,
            mode='linear',
            mz_bins=420,
            device=device,
            mean=0,
            std=1,
            num_F=2,
            m_F=42,
            T=2,
            m_T=5)
        self.student = MultiCropWrapper1d(
            model,
            LinearFeedForwardNetwork(
                in_dim=model.transformer_channels,
                out_dim=4096))
        self.teacher = MultiCropWrapper1d(
            copy.deepcopy(model),
            LinearFeedForwardNetwork(
                in_dim=model.transformer_channels,
                out_dim=4096))

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.device = device
        self.num_local_crops = num_local_crops

        self.loss = SelfSupervisedLoss1d(
            out_dim=4096,
            ncrops= 2 + self.num_local_crops,
            warmup_teacher_temp=0.04,
            teacher_temp=0.04,
            warmup_teacher_temp_epochs=30,
            max_epochs=max_epochs
        )
    
    def forward(self, batch, epoch):
        crops = []
        crops.append(self.global_augmentator_1(batch))
        crops.append(self.global_augmentator_2(batch))

        for i in range(self.num_local_crops):
            crops.append(self.local_augmentator(batch))
        
        teacher_output = self.teacher(crops[:2])
        student_output = self.student(crops)
        loss = self.loss(student_output, teacher_output, epoch)

        return loss
