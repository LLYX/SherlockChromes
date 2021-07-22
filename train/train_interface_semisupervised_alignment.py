import torch.optim as optim

from models.semisupervised_learner_1d import SemiSupervisedAlignmentLearner1d
from train.train_semisupervised_alignment import train


def main(
        data,
        template_data,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        train_kwargs,
        device):
    model_kwargs = {}
    for kw in [
        'semisupervised',
        'wu',
        'threshold',
        'use_weak_labels',
        'enforce_weak_consistency',
        'enforce_sparse_loc',
        'enforce_sparse_attn',
        'sparsity_modulator',
        'augmentator_mz_bins',
        'augmentator_scale',
        'augmentator_mean',
        'augmentator_std',
        'augmentator_F',
        'augmentator_m_F',
        'augmentator_T',
        'augmentator_m_T',
        'normalize',
        'normalization_mode',
        'regularizer_mode',
        'normalization_channels',
        'regularizer_sigma_min',
        'regularizer_sigma_max',
        'regularizer_p_min',
        'regularizer_p_max',
        'loss_alpha',
        'loss_gamma',
        'loss_logits',
        'loss_reduction',
        'model_device',
        'debug'
    ]:
        if kw in train_kwargs:
            model_kwargs[kw] = train_kwargs[kw]

    model = SemiSupervisedAlignmentLearner1d(
        model,
        **model_kwargs)

    optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    train(
        data,
        template_data,
        model,
        optimizer,
        loss,
        sampling_fn,
        collate_fn,
        device,
        **train_kwargs)
