import sys
import torch.optim as optim

from .train_semisupervised_alignment import train
from models.semisupervised_learner_1d import SemiSupervisedAlignmentLearner1d

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
        'wu',
        'threshold',
        'augmentator_num_channels',
        'augmentator_normalize',
        'augmentator_normalization_mode',
        'augmentator_scale_independently',
        'augmentator_scale_precursors',
        'augmentator_lower',
        'augmentator_upper',
        'regularizer_mode',
        'regularizer_sigma_min',
        'regularizer_sigma_max',
        'regularizer_p_min',
        'regularizer_p_max',
        'modulation_mode',
        'loss_alpha',
        'loss_gamma',
        'loss_logits',
        'loss_reduction',
        'model_device',
        'debug']:
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
