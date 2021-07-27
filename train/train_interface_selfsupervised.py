import torch.optim as optim

from models.selfsupervised_learner_1d import SelfSupervisedLearner1d
from train.train_selfsupervised import train


def main(
        data,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        scheduler_kwargs,
        train_kwargs,
        device):
    model_kwargs = {}

    for kw in [
        'device',
        'num_local_crops',
        'max_epochs'
    ]:
        if kw in train_kwargs:
            if kw == 'device':
                model_kwargs[kw] = train_kwargs.pop(kw)
            else:
                model_kwargs[kw] = train_kwargs[kw]

    model = SelfSupervisedLearner1d(model, **model_kwargs)
    optimizer = train_kwargs.pop('optimizer', None)

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = train_kwargs.pop('scheduler', None)

    if scheduler == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, **scheduler_kwargs)
    elif scheduler == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **scheduler_kwargs)

    train(
        data,
        model,
        optimizer,
        scheduler,
        sampling_fn,
        collate_fn,
        device,
        **train_kwargs)
