import sys
import torch.optim as optim

sys.path.insert(0, '../models')
sys.path.insert(0, '../datasets')
sys.path.insert(0, '../optimizers')

from train_alignment import train

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
