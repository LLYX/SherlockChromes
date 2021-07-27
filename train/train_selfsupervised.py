import importlib
import numpy as np
import os
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from datasets.chromatograms_dataset import Subset
from utils.general_utils import cosine_scheduler


def get_data_loader(
    data,
    test_batch_proportion=0.1,
    batch_size=1,
    sampling_fn=None,
    collate_fn=None,
    outdir_path=None
):
    if sampling_fn:
        unlabeled_idx = sampling_fn(data, test_batch_proportion)[0]
    else:
        raise NotImplementedError

    if outdir_path:
        if not os.path.isdir(outdir_path):
            os.mkdir(outdir_path)

        np.savetxt(
            os.path.join(outdir_path, 'unlabeled_idx.txt'),
            np.array(unlabeled_idx),
            fmt='%i')

    unlabeled_set = Subset(data, unlabeled_idx, False)

    if collate_fn:
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
    else:
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=batch_size)

    return unlabeled_loader


def train(
    data,
    model,
    optimizer=None,
    scheduler=None,
    sampling_fn=None,
    collate_fn=None,
    device='cpu',
    **kwargs
):
    unlabeled_loader = get_data_loader(
        data,
        kwargs['test_batch_proportion'],
        kwargs['batch_size'],
        sampling_fn,
        collate_fn,
        kwargs['outdir_path'])

    if 'momentum_teacher' not in kwargs:
        kwargs['momentum_teacher'] = 0.996

    momentum_schedule = cosine_scheduler(
        kwargs['momentum_teacher'],
        1,
        kwargs['max_epochs'],
        len(unlabeled_loader))

    wandb_available = False

    if 'visualize' in kwargs and kwargs['visualize']:
        wandb_spec = importlib.util.find_spec('wandb')
        wandb_available = wandb_spec is not None

        if wandb_available:
            print('wandb detected!')
            import wandb

            wandb.init(
                settings=wandb.Settings(start_method='thread'),
                project='SherlockChromes',
                group=kwargs['model_savename'],
                name=wandb.util.generate_id(),
                job_type='train-selfsupervised',
                config=kwargs)

    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters())

    if not scheduler:
        scheduler = CosineAnnealingWarmRestarts(optimizer, 10)

    if 'transfer_model_path' in kwargs:
        model.load_state_dict(
            torch.load(kwargs['transfer_model_path']).state_dict(),
            strict=False)

    lowest_loss = 100
    model.to(device)

    for epoch in range(kwargs['max_epochs']):
        iters, train_loss = 0, 0
        model.train()

        for batch, _ in unlabeled_loader:
            batch = batch.to(device=device)
            optimizer.zero_grad()
            loss_out = model(batch, epoch)
            loss_out.backward()
            # implement cancel last layer gradient here
            # if training loss does not decrease
            optimizer.step()

            with torch.no_grad():
                m = momentum_schedule[iters]  # momentum parameter

                for param_q, param_k in zip(
                    model.student.parameters(), model.teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if ('scheduler_step_on_iter' in kwargs and
                kwargs['scheduler_step_on_iter']):
                scheduler.step()

            iters += 1
            iter_loss = loss_out.item()
            train_loss += iter_loss

            print(f'Training - Iter: {iters} Iter Loss: {iter_loss:.8f}')

        if not ('scheduler_step_on_iter' in kwargs and
                kwargs['scheduler_step_on_iter']):
            scheduler.step()

        train_loss = train_loss / iters
        print(f'Training - Epoch: {epoch} Avg Loss: {train_loss:.8f}')

        if wandb_available:
            wandb.log({'Train Loss': train_loss})

        save_path = ''

        if train_loss < lowest_loss:
            save_path = os.path.join(
                kwargs['outdir_path'],
                f"{kwargs['model_savename']}_model_{epoch}_loss={train_loss}"
                '.pth')
            lowest_loss = train_loss

        if save_path:
            if 'save_whole' in kwargs and kwargs['save_whole']:
                torch.save(model, save_path)
            else:
                torch.save(model.state_dict(), save_path)

    if 'save_whole' in kwargs and kwargs['save_whole']:
        torch.save(model, save_path)
    else:
        torch.save(model.state_dict(), save_path)
