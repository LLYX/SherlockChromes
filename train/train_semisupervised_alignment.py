import numpy as np
import os
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset

from optimizers.focal_loss import FocalLossBinary


def get_data_loaders(
    data,
    template_data,
    batch_size=32,
    u_ratio=16,
    template_batch_size=4,
    sampling_fn=None,
    collate_fn=None,
    outdir_path=None
):

    # LoadingSampler supported only
    if sampling_fn:
        (
            labeled_idx,
            unlabeled_idx,
            val_idx,
            train_template_idx,
            val_template_idx
        ) = sampling_fn()
    else:
        raise NotImplementedError

    if outdir_path:
        if not os.path.isdir(outdir_path):
            os.mkdir(outdir_path)

        np.savetxt(
            os.path.join(outdir_path, 'labeled_idx.txt'),
            np.array(labeled_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'unlabeled_idx.txt'),
            np.array(unlabeled_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'val_idx.txt'),
            np.array(val_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'train_template_idx.txt'),
            np.array(train_template_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'val_template_idx.txt'),
            np.array(val_template_idx),
            fmt='%i')

    labeled_set = Subset(data, labeled_idx)
    unlabeled_set = Subset(data, unlabeled_idx)
    val_set = Subset(data, val_idx)
    train_template_set = Subset(template_data, train_template_idx)
    val_template_set = Subset(template_data, val_template_idx)

    if collate_fn:
        labeled_loader = DataLoader(
            labeled_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=u_ratio * batch_size,
            collate_fn=collate_fn)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        train_template_loader = DataLoader(
            train_template_set,
            batch_size=template_batch_size,
            collate_fn=collate_fn)
        val_template_loader = DataLoader(
            val_template_set,
            batch_size=template_batch_size,
            collate_fn=collate_fn)
    else:
        labeled_loader = DataLoader(
            labeled_set,
            batch_size=batch_size)
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=u_ratio * batch_size)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size)
        train_template_loader = DataLoader(
            train_template_set,
            batch_size=1)
        val_template_loader = DataLoader(
            val_template_set,
            batch_size=1)

    return (
        labeled_loader,
        unlabeled_loader,
        val_loader,
        train_template_loader,
        val_template_loader)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train(
    data,
    template_data,
    model,
    optimizer=None,
    loss=None,
    sampling_fn=None,
    collate_fn=None,
    device='cpu',
    **kwargs
):
    (
        labeled_loader,
        unlabeled_loader,
        val_loader,
        train_template_loader,
        val_template_loader
    ) = get_data_loaders(
            data,
            template_data,
            kwargs['batch_size'],
            kwargs['uratio'],
            kwargs['template_batch_size'],
            sampling_fn,
            collate_fn,
            kwargs['outdir_path'])

    if not optimizer:
        optimizer = AdamW(model.parameters())

    if not loss:
        loss = FocalLossBinary()

    if 'transfer_model_path' in kwargs:
        model.load_state_dict(
            torch.load(kwargs['transfer_model_path']).state_dict(),
            strict=False,
            map_location=device)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, kwargs['T_0'], T_mult=kwargs['T_mult'])

    unlabeled_loader = iter(cycle(unlabeled_loader))
    train_template_loader = iter(cycle(train_template_loader))
    val_template_loader = iter(cycle(val_template_loader))
    highest_dice, highest_iou, lowest_loss = 0, 0, 100
    model.to(device)

    for epoch in range(kwargs['max_epochs']):
        iters, avg_loss = 0, 0

        for labeled_batch, labels in labeled_loader:
            unlabeled_batch, _ = next(unlabeled_loader)
            labeled_batch = labeled_batch.to(device=device)
            labels = labels.to(device=device)
            unlabeled_batch = unlabeled_batch.to(device=device)
            template_batch, template_labels = next(train_template_loader)
            template_batch = template_batch.to(device=device)
            template_labels = template_labels.to(device=device)

            model.train()

            if ('scheduler_step_on_iter' in kwargs and
                    kwargs['scheduler_step_on_iter']):
                scheduler.step(epoch + iters / kwargs['max_epochs'])

            optimizer.zero_grad()

            loss_out = model(
                unlabeled_batch,
                template_batch,
                template_labels,
                labeled_batch,
                labels)
            loss_out.backward()
            optimizer.step()
            iters += 1
            iter_loss = loss_out.item()
            avg_loss += iter_loss
            print(f'Training - Iter: {iters} Iter loss: {iter_loss:.8f}')

        if not ('scheduler_step_on_iter' in kwargs and
                kwargs['scheduler_step_on_iter']):
            scheduler.step(epoch + iters / kwargs['max_epochs'])
        print(f'Training - Epoch: {epoch} Avg loss: {(avg_loss / iters):.8f}')

        labels_for_metrics = []
        outputs_for_metrics = []
        losses = []
        for batch, labels in val_loader:
            template_batch, template_labels = next(val_template_loader)
            model.eval()
            with torch.no_grad():
                batch = batch.to(device=device)
                labels = labels.to(device=device)
                template_batch = template_batch.to(device=device)
                template_labels = template_labels.to(device=device)
                labels_for_metrics.append(labels.cpu().detach().numpy())
                preds = model(batch, template_batch, template_labels)
                outputs_for_metrics.append(preds.cpu().detach().numpy())
                loss_out = loss(preds, labels)
                losses.append(loss_out.cpu().detach().numpy())

        labels_for_metrics = np.concatenate(labels_for_metrics).reshape(-1, 1)
        outputs_for_metrics = (
            np.concatenate(outputs_for_metrics) >= 0.5
        ).reshape(-1, 1)
        accuracy = accuracy_score(
            labels_for_metrics, outputs_for_metrics)
        balanced_accuracy = balanced_accuracy_score(
            labels_for_metrics, outputs_for_metrics)
        precision = precision_score(labels_for_metrics, outputs_for_metrics)
        recall = recall_score(labels_for_metrics, outputs_for_metrics)
        dice = f1_score(labels_for_metrics, outputs_for_metrics)
        iou = jaccard_score(labels_for_metrics, outputs_for_metrics)
        avg_loss = np.mean(losses)

        print(
            f'Validation - Epoch: {epoch} '
            f'Accuracy: {accuracy:.8f} '
            f'Balanced accuracy: {balanced_accuracy:.8f} '
            f'Precision: {precision:.8f} '
            f'Recall: {recall:.8f} '
            f'Dice: {dice:.8f} '
            f'IoU: {iou:.8f} '
            f'Avg loss: {avg_loss:.8f} ')

        save_path = ''

        if (
            dice > highest_dice or
            iou > highest_iou or
            avg_loss < lowest_loss
        ):
            save_path = f"{kwargs['model_savename']}_linear_{epoch}"

            if dice > highest_dice:
                save_path += f'_dice={dice}'
                highest_dice = dice

            if iou > highest_iou:
                save_path += f'_iou={iou}'
                highest_iou = iou

            if avg_loss < lowest_loss:
                save_path += f'_loss={avg_loss}'
                lowest_loss = avg_loss

            save_path = os.path.join(kwargs['outdir_path'], save_path + '.pth')

        if save_path:
            if 'save_whole' in kwargs and kwargs['save_whole']:
                torch.save(model, save_path)
            else:
                torch.save(model.state_dict(), save_path)
