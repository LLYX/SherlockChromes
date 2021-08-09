import importlib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score)
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts)
from torch.utils.data import DataLoader

from datasets.chromatograms_dataset import ExternalDataset, Subset
from utils.general_utils import cosine_scheduler


def get_data_loaders(
    data,
    test_batch_proportion=0.1,
    external_path=None,
    batch_size=1,
    sampling_fn=None,
    collate_fn=None,
    outdir_path=None
):
    # Currently only LoadingSampler returns 4 sets of idxs
    if sampling_fn:
        unlabeled_idx, train_idx, val_idx, test_idx = sampling_fn(
            data, test_batch_proportion)
    else:
        raise NotImplementedError

    if outdir_path:
        if not os.path.isdir(outdir_path):
            os.mkdir(outdir_path)

        np.savetxt(
            os.path.join(outdir_path, 'unlabeled_idx.txt'),
            np.array(unlabeled_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'train_idx.txt'),
            np.array(train_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'val_idx.txt'),
            np.array(val_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'test_idx.txt'),
            np.array(test_idx),
            fmt='%i')

    unlabeled_set = Subset(data, unlabeled_idx, False)

    if external_path:
        train_set = ExternalDataset(
            Subset(data, train_idx, True), external_path)
        val_set = ExternalDataset(
            Subset(data, val_idx, True), external_path)
        test_set = ExternalDataset(
            Subset(data, test_idx, True), external_path)
    else:
        train_set = Subset(data, train_idx, True)
        val_set = Subset(data, val_idx, True)
        test_set = Subset(data, test_idx, True)

    if collate_fn:
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
    else:
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=batch_size)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size)

    return unlabeled_loader, train_loader, val_loader, test_loader


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
    unlabeled_loader, train_loader, val_loader, test_loader = get_data_loaders(
        data,
        kwargs['test_batch_proportion'],
        kwargs['external_path'],
        kwargs['batch_size'],
        sampling_fn,
        collate_fn,
        kwargs['outdir_path'])
    
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
                job_type='train-selfsupervised-repr',
                config=kwargs)
    
    if 'transfer_model_path' in kwargs:
        model.load_state_dict(
            torch.load(
                kwargs['transfer_model_path'],
                map_location=device).state_dict(),
            strict=False)
    
    best_save_path = ''
    model.to(device)

    if 'train_linear_only' not in kwargs:
        kwargs['train_linear_only'] = False
        
    if not kwargs['train_linear_only']:
        if 'momentum_teacher' not in kwargs:
            kwargs['momentum_teacher'] = 0.996

        momentum_schedule = cosine_scheduler(
            kwargs['momentum_teacher'],
            1,
            kwargs['max_epochs'],
            len(unlabeled_loader))

        if not optimizer:
            optimizer = AdamW(model.parameters())

        if not scheduler:
            scheduler = CosineAnnealingWarmRestarts(optimizer, 10)

        lowest_loss = 100

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
                        model.student.parameters(),
                        model.teacher.parameters()
                    ):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                if ('scheduler_step_on_iter' in kwargs and
                    kwargs['scheduler_step_on_iter']):
                    scheduler.step(epoch + iters / kwargs['max_epochs'])

                iters += 1
                iter_loss = loss_out.item()
                train_loss += iter_loss
                print(f'Repr Training - Iter: {iters} Iter Loss: {iter_loss:.8f}')

            if not ('scheduler_step_on_iter' in kwargs and
                    kwargs['scheduler_step_on_iter']):
                scheduler.step(epoch + iters / kwargs['max_epochs'])

            train_loss = train_loss / iters
            print(f'Repr Training - Epoch: {epoch} Avg Loss: {train_loss:.8f}')

            if wandb_available:
                wandb.log({'Repr Train Loss': train_loss})

            if train_loss < lowest_loss:
                save_path = os.path.join(
                    kwargs['outdir_path'],
                    f"{kwargs['model_savename']}_model_{epoch}_loss={train_loss}"
                    '.pth')
                lowest_loss = train_loss
                best_save_path = save_path

                if 'save_whole' in kwargs and kwargs['save_whole']:
                    torch.save(model, save_path)
                else:
                    torch.save(model.state_dict(), save_path)

        # Train linear layer on top of features
        model.load_state_dict(
            torch.load(best_save_path, map_location=device).state_dict(),
            strict=False)

    model.teacher.model.output_mode = 'tkn'
    linear_classifier = nn.Linear(
        model.teacher.model.transformer_channels + 7, 1).to(device=device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = SGD(
        linear_classifier.parameters(), lr=0.0003, momentum=0.9, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, 100, eta_min=0)

    for epoch in range(100):
        iters, train_loss = 0, 0
        linear_classifier.train()

        for batch, labels, external in train_loader:
            batch = batch.to(device=device)
            labels = labels.to(device=device)
            external = external.to(device=device)

            with torch.no_grad():
                output = model.return_intermediate_repr(batch)
                output = torch.cat([output, external], dim=1)
            
            output = linear_classifier(output)
            optimizer.zero_grad()
            loss_out = loss(output, labels)
            loss_out.backward()
            optimizer.step()
            iters += 1
            iter_loss = loss_out.item()
            train_loss += iter_loss
            print(f'Linear Training - Iter: {iters} Iter Loss: {iter_loss:.8f}')
        
        train_loss = train_loss / iters
        print(f'Linear Training - Epoch: {epoch} Avg Loss: {train_loss:.8f}')
        scheduler.step()

        # Evaluate linear classifier

        linear_classifier.eval()
        labels_for_metrics = []
        outputs_for_metrics = []
        losses = []
        highest_bacc, highest_dice, highest_iou, lowest_loss = 0, 0, 0, 100

        for batch, labels, external in val_loader:
            with torch.no_grad():
                batch = batch.to(device=device)
                labels = labels.to(device=device)
                external = labels.to(device=device)
                labels_for_metrics.append(labels.cpu().numpy())
                output = model.return_intermediate_repr(batch)
                output = torch.cat([output, external], dim=1)
                output = linear_classifier(output)
                outputs_for_metrics.append(output.cpu().detach().numpy())
                loss_out = loss(output, labels).cpu().numpy()
                losses.append(loss_out)

        labels_for_metrics = np.concatenate(
            labels_for_metrics, axis=0).reshape(-1, 1)
        outputs_for_metrics = (
            np.concatenate(outputs_for_metrics, axis=0) >= 0.5).reshape(-1, 1)
        accuracy = accuracy_score(labels_for_metrics, outputs_for_metrics)
        avg_precision = average_precision_score(
            labels_for_metrics, outputs_for_metrics)
        bacc = balanced_accuracy_score(labels_for_metrics, outputs_for_metrics)
        precision = precision_score(labels_for_metrics, outputs_for_metrics)
        recall = recall_score(labels_for_metrics, outputs_for_metrics)
        dice = f1_score(labels_for_metrics, outputs_for_metrics)
        iou = jaccard_score(labels_for_metrics, outputs_for_metrics)
        val_loss = np.mean(losses)

        print(
            f'Linear Validation - Epoch: {epoch} '
            f'Accuracy: {accuracy:.8f} '
            f'Avg Precision: {avg_precision:.8f} '
            f'Balanced Accuracy: {bacc:.8f} '
            f'Precision: {precision:.8f} '
            f'Recall: {recall:.8f} '
            f'Dice/F1: {dice:.8f} '
            f'IoU/Jaccard: {iou:.8f} '
            f'Avg Loss: {val_loss:.8f}')

        if wandb_available:
            wandb.log(
                {
                    'Linear Train Loss': train_loss,
                    'Linear Accuracy': accuracy,
                    'Linear Average Precision': avg_precision,
                    'Linear Balanced Accuracy': bacc,
                    'Linear Precision': precision,
                    'Linear Recall': recall,
                    'Linear Dice/F1': dice,
                    'Linear IoU/Jaccard': iou,
                    'Linear Validation Loss': val_loss})

        save_path = ''

        if (
            dice > highest_dice or
            bacc > highest_bacc or
            iou > highest_iou or
            val_loss < lowest_loss
        ):
            save_path = f"{kwargs['model_savename']}_linear_{epoch}"

            if dice > highest_dice:
                save_path += f'_dice={dice}'
                highest_dice = dice

            if bacc > highest_bacc:
                save_path += f'_bacc={bacc}'
                highest_bacc = bacc

            if iou > highest_iou:
                save_path += f'_iou={iou}'
                highest_iou = iou

            if val_loss < lowest_loss:
                save_path += f'_loss={val_loss}'
                lowest_loss = val_loss

            save_path = os.path.join(kwargs['outdir_path'], save_path + '.pth')
            best_save_path = save_path

        if save_path:
            if 'save_whole' in kwargs and kwargs['save_whole']:
                torch.save(linear_classifier, save_path)
            else:
                torch.save(linear_classifier.state_dict(), save_path)

    labels_for_metrics = []
    outputs_for_metrics = []
    losses = []
    highest_bacc, highest_dice, highest_iou, lowest_loss = 0, 0, 0, 100
    linear_classifier.load_state_dict(
        torch.load(best_save_path, map_location=device).state_dict(),
        strict=False)

    for batch, labels, external in test_loader:
        with torch.no_grad():
            batch = batch.to(device=device)
            labels = labels.to(device=device)
            labels_for_metrics.append(labels.cpu().numpy())
            output = model.return_intermediate_repr(batch)
            output = torch.cat([output, external], dim=1)
            output = linear_classifier(output)
            outputs_for_metrics.append(output.cpu().detach().numpy())
            loss_out = loss(output, labels).cpu().numpy()
            losses.append(loss_out)

    labels_for_metrics = np.concatenate(
        labels_for_metrics, axis=0).reshape(-1, 1)
    outputs_for_metrics = (
        np.concatenate(outputs_for_metrics, axis=0) >= 0.5).reshape(-1, 1)
    accuracy = accuracy_score(labels_for_metrics, outputs_for_metrics)
    avg_precision = average_precision_score(
        labels_for_metrics, outputs_for_metrics)
    bacc = balanced_accuracy_score(labels_for_metrics, outputs_for_metrics)
    precision = precision_score(labels_for_metrics, outputs_for_metrics)
    recall = recall_score(labels_for_metrics, outputs_for_metrics)
    dice = f1_score(labels_for_metrics, outputs_for_metrics)
    iou = jaccard_score(labels_for_metrics, outputs_for_metrics)
    test_loss = np.mean(losses)

    print(
        f'Linear Test - Epoch: {epoch} '
        f'Accuracy: {accuracy:.8f} '
        f'Avg Precision: {avg_precision:.8f} '
        f'Balanced Accuracy: {bacc:.8f} '
        f'Precision: {precision:.8f} '
        f'Recall: {recall:.8f} '
        f'Dice/F1: {dice:.8f} '
        f'IoU/Jaccard: {iou:.8f} '
        f'Avg Loss: {test_loss:.8f}')
