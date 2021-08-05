import importlib
import numpy as np
import os
import scipy.ndimage
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from datasets.chromatograms_dataset import Subset
from optimizers.focal_loss import FocalLossBinary
from utils.general_utils import overlaps


def get_data_loaders(
    data,
    test_batch_proportion=0.1,
    use_weak_labels=False,
    eval_by_cla=True,
    batch_size=1,
    u_ratio=7,
    sampling_fn=None,
    collate_fn=None,
    outdir_path=None
):
    # Currently only LoadingSampler returns 4 sets of idxs
    if sampling_fn:
        labeled_idx, unlabeled_idx, val_idx, test_idx = sampling_fn(
            data, test_batch_proportion)
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
            os.path.join(outdir_path, 'test_idx.txt'),
            np.array(test_idx),
            fmt='%i')

    labeled_set = Subset(data, labeled_idx, use_weak_labels)
    unlabeled_set = Subset(data, unlabeled_idx, False)
    val_set = Subset(data, val_idx, eval_by_cla)
    test_set = Subset(data, test_idx, False)

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
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
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
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size)

    return labeled_loader, unlabeled_loader, val_loader, test_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train(
    data,
    model,
    optimizer=None,
    scheduler=None,
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
        test_loader
    ) = get_data_loaders(
            data,
            kwargs['test_batch_proportion'],
            kwargs['use_weak_labels'],
            kwargs['eval_by_cla'],
            kwargs['batch_size'],
            kwargs['uratio'],
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
                job_type='train-semisupervised',
                config=kwargs)

    if not optimizer:
        optimizer = AdamW(model.parameters())

    if not loss:
        loss = FocalLossBinary()

    if not scheduler:
        scheduler = CosineAnnealingWarmRestarts(optimizer, 10)

    if 'transfer_model_path' in kwargs:
        model.load_state_dict(
            torch.load(kwargs['transfer_model_path']).state_dict(),
            strict=False,
            map_location=device)

    unlabeled_loader = iter(cycle(unlabeled_loader))
    highest_bacc, highest_dice, highest_iou, lowest_loss = 0, 0, 0, 100
    model.to(device)

    for epoch in range(kwargs['max_epochs']):
        iters, train_loss = 0, 0
        model.train()

        for labeled_batch, labels in labeled_loader:
            unlabeled_batch, _ = next(unlabeled_loader)
            labeled_batch = labeled_batch.to(device=device)
            labels = labels.to(device=device)
            unlabeled_batch = unlabeled_batch.to(device=device)
            optimizer.zero_grad()
            loss_out = model(unlabeled_batch, labeled_batch, labels)
            loss_out.backward()
            optimizer.step()

            if ('scheduler_step_on_iter' in kwargs and
                kwargs['scheduler_step_on_iter']):
                scheduler.step(epoch + iters / kwargs['max_epochs'])

            iters += 1
            iter_loss = loss_out.item()
            train_loss += iter_loss
            print(f'Training - Iter: {iters} Iter Loss: {iter_loss:.8f}')

        if not ('scheduler_step_on_iter' in kwargs and
                kwargs['scheduler_step_on_iter']):
            scheduler.step(epoch + iters / kwargs['max_epochs'])

        train_loss = train_loss / iters
        print(f'Training - Epoch: {epoch} Avg Loss: {train_loss:.8f}')

        labels_for_metrics = []
        outputs_for_metrics = []
        losses = []
        num_pos, num_neg = 0, 0
        model.eval()
        orig_output_mode, model.model.output_mode = (
            model.model.output_mode, 'all')

        for batch, labels in val_loader:
            with torch.no_grad():
                batch = batch.to(device=device)
                labels = labels.to(device=device)
                labels_for_metrics.append(labels.cpu().numpy())
                preds = model(batch)
                strong_preds = preds['loc']
                num_pos += np.sum(strong_preds.cpu().detach().numpy() >= 0.5)
                num_neg += strong_preds.numel() - num_pos
                weak_preds = preds['cla']
                loss_preds = None

                if labels_for_metrics[-1].size == strong_preds.numel():
                    loss_preds = strong_preds
                    outputs_for_metrics.append(
                        loss_preds.cpu().detach().numpy())
                else:
                    loss_preds = weak_preds
                    outputs_for_metrics.append(
                        loss_preds.cpu().detach().numpy())

                loss_out = loss(loss_preds, labels).cpu().numpy()
                losses.append(loss_out)
        
        labels_for_metrics = np.concatenate(
            labels_for_metrics, axis=0).reshape(-1, 1)
        outputs_for_metrics = (
            np.concatenate(outputs_for_metrics, axis=0) >= 0.5).reshape(-1, 1)
        model.model.output_mode = orig_output_mode
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
            f'Validation - Epoch: {epoch} '
            f'Accuracy: {accuracy:.8f} '
            f'Balanced Accuracy: {bacc:.8f} '
            f'Precision: {precision:.8f} '
            f'Recall: {recall:.8f} '
            f'Dice/F1: {dice:.8f} '
            f'IoU/Jaccard: {iou:.8f} '
            f'Avg Loss: {val_loss:.8f} '
            f'Positive Pixel Count: {num_pos} '
            f'Negative Pixel Count: {num_neg}')

        if wandb_available:
            wandb.log(
                {
                    'Train Loss': train_loss,
                    'Accuracy': accuracy,
                    'Average Precision': avg_precision,
                    'Balanced Accuracy': bacc,
                    'Precision': precision,
                    'Recall': recall,
                    'Dice/F1': dice,
                    'IoU/Jaccard': iou,
                    'Validation Loss': val_loss})

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

        if save_path:
            if 'save_whole' in kwargs and kwargs['save_whole']:
                torch.save(model, save_path)
            else:
                torch.save(model.state_dict(), save_path)

    y_true, y_pred, y_score = [], [], []
    gt, masks = [], []
    model.eval()
    orig_output_mode, model.model.output_mode = (
        model.model.output_mode, 'all')

    for batch, labels in test_loader:
        with torch.no_grad():
            batch = batch.to(device=device)
            strong_labels = labels.to(device=device)
            weak_labels = torch.max(strong_labels, dim=1)[0].cpu().numpy()
            strong_labels = strong_labels.cpu().numpy()
            gt.append(strong_labels)
            negative = 1 - weak_labels
            preds = model(batch)
            strong_preds = preds['loc']
            weak_preds = preds['cla']

            if (
                kwargs['use_weak_labels']
                or kwargs['enforce_weak_consistency']
            ):
                strong_preds = strong_preds * weak_preds

            strong_preds = strong_preds.cpu().numpy()
            weak_preds = weak_preds.cpu().numpy()
            binarized_preds = np.where(
                strong_preds >= 0.5, 1, 0).astype(np.int32)
            inverse_binarized_preds = (1 - binarized_preds)
            mask = np.zeros(binarized_preds.shape)

            for i in range(len(strong_preds)):
                if 'fill_gaps' in kwargs and kwargs['fill_gaps']:
                    gaps = scipy.ndimage.find_objects(
                        scipy.ndimage.label(inverse_binarized_preds[i])[0])

                    for gap in gaps:
                        gap = gap[0]
                        gap_length = gap.stop - gap.start

                        if gap_length < 3:
                            binarized_preds[i][gap.start:gap.stop] = 1

                label_left_width, label_right_width = None, None

                if not negative[i]:
                    label_idx = np.argwhere(
                        strong_labels[i] == 1).astype(np.int32).ravel()
                    label_left_width, label_right_width = (
                        label_idx[0], label_idx[-1])

                regions_of_interest = scipy.ndimage.find_objects(
                    scipy.ndimage.label(binarized_preds[i])[0])
                regions_of_interest = [
                    roi[0] for roi in regions_of_interest
                    if 3 <= roi[0].stop - roi[0].start <= 36]
                overlap_found = False

                if negative[i] and not regions_of_interest:
                    # True Negative
                    y_true.append(0)
                    y_pred.append(0)
                    y_score.append(0)

                for roi in regions_of_interest:
                    mod_left_width, mod_right_width = None, None
                    score = np.max(strong_preds[i][roi.start:roi.stop])
                    mask[i][roi.start:roi.stop] = 1
                    mod_left_width, mod_right_width = (
                        roi.start, roi.stop - 1)

                    if negative[i] or not overlaps(
                        mod_left_width,
                        mod_right_width + 1,
                        label_left_width,
                        label_right_width + 1,
                        iou_threshold=kwargs['iou_threshold']
                    ):
                        # False Positive
                        y_true.append(0)
                    else:
                        # True Positive
                        y_true.append(1)
                        overlap_found = True

                    y_pred.append(1)
                    y_score.append(score)

                if not negative[i] and not overlap_found:
                    # False Negative
                    label_region_score = np.max(
                        strong_preds[i][
                            label_left_width:label_right_width + 1])
                    y_true.append(1)
                    y_pred.append(0)
                    y_score.append(label_region_score)

            masks.append(mask)

    model.model.output_mode = orig_output_mode
    y_true, y_pred, y_score = (
        np.array(y_true, dtype=np.int32),
        np.array(y_pred, dtype=np.int32),
        np.array(y_score, dtype=np.float32))
    gt = np.concatenate(gt, axis=0).reshape(-1, 1)
    masks = np.concatenate(masks, axis=0).reshape(-1, 1)
    accuracy = accuracy_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_score)
    bacc = balanced_accuracy_score(
        y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    dice = f1_score(gt, masks)
    iou = jaccard_score(gt, masks)

    print(
        f'Test - Epoch: {epoch} '
        f'RoI Accuracy: {accuracy:.4f} '
        f'RoI Avg Precision: {avg_precision:.4f} '
        f'RoI Balanced Accuracy: {bacc:.4f} '
        f'RoI Precision: {precision:.4f} '
        f'RoI Recall: {recall:.4f} '
        f'Pixel Dice/F1: {dice:.4f} '
        f'Pixel IoU/Jaccard: {iou:.4f}')
