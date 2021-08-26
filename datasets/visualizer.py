import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import sys
import torch

from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve)

from chromatograms_dataset import NpyChromatogramsDataset


def plot_whole_chromatogram(
    chromatogram_id,
    chromatogram,
    labels=None,
    bb_start=None,
    bb_end=None,
    threshold=0.5,
    num_traces=6,
    mode='subsection',
    width=30
):
    """
    Args:
        chromatogram_id (str): chromatogram id in dataset
        chromatogram: (len_of_chromatogram, num_dimensions)
        labels: (len_of_chromatogram, )
        bb_start (int): idx of label bounding box start
        bb_end (int): idx of label bounding box end
        threshold (float): value to threshold label probabilities
        width (int): width of evaluation window of model
    """
    # Convert from single multivariate time series
    # back to separate single variate time series
    # with shape (num_dimensions, len_of_chromatogram)
    if chromatogram.shape[0] == max(chromatogram.shape):
        chromatogram = chromatogram.T

    traces, timepoints, intensities = [], [], []

    for trace in range(num_traces):
        for timepoint in range(len(chromatogram[trace])):
            traces.append('chromatogram_' + str(trace))
            timepoints.append(timepoint)
            intensities.append(chromatogram[trace][timepoint])

    df = pd.DataFrame(
        {'trace': traces, 'timepoint': timepoints, 'intensity': intensities})

    by_trace = df.groupby('trace')

    plt.subplot(211)

    plt.title('chromatogram_id=' + chromatogram_id)

    for trace, group in by_trace:
        plt.plot(group['timepoint'], group['intensity'], label=trace)

    if labels is not None:
        max_idx = np.argmax(labels)

        if labels[max_idx] >= threshold:
            if mode == 'subsection':
                plt.axvline(max_idx, color='r')
                plt.axvline(max_idx + width, color='r')
            elif mode == 'point':
                binarized_labels = np.where(labels >= threshold, 1, 0)

                regions_of_interest = scipy.ndimage.find_objects(
                    scipy.ndimage.label(binarized_labels)[0])

                if regions_of_interest:
                    scores = [np.sum(labels[r])
                              for r
                              in regions_of_interest]
                    best_region_idx = np.argmax(scores)
                    best_region = regions_of_interest[best_region_idx][0]

                    start_idx, end_idx = best_region.start, best_region.stop

                    plt.axvline(start_idx, color='r')
                    plt.axvline(end_idx, color='r')

    if bb_start and bb_end:
        plt.axvline(bb_start, color='b')
        plt.axvline(bb_end, color='b')

    plt.legend()

    plt.subplot(212)

    if mode == 'subsection':
        plt.plot(
            np.pad(labels, (15, 0), 'constant'), label='raw network output')
    elif mode == 'point':
        plt.plot(labels, label='raw network output')

    plt.legend()

    plt.show()


def plot_chromatogram_subsection(chromatogram, labels=None):
    """
    Args:
        chromatogram: (len_of_chromatogram, num_dimensions)
        labels: (len_of_chromatogram, )
    """
    # Convert from single multivariate time series
    # back to separate single variate time series
    # with shape (num_dimensions, len_of_chromatogram)
    if chromatogram.shape[0] == max(chromatogram.shape):
        chromatogram = chromatogram.T

    traces, timepoints, intensities = [], [], []

    for trace in range(len(chromatogram)):
        for timepoint in range(len(chromatogram[trace])):
            traces.append('chromatogram_' + str(trace))
            timepoints.append(timepoint)
            intensities.append(chromatogram[trace][timepoint])

    df = pd.DataFrame(
        {'trace': traces, 'timepoint': timepoints, 'intensity': intensities})

    by_trace = df.groupby('trace')

    for trace, group in by_trace:
        plt.plot(group['timepoint'], group['intensity'], label=trace)

    if labels:
        labels = list(labels)

        peak_start, peak_end = (
            labels.index(1) - 1, len(labels) - labels[::-1].index(1))

        plt.axvline(peak_start, color='b')
        plt.axvline(peak_end, color='b')

    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

    return ax


def plot_binary_precision_recall_curve(
    y_true,
    y_pred,
    y_true_2=None,
    y_pred_2=None
):
    """
    This function plots a binary precision recall curve.
    """
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    plt.step(recall, precision, color='b', alpha=0.5, where='post', label='1')

    if y_true_2 and y_pred_2:
        average_precision_2 = average_precision_score(
            y_true_2, y_pred_2)
        precision_2, recall_2, thresholds_2 = precision_recall_curve(
            y_true_2, y_pred_2)

        plt.step(
            recall_2,
            precision_2,
            color='r',
            alpha=0.5,
            where='post',
            label='2')
    else:
        average_precision_2 = None

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks([i*0.05 for i in range(0, 21)])
    plt.yticks([i*0.05 for i in range(0, 21)])
    plt.grid()
    plt.legend(title='Inputs: ')
    plt.title(
        '2-class Precision-Recall curve: AP=1:{0:0.4f}/2:{1:0.4f}'.format(
            average_precision, average_precision_2))

    plt.show()

    return precision, recall, thresholds


def count_outcomes(
    dataset,
    model,
    root_dir,
    idx_filename,
    threshold=0.5,
    mode='val'
):
    """
    This function counts the number of instances of each possible outcome.
    """
    counter1, counter2, counter3, counter4 = 0, 0, 0, 0
    no_boxes, less_boxes, equal_boxes, more_boxes = [], [], [], []

    idxs = np.loadtxt(os.path.join(root_dir, idx_filename))
    for idx in idxs:
        idx = int(idx)
        chromatogram, labels = dataset[idx]
        dims = chromatogram.shape
        output = model(torch.from_numpy(
                    np.asarray(chromatogram)).view(1, *dims).float())[0]

        num_boxes = (labels == 1).sum()
        num_boxes_predicted = (output >= threshold).sum()
        if num_boxes_predicted == 0:
            counter1 += 1
            no_boxes.append(idx)
        elif 0 < num_boxes_predicted < num_boxes:
            counter2 += 1
            less_boxes.append(idx)
        elif num_boxes_predicted == num_boxes:
            counter3 += 1
            equal_boxes.append(idx)
        elif num_boxes_predicted > num_boxes:
            counter4 += 1
            more_boxes.append(idx)

    np.savetxt(
        os.path.join(root_dir, mode + '_no_boxes.txt'), np.array(no_boxes))
    np.savetxt(
        os.path.join(root_dir, mode + '_less_boxes.txt'), np.array(less_boxes))
    np.savetxt(
        os.path.join(
            root_dir, mode + '_equal_boxes.txt'), np.array(equal_boxes))
    np.savetxt(
        os.path.join(root_dir, mode + '_more_boxes.txt'), np.array(more_boxes))

    print(counter1, counter2, counter3, counter4)


def pseudo_binary_precision_recall(
    dataset,
    model,
    root_dir,
    idx_filename,
    mode='val'
):
    """
    This function calculates a pseudo binary precision recall curve by taking
    the score and label of the highest value from each chromatogram.
    """
    y_true, y_pred = [], []

    idxs = np.loadtxt(os.path.join(root_dir, idx_filename))
    for idx in idxs:
        idx = int(idx)
        chromatogram, labels = dataset[idx]
        dims = chromatogram.shape
        output = model(torch.from_numpy(
                    np.asarray(
                        chromatogram)).view(1, *dims).float())[0]
        output = output.detach().numpy()

        positive_idxs = np.where(labels == 1)[0]

        if len(positive_idxs) > 0:
            largest_positive_idx = positive_idxs[np.argmax(
                output[positive_idxs])]

            largest_positive_y_true = labels[largest_positive_idx]
            largest_positive_y_pred = output[largest_positive_idx]

            y_true.append(largest_positive_y_true)
            y_pred.append(largest_positive_y_pred)

            largest_idx = np.argmax(output)

            if largest_idx != largest_positive_idx:
                y_true.append(labels[largest_idx])
                y_pred.append(np.amax(output))
        else:
            y_true.append(labels[np.argmax(output)])
            y_pred.append(np.amax(output))

    precision, recall, thresholds = plot_binary_precision_recall_curve(
        y_true, y_pred)

    np.savetxt(os.path.join(root_dir, mode + '_precision.txt'), precision)
    np.savetxt(os.path.join(root_dir, mode + '_recall.txt'), recall)
    np.savetxt(os.path.join(root_dir, mode + '_thresholds.txt'), thresholds)


def plot_layer_output(x, layer, extra=[], extra_labels=[]):
    dims = x.shape
    output = layer(
        torch.from_numpy(
            np.asarray(x)).view(*dims).float())[0]
    channels, length = output.shape

    plt.figure()

    for channel in range(channels):
        plt.subplot(channels + len(extra), 1, channel + 1)
        plt.plot(
            output.numpy()[channel, :], label='transition_' + str(channel))

    if extra:
        for i in range(len(extra)):
            plt.subplot(channels + i + 1, 1, channels + i + 1)
            plt.plot(extra[i], label=extra_labels[i])

    plt.legend()

    plt.show()


def test_model(dataset, model, mode='whole'):
    """
    This function provides an interactive command line interface for
    visualizing model testing and evaluation.
    """
    loop = True
    labels = dataset.labels
    label_mode = input('Mode [subsection, point]: ')

    if mode != 'whole':
        positives = np.where(labels == np.int64(1))[0]
        negatives = np.where(labels == np.int64(0))[0]

    num_traces = int(input('Number of traces to show: '))

    while loop:
        retrieve = input('Retrieve: ')

        idx = 0

        if mode != 'whole':
            if retrieve == 'positive':
                idx = np.random.choice(positives)
            elif retrieve == 'negative':
                idx = np.random.choice(negatives)
            else:
                idx = int(float(retrieve))
        else:
            idx = int(float(retrieve))

        chromatogram, true_label = dataset[idx]

        print('Chromatogram/subsection id: {}'.format(idx))

        if mode != 'whole':
            print('Chromatogram subsection label: {}'.format(true_label))

        with torch.no_grad():
            dims = chromatogram.shape

            if model.__class__.__name__ == 'SemiSupervisedLearner1d':
                plot_layer_output(
                    torch.from_numpy(
                        np.asarray(
                            chromatogram)).view(1, *dims).float(),
                    torch.nn.Sequential(*model.encoder))

                output = model(
                    torch.from_numpy(
                        np.asarray(chromatogram)).view(1, *dims).float())[0]
            elif model.__class__.__name__ == 'SelfSupervisedLearner1d':
                model.teacher.model.t_blocks[-1].attention.save_attn = True
                model.return_intermediate_repr(
                    torch.from_numpy(
                        np.asarray(chromatogram)).view(1, *dims).float())
                output = model.teacher.model.t_blocks[-1].attention.get_attn()
                output = output[0, 0, 0, 1:]

            output = output.numpy()

            if len(output.shape) > 1:
                output = output.reshape(output.shape[-1])

            if mode != 'whole':
                plot_chromatogram_subsection(chromatogram, output)
            else:
                threshold = float(input('Threshold val: '))

                if label_mode == 'subsection':
                    width = int(input('Eval window width: '))
                    plot_whole_chromatogram(
                        str(idx),
                        chromatogram,
                        output,
                        *dataset.get_bb(idx),
                        threshold=threshold,
                        num_traces=num_traces,
                        width=width)
                elif label_mode == 'point':
                    plot_whole_chromatogram(
                        str(idx),
                        chromatogram,
                        output,
                        *dataset.get_bb(idx),
                        threshold=threshold,
                        num_traces=num_traces,
                        mode=label_mode)

        break_loop = input("Enter 'break' to exit: ")
        if break_loop == 'break':
            break
        elif break_loop == 'reload':
            model_filename = input('Model pth: ')
            model = torch.load(model_filename)
            model.to('cpu')
            model.eval()


def create_results_file(
    root_dir,
    chromatograms_filename,
    dataset,
    model,
    save_dir,
    skyline_filename,
    results_filename,
    threshold=0.5,
    mode='subsection'
):
    chromatograms = pd.read_csv(os.path.join(
        root_dir, chromatograms_filename))

    skyline = pd.read_csv(os.path.join(
        save_dir, skyline_filename))

    model_bounding_boxes = \
        [
            [
                'Replicate Name',
                'Peptide Modified Sequence',
                'Precursor Charge',
                'Min Start Time',
                'Max End Time'
            ]
        ]

    for i in range(len(skyline)):
        row = skyline.iloc[i]

        if type(row['Replicate Name']) == str:
            key = '_'.join(
                [
                    row['Peptide Modified Sequence'],
                    row['Replicate Name'],
                    str(row['Precursor Charge'])
                ])

            idx = chromatograms.loc[chromatograms['Filename'] == key]['ID']
            if len(idx) > 1:
                idx = int(idx.iloc[0])
            else:
                idx = int(idx)

            chromatogram, labels = dataset[idx]
            dims = chromatogram.shape

            output = model(torch.from_numpy(
                        np.asarray(
                            chromatogram)).view(1, *dims).float())[0]
            output = output.detach().numpy()

            largest_idx = np.argmax(output)

            if output[largest_idx] >= threshold:
                if mode == 'subsection':
                    left_width = largest_idx * 0.0569
                    right_width = (largest_idx + 30) * 0.0569
                elif mode == 'point':
                    start_idx, end_idx = largest_idx, largest_idx

                    while output[start_idx - 1] >= threshold:
                        start_idx -= 1

                    while output[end_idx + 1] >= threshold:
                        end_idx += 1

                    left_width = start_idx * 0.0569
                    right_width = end_idx * 0.0569
            else:
                left_width, right_width = None, None

            model_bounding_boxes.append([
                    row['Replicate Name'],
                    row['Peptide Modified Sequence'],
                    str(row['Precursor Charge']),
                    left_width,
                    right_width])

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(save_dir, results_filename),
        index=False,
        header=False)


if __name__ == "__main__":
    # e.g. ../../../data/working/ManualValidationOpenSWATHWithExpRT
    root_dir = input('Dataset root dir: ')
    # e.g. chromatograms.csv
    chromatograms_filename = input('Dataset chromatograms csv: ')
    # e.g. osw_point_labels.npy
    labels_filename = input('Dataset labels npy: ')
    # e.g. hroest_Strep_600s_175pts.tar
    dataset = input('Dataset filename: ')
    num_features = int(input('Num features: '))
    dataset = NpyChromatogramsDataset(
        root_dir,
        chromatograms_filename,
        dataset,
        labels=labels_filename,
        num_features=num_features)

    # .pth file
    model_filename = input('Model pth: ')
    model = torch.load(model_filename)
    model.to('cpu')
    model.eval()

    if input('Count outcomes? [yes, no]: ') == 'yes':
        # e.g. ../../../data/output/custom_3_layer_21_kernel_osw_points_wrt
        root_dir = input('Root dir: ')
        threshold = float(input('Threshold val: '))
        # e.g. val_idx.txt
        idx_filename = input('Val Idx filename: ')
        count_outcomes(dataset, model, root_dir, idx_filename, threshold)
        # e.g. test_idx.txt
        idx_filename = input('Test Idx filename: ')
        count_outcomes(
            dataset, model, root_dir, idx_filename, threshold, mode='test')

    if input('Calculate pseudo binary precision recall? [yes, no]: ') == 'yes':
        # e.g. ../../../data/output/custom_3_layer_21_kernel_osw_points_wrt
        root_dir = input('Root dir: ')
        # e.g. val_idx.txt
        idx_filename = input('Val Idx filename: ')
        pseudo_binary_precision_recall(dataset, model, root_dir, idx_filename)
        # e.g. test_idx.txt
        idx_filename = input('Test Idx filename: ')
        pseudo_binary_precision_recall(
            dataset, model, root_dir, idx_filename, mode='test')

    test_model(dataset, model)

    if input('Create results file? [yes, no]: ') == 'yes':
        threshold = float(input('Input threshold value: '))
        modelname = input('Model designation: ')
        mode = input('Mode [subsection, point]: ')

        create_results_file(
            '../../../data/working/ManualValidationOpenSWATH',
            'chromatograms.csv',
            dataset,
            model,
            'D:/Workspace/SherlockChromes/data/working/',
            'SkylineResult500PeptidesProcessed.csv',
            modelname + 'Result500PeptidesProcessedThresholded.csv',
            threshold=threshold,
            mode=mode)
