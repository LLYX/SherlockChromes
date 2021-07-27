import numpy as np
import torch

class PadChromatogramsForLSTM(object):
    """Pad chromatograms in LSTM batch to same length.
    Chromatograms shaped (N, D), where N is the length of the chromatogram,
    and D is the number of traces.
    Labels shaped (N, 1).
    """

    def __call__(self, batch):
        batch_size = len(batch)
        chromatograms = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        lengths = [len(chromatogram) for chromatogram in chromatograms]
        max_len = max(lengths)
        trailing_dims = chromatograms[0].size()[1:]
        out_dims = (max_len, batch_size) + trailing_dims
        padded_chromatograms = torch.zeros(*out_dims)
        padded_labels = torch.zeros(batch_size, max_len)

        for i, chromatogram in enumerate(chromatograms):
            length = chromatogram.size(0)
            padded_chromatograms[max_len - length:, i, ...] = chromatogram

        for i, label in enumerate(labels):
            length = label.size(0)
            padded_labels[i, max_len - length:] = label

        return [padded_chromatograms, padded_labels]

class PadChromatogramsFor1DCNN(object):
    """Pad whole chromatograms in 1DCNN batch to same size.
    Subsections shaped (D, N).
    Labels shaped (D, *), where * represents the number of labelled
    subsections.
    """

    def __call__(self, batch):
        batch_size = len(batch)
        chromatograms = [item[0] for item in batch]
        lengths = [chromatogram.size()[1] for chromatogram in chromatograms]
        max_len = max(lengths)
        channel_dim = chromatograms[0].size()[0]
        out_dims = (batch_size, channel_dim, max_len)
        padded_chromatograms = torch.zeros(*out_dims)

        for i, chromatogram in enumerate(chromatograms):
            length = chromatogram.size(1)
            padded_chromatograms[i, 0:channel_dim, 0:length] = chromatogram

        labels = [item[1] for item in batch]
        label_lengths = [label.size()[0] for label in labels]
        max_label_len = max(label_lengths)
        padded_labels = torch.zeros(batch_size, max_label_len)

        for i, label in enumerate(labels):
            length = label.size(0)
            padded_labels[i, 0:length] = label

        return [padded_chromatograms, padded_labels]

class PadChromatogramsOnlyFor1DCNN(object):
    """Pad whole chromatograms in 1DCNN batch to same size.
    Subsections shaped (D, N).
    Assumes second item in batch is a pair of bounding box boundaries.
    """

    def __call__(self, batch):
        batch_size = len(batch)
        
        assert batch_size == 1, 'batch_size=1 support only currently'

        chromatograms = [item[0] for item in batch]
        bboxes = [item[1] for item in batch]
        lengths = [chromatogram.size()[1] for chromatogram in chromatograms]
        max_len = max(lengths)
        channel_dim = chromatograms[0].size()[0]
        out_dims = (batch_size, channel_dim, max_len)
        padded_chromatograms = torch.zeros(*out_dims)

        for i, chromatogram in enumerate(chromatograms):
            length = chromatogram.size(1)
            padded_chromatograms[i, 0:channel_dim, 0:length] = chromatogram

        bboxes = bboxes[0].tolist()

        return [padded_chromatograms, bboxes]
