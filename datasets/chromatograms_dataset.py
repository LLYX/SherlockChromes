import h5py
import numpy as np
import os
import pandas as pd
import tarfile
import torch

from torch.utils.data import Dataset


class ExternalDataset(Dataset):
    """
    Modify a dataset with additional numpy array.

    Args:
        dataset (Dataset): The original Dataset.
        external_path (string): Path to external numpy array.
    """

    def __init__(self, dataset, external_path):
        self.dataset = dataset
        self.external = torch.from_numpy(np.load(external_path)).float()
    
    def __getitem__(self, idx):
        data, labels = self.dataset[idx]

        if hasattr(self.dataset, 'indices'):
            idx = self.dataset.indices[idx]
        elif hasattr(self.dataset, 'chromatograms_csv'):
            idx = self.dataset.chromatograms_csv.iloc[idx, 0]

        external = self.external[idx]

        return data, labels, external
    
    def __len__(self):
        return len(self.dataset)


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset.
        indices (sequence): Indices in the whole set selected for subset.
        load_weak_labels (bool): Whether to load weak labels or not.
    """

    def __init__(self, dataset, indices, load_weak_labels):
        self.dataset = dataset
        self.indices = indices
        self.load_weak_labels = load_weak_labels

    def __getitem__(self, idx):
        self.dataset.load_weak_labels = self.load_weak_labels
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ChromatogramsDataset(Dataset):
    """Whole Chromatograms dataset with point labels."""

    def __init__(
        self,
        root_path,
        chromatograms,
        labels=None,
        extra_path=None,
        transform=None,
        preload=False
    ):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram
                filenames.
            labels (string): Filename of the npy file with labels.
            extra_path (string, optional): Path to folder containing additional
                traces.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        if labels:
            self.labels = np.load(os.path.join(self.root_dir, labels))
        else:
            self.labels = False

        self.extra_dir = extra_path
        self.transform = transform
        self.preload = preload

        if self.preload:
            npys = []
            for i in range(len(self.chromatograms)):
                npys.append(self.load_chromatogram(i))

            self.chromatogram_npy = np.array(npys, dtype=np.float32)

    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id = self.chromatograms.iloc[idx, 0]

        if self.preload:
            chromatogram = self.chromatogram_npy[chromatogram_id]
        else:
            chromatogram = self.load_chromatogram(idx)

        if isinstance(self.labels, np.ndarray):
            label = self.labels[chromatogram_id].astype(float)
        else:
            label = np.zeros(chromatogram.shape[1])

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label

    def load_npy(self, npy_names):
        npy = np.load(npy_names[0]).astype(float)

        if len(npy_names) > 1:
            for i in range(1, len(npy_names)):
                npy = np.concatenate(
                    (npy, np.load(npy_names[i]).astype(float)), axis=0)

        return npy

    def load_chromatogram(self, idx):
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 1]) + '.npy'

        npy_names = [chromatogram_name]

        if self.extra_dir:
            extra_name = os.path.join(
                self.extra_dir,
                self.chromatograms.iloc[idx, 1]) + '_Extra.npy'

            npy_names.append(extra_name)

        chromatogram = self.load_npy(npy_names)

        return chromatogram

    def get_bb(self, idx):
        bb_start, bb_end = \
            self.chromatograms.iloc[idx, 2], self.chromatograms.iloc[idx, 3]

        return bb_start, bb_end


# TODO: Unfinished!!!
class HDF5ChromatogramsDataset(Dataset):
    """Whole Chromatograms HDF5 dataset with point labels."""

    def __init__(
        self,
        root_path,
        chromatograms,
        hdf5,
        labels=None,
        preload=False,
        transform=None,
        extra_features=[]
    ):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram
                filenames.
            hdf5 (string): Filename of the HDF5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.hdf5 = h5py.File(os.path.join(self.root_dir, hdf5), 'r')
        self.extra_features = extra_features
        self.transform = transform

        if not labels:
            self.labels = False
        elif '.npy' in labels:
            self.labels = np.load(os.path.join(self.root_dir, labels))
        else:
            self.labels = self.hdf5[labels][:]

        if self.preload:
            npys = []
            for i in range(len(self.chromatograms)):
                npys.append(self.load_chromatogram(i))

            self.chromatogram_npy = np.array(npys, dtype=np.float32)

    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id = self.chromatograms.iloc[idx, 0]

        if self.preload:
            chromatogram = self.chromatogram_npy[chromatogram_id]
        else:
            chromatogram = self.load_chromatogram(idx)

        if isinstance(self.labels, np.ndarray):
            label = self.labels[chromatogram_id].astype(float)
        else:
            label = np.zeros(chromatogram.shape[1])

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label

    def load_dataset(self, name, extra=False):
        dataset = self.hdf5['first_level'][name][:]

        return dataset

    def load_chromatogram(self, idx):
        chromatogram_name = self.chromatograms.iloc[idx, 1]
        chromatogram = self.load_dataset(chromatogram_name)

        return chromatogram

    def get_bb(self, idx):
        bb_start, bb_end = \
            self.chromatograms.iloc[idx, 2], self.chromatograms.iloc[idx, 3]

        return bb_start, bb_end


class NpyChromatogramsDataset(Dataset):
    """Whole Chromatograms npy dataset with point labels."""

    def __init__(
        self,
        root_path,
        chromatograms_csv,
        chromatograms_npy,
        labels=None,
        weak_labels=None,
        load_weak_labels=False,
        memmap=False,
        num_features=497,
        transform=None
    ):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms_csv (string): Filename of CSV with chromatogram
                filenames.
            chromatograms_npy (string): Filename of npy file with chromatogram
                data.
            labels (string, optional): Filename of the npy file with labels.
            weak_labels (string, optional): Filename of the npy file with weak
                labels.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
            memmap (bool, optional): To use memory mapping or not when loading.
        """
        self.root_dir = root_path
        self.chromatograms_csv = pd.read_csv(os.path.join(self.root_dir,
                                             chromatograms_csv))

        if memmap:
            rows = len(self.chromatograms_csv)
            cols = self.chromatograms_csv.iloc[0, 4]

        chromatograms_filename = os.path.join(self.root_dir, chromatograms_npy)

        if memmap:
            self.chromatograms_npy = np.memmap(
                chromatograms_filename,
                dtype='float32',
                mode='r',
                shape=(rows, num_features, cols))
        else:
            self.chromatograms_npy = np.load(chromatograms_filename)

        if labels:
            labels_filename = os.path.join(self.root_dir, labels)

            if memmap:
                self.labels = np.memmap(
                    labels_filename,
                    dtype='float32',
                    mode='r',
                    shape=(rows, cols))
            else:
                self.labels = np.load(labels_filename)
        else:
            self.labels = False

        if weak_labels:
            weak_labels_filename = os.path.join(self.root_dir, weak_labels)

            if memmap:
                self.weak_labels = np.memmap(
                    weak_labels_filename,
                    dtype='int32',
                    mode='r',
                    shape=(rows, 1))
            else:
                self.weak_labels = np.load(weak_labels_filename)
        else:
            self.weak_labels = False

        self.load_weak_labels = load_weak_labels and weak_labels
        self.transform = transform

    def __len__(self):
        return len(self.chromatograms_csv)

    def __getitem__(self, idx):
        chromatogram_id = self.chromatograms_csv.iloc[idx, 0]
        chromatogram = self.chromatograms_npy[chromatogram_id]

        if self.load_weak_labels:
            label = self.weak_labels[chromatogram_id].astype(float)
        elif isinstance(self.labels, np.ndarray):
            label = self.labels[chromatogram_id].astype(float)
        else:
            label = np.zeros(chromatogram.shape[1])

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label

    def get_bb(self, idx):
        bb_start, bb_end = (
            self.chromatograms_csv.iloc[idx, 5],
            self.chromatograms_csv.iloc[idx, 6])

        return bb_start, bb_end


class TarChromatogramsDataset(Dataset):
    """Whole Chromatograms tar dataset with point labels."""

    def __init__(
        self,
        root_path,
        chromatograms,
        tar,
        tar_shape=(6, 175),
        labels='osw_labels',
        internal_extra=False,
        internal_extra_shape=(8, 175),
        internal_extra_features=[],
        external_extra_paths=[],
        external_extra_features=[],
        preload=False,
        preload_path='',
        transform=None
    ):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram
                filenames.
            tar (string): Filename of the tar file.
            tar_shape (tuple of int): Shape of individual chromatogram
                (num_traces, total_len).
            labels (string): Filename of labels npy file.
            internal_extra (bool): Whether tar contains extra traces or not.
            internal_extra_shape (tuple of int): Shape of individual extra
                arrays (num_extra_traces, total_len).
            internal_extra_features (list of int): Array of indices in
                internal extra array.
            external_extra_paths (list of str): Path to folder with extra
                features.
            external_extra_features (list of list of int): List of list of
                indices in extra array.
            preload (bool): To load entire dataset into memory or not.
            preload_path (str): Path to entire dataset npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.tar = tarfile.open(os.path.join(self.root_dir, tar), 'r')
        self.tar_shape = tuple(tar_shape)
        self.internal_extra = internal_extra
        self.internal_extra_shape = tuple(internal_extra_shape)
        self.internal_extra_features = internal_extra_features
        self.external_extra_paths = external_extra_paths
        self.external_extra_features = external_extra_features
        self.preload = preload
        self.transform = transform

        if not labels:
            self.labels = None
        elif '.npy' in labels:
            self.labels = np.load(os.path.join(self.root_dir, labels))
        else:
            extracted = self.tar.extractfile(f'{labels}')
            self.labels = np.frombuffer(
                extracted.read(),
                dtype=np.int64
            ).reshape((-1, self.tar_shape[1]))

        if self.preload:
            if os.path.exists(preload_path):
                self.chromatogram_npy = np.load(preload_path)
            else:
                channels = self.tar_shape[0] + len(
                    self.internal_extra_features)
                for i in range(len(self.external_extra_paths)):
                    channels += len(self.external_extra_features[i])

                self.chromatogram_npy = np.zeros(
                    (len(self.chromatograms), channels, self.tar_shape[1]))

                for i in range(len(self.chromatograms)):
                    self.chromatogram_npy[i] = self.load_chromatogram(i)

                np.save(preload_path, self.chromatogram_npy)

    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id = self.chromatograms.iloc[idx, 0]

        if self.preload:
            chromatogram = self.chromatogram_npy[chromatogram_id]
        else:
            chromatogram = self.load_chromatogram(idx)

        if isinstance(self.labels, np.ndarray):
            label = self.labels[chromatogram_id].astype(float)
        else:
            label = np.zeros(chromatogram.shape[1])

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label

    def load_dataset(self, name):
        extracted = self.tar.extractfile(f'{name}')
        dataset = np.frombuffer(
            extracted.read(),
            dtype=np.float64
        ).reshape(self.tar_shape)

        if not self.internal_extra and len(self.external_extra_paths) < 1:
            return dataset

        dataset = [dataset]
        if self.internal_extra:
            extracted = self.tar.extractfile(f'{name}_Extra')
            internal_extra = np.frombuffer(
                extracted.read(),
                dtype=np.float64
            ).reshape(self.internal_extra_shape)[self.internal_extra_features]

            dataset.append(internal_extra)

        if len(self.external_extra_paths) > 0:
            for i in range(len(self.external_extra_paths)):
                dataset.append(
                    np.load(
                        os.path.join(
                            self.external_extra_paths[i],
                            f'{name}_Extra.npy'
                        )
                    ).astype(float)[self.external_extra_features[i]]
                )

        dataset = np.concatenate(dataset, axis=0)

        return dataset

    def load_chromatogram(self, idx):
        chromatogram_name = self.chromatograms.iloc[idx, 1]
        chromatogram = self.load_dataset(chromatogram_name)

        return chromatogram

    def get_bb(self, idx):
        bb_start, bb_end = \
            self.chromatograms.iloc[idx, 2], self.chromatograms.iloc[idx, 3]

        return bb_start, bb_end


class ChromatogramsBBoxDataset(Dataset):
    """Whole Chromatograms dataset with bounding boxes."""

    def __init__(self, root_path, chromatograms, transform=None):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram
                filenames with bounding boxes.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.transform = transform

    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id = self.chromatograms.iloc[idx, 0]
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 1]) + '.npy'
        chromatogram = np.load(chromatogram_name).astype(float)
        bb_start, bb_end = \
            self.chromatograms.iloc[idx, 2], self.chromatograms.iloc[idx, 3]
        bbox = (bb_start, bb_end)

        if self.transform:
            chromatogram, bbox = self.transform((chromatogram, bbox))

        return chromatogram, bbox


class ChromatogramSubsectionsDataset(Dataset):
    """Chromatogram Subsections dataset."""

    def __init__(self, root_path, chromatograms, transform=None):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram filenames
                and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.labels = [self.chromatograms.iloc[idx, 2] for idx in range(
            len(self.chromatograms))]
        self.transform = transform

    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 1]) + '.npy'
        chromatogram = np.load(chromatogram_name).astype(float)
        label = self.labels[idx]

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label


class ChromatogramSubsectionsInMemoryDataset(Dataset):
    """Chromatogram Subsections In Memory dataset."""

    def __init__(
        self, root_path, chromatograms, chromatograms_npy, transform=None
    ):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram
            subsection counter, chromatogram id, start pos, end pos, and
            subsection label.
            chromatograms_npy (string): Filename of npy file with whole
            chromatograms.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.labels = [self.chromatograms.iloc[idx, 4] for idx in range(
            len(self.chromatograms))]
        self.chromatograms_npy = np.load(
            os.path.join(self.root_dir, chromatograms_npy) + '.npy')
        self.transform = transform

    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id, start, end = (
            self.chromatograms.iloc[idx, 1],
            self.chromatograms.iloc[idx, 2],
            self.chromatograms.iloc[idx, 3])
        subsection = self.chromatograms_npy[chromatogram_id][:, start:end]
        label = self.labels[idx]

        if self.transform:
            subsection, label = self.transform((subsection, label))

        return subsection, label
