import bisect
import csv
import glob
import math
import numpy as np
import os
import pandas as pd
import random
import sqlite3

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score)
from sklearn.model_selection import KFold


def calc_bin_idx(mz, min_mz, bin_resolution):
    bin_idx = math.floor((mz - min_mz) / bin_resolution)

    return bin_idx


def get_subsequence_idxs(sequence, value, subsequence_size=-1):
    value_idx = bisect.bisect_left(sequence, value)

    if subsequence_size < 0:
        return value_idx, None, None

    length = len(sequence)
    half_span = subsequence_size // 2
    subsequence_left, subsequence_right = (
        value_idx - half_span, value_idx + half_span + 1)

    if subsequence_left < 0:
        subsequence_left, subsequence_right = 0, subsequence_size
        value_idx = half_span
    elif subsequence_right >= length:
        subsequence_left = length - subsequence_size
        value_idx = length - half_span - 1

    return subsequence_left, subsequence_right


def merge_repl_files(pattern):
    repl_names = sorted([f.split('.')[0] for f in glob.glob(f'{pattern}*gz')])

    combined_csv = pd.concat(
        [pd.read_csv(f'{f}_chromatograms.csv') for f in repl_names],
        ignore_index=True)
    combined_csv.iloc[:, 0] = combined_csv.index
    combined_csv.to_csv(f'{pattern}_csv.csv', index=False)

    extensions = [
        '_chromatograms_array',
        '_segmentation_labels_array',
        '_classification_labels_array']

    for extension in extensions:
        combined_npy = []

        for repl in repl_names:
            combined_npy.append(np.load(f'{repl}{extension}.npy'))

        dtype = np.int32 if 'labels' in extension else np.float32
        np.save(f'{pattern}{extension}', np.concatenate(combined_npy).astype(
            dtype))


def overlaps(
    pred_min,
    pred_max,
    target_min,
    target_max,
    iou_threshold=0.5
):
    if not pred_min or not pred_max or not target_min or not target_max:
        return False

    intersection = min(pred_max, target_max) - max(pred_min, target_min)
    intersection = max(intersection, 0)
    union = (pred_max - pred_min) + (target_max - target_min) - intersection
    iou = intersection / union

    if iou >= iou_threshold:
        return True

    return False


def get_high_quality_training_labels(
    target_csv,
    og_train_idx_filename,
    new_train_idx_filename
):
    idxs = {}
    with open(og_train_idx_filename, 'r') as idx_file:
        for line in idx_file:
            line = line.rstrip('\r\n')
            line = str(int(float(line)))
            idxs[line] = True

    new_idxs = []
    with open(target_csv, 'r') as csv:
        next(csv)
        for line in csv:
            line = line.rstrip('\r\n').split(',')
            idx, filename, high_quality = line[0], line[1], line[10]
            high_quality = high_quality == '1'

            if idx in idxs and ('DECOY_' in filename or high_quality):
                new_idxs.append(int(idx))

    np.savetxt(new_train_idx_filename, np.array(new_idxs))


def parse_manual_annotation_file(manual_annotation_filename, rt_gap=3.4):
    filename_to_annotation = {}

    with open(manual_annotation_filename, 'r') as csv:
        next(csv)
        for line in csv:
            idx, filename, start, end, lib_rt, win_size = line.rstrip(
                '\r\n').split(',')

            win_size = int(win_size)
            lib_rt = float(lib_rt)
            lib_rt_idx = (win_size - 1) / 2

            if start and end:
                start, end = int(start), int(end)
                start_time = lib_rt - ((lib_rt_idx - start) * rt_gap)
                end_time = lib_rt - ((lib_rt_idx - end) * rt_gap)
            else:
                start_time, end_time = None, None

            filename_to_annotation[filename] = {
                'start_idx': start,
                'end_idx': end,
                'start_time': start_time,
                'end_time': end_time,
                'lib_rt_idx': lib_rt_idx,
                'lib_rt': lib_rt,
                'win_size': win_size}

    return filename_to_annotation


def get_naked_seq(seq):
    if '_' in seq:
        seq = seq.split('_')[0]

    mod_start = seq.rfind('(')
    mod_end = seq.find(')')

    if mod_start != -1 and mod_end != -1:
        seq = seq[:mod_start] + seq[mod_end + 1:]

    return seq


def get_feature_data_table(osw_filename, decoy=0, spectral_info=True):
    con = sqlite3.connect(osw_filename)
    cursor = con.cursor()

    if spectral_info:
        query = \
            """SELECT r.FILENAME, p2.MODIFIED_SEQUENCE,
            f.LEFT_WIDTH, f.RIGHT_WIDTH, p1.CHARGE, ms1.VAR_MASSDEV_SCORE,
            ms1.VAR_ISOTOPE_CORRELATION_SCORE, ms1.VAR_ISOTOPE_OVERLAP_SCORE,
            ms1.VAR_XCORR_COELUTION, ms1.VAR_XCORR_SHAPE,
            ms2.VAR_BSERIES_SCORE, ms2.VAR_DOTPROD_SCORE,
            ms2.VAR_INTENSITY_SCORE, ms2.VAR_ISOTOPE_CORRELATION_SCORE,
            ms2.VAR_ISOTOPE_OVERLAP_SCORE, ms2.VAR_LIBRARY_CORR,
            ms2.VAR_LIBRARY_DOTPROD, ms2.VAR_LIBRARY_MANHATTAN,
            ms2.VAR_LIBRARY_RMSD, ms2.VAR_LIBRARY_ROOTMEANSQUARE,
            ms2.VAR_LIBRARY_SANGLE, ms2.VAR_LOG_SN_SCORE,
            ms2.VAR_MANHATTAN_SCORE, ms2.VAR_MASSDEV_SCORE,
            ms2.VAR_MASSDEV_SCORE_WEIGHTED, ms2.VAR_NORM_RT_SCORE,
            ms2.VAR_XCORR_COELUTION, ms2.VAR_XCORR_COELUTION_WEIGHTED,
            ms2.VAR_XCORR_SHAPE, ms2.VAR_XCORR_SHAPE_WEIGHTED,
            ms2.VAR_YSERIES_SCORE
            FROM PRECURSOR p1
            LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ppm ON p1.ID = ppm.PRECURSOR_ID
            LEFT JOIN PEPTIDE p2 ON p2.ID = ppm.PEPTIDE_ID
            LEFT JOIN FEATURE f ON p1.ID = f.PRECURSOR_ID
            LEFT JOIN RUN r on f.RUN_ID = r.ID
            LEFT JOIN FEATURE_MS1 ms1 ON f.ID = ms1.FEATURE_ID
            LEFT JOIN FEATURE_MS2 ms2 on f.ID = ms2.FEATURE_ID
            WHERE p1.DECOY = {0}""".format(decoy)
    else:
        query = \
            """SELECT r.FILENAME, p2.MODIFIED_SEQUENCE,
            f.LEFT_WIDTH, f.RIGHT_WIDTH, p1.CHARGE, ms1.VAR_XCORR_COELUTION,
            ms1.VAR_XCORR_SHAPE, ms2.VAR_DOTPROD_SCORE,
            ms2.VAR_INTENSITY_SCORE, ms2.VAR_LIBRARY_CORR,
            ms2.VAR_LIBRARY_DOTPROD, ms2.VAR_LIBRARY_MANHATTAN,
            ms2.VAR_LIBRARY_RMSD, ms2.VAR_LIBRARY_ROOTMEANSQUARE,
            ms2.VAR_LIBRARY_SANGLE, ms2.VAR_LOG_SN_SCORE,
            ms2.VAR_MANHATTAN_SCORE, ms2.VAR_NORM_RT_SCORE,
            ms2.VAR_XCORR_COELUTION, ms2.VAR_XCORR_COELUTION_WEIGHTED,
            ms2.VAR_XCORR_SHAPE, ms2.VAR_XCORR_SHAPE_WEIGHTED
            FROM PRECURSOR p1
            LEFT JOIN PRECURSOR_PEPTIDE_MAPPING ppm ON p1.ID = ppm.PRECURSOR_ID
            LEFT JOIN PEPTIDE p2 ON p2.ID = ppm.PEPTIDE_ID
            LEFT JOIN FEATURE f ON p1.ID = f.PRECURSOR_ID
            LEFT JOIN RUN r on f.RUN_ID = r.ID
            LEFT JOIN FEATURE_MS1 ms1 ON f.ID = ms1.FEATURE_ID
            LEFT JOIN FEATURE_MS2 ms2 on f.ID = ms2.FEATURE_ID
            WHERE p1.DECOY = {0}""".format(decoy)

    res = cursor.execute(query)
    tmp = res.fetchall()
    con.close()

    return tmp


def create_feature_data(
    manual_annotation_dir,
    manual_annotation_filename,
    rt_gap,
    osw_dir,
    osw_filename,
    threshold,
    out_dir,
    data_npy,
    labels_npy,
    csv_filename,
    spectral_info=True,
    decoys=False
):
    manual_annotation_filename = os.path.join(
        manual_annotation_dir, manual_annotation_filename)
    osw_filename = os.path.join(osw_dir, osw_filename)
    filename_to_annotation = parse_manual_annotation_file(
        manual_annotation_filename, rt_gap)
    feature_data_table = get_feature_data_table(
        osw_filename, 0, spectral_info)
    num_samples = len(feature_data_table)
    feature_data_array = []
    labels = []
    feature_data_csv = []
    feature_idx = 0

    for i in range(num_samples):
        if None in feature_data_table[i]:
            continue

        filename, mod_seq, left, right, charge = feature_data_table[i][0:5]
        filename = '_'.join(
            [filename.split('/')[-1].split('.')[0], mod_seq, str(charge)])
        label = 0

        if filename not in filename_to_annotation:
            continue
        elif overlaps(
            left,
            right + 1,
            filename_to_annotation[filename]['start_time'],
            filename_to_annotation[filename]['end_time'] + 1,
            threshold
        ):
            label = 1

        left_idx = round(
            ((left - filename_to_annotation[filename]['lib_rt']) / rt_gap)
            + filename_to_annotation[filename]['lib_rt_idx'])
        right_idx = round(
            ((right - filename_to_annotation[filename]['lib_rt']) / rt_gap)
            + filename_to_annotation[filename]['lib_rt_idx'])
        labels.append([label])
        feature_data_csv.append(
            [
                feature_idx,
                filename,
                left_idx,
                right_idx,
                filename_to_annotation[filename]['start_idx'],
                filename_to_annotation[filename]['end_idx'],
                label,
                filename_to_annotation[filename]['lib_rt'],
                filename_to_annotation[filename]['win_size']])

        feature_idx += 1

        feature_data_array.append(feature_data_table[i][4:])

    if decoys:
        decoy_feature_data_table = get_feature_data_table(
            osw_filename, 1, spectral_info)

        num_decoys = len(decoy_feature_data_table)

        for i in range(num_decoys):
            if None in decoy_feature_data_table[i]:
                continue

            filename, mod_seq, left, right, charge = (
                decoy_feature_data_table[i][0:5])
            filename = '_'.join(
                [filename.split('/')[-1].split('.')[0], mod_seq, str(charge)])
            filename = 'DECOY_' + filename

            label = 0

            labels.append([label])
            feature_data_csv.append(
                [
                    feature_idx,
                    filename,
                    None,
                    None,
                    None,
                    None,
                    label,
                    None,
                    None])
            feature_idx += 1
            feature_data_array.append(decoy_feature_data_table[i][4:])

    feature_data_array = np.array(feature_data_array, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    np.save(os.path.join(out_dir, data_npy), feature_data_array)
    np.save(os.path.join(out_dir, labels_npy), labels)

    with open(os.path.join(out_dir, csv_filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['ID', 'Filename', 'OSW BB Start', 'OSW BB End',
             'Manual BB Start', 'Manual BB End', 'Label', 'Lib RT',
             'Window Size'])
        writer.writerows(feature_data_csv)


def get_seqs_from_csv(
    seq_csv,
    naked=True,
    exclusion_seqs=None,
    inclusion_idxs=None
):
    excluded = {}
    if exclusion_seqs:
        for seq in exclusion_seqs:
            excluded[seq] = True

    seqs = {}
    with open(seq_csv, 'r') as seq_file:
        next(seq_file)
        for line in seq_file:
            line = line.split(',')
            idx, seq = int(line[0]), '_'.join(line[1].split('_')[-2:])

            if inclusion_idxs and idx not in inclusion_idxs:
                continue

            if naked:
                seq = get_naked_seq(seq)

            if seq not in seqs and seq not in excluded:
                seqs[seq] = True

        seq_list = [seq for seq in seqs]

    return seq_list


def get_filenames_from_csv(seq_csv):
    filenames = {}
    with open(seq_csv, 'r') as seq_file:
        next(seq_file)
        for line in seq_file:
            filename = line.split(',')[1]

            if filename not in filenames:
                filenames[filename] = True

    return filenames


def get_train_val_test_sequence_splits(
    seq_csv,
    out_dir,
    naked=True,
    inclusion_idxs=None,
    test_proportion=0.1,
    train_seqs_filename='train_seqs.txt',
    val_seqs_filename='val_seqs.txt',
    test_seqs_filename='test_seqs.txt'
):
    seq_list = get_seqs_from_csv(
        seq_csv, naked=naked, inclusion_idxs=inclusion_idxs)

    random.shuffle(seq_list)

    n = len(seq_list)
    n_test = int(n * test_proportion)
    n_train = n - 2 * n_test

    train_seqs = seq_list[:n_train]
    val_seqs = seq_list[n_train:(n_train + n_test)]
    test_seqs = seq_list[(n_train + n_test):]

    with open(os.path.join(out_dir, train_seqs_filename), 'w') as f:
        for seq in train_seqs:
            f.write(seq + '\n')

    with open(os.path.join(out_dir, val_seqs_filename), 'w') as f:
        for seq in val_seqs:
            f.write(seq + '\n')

    with open(os.path.join(out_dir, test_seqs_filename), 'w') as f:
        for seq in test_seqs:
            f.write(seq + '\n')

    train_seqs = {seq: True for seq in train_seqs}
    val_seqs = {seq: True for seq in val_seqs}
    test_seqs = {seq: True for seq in test_seqs}

    return train_seqs, val_seqs, test_seqs


def get_train_val_test_idx_from_sequences(
    split_csv,
    train_seqs,
    val_seqs,
    test_seqs,
    naked=True,
    out_dir='.',
    prefix='dl',
    shuffle=False
):
    train_idx, val_idx, test_idx = [], [], []

    with open(split_csv, 'r') as seqs:
        next(seqs)
        for line in seqs:
            line = line.split(',')
            idx, seq = int(line[0]), '_'.join(line[1].split('_')[-2:])

            if naked:
                seq = get_naked_seq(seq)

            if seq in train_seqs:
                train_idx.append(idx)
            elif seq in val_seqs:
                val_idx.append(idx)
            elif seq in test_seqs:
                test_idx.append(idx)

    if shuffle:
        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

    np.savetxt(
        os.path.join(out_dir, f'{prefix}_train_idx.txt'),
        np.array(train_idx),
        fmt='%i')
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_val_idx.txt'),
        np.array(val_idx),
        fmt='%i')
    np.savetxt(
        os.path.join(out_dir, f'{prefix}_test_idx.txt'),
        np.array(test_idx),
        fmt='%i')

    return train_idx, val_idx, test_idx


def create_train_val_test_split_by_sequence(
    in_dir,
    seq_csv,
    out_dir,
    split_csv=None,
    inclusion_idxs=None,
    naked=True,
    test_proportion=0.1,
    prefix='dl',
    shuffle=False
):
    seq_csv = os.path.join(in_dir, seq_csv)

    if inclusion_idxs:
        filename = os.path.join(in_dir, inclusion_idxs)
        inclusion_idxs = list(np.loadtxt(filename, dtype='int'))

    train_seqs, val_seqs, test_seqs = get_train_val_test_sequence_splits(
        seq_csv,
        out_dir,
        naked=naked,
        inclusion_idxs=inclusion_idxs,
        test_proportion=test_proportion)

    if split_csv:
        split_csv = os.path.join(in_dir, split_csv)
        seq_csv = split_csv

    get_train_val_test_idx_from_sequences(
        seq_csv,
        train_seqs,
        val_seqs,
        test_seqs,
        naked,
        out_dir,
        prefix,
        shuffle)


def create_supervised_data_split(
    in_dir,
    dl_csv,
    trad_csv,
    out_dir,
    naked=True,
    test_proportion=0.1,
    dl_prefix='dl',
    trad_prefix='trad'
):
    create_train_val_test_split_by_sequence(
        in_dir,
        dl_csv,
        out_dir,
        naked=naked,
        test_proportion=test_proportion,
        prefix=dl_prefix)

    create_train_val_test_split_by_sequence(
        in_dir,
        dl_csv,
        out_dir,
        split_csv=trad_csv,
        naked=naked,
        test_proportion=test_proportion,
        prefix=trad_prefix)


def get_kfold_sequence_splits(
    seq_csv,
    out_dir,
    exclusion_seqs=None,
    naked=True,
    n_splits=5,
    prefix='mv'
):
    kf = KFold(n_splits=n_splits, shuffle=True)
    seq_splits = {}
    counter = 1

    seq_list = np.array(get_seqs_from_csv(seq_csv, naked, exclusion_seqs))

    for train_idx, test_idx in kf.split(seq_list):
        seq_splits[f'split_{counter}'] = {
            'train_seqs': {item: True for item in seq_list[train_idx]},
            'test_seqs': {item: True for item in seq_list[test_idx]}
        }
        counter += 1

    for split in seq_splits:
        for split_part in seq_splits[split]:
            with open(
                os.path.join(
                    out_dir,
                    f'{prefix}_{split}_{split_part}.txt'), 'w') as f:
                for seq in seq_splits[split][split_part]:
                    f.write(seq + '\n')

    return seq_splits


def get_kfold_idx_from_sequences(
    seq_csv,
    seq_splits,
    inclusion_filenames={},
    naked=True,
    out_dir='.',
    prefix='labeled'
):
    idx_splits = {
        split: {'train_idx': [], 'test_idx': []} for split in seq_splits
    }

    with open(seq_csv, 'r') as seqs:
        next(seqs)
        for line in seqs:
            line = line.split(',')
            idx, filename = int(line[0]), line[1]
            seq = '_'.join(filename.split('_')[-2:])

            if inclusion_filenames and filename not in inclusion_filenames:
                continue

            if naked:
                seq = get_naked_seq(seq)

            for split in seq_splits:
                for split_part in seq_splits[split]:
                    if seq in seq_splits[split][split_part]:
                        if 'train' in split_part:
                            idx_splits[split]['train_idx'].append(idx)
                        else:
                            idx_splits[split]['test_idx'].append(idx)

    for split in idx_splits:
        for split_part in idx_splits[split]:
            random.shuffle(idx_splits[split][split_part])
            np.savetxt(
                os.path.join(
                    out_dir, f'{prefix}_{split}_{split_part}.txt'
                ),
                np.array(idx_splits[split][split_part]),
                fmt='%i'
            )

    return idx_splits


def create_kfold_split_by_sequence(
    in_dir,
    seq_csv,
    out_dir,
    idx_csv=None,
    exclusion_seq_csvs=[],
    inclusion_filename_csvs=[],
    naked=True,
    n_splits=5,
    prefix='labeled'
):
    seq_csv = os.path.join(in_dir, seq_csv)

    if idx_csv:
        idx_csv = os.path.join(in_dir, idx_csv)

    exclusion_seqs = []

    if exclusion_seq_csvs:
        for exclusion_seq_csv in exclusion_seq_csvs:
            exclusion_seq_csv = os.path.join(in_dir, exclusion_seq_csv)
            exclusion_seqs += get_seqs_from_csv(exclusion_seq_csv, naked)

    seq_splits = get_kfold_sequence_splits(
        seq_csv,
        out_dir,
        exclusion_seqs,
        naked,
        n_splits,
        prefix)

    if idx_csv:
        seq_csv = idx_csv

    inclusion_filenames = {}
    if inclusion_filename_csvs:
        for inclusion_csv in inclusion_filename_csvs:
            inclusion_csv = os.path.join(in_dir, inclusion_csv)
            inclusion_filenames = {
                **inclusion_filenames,
                **get_filenames_from_csv(inclusion_csv)
            }

    idx_splits = get_kfold_idx_from_sequences(
        seq_csv, seq_splits, inclusion_filenames, naked, out_dir, prefix)


def create_kfold_train_and_validation_and_holdout_test_by_sequence(
    in_dir,
    seq_csv,
    out_dir,
    naked=True,
    special_seq_csv=None,
    n_splits=5,
    holdout_proportion=0.1
):
    special_seq_filenames = {}

    if special_seq_csv:
        with open(special_seq_csv, 'r') as special:
            next(special)
            for line in special:
                filename = line.split(',')[1]
                special_seq_filenames[filename] = True

    special_seqs, decoy_seqs, non_decoy_seqs, seq_to_idx = {}, {}, {}, {}
    with open(os.path.join(in_dir, seq_csv), 'r') as seqs:
        next(seqs)
        for line in seqs:
            line = line.split(',')
            idx, filename = line[0], line[1]
            seq = '_'.join(filename.split('_')[-2:])

            if naked:
                seq = get_naked_seq(seq)

            if seq in seq_to_idx:
                seq_to_idx[seq].append(idx)
            else:
                seq_to_idx[seq] = [idx]

            if filename in special_seq_filenames:
                special_seqs[seq] = True
            elif 'DECOY' in filename:
                decoy_seqs[seq] = True
            else:
                non_decoy_seqs[seq] = True

    for seq in special_seqs:
        non_decoy_seqs.pop(seq, None)

    special_seqs = list(special_seqs.keys())
    decoy_seqs = list(decoy_seqs.keys())
    non_decoy_seqs = list(non_decoy_seqs.keys())

    random.shuffle(special_seqs)
    random.shuffle(decoy_seqs)
    random.shuffle(non_decoy_seqs)

    n = len(special_seqs)
    n_holdout = int(n * holdout_proportion)
    n_non_holdout = n - n_holdout

    non_holdout_special_seqs = special_seqs[:n_non_holdout]
    holdout_special_seqs = special_seqs[n_non_holdout:]

    n = len(decoy_seqs)
    n_holdout = int(n * holdout_proportion)
    n_non_holdout = n - n_holdout

    non_holdout_decoy_seqs = decoy_seqs[:n_non_holdout]
    holdout_decoy_seqs = decoy_seqs[n_non_holdout:]

    n = len(non_decoy_seqs)
    n_holdout = int(n * holdout_proportion)
    n_non_holdout = n - n_holdout

    non_holdout_non_decoy_seqs = non_decoy_seqs[:n_non_holdout]
    holdout_non_decoy_seqs = non_decoy_seqs[n_non_holdout:]

    with open(
        os.path.join(out_dir, 'special_holdout_test_idx.txt'), 'w'
    ) as f:
        for seq in holdout_special_seqs:
            for idx in seq_to_idx[seq]:
                f.write(idx + '\n')

    with open(
        os.path.join(out_dir, 'decoy_holdout_test_idx.txt'), 'w'
    ) as f:
        for seq in holdout_decoy_seqs:
            for idx in seq_to_idx[seq]:
                f.write(idx + '\n')

    with open(
        os.path.join(out_dir, 'non_decoy_holdout_test_idx.txt'), 'w'
    ) as f:
        for seq in holdout_non_decoy_seqs:
            for idx in seq_to_idx[seq]:
                f.write(idx + '\n')

    kf = KFold(n_splits=n_splits, shuffle=True)
    seq_splits = {}
    counter = 1
    seq_list = np.array(non_holdout_special_seqs)

    for train_idx, val_idx in kf.split(seq_list):
        seq_splits[f'split_{counter}'] = {
            'train_idx': {item: True for item in seq_list[train_idx]},
            'val_idx': {item: True for item in seq_list[val_idx]}}
        counter += 1

    for split in seq_splits:
        for split_part in seq_splits[split]:
            with open(
                os.path.join(
                    out_dir, f'special_{split}_{split_part}.txt'), 'w') as f:
                for seq in seq_splits[split][split_part]:
                    for idx in seq_to_idx[seq]:
                        f.write(idx + '\n')

    seq_splits = {}
    counter = 1
    seq_list = np.array(non_holdout_decoy_seqs)

    for train_idx, val_idx in kf.split(seq_list):
        seq_splits[f'split_{counter}'] = {
            'train_idx': {item: True for item in seq_list[train_idx]},
            'val_idx': {item: True for item in seq_list[val_idx]}}
        counter += 1

    for split in seq_splits:
        for split_part in seq_splits[split]:
            with open(
                os.path.join(
                    out_dir, f'decoy_{split}_{split_part}.txt'), 'w') as f:
                for seq in seq_splits[split][split_part]:
                    for idx in seq_to_idx[seq]:
                        f.write(idx + '\n')

    seq_splits = {}
    counter = 1
    seq_list = np.array(non_holdout_non_decoy_seqs)

    for train_idx, val_idx in kf.split(seq_list):
        seq_splits[f'split_{counter}'] = {
            'train_idx': {item: True for item in seq_list[train_idx]},
            'val_idx': {item: True for item in seq_list[val_idx]}
        }
        counter += 1

    for split in seq_splits:
        for split_part in seq_splits[split]:
            with open(
                os.path.join(
                    out_dir, f'non_decoy_{split}_{split_part}.txt'), 'w') as f:
                for seq in seq_splits[split][split_part]:
                    for idx in seq_to_idx[seq]:
                        f.write(idx + '\n')


def compare_label_npys_pointwise(
    file_dir,
    idx_filename,
    target_labels_npy,
    source_labels_npy
):
    target_labels = np.load(os.path.join(file_dir, target_labels_npy))
    source_labels = np.load(os.path.join(file_dir, source_labels_npy))
    target, source = [], []

    idxs = np.loadtxt(os.path.join(file_dir, idx_filename), dtype='int')
    for idx in idxs:
        target.append(target_labels[idx])
        source.append(source_labels[idx])

    target = np.concatenate(target, axis=0).reshape(-1, 1)
    source = np.concatenate(source, axis=0).reshape(-1, 1)
    acc = accuracy_score(target, source)
    bacc = balanced_accuracy_score(target, source)
    prec = precision_score(target, source)
    recall = recall_score(target, source)
    dice = f1_score(target, source)
    iou = jaccard_score(target, source)

    print(
        f'Pixel Accuracy: {acc:.4f} '
        f'Pixel Balanced accuracy: {bacc:.4f} '
        f'Pixel Precision: {prec:.4f} '
        f'Pixel Recall: {recall:.4f} '
        f'Pixel Dice/F1: {dice:.4f} '
        f'Pixel IoU/Jaccard: {iou:.4f}')

def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = (
        final_value + 0.5 * (base_value - final_value) * 
        (1 + np.cos(np.pi * iters / len(iters))))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep

    return schedule
