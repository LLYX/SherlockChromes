import bisect
import numpy as np
import os

from general_utils import get_subsequence_idxs
from pyopenms import AASequence


def parse_skyline_exported_annotations(
    annotations_dir, annotations_filename, transition_results=False
):
    annotations = {}

    with open(os.path.join(annotations_dir, annotations_filename)) as infile:
        next(infile)

        for line in infile:
            line = line.rstrip('\r\n').split(',')

            if transition_results:
                repl, rt, start, end, seq, charge = (
                    line[2], line[6], line[7], line[8], line[13], line[14])
            else:
                repl, rt, seq, charge, start, end = (
                    line[2], line[4], line[13], line[14], line[15], line[16])

            repl = repl.replace('0R', '0PlasmaBiolR')
            seq = AASequence.fromString(seq).toUniModString().decode('utf-8')
            repl_prefix = (
                'hroest_K120808' if 'R04' not in repl else 'hroest_K120809')
            key = f'{repl_prefix}_{repl}_{seq}_{charge}'

            if start == end == rt == '#N/A':
                annotations[key] = {'start': None, 'end': None, 'rt': None}
            else:
                annotations[key] = (
                    {
                        'start': float(start) * 60,
                        'end': float(end) * 60,
                        'rt': float(rt) * 60})

    return annotations


def create_skyline_augmented_npy_dataset(
    annotations,
    osw_dir,
    osw_csv,
    osw_strong_labels_npy,
    osw_weak_labels_npy,
    out_dir,
    prefix='skyline_augmented',
    peak_only=False
):
    orig_strong_labels = np.load(os.path.join(osw_dir, osw_strong_labels_npy))
    orig_weak_labels = np.load(os.path.join(osw_dir, osw_weak_labels_npy))
    ms1_rt_arrays = {}
    decoy_counter = 0
    skyline_strong_counter = 0
    skyline_weak_counter = 0

    with open(os.path.join(osw_dir, osw_csv)) as infile:
        next(infile)

        for line in infile:
            line = line.rstrip('\r\n').split(',')
            idx, filename, lib_rt, win_size = (
                line[0], line[1], line[3], line[4])

            if 'DECOY' in filename:
                # By default, includes positive labels even for decoys
                # since based on OSW unscored boundaries
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                decoy_counter += 1
                continue
            elif filename not in annotations:
                continue
            elif not annotations[filename]['start']:
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                orig_weak_labels[int(idx)] = 0
                skyline_strong_counter += 1
                skyline_weak_counter += 1
                continue

            skyline_strong_counter += 1
            repl = '_'.join(filename.split('_')[:-2])

            if repl not in ms1_rt_arrays:
                ms1_rt_arrays[repl] = np.load(f'{repl}_ms1_rt_array.npy')

            try:
                lib_rt = int(lib_rt)
                lib_rt = ms1_rt_arrays[repl][lib_rt]
            except Exception:
                lib_rt = float(lib_rt)

            segment_l, segment_r = get_subsequence_idxs(
                ms1_rt_arrays[repl], lib_rt, subsequence_size=int(win_size))

            rt_segment = ms1_rt_arrays[repl][segment_l:segment_r]

            if (
                annotations[filename]['end'] < rt_segment[0]
                or annotations[filename]['start'] > rt_segment[-1]
                or (
                    peak_only
                    and not (
                        rt_segment[0]
                        <= annotations[filename]['rt']
                        <= rt_segment[-1]))
            ):
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                orig_weak_labels[int(idx)] = 0
                skyline_weak_counter += 1
                continue

            if peak_only:
                skyline_left_idx = skyline_right_idx = bisect.bisect(
                    rt_segment, annotations[filename]['rt'])
            else:
                skyline_left_idx = bisect.bisect_left(
                    rt_segment, annotations[filename]['start'])
                skyline_right_idx = bisect.bisect_left(
                    rt_segment, annotations[filename]['end'])

            if skyline_right_idx >= rt_segment.shape[0]:
                skyline_right_idx = rt_segment.shape[0] - 1

            orig_strong_labels[int(idx)] = np.where(
                np.logical_and(
                    rt_segment >= rt_segment[skyline_left_idx],
                    rt_segment <= rt_segment[skyline_right_idx]),
                1,
                0)

            new_weak_label = min(1, np.sum(orig_strong_labels[int(idx)]))

            if new_weak_label != orig_weak_labels[int(idx)]:
                orig_weak_labels[int(idx)] = new_weak_label
                skyline_weak_counter += 1

    print(
        f'Saving skyline augmented segmentation labels array of shape '
        f'{orig_strong_labels.shape} with '
        f'{skyline_strong_counter} Skyline substitutions '
        f'and {decoy_counter} decoy substitutions')

    np.save(
        os.path.join(out_dir, f'{prefix}_{osw_strong_labels_npy}'),
        orig_strong_labels.astype(np.int32))

    print(
        f'Saving skyline augmented classification labels array of shape '
        f'{orig_weak_labels.shape} with '
        f'{skyline_weak_counter} Skyline substitutions')

    np.save(
        os.path.join(out_dir, f'{prefix}_{osw_weak_labels_npy}'),
        orig_weak_labels.astype(np.int32))


def create_skyline_augmented_osw_dataset(
    annotations,
    osw_dir,
    osw_csv,
    osw_strong_labels_npy,
    osw_weak_labels_npy,
    osw_times_npy,
    out_dir,
    prefix='skyline_augmented',
    peak_only=False
):
    orig_strong_labels = np.load(os.path.join(osw_dir, osw_strong_labels_npy))
    orig_weak_labels = np.zeros(orig_strong_labels.shape[0])
    times_npy = np.load(os.path.join(osw_dir, osw_times_npy))
    decoy_counter = 0
    skyline_strong_counter = 0
    skyline_weak_counter = 0
    decoy_idxs, non_decoy_idxs, skyline_idxs = [], [], []

    with open(os.path.join(osw_dir, osw_csv)) as infile:
        next(infile)

        for line in infile:
            line = line.rstrip('\r\n').split(',')
            idx, filename, lib_rt_idx = line[0], line[1], line[3]

            if not lib_rt_idx:
                lib_rt_idx = times_npy.shape[1] // 2

            if 'DECOY' in filename:
                # By default, includes positive labels even for decoys
                # since based on OSW unscored boundaries
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                decoy_counter += 1
                decoy_idxs.append(int(idx))
                continue
            elif filename not in annotations:
                orig_weak_labels[int(idx)] = 1
                non_decoy_idxs.append(int(idx))
                continue
            elif not annotations[filename]['start']:
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                skyline_strong_counter += 1
                skyline_weak_counter += 1
                skyline_idxs.append(int(idx))
                continue

            orig_weak_labels[int(idx)] = 1
            skyline_strong_counter += 1
            skyline_idxs.append(int(idx))

            try:
                lib_rt_idx = int(lib_rt_idx)
                lib_rt = times_npy[int(idx)][lib_rt_idx]
            except Exception:
                lib_rt = float(lib_rt)

            if (
                annotations[filename]['end'] < times_npy[int(idx)][0]
                or annotations[filename]['start'] > times_npy[int(idx)][-1]
                or (
                    peak_only
                    and not (
                        times_npy[int(idx)][0]
                        <= annotations[filename]['rt']
                        <= times_npy[int(idx)][-1]))
            ):
                orig_strong_labels[int(idx)] = np.zeros(
                    orig_strong_labels[int(idx)].shape)
                skyline_weak_counter += 1
                continue

            if peak_only:
                skyline_left_idx = skyline_right_idx = bisect.bisect(
                    times_npy[int(idx)], annotations[filename]['rt'])
            else:
                skyline_left_idx = bisect.bisect_left(
                    times_npy[int(idx)], annotations[filename]['start'])
                skyline_right_idx = bisect.bisect_left(
                    times_npy[int(idx)], annotations[filename]['end'])
            
            if skyline_right_idx >= times_npy[int(idx)].shape[0]:
                print(
                    skyline_left_idx,
                    skyline_right_idx,
                    annotations[filename],
                    times_npy[int(idx)][0],
                    times_npy[int(idx)][-1])
                skyline_right_idx = times_npy[int(idx)].shape[0] - 1

            orig_strong_labels[int(idx)] = np.where(
                np.logical_and(
                    times_npy[int(idx)]
                    >= times_npy[int(idx)][skyline_left_idx],
                    times_npy[int(idx)]
                    <= times_npy[int(idx)][skyline_right_idx]),
                1,
                0)

            new_weak_label = min(1, np.sum(orig_strong_labels[int(idx)]))

            if new_weak_label != orig_weak_labels[int(idx)]:
                orig_weak_labels[int(idx)] = new_weak_label
                skyline_weak_counter += 1

    print(
        f'Saving skyline augmented segmentation labels array of shape '
        f'{orig_strong_labels.shape} with '
        f'{skyline_strong_counter} Skyline substitutions '
        f'and {decoy_counter} decoy substitutions')

    np.save(
        os.path.join(out_dir, f'{prefix}_{osw_strong_labels_npy}'),
        orig_strong_labels.astype(np.int32))

    print(
        f'Saving skyline augmented classification labels array of shape '
        f'{orig_weak_labels.shape} with '
        f'{skyline_weak_counter} Skyline substitutions')

    np.save(
        os.path.join(out_dir, f'{prefix}_{osw_weak_labels_npy}'),
        orig_weak_labels.astype(np.int32))

    print('Saving indices in text files')

    np.savetxt(
        os.path.join(out_dir, 'decoy_idxs.txt'),
        np.array(decoy_idxs),
        fmt='%i')
    np.savetxt(
        os.path.join(out_dir, 'non_decoy_idxs.txt'),
        np.array(non_decoy_idxs),
        fmt='%i')
    np.savetxt(
        os.path.join(out_dir, 'skyline_idxs.txt'),
        np.array(skyline_idxs),
        fmt='%i')
