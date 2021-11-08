import argparse
import bisect
import csv
import numpy as np
import os
import sqlite3
import time

from general_utils import calc_bin_idx, get_subsequence_idxs


def get_specs_from_sql(cursor, repl):
    query = \
        f"""SELECT
        PRECURSOR_ID,
        MODIFIED_SEQUENCE,
        CHARGE,
        PRECURSOR_MZ,
        group_concat(PRODUCT_MZ, '|') AS PRODUCT_MZS,
        group_concat(LIBRARY_INTENSITY, '|') AS LIBRARY_INTENSITIES,
        DECOY,
        EXP_RT,
        DELTA_RT,
        LEFT_WIDTH,
        RIGHT_WIDTH,
        SCORE
            FROM (
                SELECT
                prec.ID AS PRECURSOR_ID,
                pep.MODIFIED_SEQUENCE,
                prec.CHARGE AS CHARGE,
                prec.PRECURSOR_MZ,
                trans.PRODUCT_MZ,
                trans.LIBRARY_INTENSITY AS LIBRARY_INTENSITY,
                prec.DECOY AS DECOY,
                feat2.EXP_RT AS EXP_RT,
                feat2.DELTA_RT AS DELTA_RT,
                feat2.LEFT_WIDTH AS LEFT_WIDTH,
                feat2.RIGHT_WIDTH AS RIGHT_WIDTH,
                feat2.SCORE AS SCORE
                FROM PRECURSOR AS prec
                LEFT JOIN PRECURSOR_PEPTIDE_MAPPING AS prec_to_pep
                ON prec.ID = prec_to_pep.PRECURSOR_ID
                LEFT JOIN PEPTIDE AS pep
                ON prec_to_pep.PEPTIDE_ID = pep.ID
                LEFT JOIN TRANSITION_PRECURSOR_MAPPING AS trans_to_prec
                ON prec.ID = trans_to_prec.PRECURSOR_ID
                LEFT JOIN TRANSITION AS trans
                ON trans_to_prec.TRANSITION_ID = trans.ID
                LEFT JOIN (
                    SELECT
                    feat1.PRECURSOR_ID AS PRECURSOR_ID,
                    feat1.EXP_RT AS EXP_RT,
                    feat1.DELTA_RT AS DELTA_RT,
                    feat1.LEFT_WIDTH AS LEFT_WIDTH,
                    feat1.RIGHT_WIDTH AS RIGHT_WIDTH,
                    score.SCORE AS SCORE
                    FROM FEATURE AS feat1
                    LEFT JOIN (
                        SELECT
                        ID
                        FROM RUN
                        WHERE FILENAME LIKE '%{repl}%') AS run
                    ON feat1.RUN_ID = run.ID
                    LEFT JOIN SCORE_MS2 AS score
                    ON feat1.ID = score.FEATURE_ID
                    WHERE NOT run.ID IS NULL AND score.RANK = 1
                    GROUP BY PRECURSOR_ID) AS feat2
                ON prec.ID = feat2.PRECURSOR_ID
                ORDER BY PRECURSOR_ID, TRANSITION_ID ASC)
        GROUP BY PRECURSOR_ID;"""
    res = cursor.execute(query)
    tmp = res.fetchall()

    return tmp


def extract_target_strip(
    lcms_map,
    target_mz,
    min_mz=0,
    bin_resolution=0.01,
    lower_span=5,
    upper_span=5
):
    bin_idx = calc_bin_idx(target_mz, min_mz, bin_resolution)
    max_idx = lcms_map.shape[0] - 1
    lower = max(bin_idx - lower_span, 0)
    upper = max(min(bin_idx + upper_span, max_idx), 0)
    strip = lcms_map[lower:upper]
    tgt_height = lower_span + upper_span

    # Edge cases: Indices below or above actual data array
    # Solution: Pad with individual components for regions below and above
    if strip.shape[0] != tgt_height:
        width = strip.shape[-1]
        padded_strip = []

        if lower == 0:
            height = min(-(bin_idx - lower_span), tgt_height)
            padded_strip.append(np.zeros((height, width)))

        padded_strip.append(strip)

        if upper == max_idx:
            curr_height = sum([item.shape[0] for item in padded_strip])
            height = tgt_height - curr_height
            padded_strip.append(np.zeros((height, width)))

        strip = np.concatenate(padded_strip, axis=0)

    return strip


def create_chromatogram(
    ms1_map,
    ms2_map,
    ms1_rt_array,
    prec_mz,
    lib_rt,
    prod_mzs,
    lib_intensities,
    prec_charge,
    osw_label_left,
    osw_label_right,
    min_swath_win=399.5,
    swath_win_size=25,
    monoisotope_only=False,
    prod_charges=[],
    num_traces=6,
    analysis_win_size=175
):
    ms2_map_idx = calc_bin_idx(prec_mz, min_swath_win, swath_win_size)
    chromatogram = []

    if not prod_charges:
        prod_charges = [None for mz in prod_mzs]

    for mz, charge in zip(prod_mzs, prod_charges):
        if not monoisotope_only:
            charge = charge if charge else 1

            for i in range(3, 1, -1):
                delta = 1 / charge * i
                chromatogram.append(
                    extract_target_strip(ms2_map[ms2_map_idx], mz + delta))

        chromatogram.append(extract_target_strip(ms2_map[ms2_map_idx], mz))

        if not monoisotope_only:
            for i in range(4, 0, -1):
                delta = 1 / i
                chromatogram.append(
                    extract_target_strip(ms2_map[ms2_map_idx], mz - delta))

    if len(prod_mzs) < num_traces:
        shape = chromatogram[-1].shape

        if not monoisotope_only:
            shape = (70, chromatogram[-1].shape[-1])

        for i in range(num_traces - len(prod_mzs)):
            chromatogram.append(np.zeros(shape))

    if not monoisotope_only:
        for i in range(3, 1, -1):
            delta = 1 / prec_charge * i
            chromatogram.append(
                extract_target_strip(ms1_map, prec_mz + delta, min_mz=400))

    chromatogram.append(extract_target_strip(ms1_map, prec_mz, min_mz=400))

    if not monoisotope_only:
        for i in range(4, 0, -1):
            delta = 1 / i
            chromatogram.append(
                extract_target_strip(ms1_map, prec_mz - delta, min_mz=400))

    if len(lib_intensities) < num_traces:
        for i in range(num_traces - len(lib_intensities)):
            lib_intensities.append(0)

    chromatogram.append(
        np.repeat(
            lib_intensities,
            ms1_rt_array.shape[-1]).reshape(num_traces, -1))
    chromatogram.append(np.expand_dims(abs(ms1_rt_array - lib_rt), axis=0))
    chromatogram.append(
        np.expand_dims(np.repeat(prec_charge, ms1_rt_array.shape[-1]), axis=0))
    chromatogram = np.concatenate(chromatogram, axis=0)

    # M = number of MZ buckets per strip
    # T = number of retention time steps
    # final shape = (7 * M + 8, T)
    # first 6 * M are the MS2 strips
    # following M are the MS1 strips
    # followed by the 6 MS2 library intensities
    # followed by the retention time diff
    # followed by the precursor charge

    osw_label_left_idx, osw_label_right_idx = None, None
    ss_left_idx, ss_right_idx = None, None
    ss_left_idx, ss_right_idx = get_subsequence_idxs(
        ms1_rt_array, lib_rt, analysis_win_size)

    if analysis_win_size > -1:
        chromatogram = chromatogram[:, ss_left_idx:ss_right_idx]
        ms1_rt_array = ms1_rt_array[ss_left_idx:ss_right_idx]

    osw_label_left_idx = bisect.bisect_left(ms1_rt_array, osw_label_left)
    osw_label_right_idx = bisect.bisect_left(ms1_rt_array, osw_label_right)

    if osw_label_left_idx == osw_label_right_idx:
        osw_label_left_idx = osw_label_right_idx = None
    elif osw_label_right_idx >= ms1_rt_array.shape[0]:
        osw_label_right_idx = ms1_rt_array.shape[0] - 1

    return (
        chromatogram,
        lib_rt,
        ss_left_idx,
        ss_right_idx,
        osw_label_left_idx,
        osw_label_right_idx
    )


def create_repl_chromatograms_array(
    work_dir,
    osw_filename,
    repl,
    num_traces=6,
    min_swath_win=399.5,
    swath_win_size=25,
    analysis_win_size=175,
    create_label_arrays=True
):
    repl = repl.split('.')[0]

    ms1_map = np.load(f'{repl}_ms1_array.npy')
    ms2_map = np.load(f'{repl}_ms2_array.npy')
    ms1_rt_array = np.load(f'{repl}_ms1_rt_array.npy')

    con = sqlite3.connect(os.path.join(work_dir, osw_filename))
    cursor = con.cursor()

    specs = get_specs_from_sql(cursor, repl)

    chromatograms_array = []
    out_csv = [
        [
            'ID',
            'Filename',
            'External Precursor ID',
            'External Library RT',
            'Window Size',
            'External Label Left IDX',
            'External Label Right IDX',
            'External Score'
        ]
    ]

    if create_label_arrays:
        segmentation_labels_array, classification_labels_array = [], []

    idx = 0
    for i in range(len(specs)):
        (
            prec_id,
            mod_seq,
            prec_charge,
            prec_mz,
            prod_mzs,
            lib_intensities,
            decoy,
            exp_rt,
            delta_rt,
            left_width,
            right_width,
            score
        ) = specs[i]

        mod_seq_and_prec_charge = f'{mod_seq}_{prec_charge}'

        if exp_rt and delta_rt:
            lib_rt = exp_rt - delta_rt
        else:
            print(
                f'Precursor {prec_id}: {mod_seq_and_prec_charge} '
                'missing lib_rt')
            continue

        prod_mzs = [float(x) for x in prod_mzs.split('|')]
        lib_intensities = [float(x) for x in lib_intensities.split('|')]

        (
            chromatogram,
            lib_rt,
            ss_left_idx,
            ss_right_idx,
            osw_label_left_idx,
            osw_label_right_idx
        ) = create_chromatogram(
                ms1_map,
                ms2_map,
                ms1_rt_array,
                prec_mz,
                lib_rt,
                prod_mzs,
                lib_intensities,
                prec_charge,
                left_width,
                right_width,
                min_swath_win=min_swath_win,
                swath_win_size=swath_win_size,
                num_traces=num_traces,
                analysis_win_size=analysis_win_size
            )

        filename = f'{repl}_{mod_seq_and_prec_charge}'

        if decoy == 1:
            filename = f'DECOY_{filename}'

        chromatograms_array.append(chromatogram)

        if create_label_arrays:
            ms1_rt_array_subsequence = ms1_rt_array

            if analysis_win_size > -1:
                ms1_rt_array_subsequence = (
                    ms1_rt_array[ss_left_idx:ss_right_idx])

            segmentation_label = np.zeros((ms1_rt_array_subsequence.shape))

            if osw_label_left_idx and osw_label_right_idx:
                segmentation_label = np.where(
                    np.logical_and(
                        ms1_rt_array_subsequence >= ms1_rt_array_subsequence[
                            osw_label_left_idx],
                        ms1_rt_array_subsequence <= ms1_rt_array_subsequence[
                            osw_label_right_idx]
                    ),
                    1,
                    0
                )

            segmentation_labels_array.append(segmentation_label)
            classification_labels_array.append(decoy)

        out_csv.append([
            idx,
            filename,
            prec_id,
            lib_rt,
            analysis_win_size,
            osw_label_left_idx,
            osw_label_right_idx,
            score
        ])
        idx += 1

    chromatograms_array = np.array(chromatograms_array, dtype=np.float32)

    print(
        f'Saving chromatograms array for {repl} of shape '
        f'{chromatograms_array.shape}'
    )

    np.save(
        os.path.join(work_dir, f'{repl}_chromatograms_array'),
        chromatograms_array
    )

    if create_label_arrays:
        segmentation_labels_array = np.array(
            segmentation_labels_array, dtype=np.int32)

        print(
            f'Saving segmentation labels array for {repl} of shape '
            f'{segmentation_labels_array.shape}'
        )

        np.save(
            os.path.join(work_dir, f'{repl}_segmentation_labels_array'),
            segmentation_labels_array
        )

        classification_labels_array = np.array(
            classification_labels_array, dtype=np.int32).reshape((-1, 1))

        # Change labels from decoy to non-decoy as positive class
        classification_labels_array = 1 - classification_labels_array

        print(
            f'Saving classification labels array for {repl} of shape '
            f'{classification_labels_array.shape}'
        )

        np.save(
            os.path.join(work_dir, f'{repl}_classification_labels_array'),
            classification_labels_array
        )

    with open(os.path.join(work_dir, f'{repl}_chromatograms.csv'), 'w') as out:
        writer = csv.writer(out, lineterminator='\n')
        writer.writerows(out_csv)


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-work_dir', '--work_dir', type=str, default='.')
    parser.add_argument('-osw', '--osw', type=str, default='merged.osw')
    parser.add_argument('-repl_name', '--repl_name', type=str, default='')

    args = parser.parse_args()

    create_repl_chromatograms_array(args.work_dir, args.osw, args.repl_name)

    print('It took {0:0.1f} seconds'.format(time.time() - start))
