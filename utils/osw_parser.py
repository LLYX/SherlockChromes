import argparse
import csv
import numpy as np
import os
import sqlite3
import time

from general_utils import get_subsequence_idxs
from sql_data_access import SqlDataAccess


def get_run_id_from_folder_name(cursor, folder_name):
    query = \
        """SELECT ID FROM RUN WHERE FILENAME LIKE '%{0}%'""".format(
            folder_name)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) == 1

    return tmp[0][0]


def get_mod_seqs_and_charges(cursor):
    query = \
        """SELECT precursor.ID, peptide.MODIFIED_SEQUENCE, precursor.CHARGE,
        precursor.DECOY
        FROM PRECURSOR precursor LEFT JOIN PRECURSOR_PEPTIDE_MAPPING mapping
        ON precursor.ID = mapping.PRECURSOR_ID LEFT JOIN PEPTIDE peptide
        ON mapping.PEPTIDE_ID = peptide.ID
        ORDER BY precursor.ID ASC"""
    res = cursor.execute(query)
    tmp = res.fetchall()

    return tmp


def get_feature_info_from_run(cursor, run_id):
    query = \
        """SELECT p.ID, f.EXP_RT, f.DELTA_RT, f.LEFT_WIDTH, f.RIGHT_WIDTH,
        s.SCORE
        FROM PRECURSOR p
        LEFT JOIN FEATURE f ON p.ID = f.PRECURSOR_ID
        AND (f.RUN_ID = {0} OR f.RUN_ID IS NULL)
        LEFT JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID
        WHERE (s.RANK = 1 OR s.RANK IS NULL)
        ORDER BY p.ID ASC""".format(run_id)
    res = cursor.execute(query)
    tmp = res.fetchall()

    return tmp


def get_transition_ids_and_library_intensities_from_prec_id(cursor, prec_id):
    query = \
        """SELECT ID, LIBRARY_INTENSITY
        FROM TRANSITION LEFT JOIN TRANSITION_PRECURSOR_MAPPING
        ON TRANSITION.ID = TRANSITION_ID
        WHERE PRECURSOR_ID = {0}""".format(prec_id)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) > 0, prec_id

    return tmp


def get_ms2_chromatogram_ids_from_transition_ids(cursor, transition_ids):
    sql_query = "SELECT ID FROM CHROMATOGRAM WHERE NATIVE_ID IN ("

    for current_id in transition_ids:
        sql_query += "'" + current_id + "', "

    sql_query = sql_query[:-2]
    sql_query = sql_query + ') ORDER BY NATIVE_ID ASC'

    res = cursor.execute(sql_query)
    tmp = res.fetchall()

    # assert len(tmp) > 0, str(transition_ids)

    return tmp


def get_ms1_chromatogram_ids_from_precursor_id_and_isotope(
    cursor,
    prec_id,
    isotopes
):
    sql_query = "SELECT ID FROM CHROMATOGRAM WHERE NATIVE_ID IN ("

    for isotope in isotopes:
        sql_query += "'{0}_Precursor_i{1}', ".format(prec_id, isotope)

    sql_query = sql_query[:-2]
    sql_query = sql_query + ') ORDER BY NATIVE_ID ASC'

    res = cursor.execute(sql_query)
    tmp = res.fetchall()

    assert len(tmp) > 0, str(prec_id) + ' ' + str(isotope)

    return tmp


def get_chromatogram_labels_and_bbox(
    left_width,
    right_width,
    times
):
    row_labels = []

    for time in times:
        if left_width and right_width:
            if left_width <= time <= right_width:
                row_labels.append(1)
            else:
                row_labels.append(0)
        else:
            row_labels.append(0)

    row_labels = np.array(row_labels)

    label_idxs = np.where(row_labels == 1)[0]

    if len(label_idxs) > 0:
        bb_start, bb_end = label_idxs[0], label_idxs[-1]
    else:
        bb_start, bb_end = None, None

    return row_labels, bb_start, bb_end


def create_data_from_transition_ids(
    sqMass_dir,
    sqMass_filename,
    transition_ids,
    chromatogram_filename,
    left_width=None,
    right_width=None,
    prec_id=None,
    prec_charge=None,
    isotopes=[],
    library_intensities=[],
    lib_rt=None,
    extra_features=[],
    csv_only=False,
    window_size=175
):
    con = sqlite3.connect(os.path.join(sqMass_dir, sqMass_filename))
    cursor = con.cursor()
    ms2_transition_ids = get_ms2_chromatogram_ids_from_transition_ids(
        cursor, transition_ids)

    if len(ms2_transition_ids) == 0:
        print(f'Skipped {chromatogram_filename}, no transitions found')

        return -1, -1, -1, None

    ms2_transition_ids = [item[0] for item in ms2_transition_ids]
    transitions = SqlDataAccess(os.path.join(sqMass_dir, sqMass_filename))
    ms2_transitions = transitions.getDataForChromatograms(ms2_transition_ids)
    times = ms2_transitions[0][0]
    len_times = len(times)
    subsection_left, subsection_right = 0, len_times

    if left_width and right_width:
        row_labels, bb_start, bb_end = get_chromatogram_labels_and_bbox(
                left_width,
                right_width,
                times)
    else:
        row_labels, bb_start, bb_end = None, 'NA', 'NA'

    if not csv_only:
        num_expected_features = 6
        num_expected_extra_features = 0
        free_idx = 0

        if 'ms1' in extra_features:
            num_expected_extra_features += len(isotopes)

        if 'lib_int' in extra_features:
            num_expected_extra_features += 6

        if 'dist_rt' in extra_features:
            num_expected_extra_features += 1

        if 'prec_charge' in extra_features:
            num_expected_extra_features += 1

        chromatogram = np.zeros((num_expected_features, len_times))
        extra = np.zeros((num_expected_extra_features, len_times))
        ms2_transitions = np.array(
            [transition[1] for transition in ms2_transitions])

        assert ms2_transitions.shape[1] > 1, print(chromatogram_filename)

        chromatogram[0:ms2_transitions.shape[0]] = ms2_transitions

        if extra_features:
            extra_meta = {}

        if 'ms1' in extra_features:
            ms1_transition_ids = \
                get_ms1_chromatogram_ids_from_precursor_id_and_isotope(
                    cursor, prec_id, isotopes)
            ms1_transition_ids = [item[0] for item in ms1_transition_ids]
            ms1_transitions = transitions.getDataForChromatograms(
                ms1_transition_ids)
            ms1_transitions = np.array(
                [transition[1] for transition in ms1_transitions])

            if ms1_transitions.shape[1] > len_times:
                ms1_transitions = ms1_transitions[:, 0:len_times]
            elif ms1_transitions.shape[1] < len_times:
                padding = np.zeros((
                    ms1_transitions.shape[0],
                    len_times - ms1_transitions.shape[1]))
                ms1_transitions = np.concatenate(
                    (ms1_transitions, padding),
                    axis=1)

            extra[free_idx:free_idx + ms1_transitions.shape[0]] = (
                ms1_transitions)
            extra_meta['ms1_start'] = free_idx
            free_idx += len(isotopes)
            extra_meta['ms1_end'] = free_idx

        if 'lib_int' in extra_features:
            lib_int_features = np.repeat(
                library_intensities,
                len_times).reshape(len(library_intensities), len_times)
            extra[free_idx:free_idx + lib_int_features.shape[0]] = (
                lib_int_features)
            extra_meta['lib_int_start'] = free_idx
            free_idx += 6
            extra_meta['lib_int_end'] = free_idx

        if 'dist_rt' in extra_features:
            if not lib_rt:
                mid = len(times) // 2
                lib_rt = times[mid]

            dist_from_lib_rt = np.absolute(
                np.repeat(lib_rt, len_times) - np.array(times))
            extra[free_idx:free_idx + 1] = dist_from_lib_rt
            extra_meta['lib_rt'] = free_idx
            free_idx += 1

        if 'prec_charge' in extra_features:
            extra[free_idx:free_idx + 1] = prec_charge
            extra_meta['prec_charge'] = free_idx
            free_idx += 1

        if window_size >= 0:
            if lib_rt:
                subsection_left, subsection_right = get_subsequence_idxs(
                    times, lib_rt, window_size)
            else:
                mid = len(times) // 2
                subsection_left = mid - window_size // 2
                subsection_right = mid + window_size // 2

                if window_size % 2 != 0:
                    subsection_right += 1

            chromatogram = chromatogram[:, subsection_left:subsection_right]
            extra = extra[:, subsection_left:subsection_right]

            if left_width and right_width:
                row_labels = row_labels[subsection_left:subsection_right]

            if chromatogram.shape[1] != window_size:
                print(f'Skipped {chromatogram_filename}, misshapen matrix')

                return -1, -1, -1, None
            elif extra.shape[1] != window_size:
                print(f'Skipped {chromatogram_filename}, misshapen matrix')

                return -1, -1, -1, None
            
            if left_width and right_width and len(row_labels) != window_size:
                print(f'Skipped {chromatogram_filename}, misshapen matrix')

                return -1, -1, -1, None

            if left_width and right_width:
                label_idxs = np.where(row_labels == 1)[0]

                if len(label_idxs) > 0:
                    bb_start, bb_end = label_idxs[0], label_idxs[-1]

        chromatogram = np.concatenate((chromatogram, extra), axis=0)

        return row_labels, bb_start, bb_end, chromatogram

def get_cnn_data(
    out,
    osw_dir='.',
    osw_filename='merged.osw',
    sqMass_roots=[],
    extra_features=['ms1', 'lib_int', 'lib_rt', 'prec_charge'],
    isotopes=[0],
    csv_only=False,
    window_size=201,
    use_lib_rt=False,
    scored=False
):
    segmentation_labels_matrix, chromatograms_array, chromatograms_csv = [], [], []
    chromatogram_id = 0
    con = sqlite3.connect(os.path.join(osw_dir, osw_filename))
    cursor = con.cursor()
    prec_id_and_prec_mod_seqs_and_charges = get_mod_seqs_and_charges(cursor)
    labels_filename = f'{out}_osw_segmentation_labels_array'
    chromatograms_filename = f'{out}_chromatograms_array'
    csv_filename = f'{out}_chromatograms_csv.csv'

    for sqMass_root in sqMass_roots:
        print(sqMass_root)

        run_id = get_run_id_from_folder_name(cursor, sqMass_root)
        feature_info = get_feature_info_from_run(cursor, run_id)

        assert len(
            prec_id_and_prec_mod_seqs_and_charges) == len(
                feature_info), print(
                    len(prec_id_and_prec_mod_seqs_and_charges),
                    len(feature_info))

        for i in range(len(prec_id_and_prec_mod_seqs_and_charges)):
            print(i)

            prec_id, prec_mod_seq, prec_charge, decoy = (
                prec_id_and_prec_mod_seqs_and_charges[i])
            transition_ids_and_library_intensities = (
                get_transition_ids_and_library_intensities_from_prec_id(
                    cursor,
                    prec_id))
            transition_ids = \
                [str(x[0]) for x in transition_ids_and_library_intensities]
            library_intensities = \
                [x[1] for x in transition_ids_and_library_intensities]

            if scored:
                prec_id_2, exp_rt, delta_rt, left_width, right_width, score = (
                    feature_info[i])

                assert prec_id == prec_id_2, print(prec_id, prec_id_2)

                if use_lib_rt:
                    if exp_rt and delta_rt:
                        lib_rt = exp_rt - delta_rt
                    else:
                        print(
                            f'Skipped {chromatograms_filename} due to missing rt')

                        continue
                else:
                    lib_rt = None
            else:
                lib_rt, left_width, right_width, score = None, None, None, 'NA'

            repl_name = sqMass_root
            chromatogram_filename = [repl_name, prec_mod_seq, str(prec_charge)]

            if decoy == 1:
                chromatogram_filename.insert(0, 'DECOY')

            chromatogram_filename = '_'.join(chromatogram_filename)
            (
                labels,
                bb_start,
                bb_end,
                chromatogram) = create_data_from_transition_ids(
                sqMass_root,
                'output.sqMass',
                transition_ids,
                chromatogram_filename,
                left_width=left_width,
                right_width=right_width,
                prec_id=prec_id,
                prec_charge=prec_charge,
                isotopes=isotopes,
                library_intensities=library_intensities,
                lib_rt=lib_rt,
                extra_features=extra_features,
                csv_only=csv_only,
                window_size=window_size)

            if scored and not isinstance(labels, np.ndarray):
                continue

            if not isinstance(chromatogram, np.ndarray):
                continue

            chromatograms_array.append(chromatogram)

            if not csv_only and scored:
                segmentation_labels_matrix.append(labels)

            chromatograms_csv.append(
                [
                    chromatogram_id,
                    chromatogram_filename,
                    prec_id,
                    lib_rt,
                    window_size,
                    bb_start,
                    bb_end,
                    score])
            chromatogram_id += 1

    con.close()

    if not csv_only:
        np.save(
            chromatograms_filename,
            np.array(chromatograms_array, dtype=np.float32))

        if scored:
            np.save(
                labels_filename,
                np.array(segmentation_labels_matrix, dtype=np.float32))

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'ID',
                'Filename',
                'External Precursor ID',
                'External Library RT/RT IDX',
                'Window Size',
                'External Label Left IDX',
                'External Label Right IDX',
                'External Score'])
        writer.writerows(chromatograms_csv)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-out', '--out', type=str, default='osw_parser_out')
    parser.add_argument('-osw_dir', '--osw_dir', type=str, default='.')
    parser.add_argument('-osw_in', '--osw_in', type=str, default='merged.osw')
    parser.add_argument(
        '-in_folder',
        '--in_folder',
        type=str,
        default='hroest_K120808_Strep0PlasmaBiolRepl1_R01_SW')
    parser.add_argument(
        '-extra_features',
        '--extra_features',
        type=str,
        default='ms1,lib_int,dist_rt,prec_charge')
    parser.add_argument('-isotopes', '--isotopes', type=str, default='0')
    parser.add_argument(
        '-csv_only',
        '--csv_only',
        action='store_true',
        default=False)
    parser.add_argument('-window_size', '--window_size', type=int, default=175)
    parser.add_argument(
        '-use_lib_rt',
        '--use_lib_rt',
        action='store_true',
        default=False)
    parser.add_argument(
        '-scored',
        '--scored',
        action='store_true',
        default=False)
    args = parser.parse_args()

    args.in_folder = args.in_folder.split(',')
    args.isotopes = args.isotopes.split(',')
    args.extra_features = args.extra_features.split(',')

    print(args)

    out = None

    if not args.csv_only:
        out = args.out

    get_cnn_data(
        out=out,
        osw_dir=args.osw_dir,
        osw_filename=args.osw_in,
        sqMass_roots=args.in_folder,
        extra_features=args.extra_features,
        isotopes=args.isotopes,
        csv_only=args.csv_only,
        window_size=args.window_size,
        use_lib_rt=args.use_lib_rt,
        scored=args.scored)

    print('It took {0:0.1f} seconds'.format(time.time() - start))
