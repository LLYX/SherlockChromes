import argparse
import csv
import h5py
import io
import numpy as np
import os
import sqlite3
import tarfile
import time
import click

from general_utils import get_subsequence_idxs
from sql_data_access import SqlDataAccess


def get_run_id_from_folder_name(
    con,
    cursor,
    folder_name
):
    query = \
        """SELECT ID FROM RUN WHERE FILENAME LIKE '%{0}%'""".format(
            os.path.basename(folder_name) )
        
    print( query )
    res = cursor.execute(query)
    tmp = res.fetchall()
    print(tmp)
    assert len(tmp) == 1

    return tmp[0][0]


def get_mod_seqs_and_charges(
    con,
    cursor
):
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


def get_feature_info_from_run(
    con,
    cursor,
    run_id
):
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


def get_transition_ids_and_library_intensities_from_prec_id(
    con,
    cursor,
    prec_id,
    detecting=True,
    identifying=False
):
    query = \
        """SELECT ID, LIBRARY_INTENSITY
        FROM TRANSITION LEFT JOIN TRANSITION_PRECURSOR_MAPPING
        ON TRANSITION.ID = TRANSITION_ID
        WHERE PRECURSOR_ID = {0}
        AND (
      TRANSITION.DETECTING = {1} --- #detecting Include detecting transitions
      OR TRANSITION.IDENTIFYING = {2} --- #identifying Include identifying transitions
          )""".format(prec_id, detecting, identifying)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) > 0, prec_id

    return tmp


def get_ms2_chromatogram_ids_from_transition_ids(con, cursor, transition_ids):
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
    con,
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
    out,
    chromatogram_filename,
    left_width,
    right_width,
    prec_id=None,
    isotopes=[],
    library_intensities=[],
    exp_rt=None,
    extra_features=[],
    csv_only=False,
    window_size=201,
    mode='tar'
):
    con = sqlite3.connect(os.path.join(sqMass_dir, sqMass_filename))

    cursor = con.cursor()

    ms2_transition_ids = get_ms2_chromatogram_ids_from_transition_ids(
        con, cursor, transition_ids)

    if len(ms2_transition_ids) == 0:
        print(f'Skipped {chromatogram_filename}, no transitions found')

        return -1, -1, -1, None

    ms2_transition_ids = [item[0] for item in ms2_transition_ids]

    transitions = SqlDataAccess(os.path.join(sqMass_dir, sqMass_filename))

    ms2_transitions = transitions.getDataForChromatograms(
        ms2_transition_ids)

    times = ms2_transitions[0][0]
    len_times = len(times)
    subsection_left, subsection_right = 0, len_times

    row_labels, bb_start, bb_end = get_chromatogram_labels_and_bbox(
            left_width,
            right_width,
            times)

    if not csv_only:
        num_expected_features = 6
        num_expected_extra_features = 0
        free_idx = 0

        if 'ms1' in extra_features:
            num_expected_extra_features += len(isotopes)

        if 'lib_int' in extra_features:
            num_expected_extra_features += 6

        if 'exp_rt' in extra_features:
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
                    con, cursor, prec_id, isotopes)

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
                    len_times - ms1_transitions.shape[1]
                ))

                ms1_transitions = np.concatenate(
                    (ms1_transitions, padding),
                    axis=1
                )

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

        if 'exp_rt' in extra_features:
            dist_from_exp_rt = np.absolute(
                np.repeat(exp_rt, len_times) - np.array(times))

            extra[free_idx:free_idx + 1] = dist_from_exp_rt
            extra_meta['exp_rt'] = free_idx
            free_idx += 1

        if window_size >= 0:
            subsection_left, subsection_right = get_subsequence_idxs(
                times, exp_rt, window_size)

            chromatogram = chromatogram[:, subsection_left:subsection_right]
            extra = extra[:, subsection_left:subsection_right]
            row_labels = row_labels[subsection_left:subsection_right]

            if chromatogram.shape[1] != window_size:
                print(f'Skipped {chromatogram_filename}, misshapen matrix')

                return -1, -1, -1, None
            elif extra.shape[1] != window_size:
                print(f'Skipped {chromatogram_filename}, misshapen matrix')

                return -1, -1, -1, None
            elif len(row_labels) != window_size:
                print(f'Skipped {chromatogram_filename}, misshapen matrix')

                return -1, -1, -1, None

            label_idxs = np.where(row_labels == 1)[0]

            if len(label_idxs) > 0:
                bb_start, bb_end = label_idxs[0], label_idxs[-1]
            else:
                bb_start, bb_end = None, None

        if mode == 'npy':
            chromatogram = np.concatenate([chromatogram, extra], axis=0)

            return row_labels, bb_start, bb_end, chromatogram
        elif mode == 'hdf5':
            out.create_dataset(chromatogram_filename, data=chromatogram)

            if extra_features:
                extra_dset = out.create_dataset(
                    chromatogram_filename + '_Extra', data=extra)

                for feature in extra_meta:
                    extra_dset.attrs[feature] = extra_meta[feature]
        elif mode == 'tar':
            data = chromatogram.tobytes()
            with io.BytesIO(data) as f:
                info = tarfile.TarInfo(chromatogram_filename)
                info.size = len(data)
                out.addfile(info, f)

            if extra_features:
                data = extra.tobytes()
                with io.BytesIO(data) as f:
                    info = tarfile.TarInfo(chromatogram_filename + '_Extra')
                    info.size = len(data)
                    out.addfile(info, f)

    return row_labels, bb_start, bb_end, None


def get_cnn_data(
    out,
    osw_dir='.',
    osw_filename='merged.osw',
    sqMass_roots=[],
    detecting=True,
    identifying=False,
    extra_features=['ms1', 'lib_int', 'exp_rt'],
    isotopes=[0],
    csv_only=False,
    window_size=201,
    use_rt=False,
    scored=False,
    mode='tar'
):
    label_matrix, chromatograms_array, chromatograms_csv = [], [], []

    chromatogram_id = 0

    con = sqlite3.connect(os.path.join(osw_dir, osw_filename))
    cursor = con.cursor()

    prec_id_and_prec_mod_seqs_and_charges = get_mod_seqs_and_charges(
            con,
            cursor)

    labels_filename = f'{out}_osw_labels_array'
    chromatograms_filename = f'{out}_chromatograms_array'
    csv_filename = f'{out}_chromatograms_csv.csv'

    for sqMass_root in sqMass_roots:
        click.echo( "INFO: Working on Chrom File %s " % (sqMass_root) )

        run_id = get_run_id_from_folder_name(con, cursor, sqMass_root)

        if use_rt and scored:
            feature_info = get_feature_info_from_run(
                con,
                cursor,
                run_id)

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
                    con,
                    cursor,
                    prec_id,
                    detecting,
                    identifying))
            transition_ids = \
                [str(x[0]) for x in transition_ids_and_library_intensities]
            library_intensities = \
                [x[1] for x in transition_ids_and_library_intensities]

            if use_rt and scored:
                prec_id_2, exp_rt, delta_rt, left_width, right_width, score = (
                    feature_info[i])

                assert prec_id == prec_id_2, print(prec_id, prec_id_2)

                if exp_rt and delta_rt:
                    exp_rt = exp_rt - delta_rt
                else:
                    print(f'Skipped {chromatogram_filename} due to missing rt')

                    continue
            else:
                assert window_size == -1, print(
                    'Cannot subset without using library RT!')

            if not scored:
                # TODO: Implement extraction of OSW features only
                exp_rt, left_width, right_width, score = None, None, None, None
                bb_start, bb_end = None, None

            repl_name = sqMass_root

            chromatogram_filename = [repl_name, prec_mod_seq, str(prec_charge)]
            if decoy == 1:
                chromatogram_filename.insert(0, 'DECOY')

            chromatogram_filename = '_'.join(chromatogram_filename)

            if scored:
                (
                    labels,
                    bb_start,
                    bb_end,
                    chromatogram) = create_data_from_transition_ids(
                    sqMass_root,
                    'output.sqMass',
                    transition_ids,
                    out,
                    chromatogram_filename,
                    left_width,
                    right_width,
                    prec_id=prec_id,
                    isotopes=isotopes,
                    library_intensities=library_intensities,
                    exp_rt=exp_rt,
                    extra_features=extra_features,
                    csv_only=csv_only,
                    window_size=window_size,
                    mode=mode)

                if not isinstance(labels, np.ndarray):
                    continue

                if not isinstance(chromatogram, np.ndarray):
                    continue

                chromatograms_array.append(chromatogram)

            if not csv_only and scored:
                label_matrix.append(labels)

            chromatograms_csv.append(
                [
                    chromatogram_id,
                    chromatogram_filename,
                    prec_id,
                    exp_rt,
                    window_size,
                    bb_start,
                    bb_end,
                    score])
            chromatogram_id += 1

    con.close()

    if not csv_only and scored:
        if mode == 'npy':
            np.save(
                chromatograms_filename,
                np.vstack(chromatograms_array).astype(np.float32)
            )
            np.save(labels_filename, np.vstack(label_matrix).astype(np.int32))
        elif mode == 'hdf5':
            out.create_dataset(
                labels_filename, data=np.vstack(label_matrix))
        elif mode == 'tar':
            data = np.vstack(label_matrix).tobytes()
            with io.BytesIO(data) as f:
                info = tarfile.TarInfo(labels_filename)
                info.size = len(data)
                out.addfile(info, f)

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
                'External Score'
            ]
        )
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
    parser.add_argument('-no_detecting', '--no_detecting', default=False, help="Enable to exclude detecting transitions", action="store_false")
    parser.add_argument('-identifying', '--identifying', default=False, help="Enable to include identifying transitions", action="store_false")
    parser.add_argument(
        '-extra_features',
        '--extra_features',
        type=str,
        default='ms1,lib_int,exp_rt')
    parser.add_argument('-isotopes', '--isotopes', type=str, default='0')
    parser.add_argument(
        '-csv_only',
        '--csv_only',
        action='store_true',
        default=False)
    parser.add_argument('-window_size', '--window_size', type=int, default=201)
    parser.add_argument(
        '-use_rt',
        '--use_rt',
        action='store_true',
        default=False)
    parser.add_argument(
        '-scored',
        '--scored',
        action='store_true',
        default=False)
    parser.add_argument('-mode', '--mode', type=str, default='tar')
    args = parser.parse_args()

    args.in_folder = args.in_folder.split(',')
    args.isotopes = args.isotopes.split(',')
    args.extra_features = args.extra_features.split(',')

    print(args)

    out = None

    if not args.csv_only:
        if args.mode == 'npy':
            out = args.out
        elif args.mode == 'hdf5':
            out = hfpy.File(args.out + '.hdf5', 'w')
        elif args.mode == 'tar':
            out = tarfile.open(args.out + '.tar', 'w|')


    ## Set Detecting off
    if args.no_detecting:
        click.echo("WARN: Excluding Detecting Transitions...")
        detecting=False
    else:
        detecting=True

    ## Set Identifying on
    if args.identifying:
        click.echo("WARN: Including Identifying Transitions...")
        identifying=True
    else:
        identifying=False
    
    get_cnn_data(
        out=out,
        osw_dir=args.osw_dir,
        osw_filename=args.osw_in,
        sqMass_roots=args.in_folder,
        detecting=detecting,
        identifying=identifying,
        extra_features=args.extra_features,
        isotopes=args.isotopes,
        csv_only=args.csv_only,
        window_size=args.window_size,
        use_rt=args.use_rt,
        scored=args.scored,
        mode=args.mode)

    print('It took {0:0.1f} seconds'.format(time.time() - start))
