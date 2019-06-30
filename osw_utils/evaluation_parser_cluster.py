import time

def overlaps(
    pred_min,
    pred_max,
    target_min,
    target_max):
    if ((pred_min <= target_min and target_min <= pred_max <= target_max) or
        (pred_min <= target_min and pred_max >= target_max) or
        (target_min <= pred_min <= target_max and pred_max <= target_max) or
        (target_min <= pred_min <= target_max and pred_max >= target_max)):
        return True
    return False

def parse_model_evaluation_file(
    filename,
    osw_threshold=2.5,
    mod_threshold=0.5,
    mod_min_pts=1,
    exclusion_list=None):
    exclude = {}
    if exclusion_list:
        with open(exclusion_list, 'r') as exclusions:
            next(exclusions)
            for line in exclusions:
                line = line.split(',')
                seq_source = line[1]
                seq = seq_source.split('_')[-2]
                exclude[seq] = True

    mod_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    mod_tp, mod_fp, mod_tn, mod_fn = [], [], [], []

    mod_tp, mod_fp, mod_tn, mod_fn, osw_scores, target, pred = \
        [], [], [], [], [], [], []

    with open(filename, 'r') as infile:
        next(infile)
        for line in infile:
            line = line.rstrip('\r\n').split(',')
            
            (
                chrom_id,
                seq_source,
                osw_start,
                osw_end,
                mod_start,
                mod_end,
                osw_score,
                mod_score
            ) = line

            seq = seq_source.split('_')[-2]

            if seq not in exclude:
                osw_start = int(osw_start)
                osw_end = int(osw_end)

                if mod_start:
                    mod_start = int(mod_start)
                else:
                    mod_start = None

                if mod_end:
                    mod_end = int(mod_end)
                else:
                    mod_end = None

                if mod_start and mod_end:
                    if (mod_end - mod_start + 1) < mod_min_pts:
                        mod_start, mod_end = None, None

                osw_score = float(osw_score)
                mod_score = float(mod_score)

                osw_scores.append(osw_score)
                pred.append(mod_score)

                if osw_score < osw_threshold:
                    osw_start, osw_end = None, None

                if mod_score < mod_threshold:
                    mod_start, mod_end = None, None

                if not osw_start and mod_start:
                    mod_stats['fp']+= 1
                    mod_fp.append(
                        (chrom_id, osw_start, osw_end, mod_start, mod_end))
                    target.append(0)
                elif osw_start and not mod_start:
                    mod_stats['fn']+= 1
                    mod_fn.append(
                        (chrom_id, osw_start, osw_end, mod_start, mod_end))
                    target.append(1)
                elif not osw_start and not mod_start:
                    mod_stats['tn']+= 1
                    mod_tn.append(
                        (chrom_id, osw_start, osw_end, mod_start, mod_end))
                    target.append(0)
                else:
                    if overlaps(osw_start, osw_end, mod_start, mod_end):
                        mod_stats['tp']+= 1
                        mod_tp.append(
                            (chrom_id, osw_start, osw_end, mod_start, mod_end))
                        target.append(1)
                    else:
                        mod_stats['fp']+= 1
                        mod_fp.append(
                            (chrom_id, osw_start, osw_end, mod_start, mod_end))
                        target.append(0)

    print(mod_stats)

    return mod_tp, mod_fp, mod_tn, mod_fn, osw_scores, target, pred

if __name__ == '__main__':
    """
    Usage:

    tp, fp, tn, fn, osw_scores, target, pred = parse_model_evaluation_file(
        'evaluation_results/evaluation_results_all_32_decoy.csv',
        osw_threshold=2.1
        mod_threshold=0.5,
        mod_min_pts=5,
        exclusion_list='chromatograms_mixed.csv'
    )
    """
    start = time.time()

    print('It took {0:0.1f} seconds'.format(time.time() - start))

    pass
