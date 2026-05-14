"""Extract brain-age features from per-subject spectrogram CSVs (parallel)."""
import os
import glob
import time
import traceback
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASETS    = ['SHHS1', 'SHHS2']  # datasets to process
OUTPUT_ROOT = '/home/ubuntu/pyDYNAM-O/output'
N_JOBS      = cpu_count()
# ─────────────────────────────────────────────────────────────────────────────

NREM_STAGES = [1, 2, 3]  # N3, N2, N1 in pyDYNAM-O convention


def extract_features(subj_dir, dataset):
    """Compute per-stage and pooled-NREM features for one subject."""
    subject = os.path.basename(subj_dir.rstrip('/'))
    spect = pd.read_csv(f'{subj_dir}/spect.csv', index_col='freq_hz')
    stimes = pd.read_csv(f'{subj_dir}/stimes.csv')['time_sec'].values
    stages = pd.read_csv(f'{subj_dir}/stages.csv')

    # Align stage to each spectrogram time bin
    stage_of_bin = np.searchsorted(stages.Time.values, stimes, side='right') - 1
    stage_per_bin = stages.Stage.values[np.clip(stage_of_bin, 0, len(stages) - 1)]

    row = {'subject': subject, 'dataset': dataset}
    n_freqs = spect.shape[0]

    # Per-stage features (N3, N2, N1)
    for stage_id, stage_name in [(1, 'N3'), (2, 'N2'), (3, 'N1')]:
        mask = stage_per_bin == stage_id
        if mask.sum() < 100:  # skip if <5s of this stage
            for i in range(n_freqs):
                row[f'{stage_name}_mean_f{i}'] = np.nan
            row[f'{stage_name}_pct'] = mask.sum() / len(mask)
            continue
        means = spect.values[:, mask].mean(axis=1)
        for i, v in enumerate(means):
            row[f'{stage_name}_mean_f{i}'] = v
        row[f'{stage_name}_pct'] = mask.sum() / len(mask)

    # Pooled NREM features
    nrem_mask = np.isin(stage_per_bin, NREM_STAGES)
    if nrem_mask.sum() >= 100:
        means = spect.values[:, nrem_mask].mean(axis=1)
        stds  = spect.values[:, nrem_mask].std(axis=1)
        for i in range(n_freqs):
            row[f'NREM_mean_f{i}'] = means[i]
            row[f'NREM_std_f{i}']  = stds[i]
        row['NREM_pct'] = nrem_mask.sum() / len(nrem_mask)
    else:
        for i in range(n_freqs):
            row[f'NREM_mean_f{i}'] = np.nan
            row[f'NREM_std_f{i}']  = np.nan
        row['NREM_pct'] = nrem_mask.sum() / len(nrem_mask) if len(nrem_mask) else 0

    return row


def extract_safe(subj_dir, dataset):
    """Wrapper that returns None on failure instead of crashing the pool."""
    try:
        return extract_features(subj_dir, dataset)
    except Exception as err:
        subject = os.path.basename(subj_dir.rstrip('/'))
        print(f"FAILED {dataset}/{subject}: {type(err).__name__}: {err}")
        traceback.print_exc()
        return None


def main():
    # Collect (subj_dir, dataset) pairs from all datasets
    jobs = []
    for dataset in DATASETS:
        input_dir = f'{OUTPUT_ROOT}/{dataset}'
        subj_dirs = sorted(d for d in glob.glob(f'{input_dir}/*')
                           if os.path.isdir(d) and not os.path.basename(d).startswith('_'))
        print(f"{dataset}: found {len(subj_dirs)} subject directories")
        jobs.extend((d, dataset) for d in subj_dirs)

    print(f"\nTotal jobs: {len(jobs)}")
    print(f"Running with {N_JOBS} parallel workers...")

    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(extract_safe)(d, ds) for d, ds in jobs
    )
    elapsed = time.time() - t0

    rows = [r for r in results if r is not None]
    n_failed = len(results) - len(rows)
    df = pd.DataFrame(rows).set_index('subject')

    # Combined output
    combined_csv = f'{OUTPUT_ROOT}/features.csv'
    df.to_csv(combined_csv)

    # Per-dataset outputs
    for dataset in DATASETS:
        sub_df = df[df['dataset'] == dataset]
        per_ds_csv = f'{OUTPUT_ROOT}/{dataset}_features.csv'
        sub_df.to_csv(per_ds_csv)
        print(f"  {dataset}: {len(sub_df)} subjects → {per_ds_csv}")

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Subjects processed: {len(rows)}")
    print(f"  Subjects failed:    {n_failed}")
    print(f"  Feature matrix:     {df.shape}")
    print(f"  Combined CSV:       {combined_csv}")


if __name__ == '__main__':
    main()
