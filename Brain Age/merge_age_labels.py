"""Merge ground-truth age labels onto the feature matrix from extract_features.py."""
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURES_CSV = '/home/ubuntu/pyDYNAM-O/output/features.csv'
SHHS1_DEMO   = '/home/ubuntu/sleepdata/shhs/datasets/shhs1-dataset-0.21.0.csv'
SHHS2_DEMO   = '/home/ubuntu/sleepdata/shhs/datasets/shhs2-dataset-0.21.0.csv'
OUTPUT_CSV   = '/home/ubuntu/pyDYNAM-O/output/labeled_features.csv'
# ─────────────────────────────────────────────────────────────────────────────


def load_demographics():
    """Load SHHS1 + SHHS2 demographics into a single (subject, age) dataframe."""
    rows = []
    for ds, path, age_col, prefix in [
        ('SHHS1', SHHS1_DEMO, 'age_s1', 'shhs1-'),
        ('SHHS2', SHHS2_DEMO, 'age_s2', 'shhs2-'),
    ]:
        df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        # Build subject id matching the feature matrix (e.g. 'shhs1-200001')
        df['subject'] = prefix + df['nsrrid'].astype(int).astype(str).str.zfill(6)
        rows.append(df[['subject', age_col]].rename(columns={age_col: 'age'}).assign(dataset=ds))
        print(f"  {ds}: {len(df)} demographics rows, {df[age_col].notna().sum()} with age")
    return pd.concat(rows, ignore_index=True)


def main():
    print(f"Loading features from {FEATURES_CSV}...")
    features = pd.read_csv(FEATURES_CSV, index_col='subject')
    print(f"  {len(features)} subjects, {features.shape[1]} columns")

    print("\nLoading demographics...")
    demo = load_demographics()

    # Join age onto features (inner join — drop subjects without age)
    print("\nJoining age labels...")
    labeled = features.join(demo.set_index('subject')[['age']], how='inner')

    n_before = len(features)
    n_after = len(labeled)
    print(f"  {n_before} → {n_after} subjects after join ({n_before - n_after} dropped)")

    # Drop rows missing age
    n_missing_age = labeled['age'].isna().sum()
    if n_missing_age > 0:
        labeled = labeled.dropna(subset=['age'])
        print(f"  Dropped {n_missing_age} subjects with NaN age → {len(labeled)} remaining")

    # Reorder: dataset, age first, then features
    cols = ['dataset', 'age'] + [c for c in labeled.columns if c not in ('dataset', 'age')]
    labeled = labeled[cols]

    labeled.to_csv(OUTPUT_CSV)
    print(f"\nSaved {len(labeled)} labeled subjects to {OUTPUT_CSV}")
    print(f"  Shape: {labeled.shape}")
    print(f"  Per dataset:")
    print(labeled.groupby('dataset').agg(n=('age', 'count'),
                                          age_min=('age', 'min'),
                                          age_max=('age', 'max'),
                                          age_mean=('age', 'mean')).round(1))


if __name__ == '__main__':
    main()
