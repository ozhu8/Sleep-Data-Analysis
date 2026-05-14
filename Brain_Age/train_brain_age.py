"""Train a brain-age regression model from the labeled feature matrix."""
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

# ── CONFIG ────────────────────────────────────────────────────────────────────
LABELED_CSV     = '/home/ubuntu/pyDYNAM-O/output/labeled_features.csv'
PREDICTIONS_CSV = '/home/ubuntu/pyDYNAM-O/output/predictions.csv'
MODEL_PKL       = '/home/ubuntu/pyDYNAM-O/output/brain_age_model.pkl'
METRICS_CSV     = '/home/ubuntu/pyDYNAM-O/output/metrics.csv'
N_SPLITS        = 5
RANDOM_STATE    = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_data():
    df = pd.read_csv(LABELED_CSV, index_col='subject')
    print(f"Loaded {len(df)} subjects with {df.shape[1]} columns")

    y = df['age'].values
    datasets = df['dataset'].values
    X = df.drop(columns=['age', 'dataset'])

    # Drop columns that are entirely NaN
    n_cols_before = X.shape[1]
    X = X.dropna(axis=1, how='all')
    if X.shape[1] < n_cols_before:
        print(f"  Dropped {n_cols_before - X.shape[1]} all-NaN columns")

    # Fill remaining NaN with per-column median (subjects missing some stages)
    n_nan = int(X.isna().sum().sum())
    if n_nan > 0:
        X = X.fillna(X.median())
        print(f"  Filled {n_nan} NaN values with column medians")

    print(f"  Feature matrix: {X.shape}, age range {y.min():.0f}-{y.max():.0f}, mean {y.mean():.1f}")
    return X, y, datasets, df.index.values


def evaluate(y_true, y_pred, datasets, label=''):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    print(f"\n{label}")
    print(f"  MAE:     {mae:.2f} years")
    print(f"  R²:      {r2:.3f}")
    print(f"  Pearson: {r:.3f}")

    rows = [{'group': label or 'overall', 'n': len(y_true),
             'mae': mae, 'r2': r2, 'pearson_r': r}]

    # Per-dataset breakdown
    for ds in np.unique(datasets):
        mask = datasets == ds
        if mask.sum() < 10:
            continue
        mae_ds = mean_absolute_error(y_true[mask], y_pred[mask])
        r2_ds = r2_score(y_true[mask], y_pred[mask])
        r_ds, _ = pearsonr(y_true[mask], y_pred[mask])
        rows.append({'group': ds, 'n': int(mask.sum()),
                     'mae': mae_ds, 'r2': r2_ds, 'pearson_r': r_ds})
        print(f"  {ds}: MAE={mae_ds:.2f}, R²={r2_ds:.3f}, r={r_ds:.3f}  (n={mask.sum()})")

    return rows


def main():
    X, y, datasets, subjects = load_data()

    # 5-fold CV with out-of-fold predictions
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros_like(y, dtype=float)

    print(f"\nRunning {N_SPLITS}-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(X.iloc[test_idx])
        fold_mae = mean_absolute_error(y[test_idx], oof_pred[test_idx])
        print(f"  Fold {fold}/{N_SPLITS}: MAE={fold_mae:.2f}  (train={len(train_idx)}, test={len(test_idx)})")

    # Overall metrics
    metrics_rows = evaluate(y, oof_pred, datasets, label='5-fold CV (out-of-fold)')

    # Brain-age gap (predicted - actual)
    bag = oof_pred - y

    # Save predictions for downstream analysis
    pred_df = pd.DataFrame({
        'subject': subjects,
        'dataset': datasets,
        'age': y,
        'predicted_age': oof_pred,
        'brain_age_gap': bag,
    }).set_index('subject')
    pred_df.to_csv(PREDICTIONS_CSV)
    print(f"\nPredictions saved to {PREDICTIONS_CSV}")

    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    print(f"Metrics saved to {METRICS_CSV}")

    # Train final model on all data and save
    print("\nTraining final model on all data...")
    final_model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X, y)
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump({'model': final_model, 'feature_names': X.columns.tolist()}, f)
    print(f"Final model saved to {MODEL_PKL}")

    # Top feature importances
    imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(imp.head(15).to_string())


if __name__ == '__main__':
    main()
