#!/usr/bin/env python3
"""
Generate missing evaluation results from the scores feather file.

This script fills in items 4-11 from model_report_missing_results.md that
can be derived from the existing scores_all_splits.feather file.

Items that CANNOT be generated:
  - Item 1: Variable bin plots (need raw feature values, not in feather)
  - Item 2: Feature Chinese descriptions (need business knowledge)
  - Item 3: MOB risk metrics (need future-period repayment data)
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
FEATHER = '/root/notebook/自动化建模/jy-model-agent/projects/2026-05-fujie-gcard-v1/runs/model_scores/scores_all_splits.feather'
OUT_DIR = BASE  # same directory as this script

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading feather data...")
df = pd.read_feather(FEATHER)
df['mdl_month'] = pd.to_datetime(df['mdl_dte']).dt.strftime('%Y-%m')

# Map blue_customer_flag to segment
SEGMENT_MAP = {'E3': '老户', 'E2': '次新', 'B2': '流失户'}
df['segment'] = df['blue_customer_flag'].map(SEGMENT_MAP)
# 老户次新 = 老户 + 次新
df['segment_ln'] = df['blue_customer_flag'].map({'E2': '老户次新', 'E3': '老户次新', 'B2': '流失户'})

# Ensure numeric score columns
for c in ['gcard_v2', 'gcard_v4', 'gcard_v5', 'gcard_v6']:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors='coerce')

SCORE_COLS = ['model_score', 'gcard_v2', 'gcard_v4', 'gcard_v5', 'gcard_v6']
LABEL = 'ftr_30d_ord_flag'

# ── Helper: compute KS ─────────────────────────────────────────────────────
def compute_ks(y_true, y_score):
    """Compute KS statistic."""
    df_ks = pd.DataFrame({'y': y_true, 'score': y_score})
    df_ks = df_ks.sort_values('score', ascending=False)
    df_ks['cum_pos'] = df_ks['y'].cumsum() / df_ks['y'].sum()
    df_ks['cum_neg'] = (1 - df_ks['y']).cumsum() / (1 - df_ks['y']).sum()
    return (df_ks['cum_pos'] - df_ks['cum_neg']).abs().max()

def compute_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan

# ══════════════════════════════════════════════════════════════════════════════
# ITEM 4: Decile lift bins with score boundaries
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Item 4] Generating decile_lift_bins.csv ...")

def decile_label(i):
    """Return decile label 001-010 (1=lowest score, 10=highest)."""
    return f"{i:03d}"

def compute_decile_bins(group_df, score_col):
    """Compute decile lift with score bounds for a group."""
    n = len(group_df)
    if n == 0:
        return None
    # Decile 1 = lowest scores, Decile 10 = highest
    group_df = group_df.sort_values(score_col)
    group_df['decile'] = pd.qcut(group_df[score_col], q=10, labels=False, duplicates='drop') + 1
    # But qcut labels 0-based; we want decile 10 = highest score
    # Actually qcut gives 0-9, +1 gives 1-10. But qcut's 0 is lowest, 9 is highest
    # That means decile 1 = lowest score, decile 10 = highest. Correct.

    bins = []
    total_bad = group_df[LABEL].sum()
    total_n = len(group_df)
    overall_bad_rate = total_bad / total_n if total_n > 0 else 0

    for d in range(10, 0, -1):  # from high to low (decile 10 -> 1)
        chunk = group_df[group_df['decile'] == d]
        n_samples = len(chunk)
        bad = chunk[LABEL].sum()
        bad_rate = bad / n_samples if n_samples > 0 else 0
        pct = n_samples / total_n if total_n > 0 else 0
        score_min = chunk[score_col].min()
        score_max = chunk[score_col].max()

        bins.append({
            'decile': f"{d:03d}",
            'decile_int': d,
            'n_samples': n_samples,
            'pct': pct,
            'bad': int(bad),
            'bad_rate': bad_rate,
            'score_min': score_min,
            'score_max': score_max,
            'lower_bound': f"({score_min:.6f}, {score_max:.6f}]",
        })
    return pd.DataFrame(bins)

decile_rows = []
for segment in ['全客群', '老户', '次新', '流失户', '老户次新']:
    for final_flag in ['DEV', 'DEV-OOS', 'OOT', 'OOT-OOS']:
        if segment == '全客群':
            mask = df['final_flag'] == final_flag
        elif segment == '老户次新':
            mask = (df['final_flag'] == final_flag) & (df['segment_ln'] == '老户次新')
        else:
            mask = (df['final_flag'] == final_flag) & (df['segment'] == segment)

        sub = df[mask]
        if len(sub) == 0:
            continue

        for score_col in ['model_score']:  # focus on main model_score
            bins_df = compute_decile_bins(sub.copy(), score_col)
            if bins_df is not None:
                bins_df['segment'] = segment
                bins_df['final_flag'] = final_flag
                bins_df['score_column'] = score_col
                decile_rows.append(bins_df)

decile_bins_all = pd.concat(decile_rows, ignore_index=True)
decile_bins_all.to_csv(os.path.join(OUT_DIR, 'decile_lift_bins.csv'), index=False)
print(f"  -> {len(decile_bins_all)} rows written to decile_lift_bins.csv")


# ══════════════════════════════════════════════════════════════════════════════
# ITEMS 5-8: Segment-level intent/zc matrices for DEV-OOS (老户/流失户)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Items 5-8] Generating segment-level intent/zc matrices ...")

# Focus: DEV-OOS, segment = 老户 or 流失户
# intent_level is computed per segment (等频三份 within each segment)
dev_oos_mask = df['final_flag'] == 'DEV-OOS'
seg_mask = df['segment'].isin(['老户', '流失户'])
sub_df = df[dev_oos_mask & seg_mask].copy()

dist_parts = []
ftr_parts = []
amt_parts = []

for seg_name in ['老户', '流失户']:
    seg_df = sub_df[sub_df['segment'] == seg_name].copy()
    seg_total = len(seg_df)
    # Compute intent_level within this segment (等频三份 per segment)
    seg_df['intent_level'] = pd.qcut(seg_df['model_score'], q=3,
                                      labels=['低意愿', '中意愿', '高意愿'])

    # --- Item 5: Distribution matrix (n_samples, sample_pct, row_pct, col_pct) ---
    dist = seg_df.groupby(['intent_level', 'zc_level']).agg(
        n_samples=('uid', 'count')
    ).reset_index()
    dist['sample_pct'] = dist['n_samples'] / seg_total
    row_totals = dist.groupby('intent_level')['n_samples'].transform('sum')
    dist['row_pct'] = dist['n_samples'] / row_totals
    col_totals = dist.groupby('zc_level')['n_samples'].transform('sum')
    dist['col_pct'] = dist['n_samples'] / col_totals

    # Row sums
    row_sums = seg_df.groupby('intent_level').agg(n_samples=('uid', 'count')).reset_index()
    row_sums['zc_level'] = '合计'
    row_sums['sample_pct'] = row_sums['n_samples'] / seg_total
    row_sums['row_pct'] = 1.0
    row_sums['col_pct'] = np.nan

    # Column sums
    col_sums = seg_df.groupby('zc_level').agg(n_samples=('uid', 'count')).reset_index()
    col_sums['intent_level'] = '合计'
    col_sums['sample_pct'] = col_sums['n_samples'] / seg_total
    col_sums['col_pct'] = 1.0
    col_sums['row_pct'] = np.nan

    dist_full = pd.concat([dist, row_sums, col_sums], ignore_index=True)
    dist_full['segment'] = seg_name
    dist_full['score_version'] = 'model_score'
    dist_full['final_flag'] = 'DEV-OOS'
    dist_parts.append(dist_full)

    # --- Item 6: FTR 30d rate matrix ---
    ftr = seg_df.groupby(['intent_level', 'zc_level']).agg(
        n_samples=('uid', 'count'),
        ftr_30d_count=(LABEL, 'sum'),
    ).reset_index()
    row_ftr = seg_df.groupby('intent_level').agg(
        n_samples=('uid', 'count'), ftr_30d_count=(LABEL, 'sum')).reset_index()
    row_ftr['zc_level'] = '合计'
    col_ftr = seg_df.groupby('zc_level').agg(
        n_samples=('uid', 'count'), ftr_30d_count=(LABEL, 'sum')).reset_index()
    col_ftr['intent_level'] = '合计'
    ftr_full = pd.concat([ftr, row_ftr, col_ftr], ignore_index=True)
    ftr_full['ftr_30d_rate'] = ftr_full['ftr_30d_count'] / ftr_full['n_samples']
    ftr_full['segment'] = seg_name
    ftr_full['score_version'] = 'model_score'
    ftr_full['final_flag'] = 'DEV-OOS'
    ftr_parts.append(ftr_full)

    # --- Item 7: Amount overdue rate (3-period) matrix ---
    amt = seg_df.groupby(['intent_level', 'zc_level']).agg(
        n_samples=('uid', 'count'),
        total_principal=('prc_amt_xz_30d_3m', 'sum'),
        total_overdue=('ovd_amt_xz_30d_3m', 'sum'),
    ).reset_index()
    row_amt = seg_df.groupby('intent_level').agg(
        n_samples=('uid', 'count'),
        total_principal=('prc_amt_xz_30d_3m', 'sum'),
        total_overdue=('ovd_amt_xz_30d_3m', 'sum'),
    ).reset_index()
    row_amt['zc_level'] = '合计'
    col_amt = seg_df.groupby('zc_level').agg(
        n_samples=('uid', 'count'),
        total_principal=('prc_amt_xz_30d_3m', 'sum'),
        total_overdue=('ovd_amt_xz_30d_3m', 'sum'),
    ).reset_index()
    col_amt['intent_level'] = '合计'
    amt_full = pd.concat([amt, row_amt, col_amt], ignore_index=True)
    amt_full['amount_overdue_rate'] = np.where(
        amt_full['total_principal'] > 0,
        amt_full['total_overdue'] / amt_full['total_principal'], 0)
    amt_full['segment'] = seg_name
    amt_full['score_version'] = 'model_score'
    amt_full['final_flag'] = 'DEV-OOS'
    amt_parts.append(amt_full)

# Write combined CSVs
pd.concat(dist_parts, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, 'intent_zc_segment_distribution.csv'), index=False)
pd.concat(ftr_parts, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, 'intent_zc_segment_ftr_rate.csv'), index=False)
pd.concat(amt_parts, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, 'intent_zc_segment_amount_risk.csv'), index=False)
print(f"  -> intent_zc_segment_distribution.csv ({sum(len(p) for p in dist_parts)} rows)")
print(f"  -> intent_zc_segment_ftr_rate.csv ({sum(len(p) for p in ftr_parts)} rows)")
print(f"  -> intent_zc_segment_amount_risk.csv ({sum(len(p) for p in amt_parts)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 9: Monthly segment metrics for OOT-OOS (老户次新, 流失户)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Item 9] Generating monthly segment OOT-OOS metrics ...")

oot_oos_df = df[(df['final_flag'] == 'OOT-OOS') & (df['segment_ln'].isin(['老户次新', '流失户']))].copy()

monthly_rows = []
for (mdl_month, segment), group in oot_oos_df.groupby(['mdl_month', 'segment_ln']):
    row = {'mdl_month': mdl_month, 'segment': segment, 'final_flag': 'OOT-OOS'}
    row['n_samples'] = len(group)
    row['positive'] = int(group[LABEL].sum())
    row['bad_rate'] = group[LABEL].mean()

    for sc in ['model_score']:
        valid = group.dropna(subset=[sc])
        if len(valid) > 0 and valid[LABEL].nunique() >= 2:
            row['auc'] = compute_auc(valid[LABEL], valid[sc])
            row['ks'] = compute_ks(valid[LABEL], valid[sc])
        else:
            row['auc'] = np.nan
            row['ks'] = np.nan
    monthly_rows.append(row)

monthly_metrics = pd.DataFrame(monthly_rows)
monthly_metrics = monthly_metrics.sort_values(['segment', 'mdl_month'])
monthly_metrics.to_csv(os.path.join(OUT_DIR, 'monthly_segment_metrics_oot_oos.csv'), index=False)
print(f"  -> monthly_segment_metrics_oot_oos.csv ({len(monthly_metrics)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 10: Segment model comparison vs full-population model
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Item 10] Generating segment model comparison ...")

comparison_rows = []
for final_flag in ['DEV', 'DEV-OOS', 'OOT', 'OOT-OOS']:
    ff_df = df[df['final_flag'] == final_flag]

    # Full population
    valid = ff_df.dropna(subset=['model_score'])
    if valid[LABEL].nunique() >= 2:
        all_auc = compute_auc(valid[LABEL], valid['model_score'])
        all_ks = compute_ks(valid[LABEL], valid['model_score'])
    else:
        all_auc, all_ks = np.nan, np.nan

    # By segment
    for seg in ['老户', '次新', '流失户']:
        seg_df = ff_df[ff_df['segment'] == seg]
        valid_seg = seg_df.dropna(subset=['model_score'])
        if len(valid_seg) > 0 and valid_seg[LABEL].nunique() >= 2:
            seg_auc = compute_auc(valid_seg[LABEL], valid_seg['model_score'])
            seg_ks = compute_ks(valid_seg[LABEL], valid_seg['model_score'])
        else:
            seg_auc, seg_ks = np.nan, np.nan

        comparison_rows.append({
            'segment': seg,
            'final_flag': final_flag,
            'score_column': 'model_score',
            'n_samples': len(seg_df),
            'positive': int(seg_df[LABEL].sum()),
            'bad_rate': seg_df[LABEL].mean(),
            'auc': seg_auc,
            'ks': seg_ks,
            'all_population_auc': all_auc,
            'all_population_ks': all_ks,
            'auc_diff_vs_all': seg_auc - all_auc if not np.isnan(seg_auc) and not np.isnan(all_auc) else np.nan,
            'ks_diff_vs_all': seg_ks - all_ks if not np.isnan(seg_ks) and not np.isnan(all_ks) else np.nan,
        })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(os.path.join(OUT_DIR, 'segment_model_comparison.csv'), index=False)
print(f"  -> segment_model_comparison.csv ({len(comparison_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 11: Model score bin distribution by month (for stability)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Item 11] Generating model score bin distribution by month ...")

# Use DEV first month (2025-06) as baseline, create fixed 10-decile boundaries
baseline_df = df[(df['final_flag'] == 'DEV') & (df['mdl_month'] == '2025-06')].copy()
_, bin_edges = pd.qcut(baseline_df['model_score'], q=10, retbins=True, duplicates='drop')
# bin_edges has 11 elements; create 10 bins

bin_rows = []
for mdl_month in sorted(df['mdl_month'].unique()):
    month_df = df[df['mdl_month'] == mdl_month].copy()
    month_df['score_bin'] = pd.cut(month_df['model_score'], bins=bin_edges,
                                    labels=False, include_lowest=True) + 1
    total_month = len(month_df)

    for d in range(1, 11):
        chunk = month_df[month_df['score_bin'] == d]
        n = len(chunk)
        pct = n / total_month if total_month > 0 else 0
        bad_rate = chunk[LABEL].mean() if n > 0 else 0
        base_pct = 0.1  # equal-frequency baseline
        if pct > 0 and base_pct > 0:
            psi_comp = (pct - base_pct) * np.log(pct / base_pct)
        else:
            psi_comp = 0

        bin_rows.append({
            'mdl_month': mdl_month,
            'score_column': 'model_score',
            'score_decile': f"{d:03d}",
            'score_min': bin_edges[d-1],
            'score_max': bin_edges[d],
            'n_samples': n,
            'pct': pct,
            'bad_rate': bad_rate,
            'baseline_pct': base_pct,
            'psi_component': psi_comp,
        })

bin_dist = pd.DataFrame(bin_rows)
month_psi = bin_dist.groupby('mdl_month')['psi_component'].sum().reset_index()
month_psi.columns = ['mdl_month', 'month_psi']
bin_dist = bin_dist.merge(month_psi, on='mdl_month', how='left')

# Add score bounds text
bin_dist['lower_bound'] = bin_dist.apply(
    lambda r: f"({r['score_min']:.6f}, {r['score_max']:.6f}]", axis=1)

out_path = os.path.join(OUT_DIR, 'model_score_bin_distribution_by_month.csv')
bin_dist.to_csv(out_path, index=False)
print(f"  -> model_score_bin_distribution_by_month.csv ({len(bin_dist)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generation complete!")
print("=" * 60)
print("\nFiles generated:")
for f in [
    'decile_lift_bins.csv',
    'intent_zc_segment_distribution.csv',
    'intent_zc_segment_ftr_rate.csv',
    'intent_zc_segment_amount_risk.csv',
    'monthly_segment_metrics_oot_oos.csv',
    'segment_model_comparison.csv',
    'model_score_bin_distribution_by_month.csv',
]:
    path = os.path.join(OUT_DIR, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  OK  {f} ({size:,} bytes)")
    else:
        print(f"  MISS  {f}")
