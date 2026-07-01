#!/usr/bin/env python3
"""
Generate versioned (multi-score-column) comparison metrics.

Covers: model_score, gcard_v2, gcard_v4, gcard_v5, gcard_v6
Outputs all go into the evaluation/ directory.
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
OUT = BASE

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading feather...")
df = pd.read_feather(FEATHER)
df['mdl_month'] = pd.to_datetime(df['mdl_dte']).dt.strftime('%Y-%m')

# Score columns
SCORE_COLS = ['model_score', 'gcard_v2', 'gcard_v4', 'gcard_v5', 'gcard_v6']

# Convert string score cols to numeric (Arrow dtypes use "str" not "object")
for c in ['gcard_v2', 'gcard_v4', 'gcard_v6']:
    if str(df[c].dtype) in ('object', 'str', 'string', 'large_string'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Segment mapping
SEGMENT_MAP = {'E3': '老户', 'E2': '次新', 'B2': '流失户'}
df['segment'] = df['blue_customer_flag'].map(SEGMENT_MAP)
df['segment_ln'] = df['blue_customer_flag'].map({'E3': '老户次新', 'E2': '老户次新', 'B2': '流失户'})

LABEL = 'ftr_30d_ord_flag'


# ── Helpers ────────────────────────────────────────────────────────────────
def calc_auc(y_true, y_score):
    valid = ~(y_score.isna() | y_true.isna())
    yt, ys = y_true[valid], y_score[valid]
    if len(yt) < 2 or yt.nunique() < 2:
        return np.nan
    try:
        return roc_auc_score(yt, ys)
    except Exception:
        return np.nan


def calc_ks(y_true, y_score):
    valid = ~(y_score.isna() | y_true.isna())
    yt, ys = y_true[valid], y_score[valid]
    if len(yt) < 2:
        return np.nan
    df_ks = pd.DataFrame({'y': yt, 's': ys}).sort_values('s', ascending=False)
    df_ks['cp'] = df_ks['y'].cumsum() / df_ks['y'].sum()
    df_ks['cn'] = (1 - df_ks['y']).cumsum() / (1 - df_ks['y']).sum()
    return (df_ks['cp'] - df_ks['cn']).abs().max()


def add_row_col_sums(data, group_cols, value_cols, seg_total):
    """
    Append row (intent/zc level) sums and column (合计) sums to a grouped DataFrame.
    data: result of groupby with reset_index, has group_cols and value_cols.
    Returns DataFrame with extra 合计 rows.
    """
    parts = [data.copy()]

    # Row sums (per intent_level)
    for gcol in group_cols:
        row_sums = data.groupby(gcol, as_index=False).agg(
            {vc: 'sum' if data[vc].dtype in ('int64', 'float64') else 'count'
             for vc in value_cols if vc not in group_cols})
        # Fill the other group column(s) with '合计'
        for oc in group_cols:
            if oc != gcol:
                row_sums[oc] = '合计'
        parts.append(row_sums)

    # Grand total row (all group cols = '合计')
    grand = {gc: '合计' for gc in group_cols}
    for vc in value_cols:
        if vc not in group_cols:
            if data[vc].dtype in ('int64', 'float64'):
                grand[vc] = data[vc].sum()
            else:
                grand[vc] = len(data)
    parts.append(pd.DataFrame([grand]))

    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILE 1. intent_zc_segment_*_by_version.csv  (distribution, ftr_rate, amount_risk)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Generating intent/zc segment by version ...")

dev_oos = df[df['final_flag'] == 'DEV-OOS'].copy()

dist_rows, ftr_rows, amt_rows = [], [], []

for seg_name in ['老户', '流失户']:
    seg_mask = dev_oos['segment'] == seg_name
    seg_df_base = dev_oos[seg_mask].copy()

    for sv in SCORE_COLS:
        seg_df = seg_df_base.dropna(subset=[sv]).copy()
        if len(seg_df) == 0:
            continue
        seg_total = len(seg_df)

        # intent_level: equal-frequency thirds per (segment, score_version)
        try:
            seg_df['intent_level'] = pd.qcut(seg_df[sv], q=3,
                                              labels=['低意愿', '中意愿', '高意愿'],
                                              duplicates='drop')
        except ValueError:
            print(f"  WARN: qcut failed for {seg_name}/{sv}, skipping")
            continue

        # ── Distribution ──
        g_dist = seg_df.groupby(['intent_level', 'zc_level']).size().reset_index(name='n_samples')
        g_dist['sample_pct'] = g_dist['n_samples'] / seg_total
        row_tot = g_dist.groupby('intent_level')['n_samples'].transform('sum')
        g_dist['row_pct'] = g_dist['n_samples'] / row_tot
        col_tot = g_dist.groupby('zc_level')['n_samples'].transform('sum')
        g_dist['col_pct'] = g_dist['n_samples'] / col_tot
        g_dist['segment'] = seg_name
        g_dist['final_flag'] = 'DEV-OOS'
        g_dist['score_version'] = sv
        dist_rows.append(g_dist)

        # Row sums for distribution
        rsum = seg_df.groupby('intent_level').size().reset_index(name='n_samples')
        rsum['zc_level'] = '合计'
        rsum['sample_pct'] = rsum['n_samples'] / seg_total
        rsum['row_pct'] = 1.0
        rsum['col_pct'] = np.nan
        rsum['segment'] = seg_name
        rsum['final_flag'] = 'DEV-OOS'
        rsum['score_version'] = sv
        dist_rows.append(rsum)

        csum = seg_df.groupby('zc_level').size().reset_index(name='n_samples')
        csum['intent_level'] = '合计'
        csum['sample_pct'] = csum['n_samples'] / seg_total
        csum['col_pct'] = 1.0
        csum['row_pct'] = np.nan
        csum['segment'] = seg_name
        csum['final_flag'] = 'DEV-OOS'
        csum['score_version'] = sv
        dist_rows.append(csum)

        # ── FTR rate ──
        g_ftr = seg_df.groupby(['intent_level', 'zc_level']).agg(
            n_samples=('uid', 'count'),
            ftr_30d_count=(LABEL, 'sum')
        ).reset_index()
        g_ftr['ftr_30d_rate'] = g_ftr['ftr_30d_count'] / g_ftr['n_samples']
        g_ftr['segment'] = seg_name
        g_ftr['final_flag'] = 'DEV-OOS'
        g_ftr['score_version'] = sv
        ftr_rows.append(g_ftr)

        # FTR row sums
        r_ftr = seg_df.groupby('intent_level').agg(
            n_samples=('uid', 'count'), ftr_30d_count=(LABEL, 'sum')).reset_index()
        r_ftr['zc_level'] = '合计'
        r_ftr['ftr_30d_rate'] = r_ftr['ftr_30d_count'] / r_ftr['n_samples']
        r_ftr['segment'] = seg_name
        r_ftr['final_flag'] = 'DEV-OOS'
        r_ftr['score_version'] = sv
        ftr_rows.append(r_ftr)

        c_ftr = seg_df.groupby('zc_level').agg(
            n_samples=('uid', 'count'), ftr_30d_count=(LABEL, 'sum')).reset_index()
        c_ftr['intent_level'] = '合计'
        c_ftr['ftr_30d_rate'] = c_ftr['ftr_30d_count'] / c_ftr['n_samples']
        c_ftr['segment'] = seg_name
        c_ftr['final_flag'] = 'DEV-OOS'
        c_ftr['score_version'] = sv
        ftr_rows.append(c_ftr)

        # ── Amount risk ──
        g_amt = seg_df.groupby(['intent_level', 'zc_level']).agg(
            n_samples=('uid', 'count'),
            total_principal=('prc_amt_xz_30d_3m', 'sum'),
            total_overdue=('ovd_amt_xz_30d_3m', 'sum')
        ).reset_index()
        g_amt['amount_overdue_rate'] = np.where(
            g_amt['total_principal'] > 0,
            g_amt['total_overdue'] / g_amt['total_principal'], 0)
        g_amt['segment'] = seg_name
        g_amt['final_flag'] = 'DEV-OOS'
        g_amt['score_version'] = sv
        amt_rows.append(g_amt)

        r_amt = seg_df.groupby('intent_level').agg(
            n_samples=('uid', 'count'),
            total_principal=('prc_amt_xz_30d_3m', 'sum'),
            total_overdue=('ovd_amt_xz_30d_3m', 'sum')).reset_index()
        r_amt['zc_level'] = '合计'
        r_amt['amount_overdue_rate'] = np.where(
            r_amt['total_principal'] > 0,
            r_amt['total_overdue'] / r_amt['total_principal'], 0)
        r_amt['segment'] = seg_name
        r_amt['final_flag'] = 'DEV-OOS'
        r_amt['score_version'] = sv
        amt_rows.append(r_amt)

        c_amt = seg_df.groupby('zc_level').agg(
            n_samples=('uid', 'count'),
            total_principal=('prc_amt_xz_30d_3m', 'sum'),
            total_overdue=('ovd_amt_xz_30d_3m', 'sum')).reset_index()
        c_amt['intent_level'] = '合计'
        c_amt['amount_overdue_rate'] = np.where(
            c_amt['total_principal'] > 0,
            c_amt['total_overdue'] / c_amt['total_principal'], 0)
        c_amt['segment'] = seg_name
        c_amt['final_flag'] = 'DEV-OOS'
        c_amt['score_version'] = sv
        amt_rows.append(c_amt)

# Write
cols_order_dist = ['segment', 'final_flag', 'score_version', 'intent_level', 'zc_level',
                    'n_samples', 'sample_pct', 'row_pct', 'col_pct']
pd.concat(dist_rows, ignore_index=True)[cols_order_dist].to_csv(
    os.path.join(OUT, 'intent_zc_segment_distribution_by_version.csv'), index=False)

cols_order_ftr = ['segment', 'final_flag', 'score_version', 'intent_level', 'zc_level',
                   'n_samples', 'ftr_30d_count', 'ftr_30d_rate']
pd.concat(ftr_rows, ignore_index=True)[cols_order_ftr].to_csv(
    os.path.join(OUT, 'intent_zc_segment_ftr_rate_by_version.csv'), index=False)

cols_order_amt = ['segment', 'final_flag', 'score_version', 'intent_level', 'zc_level',
                   'n_samples', 'total_principal', 'total_overdue', 'amount_overdue_rate']
pd.concat(amt_rows, ignore_index=True)[cols_order_amt].to_csv(
    os.path.join(OUT, 'intent_zc_segment_amount_risk_by_version.csv'), index=False)

n_dist = len(dist_rows)
n_ftr = len(ftr_rows)
n_amt = len(amt_rows)
print(f"  -> intent_zc_segment_distribution_by_version.csv ({n_dist} rows)")
print(f"  -> intent_zc_segment_ftr_rate_by_version.csv ({n_ftr} rows)")
print(f"  -> intent_zc_segment_amount_risk_by_version.csv ({n_amt} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# FILE 2. decile_lift_bins_by_version.csv
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Generating decile lift bins by version ...")

SEGMENTS = ['全客群', '老户次新', '老户', '流失户']
SPLITS = ['DEV', 'DEV-OOS', 'OOT', 'OOT-OOS']

bin_rows = []
for segment in SEGMENTS:
    for ff in SPLITS:
        # Build mask
        if segment == '全客群':
            mask = df['final_flag'] == ff
        elif segment == '老户次新':
            mask = (df['final_flag'] == ff) & (df['segment_ln'] == '老户次新')
        else:
            mask = (df['final_flag'] == ff) & (df['segment'] == segment)

        sub = df[mask]
        if len(sub) == 0:
            continue

        for sv in SCORE_COLS:
            sv_sub = sub.dropna(subset=[sv]).copy()
            if len(sv_sub) < 10:
                continue

            sv_sub = sv_sub.sort_values(sv)
            try:
                sv_sub['decile_int'] = pd.qcut(sv_sub[sv], q=10, labels=False,
                                                duplicates='drop') + 1
            except ValueError:
                continue

            total_bad = sv_sub[LABEL].sum()

            for d in range(10, 0, -1):
                chunk = sv_sub[sv_sub['decile_int'] == d]
                n = len(chunk)
                bad = int(chunk[LABEL].sum())
                br = bad / n if n > 0 else 0
                pct = n / len(sv_sub)
                smin = chunk[sv].min()
                smax = chunk[sv].max()

                bin_rows.append({
                    'segment': segment,
                    'final_flag': ff,
                    'score_version': sv,
                    'decile': f"{d:03d}",
                    'decile_int': d,
                    'n_samples': n,
                    'pct': pct,
                    'bad': bad,
                    'bad_rate': br,
                    'score_min': smin,
                    'score_max': smax,
                    'lower_bound': f"(-inf, {smax:.6f}]" if d == 1 else f"({smin:.6f}, {smax:.6f}]",
                    'upper_bound': f"({smin:.6f}, +inf)" if d == 10 else f"({smin:.6f}, {smax:.6f}]",
                })

bins_df = pd.DataFrame(bin_rows)
cols_bin = ['segment', 'final_flag', 'score_version', 'decile', 'decile_int',
            'n_samples', 'pct', 'bad', 'bad_rate',
            'score_min', 'score_max', 'lower_bound', 'upper_bound']
bins_df[cols_bin].to_csv(os.path.join(OUT, 'decile_lift_bins_by_version.csv'), index=False)
print(f"  -> decile_lift_bins_by_version.csv ({len(bins_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# FILE 3. monthly_segment_metrics_oot_oos_by_version.csv
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Generating monthly segment OOT-OOS metrics by version ...")

oot_oos = df[(df['final_flag'] == 'OOT-OOS') &
             (df['segment_ln'].isin(['老户次新', '流失户']))].copy()

monthly_rows = []
for (month, seg), group in oot_oos.groupby(['mdl_month', 'segment_ln']):
    for sv in SCORE_COLS:
        gv = group.dropna(subset=[sv])
        if len(gv) < 100:
            continue
        row = {
            'mdl_month': month,
            'segment': seg,
            'final_flag': 'OOT-OOS',
            'score_version': sv,
            'n_samples': len(gv),
            'positive': int(gv[LABEL].sum()),
            'bad_rate': gv[LABEL].mean(),
            'auc': calc_auc(gv[LABEL], gv[sv]),
            'ks': calc_ks(gv[LABEL], gv[sv]),
        }
        monthly_rows.append(row)

monthly_df = pd.DataFrame(monthly_rows)
monthly_df = monthly_df.sort_values(['segment', 'score_version', 'mdl_month'])
cols_mo = ['mdl_month', 'segment', 'final_flag', 'score_version',
           'n_samples', 'positive', 'bad_rate', 'auc', 'ks']
monthly_df[cols_mo].to_csv(
    os.path.join(OUT, 'monthly_segment_metrics_oot_oos_by_version.csv'), index=False)
print(f"  -> monthly_segment_metrics_oot_oos_by_version.csv ({len(monthly_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# FILE 4. score_bin_distribution_by_month_by_version.csv
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Generating score bin distribution by month by version ...")

# Baseline: 2025-06，full sample per score_version
base_mask = (df['mdl_month'] == '2025-06') & (df['final_flag'] == 'DEV')
base = df[base_mask].copy()

psi_rows = []
for sv in SCORE_COLS:
    sv_base = base.dropna(subset=[sv]).copy()
    if len(sv_base) < 10:
        print(f"  WARN: not enough baseline samples for {sv}")
        continue

    # Fixed decile boundaries from baseline
    try:
        _, bin_edges = pd.qcut(sv_base[sv], q=10, retbins=True, duplicates='drop')
    except ValueError:
        continue

    # Baseline distribution
    sv_base['score_bin'] = pd.cut(sv_base[sv], bins=bin_edges, labels=False,
                                  include_lowest=True) + 1
    base_dist = sv_base['score_bin'].value_counts(normalize=True).to_dict()

    for month in sorted(df['mdl_month'].unique()):
        month_df = df[df['mdl_month'] == month].dropna(subset=[sv]).copy()
        month_df['score_bin'] = pd.cut(month_df[sv], bins=bin_edges, labels=False,
                                        include_lowest=True) + 1
        t = len(month_df)
        month_psi = 0.0

        for d in range(1, 11):
            chunk = month_df[month_df['score_bin'] == d]
            n = len(chunk)
            pct = n / t if t > 0 else 0
            br = chunk[LABEL].mean() if n > 0 else 0
            bp = base_dist.get(d, 0.1)
            if pct > 0 and bp > 0:
                pc = (pct - bp) * np.log(pct / bp)
            else:
                pc = 0.0
            month_psi += pc

            psi_rows.append({
                'mdl_month': month,
                'score_version': sv,
                'score_decile': f"{d:03d}",
                'score_min': bin_edges[d-1],
                'score_max': bin_edges[d],
                'n_samples': n,
                'pct': pct,
                'bad_rate': br,
                'baseline_pct': bp,
                'psi_component': pc,
                'month_psi': month_psi,
                'lower_bound': f"(-inf, {bin_edges[d]:.6f}]" if d == 1 else f"({bin_edges[d-1]:.6f}, {bin_edges[d]:.6f}]",
                'upper_bound': f"({bin_edges[d-1]:.6f}, +inf)" if d == 10 else f"({bin_edges[d-1]:.6f}, {bin_edges[d]:.6f}]",
            })

psi_df = pd.DataFrame(psi_rows)
cols_psi = ['mdl_month', 'score_version', 'score_decile', 'score_min', 'score_max',
            'n_samples', 'pct', 'bad_rate', 'baseline_pct', 'psi_component', 'month_psi',
            'lower_bound', 'upper_bound']
psi_df[cols_psi].to_csv(
    os.path.join(OUT, 'score_bin_distribution_by_month_by_version.csv'), index=False)
print(f"  -> score_bin_distribution_by_month_by_version.csv ({len(psi_df)} rows)")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All versioned results generated!")
print("=" * 60)
for f in [
    'intent_zc_segment_distribution_by_version.csv',
    'intent_zc_segment_ftr_rate_by_version.csv',
    'intent_zc_segment_amount_risk_by_version.csv',
    'decile_lift_bins_by_version.csv',
    'monthly_segment_metrics_oot_oos_by_version.csv',
    'score_bin_distribution_by_month_by_version.csv',
]:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        print(f"  OK  {f} ({os.path.getsize(path):,} bytes)")
    else:
        print(f"  MISS  {f}")
