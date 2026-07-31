import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_PROB_COL = {0: 'bracket_prob', 1: 'win_prob', 2: 'log_prob'}


def _col(model: int) -> str:
    return _PROB_COL.get(model, 'win_prob')

# determine the Brier score, log loss, and calibration metrics for a given model
def calc_brier(df: pd.DataFrame, model: int = 1) -> float:
    prob_col = _col(model)
    brier = 0.0
    for _, row in df.iterrows():
        brier += (row['result'] - row[prob_col]) ** 2
    return brier / len(df)


def calc_logloss(df: pd.DataFrame, model: int = 1) -> float:
    prob_col = _col(model)
    logloss = 0.0
    for _, row in df.iterrows():
        win_prob = max(0.001, min(0.999, row[prob_col]))
        logloss += row['result'] * math.log(win_prob) + (1 - row['result']) * math.log(1 - win_prob)
    return -logloss / len(df)


def _bin_calibration(df: pd.DataFrame, n_bins: int = 20, model: int = 1) -> pd.DataFrame:
    prob_col = _col(model)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, df[prob_col].values, side='right') - 1, 0, n_bins - 1)
    binned = pd.DataFrame({'y': df['result'].values, 'p': df[prob_col].values, 'bin': idx})
    return binned.groupby('bin', observed=True).agg(
        n=('y', 'size'),
        mean_p=('p', 'mean'),
        rate=('y', 'mean'),
    )


def calc_cal_mae(df: pd.DataFrame, n_bins: int = 20, model: int = 1) -> float:
    g = _bin_calibration(df, n_bins, model)
    if g.empty:
        return float('nan')
    err = (g['mean_p'] - g['rate']).abs()
    w = g['n'] / g['n'].sum()
    return float((w * err).sum())


def calc_cal_rmse(df: pd.DataFrame, n_bins: int = 20, model: int = 1) -> float:
    g = _bin_calibration(df, n_bins, model)
    if g.empty:
        return float('nan')
    err = (g['mean_p'] - g['rate']).abs()
    w = g['n'] / g['n'].sum()
    return float(np.sqrt((w * err ** 2).sum()))

# Calibration curve
def plot_calibration(df: pd.DataFrame, n_bins: int = 20, model: int = 1) -> None:
    g = _bin_calibration(df, n_bins, model)
    if g.empty:
        print("No calibration data to plot.")
        return

    g = g.sort_values('mean_p')

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='perfect calibration')
    plt.plot(g['mean_p'], g['rate'], marker='o', label='model')

    plt.xlabel('Mean predicted favorite win probability')
    plt.ylabel('Actual favorite win rate')
    plt.title(f'Win Probability Calibration ({_col(model)})')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

# Plots calibration bucketed by predicted win probability, with a table of calibration metrics next to the plot
def plot_winprob_histogram(df: pd.DataFrame, bin_width: float = 0.05, model: int = 1) -> pd.DataFrame:
    prob_col = _col(model)
    edges = np.arange(0.0, 1.0 + bin_width / 2, bin_width)
    binned = df[[prob_col, 'result']].copy()
    binned['bucket'] = pd.cut(binned[prob_col], bins=edges, include_lowest=True, right=False)

    grp = (
        binned.groupby('bucket', observed=True)
        .agg(total=('result', 'size'), wins=('result', 'sum'), mean_pred=(prob_col, 'mean'))
        .reset_index()
    )
    grp['expected_wins'] = grp['total'] * grp['mean_pred']
    grp['actual_rate'] = grp['wins'] / grp['total']
    grp['delta'] = grp['actual_rate'] - grp['mean_pred']

    centers = np.array([(b.left + b.right) / 2 for b in grp['bucket']])
    bar_w = bin_width * 0.9

    _, (ax, ax_tbl) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 2]})

    ax.bar(centers, grp['total'], width=bar_w, color='lightsteelblue', edgecolor='white', label='Possessions in bucket (n)')
    ax.bar(centers, grp['wins'], width=bar_w, color='steelblue', edgecolor='white', label='Actual favorite wins')
    ax.plot(centers, grp['expected_wins'], 'k--', marker='o', markersize=4, linewidth=1, label='Expected wins (perfect calibration)')

    y_offset = max(grp['total']) * 0.01
    for c, total, wins in zip(centers, grp['total'], grp['wins']):
        if total == 0:
            continue
        ax.text(c, total + y_offset, f'{wins / total:.0%}', ha='center', va='bottom', fontsize=8)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('Predicted favorite win probability')
    ax.set_ylabel('Number of possessions')
    ax.set_title(f'Win Probability Calibration ({_col(model)})')
    ax.legend(loc='upper center')
    ax.grid(True, axis='y', alpha=0.3)

    ax_tbl.axis('off')
    tbl_df = grp.copy()
    tbl_df['bucket'] = tbl_df['bucket'].astype(str)
    tbl_df['mean_pred'] = tbl_df['mean_pred'].map(lambda v: f'{v:.3f}')
    tbl_df['actual_rate'] = tbl_df['actual_rate'].map(lambda v: f'{v:.3f}')
    tbl_df['expected_wins'] = tbl_df['expected_wins'].map(lambda v: f'{v:.0f}')
    tbl_df['delta'] = tbl_df['delta'].map(lambda v: f'{v:+.3f}')
    tbl_df = tbl_df[['bucket', 'total', 'wins', 'mean_pred', 'actual_rate', 'delta', 'expected_wins']]

    table = ax_tbl.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc='center', cellLoc='right')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    delta_col_idx = list(tbl_df.columns).index('delta')
    for r, val in enumerate(grp['delta'], start=1):
        cell = table[(r, delta_col_idx)]
        cell.set_facecolor('#e5ffe5' if abs(val) <= 0.02 else '#ffe5e5')

    plt.tight_layout()
    return grp

# Plots calibration bucketed by time remaining, with a table of calibration metrics next to the plot
def plot_time_histogram(df: pd.DataFrame, model: int = 1) -> pd.DataFrame:
    prob_col = _col(model)
    reg = df[~df['period'].str.contains('ot', case=False, na=False)].copy()

    edges = [0, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    reg['bucket'] = pd.cut(reg['time_rem'], bins=edges, include_lowest=True, right=False)

    grp = (
        reg.groupby('bucket', observed=True)
        .agg(total=('result', 'size'), wins=('result', 'sum'), mean_pred=(prob_col, 'mean'))
        .reset_index()
    )
    grp['expected_wins'] = grp['total'] * grp['mean_pred']
    grp['actual_rate'] = grp['wins'] / grp['total']
    grp['delta'] = grp['actual_rate'] - grp['mean_pred']
    grp = grp.sort_values('bucket', ascending=False).reset_index(drop=True)

    centers = np.array([(b.left + b.right) / 2 for b in grp['bucket']])
    widths = np.array([b.right - b.left for b in grp['bucket']]) * 0.9

    _, (ax, ax_tbl) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 2]})

    ax.bar(centers, grp['total'], width=widths, color='lightsteelblue', edgecolor='white', label='Possessions (n)')
    ax.bar(centers, grp['wins'], width=widths, color='steelblue', edgecolor='white', label='Actual favorite wins')
    ax.plot(centers, grp['expected_wins'], 'k--', marker='o', markersize=4, linewidth=1, label='Expected wins')

    y_offset = grp['total'].max() * 0.01
    for c, total, wins in zip(centers, grp['total'], grp['wins']):
        if total == 0:
            continue
        ax.text(c, total + y_offset, f'{wins / total:.0%}', ha='center', va='bottom', fontsize=8)

    ax.invert_xaxis()
    ax.set_xlabel('Time remaining (minutes)')
    ax.set_ylabel('Number of possessions')
    ax.set_title(f'Calibration by Time Remaining ({_col(model)})')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    ax_tbl.axis('off')
    tbl_df = grp.copy()
    tbl_df['bucket'] = tbl_df['bucket'].astype(str)
    tbl_df['mean_pred'] = tbl_df['mean_pred'].map(lambda v: f'{v:.3f}')
    tbl_df['actual_rate'] = tbl_df['actual_rate'].map(lambda v: f'{v:.3f}')
    tbl_df['expected_wins'] = tbl_df['expected_wins'].map(lambda v: f'{v:.0f}')
    tbl_df['delta'] = tbl_df['delta'].map(lambda v: f'{v:+.3f}')
    tbl_df = tbl_df[['bucket', 'total', 'wins', 'mean_pred', 'actual_rate', 'delta', 'expected_wins']]

    table = ax_tbl.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc='center', cellLoc='right')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    delta_col_idx = list(tbl_df.columns).index('delta')
    for r, val in enumerate(grp['delta'], start=1):
        cell = table[(r, delta_col_idx)]
        cell.set_facecolor('#e5ffe5' if abs(val) <= 0.02 else '#ffe5e5')

    plt.tight_layout()
    return grp

