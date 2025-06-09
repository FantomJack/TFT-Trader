#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def compute_comparison_metrics(mlp_group, tft_group, metric):
    """
    Computes comparative statistics and statistical significance tests between two groups.

    Args:
        mlp_group (pd.DataFrame): DataFrame of MLP results for a given metric/category.
        tft_group (pd.DataFrame): DataFrame of TFT results for a given metric/category.
        metric (str): Name of the metric column to compare.

    Returns:
        dict: Summary of comparative statistics, including:
            - t_statistic: Welch's t-test statistic.
            - p_value: Corresponding p-value.
            - significant: Boolean, whether the difference is significant at alpha=0.001.
            - cohens_d: Cohen's d effect size.
            - mlp_mean: Mean value for MLP group.
            - tft_mean: Mean value for TFT group.
            - mlp_iqr: IQR for MLP group.
            - tft_iqr: IQR for TFT group.
    """



    t_stat, p_val = ttest_ind(mlp_group[metric], tft_group[metric], equal_var=False)
    # t_stat, p_val = ttest_ind(tft_group[metric], mlp_group[metric], equal_var=False)

    mlp_q1, mlp_q3 = mlp_group[metric].quantile([0.25, 0.75])
    tft_q1, tft_q3 = tft_group[metric].quantile([0.25, 0.75])
    mlp_iqr = mlp_q3 - mlp_q1
    tft_iqr = tft_q3 - tft_q1

    mlp_mean = mlp_group[metric].mean()
    tft_mean = tft_group[metric].mean()
    mlp_var = mlp_group[metric].var(ddof=1)
    tft_var = tft_group[metric].var(ddof=1)
    mlp_med = mlp_group[metric].median()
    tft_med = tft_group[metric].median()

    n1 = len(mlp_group)
    n2 = len(tft_group)

    alpha = 0.001
    s1_sq = mlp_group[metric].var(ddof=1)
    s2_sq = tft_group[metric].var(ddof=1)

    df_num = (s1_sq / n1 + s2_sq / n2) ** 2
    df_den = ((s1_sq / n1) ** 2 / (n1 - 1)) + ((s2_sq / n2) ** 2 / (n2 - 1))
    df_eff = df_num / df_den
    t_crit = t.ppf(1 - alpha / 2, df_eff)
    significant = abs(t_stat) > t_crit

    pooled_var = ((n1 - 1) * mlp_var + (n2 - 1) * tft_var) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    cohen_d = (mlp_mean - tft_mean) / pooled_sd

    return {
        'metric': metric,
        't_statistic': round(t_stat, 4),
        'p_value': p_val,
        # 'alpha': alpha,
        # 'df_eff': round(df_eff, 2),
        # 't_critical': round(t_crit, 4),
        'significant': significant,
        'cohens_d': round(cohen_d, 4),
        'mlp_mean': round(mlp_mean, 4),
        # 'mlp_med': round(mlp_med, 4),
        # 'mlp_variance': round(mlp_var, 4),
        'mlp_iqr': round(mlp_iqr, 4),
        'mlp_q1': round(mlp_q1, 4),
        'mlp_q3': round(mlp_q3, 4),
        'tft_mean': round(tft_mean, 4),
        # 'tft_median': round(tft_med, 4),
        # 'tft_variance': round(tft_var, 4),
        'tft_iqr': round(tft_iqr, 4),
        'tft_q1': round(tft_q1, 4),
        'tft_q3': round(tft_q3, 4),
    }


def half_violin(ax, data: pd.DataFrame, pos: float, color: str, side: str ='left', width: float =0.6):
    """
    Plots a half-violin (left or right) at a given position on the axis.

    Args:
        ax (matplotlib.axes.Axes): Target axis.
        data (array-like): Data to plot.
        pos (float): Position on the x-axis.
        color (str): Color of the violin.
        side (str): 'left' or 'right'.
        width (float): Width of the violin plot.
    """

    parts = ax.violinplot([data], positions=[pos], widths=width, showextrema=False)
    for body in parts['bodies']:
        verts = body.get_paths()[0].vertices
        if side == 'left':
            verts[verts[:,0] > pos, 0] = pos
        else:
            verts[verts[:,0] < pos, 0] = pos
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.7)


def violin_all(mlp: pd.DataFrame, tft: pd.DataFrame, metrics: list):
    """
    Generates and saves side-by-side full violin plots for each metric (MLP vs TFT).

    Args:
        mlp (pd.DataFrame): MLP metrics.
        tft (pd.DataFrame): TFT metrics.
        metrics (list): List of metric column names.
    """

    title_translations = {
        'total_return': 'celkového výnosu',
        'sharpe_ratio': 'Sharpeho pomeru',
        'max_drawdown': 'maximálneho drawdownu'
    }

    y_translations = {
        'total_return': 'celkový výnos',
        'sharpe_ratio': 'Sharpeho pomer',
        'max_drawdown': 'maximálny drawdown'
    }

    for m in metrics:
        mlp_vals = mlp[m].dropna()
        tft_vals = tft[m].dropna()
        fig, ax = plt.subplots()
        # Side-by-side violins
        parts = ax.violinplot([mlp_vals, tft_vals], positions=[0,1], widths=0.6, showextrema=False)
        parts['bodies'][0].set_facecolor('C0')
        parts['bodies'][1].set_facecolor('C1')
        for body in parts['bodies']:
            body.set_edgecolor('black')
            body.set_alpha(0.7)

        q1_m, q3_m = np.percentile(mlp_vals, [25, 75])
        q1_t, q3_t = np.percentile(tft_vals, [25, 75])
        ax.vlines(0, q1_m, q3_m, color='black', linewidth=2)
        ax.vlines(1, q1_t, q3_t, color='black', linewidth=2)

        title = title_translations.get(m, m.replace('_', ' ').title())
        ylabel = y_translations.get(m, m.replace('_', ' ').title())
        ax.set_xticks([0,1])
        ax.set_xticklabels(['MLP','TFT'])
        ax.set_ylabel(ylabel)
        ax.set_title(f'Graf hustoty rozdelenia {title}')
        plt.tight_layout()
        plt.savefig(f"../model_compare/{m}.png")
        plt.close()
        print(f"Overall violin saved to: ../model_compare/{m}.png")


def category_ttest(mlp: pd.DataFrame, tft: pd.DataFrame, metrics: list, params: dict[str, str]):
    """
    Performs category-level t-tests for all metric/category pairs and saves to CSV.

    Args:
        mlp (pd.DataFrame): MLP group.
        tft (pd.DataFrame): TFT group.
        metrics (list): Metric column names.
        params (dict): {category: [label1, ...]} for each grouping.
    """
    stats_rows = []

    for metric in metrics:
        for category_col, labels in params.items():
            for label in labels:
                mlp_group = mlp.loc[mlp[category_col] == label].dropna()
                tft_group = tft.loc[tft[category_col] == label].dropna()

                if len(mlp_group) > 1 and len(tft_group) > 1:
                    stats = {
                        'category': category_col,
                        'label': label,
                    }
                    stats.update(compute_comparison_metrics(mlp_group, tft_group, metric))
                    stats_rows.append(stats)

    results_df = pd.DataFrame(stats_rows)
    results_df['p_value'] = results_df['p_value'].apply(lambda x: f"{x:.4e}")
    csv_fn = f"../model_compare/category_results.csv"
    results_df.to_csv(csv_fn, index=False)


def group_ttest(mlp: pd.DataFrame, tft: pd.DataFrame, metrics: list):
    """
    Performs group-level t-tests (by hyperparams, start_date, train_val_ratio) and saves CSV.

    Args:
        mlp (pd.DataFrame): MLP group.
        tft (pd.DataFrame): TFT group.
        metrics (list): List of metric column names.
    """

    df = pd.concat([mlp, tft], ignore_index=True)
    group_results = []
    grouped = df.groupby(['hyperparams', 'start_date', 'train_val_ratio'])
    for metric in metrics:
        for (hp, sd, tv), group in grouped:
            mlp_group = group[group['model'] == 'MLP']
            tft_group = group[group['model'] == 'TFT']
            if len(mlp_group) > 1 and len(tft_group) > 1:
                stats = {
                    'hyperparams': hp,
                    'start_date': sd,
                    'train_val_ratio': tv,
                }
                stats.update(compute_comparison_metrics(mlp_group, tft_group, metric))
                group_results.append(stats)

    group_df = pd.DataFrame(group_results)
    group_df['p_value'] = group_df['p_value'].apply(lambda x: f"{x:.4e}")
    group_df.to_csv("../model_compare/group_results.csv", index=False)


def plot_by_category(mlp: pd.DataFrame, tft: pd.DataFrame,
                     metric: str, category_col: str, category_labels: list, output_fn: str):
    """
    Plots and saves side-by-side half-violin plots for a metric, grouped by category.

    Args:
        mlp (pd.DataFrame): MLP results.
        tft (pd.DataFrame): TFT results.
        metric (str): Metric column.
        category_col (str): Name of the category column.
        category_labels (list): List of category values.
        output_fn (str): Output filename.
    """

    metric_translations = {
        'total_return': 'Celkový výnos',
        'sharpe_ratio': 'Sharpeho pomer',
        'max_drawdown': 'Maximálny pokles'
    }
    title = metric_translations.get(metric, metric.replace('_', ' ').title())

    col_translations = {
        'train_val_ratio': 'pomeru trénovacích a validačných dát',
        'hyperparams': 'sady hyperparametrov',
        'start_date': 'počatočného dátumu'
    }
    col = col_translations.get(category_col, category_col.replace('_', ' ').title())

    fig, ax = plt.subplots(figsize=(max(6, len(category_labels)*2), 6))
    positions = np.arange(len(category_labels))
    for i, category in enumerate(category_labels):
        mlp_vals = mlp[mlp[category_col] == category][metric].dropna()
        tft_vals = tft[tft[category_col] == category][metric].dropna()
        half_violin(ax, mlp_vals, pos=positions[i], color='C0', side='left')
        half_violin(ax, tft_vals, pos=positions[i], color='C1', side='right')
        if len(mlp_vals):
            q1, q3 = np.percentile(mlp_vals, [25,75])
            ax.vlines(positions[i] - 0.1, q1, q3, color='black', linewidth=2)
        if len(tft_vals):
            q1, q3 = np.percentile(tft_vals, [25,75])
            ax.vlines(positions[i] + 0.1, q1, q3, color='black', linewidth=2)

    ax.set_xticks(positions)
    ax.set_xticklabels(category_labels, rotation=45, ha='right')
    ax.set_ylabel(title)
    ax.set_title(f"{title} podľa {col}")

    proxy_mlp = Patch(label='MLP', facecolor='C0', edgecolor='black')
    proxy_tft = Patch(label='TFT', facecolor='C1', edgecolor='black')
    ax.legend(handles=[proxy_mlp, proxy_tft], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_fn)
    plt.close()
    print(f"Saved: {output_fn}")


def plot_return_by_ticker(mlp: pd.DataFrame, tft: pd.DataFrame, output_fn: str ='../model_compare/return_by_ticker.png'):
    """
    Plots half-violin plots of total_return for each ticker, MLP vs TFT.

    Args:
        mlp (pd.DataFrame): MLP results.
        tft (pd.DataFrame): TFT results.
        output_fn (str): Output filename for plot.
    """
    # get the sorted union of tickers in both dataframes
    tickers = sorted(set(mlp['ticker']).union(tft['ticker']))
    n = len(tickers)
    fig, ax = plt.subplots(figsize=(max(10, n * 1), 6))
    positions = np.arange(n)

    for i, ticker in enumerate(tickers):
        m_vals = mlp.loc[mlp['ticker'] == ticker, 'total_return'].dropna()
        t_vals = tft.loc[tft['ticker'] == ticker, 'total_return'].dropna()

        # left half-violin = MLP
        half_violin(ax, m_vals, pos=positions[i], color='C0', side='left', width=0.6)
        # right half-violin = TFT
        half_violin(ax, t_vals, pos=positions[i], color='C1', side='right', width=0.6)

        # overlay IQR lines
        if len(m_vals) >= 2:
            q1, q3 = np.percentile(m_vals, [25, 75])
            ax.vlines(positions[i] - 0.1, q1, q3, color='black', linewidth=2)
        if len(t_vals) >= 2:
            q1, q3 = np.percentile(t_vals, [25, 75])
            ax.vlines(positions[i] + 0.1, q1, q3, color='black', linewidth=2)

    # formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(tickers, rotation=90)
    ax.set_ylabel('Total Return')
    ax.set_title('Total Return Distribution by Ticker')
    ax.legend(
        handles=[
            Patch(label='MLP', facecolor='C0', edgecolor='black'),
            Patch(label='TFT', facecolor='C1', edgecolor='black')
        ],
        loc='upper right'
    )
    plt.tight_layout()
    plt.savefig(output_fn)
    plt.close()
    print(f"Saved: {output_fn}")


def main(mlp_path: str, tft_path: str, plot_by_ticker: bool):
    """
    Main entry point: Loads data, runs all analyses, and generates plots.

    Args:
        mlp_path (str): Path to MLP metrics CSV.
        tft_path (str): Path to TFT metrics CSV.
        plot_by_ticker (bool): Whether to plot by ticker as well.
    """

    mlp = pd.read_csv(mlp_path)
    tft = pd.read_csv(tft_path)
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
    group_ttest(mlp, tft, metrics)

    print("=== Overall Welch's T-Tests ===")
    overall_results = []
    df = pd.concat([mlp, tft], ignore_index=True)
    for metric in metrics:
        mlp_group = df[df['model'] == 'MLP']
        tft_group = df[df['model'] == 'TFT']

        stats = compute_comparison_metrics(mlp_group, tft_group, metric)
        overall_results.append(stats)

    overall_df = pd.DataFrame(overall_results)
    print(overall_df)

    violin_all(mlp, tft, metrics)
    params = {
        'start_date': sorted(mlp['start_date'].unique()), # 1, 2, 3, 4
        'hyperparams': sorted(mlp['hyperparams'].unique()), # 01, 02, 03
        'train_val_ratio': sorted(mlp['train_val_ratio'].unique()) # 85, 90, 95
    }

    for m in metrics:
        for col, labels in params.items():
            fn = f"../model_compare/{m}_by_{col}.png"
            plot_by_category(mlp, tft, m, col, labels, fn)
    category_ttest(mlp, tft, metrics, params)

    # best model my metric
    combined = pd.concat([mlp, tft], ignore_index=True)
    best = combined.loc[combined['total_return'].idxmax()]
    best_df = pd.DataFrame([best])
    print(best_df)
    best = combined.loc[combined['sharpe_ratio'].idxmax()]
    best_df = pd.DataFrame([best])
    print(best_df)
    best = combined.loc[combined['max_drawdown'].idxmin()]
    best_df = pd.DataFrame([best])
    print(best_df)

    if plot_by_ticker:
        plot_return_by_ticker(mlp, tft, output_fn='../model_compare/total_return_by_ticker.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MLP vs TFT: half-violins by one parameter at a time."
    )
    parser.add_argument('--mlp_path',
                        type=str,
                        help='Path to MLP metrics CSV',
                        default='../model_compare/mlp_metrics.csv')
    parser.add_argument('--tft_path',
                        type=str,
                        help='Path to TFT metrics CSV',
                        default='../model_compare/tft_metrics.csv')
    parser.add_argument('--ticker',
                        action = 'store_true',
                        help = 'If set, also plot the return-by-ticker violin charts')
    args = parser.parse_args()

    main(args.mlp_path, args.tft_path, args.ticker)


