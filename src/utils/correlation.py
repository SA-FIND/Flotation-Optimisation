"""
Correlation analysis utilities for Flotation Analysis.

Implements Spearman Rank Correlation as recommended by Lartey et al. (2025),
Minerals Engineering 232, 109492. Spearman is preferred over Pearson because
flotation data is typically non-normal and exhibits non-linear monotonic
relationships -- both of which violate Pearson''s assumptions.

References
----------
Lartey, C., Amankwaa-Kyeremeh, B., Skinner, W., Zanin, M., & Asamoah, R.K. (2025).
    Interpretable machine learning model for prediction of copper recovery in flotation.
    Minerals Engineering, 232, 109492.
Gomez-Flores, A. et al. (2022). Spearman correlation for flotation data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, normaltest
from typing import Optional, List, Tuple


def compute_spearman(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman rank correlation matrix and p-value matrix.

    Spearman is non-parametric: it ranks the data before computing correlation,
    so it captures any monotonic relationship (not just linear) and is robust to
    outliers and non-Gaussian distributions common in flotation process data.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric dataframe of flotation variables.

    Returns
    -------
    rho : pd.DataFrame
        Spearman correlation coefficients (-1 to +1).
    pvals : pd.DataFrame
        Two-tailed p-values (H0: no monotonic relationship).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    cols = numeric_df.columns.tolist()
    n = len(cols)
    rho_arr = np.ones((n, n))
    pval_arr = np.zeros((n, n))

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i == j:
                continue
            r, p = spearmanr(numeric_df[col_i], numeric_df[col_j])
            rho_arr[i, j] = r
            pval_arr[i, j] = p

    rho = pd.DataFrame(rho_arr, index=cols, columns=cols)
    pvals = pd.DataFrame(pval_arr, index=cols, columns=cols)
    return rho, pvals


def compute_pearson(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlation matrix and p-value matrix.

    Provided for comparison with Spearman. Pearson assumes linearity and
    bivariate normality -- assumptions that flotation data rarely satisfies.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    cols = numeric_df.columns.tolist()
    n = len(cols)
    r_arr = np.ones((n, n))
    pval_arr = np.zeros((n, n))

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i == j:
                continue
            r, p = pearsonr(numeric_df[col_i], numeric_df[col_j])
            r_arr[i, j] = r
            pval_arr[i, j] = p

    r = pd.DataFrame(r_arr, index=cols, columns=cols)
    pvals = pd.DataFrame(pval_arr, index=cols, columns=cols)
    return r, pvals


def normality_check(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    D''Agostino-Pearson normality test on every numeric column.

    If many variables fail this test (p < alpha), Pearson is inappropriate
    and Spearman should be used instead.

    Parameters
    ----------
    df : pd.DataFrame
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    pd.DataFrame with columns: variable, statistic, p_value, is_normal.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    results = []
    for col in numeric_df.columns:
        stat, p = normaltest(numeric_df[col].dropna())
        results.append({
            "variable": col,
            "statistic": round(stat, 4),
            "p_value": round(p, 6),
            "is_normal": p >= alpha,
        })
    return pd.DataFrame(results)


def plot_spearman_heatmap(
    rho: pd.DataFrame,
    pvals: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    title: str = "Spearman Rank Correlation",
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    targets: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a publication-quality Spearman correlation heatmap.

    Non-significant correlations (p >= alpha) are masked with hatching so readers
    can immediately identify which relationships are statistically meaningful.

    Parameters
    ----------
    rho : pd.DataFrame
    pvals : pd.DataFrame, optional
    alpha : float
    title : str
    figsize : tuple
    annot : bool
    targets : list of str, optional  -- highlighted in bold red on tick labels
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.zeros_like(rho.values, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        rho,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        square=True,
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Spearman rho"},
    )

    # Hatch non-significant cells
    if pvals is not None:
        cols = rho.columns.tolist()
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if j >= i:
                    continue
                if pvals.loc[col_i, col_j] >= alpha:
                    ax.add_patch(
                        plt.Rectangle(
                            (j, i), 1, 1,
                            fill=False,
                            hatch="///",
                            edgecolor="grey",
                            lw=0,
                            alpha=0.5,
                        )
                    )

    if targets:
        for lbl in ax.get_xticklabels():
            if lbl.get_text() in targets:
                lbl.set_fontweight("bold")
                lbl.set_color("darkred")
        for lbl in ax.get_yticklabels():
            if lbl.get_text() in targets:
                lbl.set_fontweight("bold")
                lbl.set_color("darkred")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    return fig


def compare_pearson_vs_spearman(
    df: pd.DataFrame,
    targets: List[str],
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Bar chart comparing Pearson r vs Spearman rho for each feature vs target.

    Highlights cases where |delta| > 0.1, signalling non-linearity where Spearman
    is the more reliable measure of association.

    Parameters
    ----------
    df : pd.DataFrame
    targets : list of str
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    pd.DataFrame with columns: feature, target, pearson_r, spearman_rho, delta.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    features = [c for c in numeric_df.columns if c not in targets]

    records = []
    for target in targets:
        for feat in features:
            pr, _ = pearsonr(numeric_df[feat], numeric_df[target])
            sr, _ = spearmanr(numeric_df[feat], numeric_df[target])
            records.append(
                {
                    "feature": feat,
                    "target": target,
                    "pearson_r": round(pr, 4),
                    "spearman_rho": round(sr, 4),
                    "delta": round(abs(sr - pr), 4),
                }
            )

    comparison = pd.DataFrame(records)

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=figsize, sharey=False)
    if n_targets == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        sub = comparison[comparison["target"] == target].sort_values("spearman_rho")
        x = np.arange(len(sub))
        w = 0.35
        ax.barh(x - w / 2, sub["pearson_r"], w, label="Pearson r", color="#4C72B0", alpha=0.8)
        ax.barh(x + w / 2, sub["spearman_rho"], w, label="Spearman rho", color="#DD8452", alpha=0.8)

        for idx, (_, row) in enumerate(sub.iterrows()):
            if row["delta"] > 0.1:
                ax.axhline(idx, color="red", lw=0.5, alpha=0.4, linestyle="--")

        ax.set_yticks(x)
        ax.set_yticklabels(sub["feature"].tolist(), fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Correlation coefficient")
        ax.set_title(f"Correlations with {target}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Pearson vs Spearman Correlation Comparison\n"
        "(dashed red = |delta| > 0.10, indicating non-linearity)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

    return comparison


def rank_features_by_spearman(
    rho: pd.DataFrame,
    targets: List[str],
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Rank features by absolute Spearman correlation with specified target(s).

    Parameters
    ----------
    rho : pd.DataFrame
        Full Spearman matrix from compute_spearman().
    targets : list of str
    threshold : float
        Minimum |rho| to be flagged as meaningfully correlated (default 0.3).

    Returns
    -------
    pd.DataFrame sorted by mean |rho| descending. Includes meets_threshold flag.
    """
    features = [c for c in rho.columns if c not in targets]
    records = []
    for feat in features:
        target_corrs = {t: rho.loc[feat, t] for t in targets if t in rho.columns}
        mean_abs = np.mean([abs(v) for v in target_corrs.values()])
        records.append({"feature": feat, **target_corrs, "mean_abs_rho": round(mean_abs, 4)})

    ranked = (
        pd.DataFrame(records)
        .sort_values("mean_abs_rho", ascending=False)
        .reset_index(drop=True)
    )
    ranked["meets_threshold"] = ranked["mean_abs_rho"] >= threshold
    return ranked
