"""
Era Analysis - How ZUN's sound evolved over time.

This script analyzes spectral and compositional differences across eras:
- PC-98 (TH01-05): FM synthesis, hardware constraints
- Early Windows (TH06-09): Foundational Windows style
- Mid Windows (TH10-14): Maturing compositional voice
- Late Windows (TH15+): Modern production
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from analysis.dataloader import load_analysis_df, get_feature_columns, filter_canonical


def main():
    # Load data
    print("Loading data...")
    df = load_analysis_df()
    df = filter_canonical(df)  # Only analyze canonical tracks

    print(f"Loaded {len(df)} canonical tracks")
    print(f"Era distribution:\n{df['era'].value_counts()}")

    # Define era order for plotting
    era_order = ["pc98", "early_windows", "mid_windows", "late_windows"]
    df["era"] = pd.Categorical(df["era"], categories=era_order, ordered=True)

    # Key features to analyze
    key_features = [
        "spectral_centroid_mean",  # Brightness
        "spectral_bandwidth_mean",  # Tonal spread
        "spectral_rolloff_mean",    # High frequency content
        "tempo",                     # BPM
        "rms_energy_mean",          # Loudness
        "onset_rate",               # Note density
        "spectral_entropy",         # Spectral complexity
        "chroma_entropy",           # Harmonic complexity
    ]

    # Create era comparison plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        ax = axes[i]
        sns.boxplot(data=df, x="era", y=feat, ax=ax, order=era_order)
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("ZUN's Compositional Evolution by Era", fontsize=14)
    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent / "outputs" / "figures" / "01_era_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Statistical tests: Era differences
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: Era Differences")
    print("="*60)

    for feat in key_features:
        groups = [df[df["era"] == era][feat].dropna() for era in era_order]
        if all(len(g) > 2 for g in groups):
            stat, pval = stats.kruskal(*groups)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"{feat:30s}: H={stat:8.2f}, p={pval:.4f} {sig}")

    # Windows-era evolution (exclude PC-98 for cleaner progression)
    print("\n" + "="*60)
    print("WINDOWS ERA EVOLUTION (excluding PC-98)")
    print("="*60)

    windows_df = df[df["era"] != "pc98"].copy()
    windows_order = ["early_windows", "mid_windows", "late_windows"]

    for feat in key_features:
        # Calculate means by era
        means = windows_df.groupby("era")[feat].mean()
        means = means.reindex(windows_order)

        # Calculate percent change from early to late
        early_mean = means["early_windows"]
        late_mean = means["late_windows"]
        pct_change = ((late_mean - early_mean) / early_mean) * 100

        direction = "increased" if pct_change > 0 else "decreased"
        print(f"{feat:30s}: {direction} {abs(pct_change):5.1f}% (early: {early_mean:.2f} -> late: {late_mean:.2f})")

    # Year-by-year trend
    print("\n" + "="*60)
    print("YEAR CORRELATIONS (Windows era)")
    print("="*60)

    for feat in key_features:
        valid = windows_df[[feat, "year"]].dropna()
        if len(valid) > 10:
            corr, pval = stats.spearmanr(valid["year"], valid[feat])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            direction = "+" if corr > 0 else "-"
            print(f"{feat:30s}: r={corr:+.3f} {sig} ({direction})")

    plt.show()


if __name__ == "__main__":
    main()
