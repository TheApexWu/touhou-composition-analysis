"""
Stage Position Analysis - The sonic arc of a Touhou game.

Hypotheses to test:
- Stage 1: Accessible, welcoming, bouncy
- Stage 4-5: Building tension, increased complexity
- Stage 6: Climactic, harmonically dense
- Boss themes: Higher energy than stage themes
- Extra/Phantasm: Unhinged energy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from analysis.dataloader import load_analysis_df, get_feature_columns, filter_canonical, filter_windows_era


def main():
    print("Loading data...")
    df = load_analysis_df()
    df = filter_canonical(df)
    df = filter_windows_era(df)  # Focus on Windows era for consistent structure

    print(f"Loaded {len(df)} canonical Windows-era tracks")
    print(f"Stage positions:\n{df['stage_position'].value_counts()}")

    # Filter to gameplay tracks (stage and boss)
    gameplay_df = df[df["stage_position"].isin(["stage", "boss", "extra", "phantasm"])].copy()
    print(f"\nGameplay tracks: {len(gameplay_df)}")

    # Stage vs Boss comparison
    print("\n" + "="*60)
    print("STAGE vs BOSS COMPARISON")
    print("="*60)

    stage_boss_df = gameplay_df[gameplay_df["stage_position"].isin(["stage", "boss"])]

    key_features = [
        "tempo",
        "rms_energy_mean",
        "spectral_centroid_mean",
        "onset_rate",
        "spectral_entropy",
        "chroma_entropy",
    ]

    for feat in key_features:
        stage_vals = stage_boss_df[stage_boss_df["stage_position"] == "stage"][feat].dropna()
        boss_vals = stage_boss_df[stage_boss_df["stage_position"] == "boss"][feat].dropna()

        if len(stage_vals) > 5 and len(boss_vals) > 5:
            stat, pval = stats.mannwhitneyu(stage_vals, boss_vals, alternative="two-sided")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

            stage_mean = stage_vals.mean()
            boss_mean = boss_vals.mean()
            direction = "Boss higher" if boss_mean > stage_mean else "Stage higher"

            print(f"{feat:25s}: Stage={stage_mean:.2f}, Boss={boss_mean:.2f} ({direction}) p={pval:.4f} {sig}")

    # Stage number progression (for tracks with stage numbers)
    numbered_df = gameplay_df[gameplay_df["stage_number"].notna()].copy()
    numbered_df["stage_number"] = numbered_df["stage_number"].astype(int)

    print("\n" + "="*60)
    print("STAGE NUMBER PROGRESSION")
    print("="*60)

    for feat in key_features:
        valid = numbered_df[[feat, "stage_number"]].dropna()
        if len(valid) > 10:
            corr, pval = stats.spearmanr(valid["stage_number"], valid[feat])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            trend = "increases" if corr > 0 else "decreases"
            print(f"{feat:25s}: {trend} with stage (r={corr:+.3f}) {sig}")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Stage vs Boss boxplots
    for i, feat in enumerate(key_features):
        ax = axes.flatten()[i]
        order = ["stage", "boss"]
        data = stage_boss_df[stage_boss_df["stage_position"].isin(order)]
        sns.boxplot(data=data, x="stage_position", y=feat, ax=ax, order=order)
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("")

    plt.suptitle("Stage vs Boss Theme Characteristics", fontsize=14)
    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "outputs" / "figures" / "02_stage_vs_boss.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    # Stage progression line plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for i, feat in enumerate(key_features):
        ax = axes.flatten()[i]

        # Separate stage and boss themes
        for pos, color, marker in [("stage", "blue", "o"), ("boss", "red", "s")]:
            pos_df = numbered_df[numbered_df["stage_position"] == pos]
            means = pos_df.groupby("stage_number")[feat].mean()
            stds = pos_df.groupby("stage_number")[feat].std()

            ax.errorbar(means.index, means.values, yerr=stds.values,
                       label=pos, marker=marker, color=color, capsize=3)

        ax.set_xlabel("Stage Number")
        ax.set_title(feat.replace("_", " ").title())
        ax.legend()
        ax.set_xticks(range(1, 7))

    plt.suptitle("Feature Progression Through Game Stages", fontsize=14)
    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "outputs" / "figures" / "02_stage_progression.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Extra stage analysis
    print("\n" + "="*60)
    print("EXTRA/PHANTASM STAGE ANALYSIS")
    print("="*60)

    extra_df = gameplay_df[gameplay_df["stage_position"].isin(["extra", "phantasm"])]
    final_df = numbered_df[numbered_df["stage_number"] == 6]

    if len(extra_df) > 3 and len(final_df) > 3:
        for feat in key_features:
            extra_vals = extra_df[feat].dropna()
            final_vals = final_df[feat].dropna()

            if len(extra_vals) > 2 and len(final_vals) > 2:
                extra_mean = extra_vals.mean()
                final_mean = final_vals.mean()
                pct_diff = ((extra_mean - final_mean) / final_mean) * 100

                comparison = "higher" if extra_mean > final_mean else "lower"
                print(f"{feat:25s}: Extra is {abs(pct_diff):.1f}% {comparison} than Stage 6")

    plt.show()


if __name__ == "__main__":
    main()
