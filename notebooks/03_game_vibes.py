"""
Game Vibes Analysis - What makes each game's atmosphere unique?

Operationalizing game-level atmospheres:
- EoSD (TH06): Gothic-whimsical
- PCB (TH07): Cherry blossom melancholy, spring/death duality
- IN (TH08): Nocturnal, lunar
- SA (TH11): Subterranean, oppressive
- UFO (TH12): Adventurous, Buddhist motifs
- TD (TH13): Ghostly, trance elements
- LoLK (TH15): Lunar sterility, intensity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from analysis.dataloader import load_analysis_df, get_feature_columns, filter_canonical, filter_windows_era


# Game descriptions for context
GAME_VIBES = {
    "TH06": "Gothic, elegant, mysterious",
    "TH07": "Melancholic, ethereal, death/spring",
    "TH08": "Nocturnal, lunar, intimate",
    "TH09": "Competitive, flower view",
    "TH10": "Mountain faith, traditional",
    "TH11": "Underground, oppressive, nuclear",
    "TH12": "Adventurous, Buddhist, hopeful",
    "TH13": "Ghostly, trance, resurrection",
    "TH14": "Rebellion, upside-down world",
    "TH15": "Lunar, sterile, intense",
    "TH16": "Seasonal, nature, four seasons",
    "TH17": "Bestial, hell, animal spirits",
    "TH18": "Commercial, ability cards",
    "TH19": "Ghostly, competitive",
}


def main():
    print("Loading data...")
    df = load_analysis_df()
    df = filter_canonical(df)
    df = filter_windows_era(df)

    print(f"Loaded {len(df)} canonical Windows-era tracks")
    print(f"Games: {sorted(df['game_id'].unique())}")

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Feature count: {len(feature_cols)}")

    # Aggregate features by game
    game_means = df.groupby("game_id")[feature_cols].mean()
    game_stds = df.groupby("game_id")[feature_cols].std()

    # Key features for game comparison
    key_features = [
        "spectral_centroid_mean",   # Brightness
        "spectral_bandwidth_mean",  # Spread
        "tempo",                    # Speed
        "rms_energy_mean",          # Loudness
        "onset_rate",               # Density
        "spectral_entropy",         # Spectral complexity
        "chroma_entropy",           # Harmonic complexity
        "spectral_flatness_mean",   # Noisiness
    ]

    # Create game comparison heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalize features for comparison
    scaler = StandardScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(game_means[key_features]),
        index=game_means.index,
        columns=key_features
    )

    # Sort by game number
    normalized = normalized.reindex(sorted(normalized.index))

    sns.heatmap(normalized, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Game Feature Profiles (Z-scores)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Game")

    output_path = Path(__file__).parent.parent / "outputs" / "figures" / "03_game_profiles_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # PCA for game clustering
    print("\n" + "="*60)
    print("PCA ANALYSIS - Game Similarity")
    print("="*60)

    # Use all features for PCA
    valid_features = [f for f in feature_cols if not game_means[f].isna().any()]
    X = game_means[valid_features].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot games in PCA space
    games = game_means.index.tolist()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], s=100, c=range(len(games)), cmap="viridis")

    for i, game in enumerate(games):
        ax.annotate(game, (X_pca[i, 0], X_pca[i, 1]),
                   xytext=(5, 5), textcoords="offset points", fontsize=10)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("Game Similarity in Feature Space")

    output_path = Path(__file__).parent.parent / "outputs" / "figures" / "03_game_pca.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Identify what drives PC1 and PC2
    print("\nTop features for PC1 (what separates games horizontally):")
    loadings = pd.DataFrame(pca.components_.T, index=valid_features, columns=["PC1", "PC2"])
    for feat in loadings["PC1"].abs().nlargest(5).index:
        sign = "+" if loadings.loc[feat, "PC1"] > 0 else "-"
        print(f"  {sign} {feat}: {loadings.loc[feat, 'PC1']:.3f}")

    print("\nTop features for PC2 (what separates games vertically):")
    for feat in loadings["PC2"].abs().nlargest(5).index:
        sign = "+" if loadings.loc[feat, "PC2"] > 0 else "-"
        print(f"  {sign} {feat}: {loadings.loc[feat, 'PC2']:.3f}")

    # Game-specific insights
    print("\n" + "="*60)
    print("GAME-SPECIFIC INSIGHTS")
    print("="*60)

    # Find distinctive features for each game
    for game in sorted(games):
        game_vals = normalized.loc[game]
        extreme_features = game_vals.abs().nlargest(3)

        print(f"\n{game} ({GAME_VIBES.get(game, 'Unknown')}):")
        for feat, val in extreme_features.items():
            direction = "high" if val > 0 else "low"
            print(f"  - {direction.upper()} {feat.replace('_', ' ')}: {val:+.2f} SD")

    # Specific comparisons
    print("\n" + "="*60)
    print("TARGETED COMPARISONS")
    print("="*60)

    # SA (underground) vs others - should be darker/lower centroid
    if "TH11" in games:
        sa_centroid = normalized.loc["TH11", "spectral_centroid_mean"]
        others_centroid = normalized.loc[normalized.index != "TH11", "spectral_centroid_mean"].mean()
        print(f"\nSA (TH11) spectral centroid: {sa_centroid:.2f} SD (others avg: {others_centroid:.2f})")
        print("  -> " + ("Confirms darker/more oppressive sound" if sa_centroid < -0.5 else "Does not clearly confirm darker sound"))

    # LoLK (lunar) - should be more intense
    if "TH15" in games:
        lolk_tempo = normalized.loc["TH15", "tempo"]
        lolk_energy = normalized.loc["TH15", "rms_energy_mean"]
        print(f"\nLoLK (TH15) tempo: {lolk_tempo:.2f} SD, energy: {lolk_energy:.2f} SD")
        print("  -> " + ("Confirms higher intensity" if (lolk_tempo > 0.5 or lolk_energy > 0.5) else "Does not clearly confirm higher intensity"))

    plt.show()


if __name__ == "__main__":
    main()
