"""
Track Clustering - Finding emergent groupings in ZUN's music.

Uses UMAP for dimensionality reduction and identifies clusters
that may not align with explicit metadata (game, stage, era).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

from analysis.dataloader import load_analysis_df, get_feature_columns, filter_canonical, filter_windows_era


def main():
    print("Loading data...")
    df = load_analysis_df()
    df = filter_canonical(df)

    print(f"Loaded {len(df)} canonical tracks")

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Prepare features matrix
    X = df[feature_cols].values

    # Handle NaN values
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"Warning: Removing {nan_mask.sum()} tracks with NaN values")
        df = df[~nan_mask].reset_index(drop=True)
        X = df[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix shape: {X_scaled.shape}")

    # UMAP embedding
    print("Computing UMAP embedding...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    df["umap_x"] = embedding[:, 0]
    df["umap_y"] = embedding[:, 1]

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Color by era
    ax = axes[0, 0]
    era_colors = {"pc98": "purple", "early_windows": "blue",
                  "mid_windows": "green", "late_windows": "orange"}
    for era, color in era_colors.items():
        mask = df["era"] == era
        ax.scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                  c=color, label=era, alpha=0.7, s=30)
    ax.legend()
    ax.set_title("Colored by Era")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Color by game
    ax = axes[0, 1]
    games = sorted(df["game_id"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(games)))
    for game, color in zip(games, colors):
        mask = df["game_id"] == game
        ax.scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                  c=[color], label=game, alpha=0.7, s=30)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title("Colored by Game")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Color by stage position (Windows only)
    ax = axes[1, 0]
    windows_df = df[df["era"] != "pc98"]
    position_colors = {"stage": "blue", "boss": "red", "extra": "purple",
                       "phantasm": "magenta", "title": "gray",
                       "ending": "lightgray", "staff": "black"}
    for pos, color in position_colors.items():
        mask = windows_df["stage_position"] == pos
        ax.scatter(windows_df.loc[mask, "umap_x"], windows_df.loc[mask, "umap_y"],
                  c=color, label=pos, alpha=0.7, s=30)
    ax.legend()
    ax.set_title("Colored by Stage Position (Windows era)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # K-means clustering
    ax = axes[1, 1]
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = cluster_labels

    scatter = ax.scatter(df["umap_x"], df["umap_y"],
                        c=cluster_labels, cmap="Set2", alpha=0.7, s=30)
    ax.set_title(f"K-Means Clustering (k={n_clusters})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()

    output_path = Path(__file__).parent.parent / "outputs" / "figures" / "04_track_clustering.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Analyze clusters
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)

    for cluster_id in range(n_clusters):
        cluster_tracks = df[df["cluster"] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_tracks)} tracks):")

        # Era distribution
        era_dist = cluster_tracks["era"].value_counts(normalize=True)
        print(f"  Eras: {dict(era_dist.round(2))}")

        # Stage position distribution (Windows only)
        windows_cluster = cluster_tracks[cluster_tracks["era"] != "pc98"]
        if len(windows_cluster) > 0:
            pos_dist = windows_cluster["stage_position"].value_counts(normalize=True)
            print(f"  Positions: {dict(pos_dist.head(3).round(2))}")

        # Top games
        game_dist = cluster_tracks["game_id"].value_counts().head(3)
        print(f"  Top games: {dict(game_dist)}")

        # Feature characteristics
        cluster_features = cluster_tracks[feature_cols].mean()
        overall_features = df[feature_cols].mean()
        diff = (cluster_features - overall_features) / overall_features.replace(0, 1)
        extreme_features = diff.abs().nlargest(3)
        print("  Distinctive features:")
        for feat in extreme_features.index:
            direction = "high" if diff[feat] > 0 else "low"
            print(f"    - {direction} {feat}: {diff[feat]*100:+.1f}%")

    # Cross-game similarity analysis
    print("\n" + "="*60)
    print("CROSS-GAME TRACK SIMILARITY")
    print("="*60)

    # Find tracks from different games that cluster together
    for cluster_id in range(n_clusters):
        cluster_tracks = df[df["cluster"] == cluster_id]
        games_in_cluster = cluster_tracks["game_id"].unique()

        if len(games_in_cluster) >= 3:
            # Find pairs of tracks from different games that are close
            print(f"\nCluster {cluster_id} spans {len(games_in_cluster)} games:")

            # Sample some track names
            for game in games_in_cluster[:3]:
                game_tracks = cluster_tracks[cluster_tracks["game_id"] == game]["title_en"].head(2).tolist()
                print(f"  {game}: {', '.join(game_tracks)}")

    plt.show()


if __name__ == "__main__":
    main()
