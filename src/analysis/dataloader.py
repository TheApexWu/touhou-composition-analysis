"""
Data loading utilities for analysis.

Loads catalog + features and merges into analysis-ready DataFrames.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def load_catalog(path: Optional[Path] = None) -> dict:
    """Load the enriched catalog."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "metadata" / "catalog.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_features(path: Optional[Path] = None) -> dict:
    """Load extracted features."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_aggregation(agg: dict, prefix: str) -> dict:
    """Flatten a FeatureAggregation dict into prefixed columns."""
    return {f"{prefix}_{k}": v for k, v in agg.items()}


def features_to_flat_dict(feat: dict) -> dict:
    """Convert a single track's features to a flat dict for DataFrame."""
    flat = {
        "track_id": feat["track_id"],
        "duration_seconds": feat["duration_seconds"],
        "tempo": feat["tempo"],
        "beat_strength": feat["beat_strength"],
        "onset_rate": feat["onset_rate"],
        "spectral_entropy": feat["spectral_entropy"],
        "chroma_entropy": feat["chroma_entropy"],
    }

    # Flatten aggregations
    for agg_name in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                     "spectral_flatness", "zero_crossing_rate", "rms_energy"]:
        flat.update(flatten_aggregation(feat[agg_name], agg_name))

    # MFCCs
    for i, (mean, std) in enumerate(zip(feat["mfcc_means"], feat["mfcc_stds"])):
        flat[f"mfcc_{i}_mean"] = mean
        flat[f"mfcc_{i}_std"] = std

    # Chroma
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, (mean, std) in enumerate(zip(feat["chroma_mean"], feat["chroma_std"])):
        flat[f"chroma_{pitch_classes[i]}_mean"] = mean
        flat[f"chroma_{pitch_classes[i]}_std"] = std

    return flat


def build_features_df(features: dict) -> pd.DataFrame:
    """Build a DataFrame from all features."""
    rows = []
    for game_id, game_features in features.items():
        for feat in game_features:
            row = features_to_flat_dict(feat)
            row["game_id"] = game_id
            rows.append(row)
    return pd.DataFrame(rows)


def build_metadata_df(catalog: dict) -> pd.DataFrame:
    """Build a DataFrame from catalog metadata."""
    rows = []
    for game_id, game_data in catalog["games"].items():
        for track in game_data["tracks"]:
            row = {
                "track_id": track["track_id"],
                "game_id": game_id,
                "game_title": game_data["title"],
                "era": game_data["era"],
                "year": game_data["year"],
                "track_number": track["track_number"],
                "title_en": track["title_en"],
                "stage_position": track.get("stage_position"),
                "stage_number": track.get("stage_number"),
                "character": track.get("character"),
                "location": track.get("location"),
                "is_canonical": track.get("is_canonical", True),
                "notes": track.get("notes"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def load_analysis_df(features_path: Optional[Path] = None,
                     catalog_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and merge features + metadata into single analysis DataFrame.

    Returns:
        DataFrame with all features and metadata columns
    """
    catalog = load_catalog(catalog_path)
    features = load_features(features_path)

    meta_df = build_metadata_df(catalog)
    feat_df = build_features_df(features)

    # Merge on track_id
    # Some metadata tracks may not have features (if extraction failed)
    df = meta_df.merge(feat_df, on=["track_id", "game_id"], how="inner")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of numeric feature columns (excluding metadata)."""
    metadata_cols = {
        "track_id", "game_id", "game_title", "era", "year",
        "track_number", "title_en", "stage_position", "stage_number",
        "character", "location", "is_canonical", "notes"
    }
    return [c for c in df.columns if c not in metadata_cols and df[c].dtype in [np.float64, np.int64]]


def filter_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only canonical (non-remix) tracks."""
    return df[df["is_canonical"] == True].copy()


def filter_windows_era(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Windows-era games only (TH06+)."""
    return df[~df["era"].isin(["pc98"])].copy()


if __name__ == "__main__":
    # Test loading
    try:
        df = load_analysis_df()
        print(f"Loaded {len(df)} tracks with {len(df.columns)} columns")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nGames: {df['game_id'].unique()}")
        print(f"\nEras: {df['era'].value_counts()}")
        print(f"\nStage positions: {df['stage_position'].value_counts()}")
    except FileNotFoundError as e:
        print(f"Data not yet extracted: {e}")
