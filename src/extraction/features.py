"""
Audio feature extraction pipeline.

Extracts spectral, temporal, and complexity features from FLAC audio files.
Uses librosa for all audio analysis.
"""

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from scipy import stats
from tqdm import tqdm


@dataclass
class FeatureAggregation:
    """Statistical aggregation of a time-series feature."""
    mean: float
    std: float
    min: float
    max: float
    p10: float
    p50: float
    p90: float
    skew: float
    range: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "FeatureAggregation":
        """Compute aggregation from numpy array."""
        arr = arr.flatten()
        arr = arr[~np.isnan(arr)]  # Remove NaN values

        if len(arr) == 0:
            return cls(
                mean=0.0, std=0.0, min=0.0, max=0.0,
                p10=0.0, p50=0.0, p90=0.0, skew=0.0, range=0.0
            )

        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p10=float(np.percentile(arr, 10)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            skew=float(stats.skew(arr)) if len(arr) > 2 else 0.0,
            range=float(np.max(arr) - np.min(arr)),
        )


@dataclass
class TrackFeatures:
    """All extracted features for a single track."""
    track_id: str
    duration_seconds: float

    # Spectral features (aggregated)
    spectral_centroid: FeatureAggregation
    spectral_bandwidth: FeatureAggregation
    spectral_rolloff: FeatureAggregation
    spectral_flatness: FeatureAggregation
    zero_crossing_rate: FeatureAggregation

    # MFCCs (first 13 coefficients, each aggregated)
    mfcc_means: list[float]  # Mean of each coefficient
    mfcc_stds: list[float]   # Std of each coefficient

    # Chroma features
    chroma_mean: list[float]  # 12 pitch classes
    chroma_std: list[float]

    # Temporal features
    tempo: float
    beat_strength: float
    onset_rate: float  # Onsets per second

    # Energy features
    rms_energy: FeatureAggregation

    # Complexity metrics
    spectral_entropy: float  # Entropy over spectral frames
    chroma_entropy: float    # Harmonic complexity


def extract_features(audio_path: Path, sr: int = 22050) -> Optional[TrackFeatures]:
    """
    Extract all features from an audio file.

    Args:
        audio_path: Path to FLAC file
        sr: Sample rate for analysis

    Returns:
        TrackFeatures object or None if extraction fails
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = [float(np.mean(mfccs[i])) for i in range(13)]
        mfcc_stds = [float(np.std(mfccs[i])) for i in range(13)]

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = [float(np.mean(chroma[i])) for i in range(12)]
        chroma_std = [float(np.std(chroma[i])) for i in range(12)]

        # Tempo and beat
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)

        # Beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = float(np.mean(onset_env)) if len(onset_env) > 0 else 0.0

        # Onset rate
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        onset_rate = len(onsets) / duration if duration > 0 else 0.0

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]

        # Entropy calculations
        # Spectral entropy: entropy over spectrogram magnitudes
        S = np.abs(librosa.stft(y))
        S_normalized = S / (S.sum(axis=0, keepdims=True) + 1e-10)
        spectral_entropy_per_frame = -np.sum(S_normalized * np.log2(S_normalized + 1e-10), axis=0)
        spectral_entropy = float(np.mean(spectral_entropy_per_frame))

        # Chroma entropy: harmonic complexity
        chroma_normalized = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-10)
        chroma_entropy_per_frame = -np.sum(chroma_normalized * np.log2(chroma_normalized + 1e-10), axis=0)
        chroma_entropy = float(np.mean(chroma_entropy_per_frame))

        # Track ID from filename
        track_id = audio_path.stem.replace(" ", "_")

        return TrackFeatures(
            track_id=track_id,
            duration_seconds=duration,
            spectral_centroid=FeatureAggregation.from_array(spectral_centroid),
            spectral_bandwidth=FeatureAggregation.from_array(spectral_bandwidth),
            spectral_rolloff=FeatureAggregation.from_array(spectral_rolloff),
            spectral_flatness=FeatureAggregation.from_array(spectral_flatness),
            zero_crossing_rate=FeatureAggregation.from_array(zcr),
            mfcc_means=mfcc_means,
            mfcc_stds=mfcc_stds,
            chroma_mean=chroma_mean,
            chroma_std=chroma_std,
            tempo=tempo,
            beat_strength=beat_strength,
            onset_rate=onset_rate,
            rms_energy=FeatureAggregation.from_array(rms),
            spectral_entropy=spectral_entropy,
            chroma_entropy=chroma_entropy,
        )

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def features_to_dict(features: TrackFeatures) -> dict:
    """Convert TrackFeatures to JSON-serializable dict."""
    d = asdict(features)

    # Convert FeatureAggregation objects
    for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                'spectral_flatness', 'zero_crossing_rate', 'rms_energy']:
        if isinstance(d[key], dict):
            continue  # Already a dict from asdict

    return d


def extract_all_features(catalog_path: Path, output_path: Path):
    """
    Extract features for all tracks in catalog.

    Args:
        catalog_path: Path to catalog.json
        output_path: Path to save features.json
    """
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    all_features = {}
    total_tracks = sum(g["track_count"] for g in catalog["games"].values())

    with tqdm(total=total_tracks, desc="Extracting features") as pbar:
        for game_id, game_data in catalog["games"].items():
            game_features = []

            for track in game_data["tracks"]:
                flac_path = Path(track["flac_path"])

                if not flac_path.exists():
                    print(f"Warning: File not found: {flac_path}")
                    pbar.update(1)
                    continue

                # Suppress librosa warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    features = extract_features(flac_path)

                if features is not None:
                    # Use catalog track_id for consistency
                    feat_dict = features_to_dict(features)
                    feat_dict["track_id"] = track["track_id"]
                    game_features.append(feat_dict)

                pbar.update(1)

            all_features[game_id] = game_features

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2)

    print(f"\nFeatures saved to: {output_path}")
    total_extracted = sum(len(f) for f in all_features.values())
    print(f"Total tracks processed: {total_extracted}")


def main():
    base_path = Path(__file__).parent.parent.parent
    catalog_path = base_path / "data" / "metadata" / "catalog.json"
    output_path = base_path / "data" / "processed" / "features.json"

    extract_all_features(catalog_path, output_path)


if __name__ == "__main__":
    main()
