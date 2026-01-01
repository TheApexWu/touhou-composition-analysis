"""
Leitmotif Detection Experiment

Attempts to find recurring melodic patterns across Touhou tracks using:
1. Chromagram extraction (pitch class profiles over time)
2. Dynamic Time Warping (DTW) for melodic similarity
3. Cross-correlation to find shared motifs

This is experimental - melodic analysis is hard without proper transcription.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import cdist
from dtaidistance import dtw
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Iconic tracks to analyze (known to share motifs or be memorable)
ICONIC_TRACKS = {
    # Main themes / Title screens
    'TH06_01': 'EoSD Title - A Dream More Scarlet than Red',
    'TH07_01': 'PCB Title - Mystic Dream',
    'TH08_01': 'IN Title - Eternal Night Vignette',

    # Reimu's themes
    'TH06_04': 'Reimu Stage 1 - Shrine Maiden',
    'TH07_18': 'Reimu Extra - Dream Land',

    # Marisa's themes
    'TH06_10': 'Marisa Stage 4 - Voile Library',
    'TH07_05': 'Marisa Stage 2 - Doll Maker',

    # Remilia/Scarlet themes
    'TH06_14': 'Remilia - Septette for Dead Princess',
    'TH06_16': 'Flandre - UN Owen Was Her',

    # Yuyuko themes
    'TH07_13': 'Yuyuko - Border of Life',
    'TH07_14': 'Resurrection Butterfly',

    # Kaguya/Eirin/Mokou
    'TH08_17': 'Kaguya - Flight of Bamboo Cutter',
    'TH08_20': 'Mokou - Reach for the Moon',

    # SA Underground themes
    'TH11_05': 'Parsee - Green-Eyed Jealousy',
    'TH11_13': 'Utsuho - Nuclear Fusion',

    # Cross-game potential connections
    'TH10_06': 'Sanae - Faith is for the Transient',
    'TH13_14': 'Miko - Shoutoku Legend',
    'TH15_16': 'Junko - Pure Furies',
}


def extract_chromagram(audio_path: Path, sr: int = 22050,
                       hop_length: int = 512) -> np.ndarray:
    """Extract chromagram from audio file."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Use CQT-based chroma for better pitch resolution
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    return chroma


def extract_melody_contour(chroma: np.ndarray) -> np.ndarray:
    """
    Extract simplified melody contour from chromagram.
    Takes the dominant pitch class at each frame.
    """
    # Get dominant pitch class per frame
    melody = np.argmax(chroma, axis=0)

    # Smooth to reduce noise
    kernel_size = 5
    melody_smooth = np.convolve(melody, np.ones(kernel_size)/kernel_size, mode='same')

    return melody_smooth


def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute DTW distance between two sequences."""
    # Subsample for speed (take every 4th frame)
    seq1_sub = seq1[::4]
    seq2_sub = seq2[::4]

    # Normalize
    seq1_norm = (seq1_sub - np.mean(seq1_sub)) / (np.std(seq1_sub) + 1e-6)
    seq2_norm = (seq2_sub - np.mean(seq2_sub)) / (np.std(seq2_sub) + 1e-6)

    # DTW distance
    distance = dtw.distance(seq1_norm, seq2_norm)

    return distance


def find_similar_segments(chroma1: np.ndarray, chroma2: np.ndarray,
                         segment_length: int = 100,
                         hop: int = 50) -> list:
    """
    Find similar melodic segments between two tracks using sliding window.
    Returns list of (pos1, pos2, similarity) tuples.
    """
    similarities = []

    n_frames1 = chroma1.shape[1]
    n_frames2 = chroma2.shape[1]

    # Slide window across track 1
    for i in range(0, n_frames1 - segment_length, hop):
        seg1 = chroma1[:, i:i+segment_length]
        seg1_flat = seg1.flatten()

        best_sim = -1
        best_pos = 0

        # Find best match in track 2
        for j in range(0, n_frames2 - segment_length, hop):
            seg2 = chroma2[:, j:j+segment_length]
            seg2_flat = seg2.flatten()

            # Cosine similarity
            sim = np.dot(seg1_flat, seg2_flat) / (
                np.linalg.norm(seg1_flat) * np.linalg.norm(seg2_flat) + 1e-6)

            if sim > best_sim:
                best_sim = sim
                best_pos = j

        if best_sim > 0.7:  # Threshold for "similar"
            similarities.append((i, best_pos, best_sim))

    return similarities


def plot_chromagram_comparison(chroma1: np.ndarray, chroma2: np.ndarray,
                               name1: str, name2: str, output_path: Path):
    """Plot two chromagrams for visual comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    # First 30 seconds only (about 1300 frames at 22050/512)
    max_frames = min(1300, chroma1.shape[1], chroma2.shape[1])

    librosa.display.specshow(chroma1[:, :max_frames], y_axis='chroma',
                            x_axis='time', ax=axes[0], hop_length=512, sr=22050)
    axes[0].set_title(f'{name1} (first 30s)')

    librosa.display.specshow(chroma2[:, :max_frames], y_axis='chroma',
                            x_axis='time', ax=axes[1], hop_length=512, sr=22050)
    axes[1].set_title(f'{name2} (first 30s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("LEITMOTIF DETECTION EXPERIMENT")
    print("=" * 70)
    print("\nExtracting chromagrams from iconic tracks...")

    # Load catalog to get file paths
    base_path = Path(__file__).parent.parent
    catalog_path = base_path / "data" / "metadata" / "catalog.json"

    with open(catalog_path) as f:
        catalog = json.load(f)

    # Build track_id -> path mapping
    track_paths = {}
    for game_id, game_data in catalog["games"].items():
        for track in game_data["tracks"]:
            track_paths[track["track_id"]] = Path(track["flac_path"])

    # Extract chromagrams for iconic tracks
    chromagrams = {}
    melodies = {}

    available_tracks = [t for t in ICONIC_TRACKS.keys() if t in track_paths]
    print(f"Found {len(available_tracks)}/{len(ICONIC_TRACKS)} iconic tracks")

    for track_id in tqdm(available_tracks, desc="Extracting chromagrams"):
        path = track_paths[track_id]
        if not path.exists():
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chroma = extract_chromagram(path)
            melody = extract_melody_contour(chroma)

        chromagrams[track_id] = chroma
        melodies[track_id] = melody

    print(f"\nExtracted {len(chromagrams)} chromagrams")

    # Compute pairwise melodic distances
    print("\nComputing melodic similarity matrix...")
    track_ids = list(chromagrams.keys())
    n = len(track_ids)

    distance_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="DTW distances"):
        for j in range(i+1, n):
            dist = compute_dtw_distance(melodies[track_ids[i]],
                                       melodies[track_ids[j]])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Convert to similarity (inverse of distance)
    max_dist = distance_matrix.max()
    similarity_matrix = 1 - (distance_matrix / max_dist)
    np.fill_diagonal(similarity_matrix, 1.0)

    # Find most similar pairs (excluding self)
    print("\n" + "=" * 70)
    print("MOST MELODICALLY SIMILAR TRACK PAIRS")
    print("=" * 70)

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((track_ids[i], track_ids[j], similarity_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    for t1, t2, sim in pairs[:15]:
        name1 = ICONIC_TRACKS.get(t1, t1)
        name2 = ICONIC_TRACKS.get(t2, t2)
        print(f"\n{sim:.3f}: {t1} <-> {t2}")
        print(f"       {name1[:40]}")
        print(f"       {name2[:40]}")

    # Plot similarity matrix
    output_path = base_path / "outputs" / "figures"

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0.5, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(track_ids, rotation=90, fontsize=8)
    ax.set_yticklabels(track_ids, fontsize=8)

    plt.colorbar(im, ax=ax, label='Melodic Similarity')
    ax.set_title('Cross-Track Melodic Similarity (DTW on melody contours)')

    plt.tight_layout()
    plt.savefig(output_path / "06_melodic_similarity_matrix.png", dpi=150)
    plt.close()
    print(f"\nSaved: {output_path / '06_melodic_similarity_matrix.png'}")

    # Find potential leitmotif candidates
    print("\n" + "=" * 70)
    print("POTENTIAL LEITMOTIF CONNECTIONS")
    print("=" * 70)

    # Group by game to find cross-game similarities
    cross_game_pairs = [(t1, t2, sim) for t1, t2, sim in pairs
                        if t1[:4] != t2[:4] and sim > 0.6]

    if cross_game_pairs:
        print("\nHigh similarity across different games (potential shared motifs):")
        for t1, t2, sim in cross_game_pairs[:10]:
            print(f"  {sim:.3f}: {t1} ({t1[:4]}) <-> {t2} ({t2[:4]})")
    else:
        print("\nNo strong cross-game melodic connections found at threshold 0.6")
        print("(This suggests ZUN's melodies are quite game-specific)")

    # Same-game high similarity (thematic unity within game)
    same_game_pairs = [(t1, t2, sim) for t1, t2, sim in pairs
                       if t1[:4] == t2[:4] and sim > 0.65]

    if same_game_pairs:
        print("\nHigh similarity within same game (thematic unity):")
        for t1, t2, sim in same_game_pairs[:8]:
            print(f"  {sim:.3f}: {t1} <-> {t2}")

    # Visualize a few interesting comparisons
    print("\nGenerating chromagram comparisons...")

    if len(pairs) > 0:
        # Most similar pair
        t1, t2, sim = pairs[0]
        plot_chromagram_comparison(
            chromagrams[t1], chromagrams[t2],
            f"{t1}: {ICONIC_TRACKS.get(t1, '')[:30]}",
            f"{t2}: {ICONIC_TRACKS.get(t2, '')[:30]}",
            output_path / "06_chroma_comparison_most_similar.png"
        )
        print(f"Saved: 06_chroma_comparison_most_similar.png")

    print("\n" + "=" * 70)
    print("EXPERIMENT CONCLUSIONS")
    print("=" * 70)
    print("""
    This chromagram-based approach has limitations:
    - Chromagrams capture harmony/pitch class, not exact melody
    - DTW on contours loses rhythmic information
    - ZUN's motifs are often rhythmic as much as melodic

    What we CAN detect:
    - Harmonic similarity (similar chord progressions)
    - Pitch class distributions (similar scales/modes)
    - General melodic shape

    For true leitmotif detection, would need:
    - Proper melody transcription (hard problem)
    - Rhythm-aware pattern matching
    - Knowledge of ZUN's known motifs as templates
    """)


if __name__ == "__main__":
    main()
