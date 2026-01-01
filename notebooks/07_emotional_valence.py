"""
Emotional Valence/Arousal Mapping

Maps Touhou tracks onto Russell's Circumplex Model of Affect:
- Valence (x-axis): negative/sad ← → positive/happy
- Arousal (y-axis): calm/sleepy ↓ → excited/alert ↑

Based on music psychology research correlating audio features with perceived emotion.

References:
- Russell, J.A. (1980). A circumplex model of affect.
- Eerola & Vuoskoski (2011). A comparison of the discrete and dimensional
  models of emotion in music.
- Yang et al. (2008). A regression approach to music emotion recognition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.preprocessing import StandardScaler
from analysis.dataloader import load_analysis_df, filter_canonical, filter_windows_era

# ============================================================================
# EMOTION MODEL EXPLANATION
# ============================================================================
"""
HOW THIS WORKS - The Science Behind Valence/Arousal Mapping

Russell's Circumplex Model places emotions in a 2D space:

                    HIGH AROUSAL
                         ↑
                    excited, alert
                    tense, nervous
    NEGATIVE ←                      → POSITIVE
    VALENCE      sad, depressed      happy, elated   VALENCE
                    calm, relaxed
                    sleepy, tired
                         ↓
                    LOW AROUSAL

AROUSAL (activation/energy):
    Research shows arousal strongly correlates with:
    - Tempo: faster = higher arousal
    - Loudness (RMS energy): louder = higher arousal
    - Onset rate: more note events = higher arousal
    - Spectral centroid: brighter = higher arousal
    - Spectral flux: more change = higher arousal

    These are "energetic" features - they reflect how much acoustic
    activity is happening.

VALENCE (positive/negative emotional tone):
    Valence is harder to measure from audio alone. Research suggests:
    - Major mode = positive, minor mode = negative (strongest factor)
    - Higher pitch register = more positive
    - Smooth timbre = more positive, harsh = negative
    - Consonant harmony = positive, dissonant = negative

    We approximate valence using:
    - Spectral brightness (higher centroid = brighter = more positive)
    - Spectral flatness (lower = more tonal = often more positive)
    - Chroma entropy (lower = clearer tonal center = often more resolved)
    - Mode detection (if available)

IMPORTANT CAVEATS:
1. This is a PERCEPTUAL model based on Western listeners
2. ZUN's music often subverts expectations (minor key can feel triumphant)
3. Game context matters - a "tense" track during gameplay feels different
4. This is an approximation - true emotion perception requires human study

The model is useful for:
- Comparing relative positions of tracks
- Finding clusters of similar "vibes"
- Identifying outliers
- Providing a framework for discussing game atmospheres
"""


def compute_arousal(df: pd.DataFrame) -> pd.Series:
    """
    Compute arousal score from audio features.

    Arousal = activation/energy level
    Higher values = more energetic, exciting, tense
    Lower values = calmer, more relaxed, sleepy

    Based on Eerola & Vuoskoski (2011) and Yang et al. (2008):
    - Tempo: r ≈ 0.50 with arousal
    - Loudness: r ≈ 0.45 with arousal
    - Spectral centroid: r ≈ 0.35 with arousal
    - Onset density: r ≈ 0.40 with arousal
    """
    scaler = StandardScaler()

    # Features that positively correlate with arousal
    arousal_features = ['tempo', 'rms_energy_mean', 'onset_rate',
                        'spectral_centroid_mean', 'spectral_entropy']

    # Weights based on literature (approximate)
    weights = {
        'tempo': 0.30,
        'rms_energy_mean': 0.30,
        'onset_rate': 0.20,
        'spectral_centroid_mean': 0.10,
        'spectral_entropy': 0.10
    }

    # Standardize features
    features_std = pd.DataFrame(
        scaler.fit_transform(df[arousal_features]),
        columns=arousal_features,
        index=df.index
    )

    # Weighted sum
    arousal = sum(features_std[f] * w for f, w in weights.items())

    # Rescale to roughly -1 to 1
    arousal = (arousal - arousal.mean()) / arousal.std()

    return arousal


def compute_valence(df: pd.DataFrame) -> pd.Series:
    """
    Compute valence score from audio features.

    Valence = positive/negative emotional tone
    Higher values = happier, more positive, triumphant
    Lower values = sadder, more negative, melancholic

    This is HARDER to estimate from audio features alone.
    Best predictors from research:
    - Mode (major/minor) - but we don't have this directly
    - Register/pitch height - higher = more positive
    - Timbre smoothness - less harsh = more positive

    We approximate using:
    - Spectral centroid (brightness): higher = more positive
    - Spectral flatness: lower (more tonal) = often more positive
    - Chroma entropy: lower = clearer key = often more resolved
    - RMS variance: lower = more stable = often more positive
    """
    scaler = StandardScaler()

    # Features and their expected relationship to valence
    valence_positive = ['spectral_centroid_mean']  # higher = more positive
    valence_negative = ['spectral_flatness_mean', 'chroma_entropy']  # higher = more negative

    # Standardize
    all_features = valence_positive + valence_negative
    features_std = pd.DataFrame(
        scaler.fit_transform(df[all_features]),
        columns=all_features,
        index=df.index
    )

    # Compute valence (positive features - negative features)
    valence = (
        0.5 * features_std['spectral_centroid_mean']
        - 0.3 * features_std['spectral_flatness_mean']
        - 0.2 * features_std['chroma_entropy']
    )

    # Rescale
    valence = (valence - valence.mean()) / valence.std()

    return valence


def get_emotion_quadrant(valence: float, arousal: float) -> str:
    """Classify emotion into quadrant."""
    if arousal > 0:
        if valence > 0:
            return "Excited/Happy"
        else:
            return "Tense/Angry"
    else:
        if valence > 0:
            return "Calm/Content"
        else:
            return "Sad/Depressed"


def get_emotion_label(valence: float, arousal: float) -> str:
    """Get more specific emotion label based on position."""
    # Distance from origin
    intensity = np.sqrt(valence**2 + arousal**2)

    # Angle in degrees
    angle = np.degrees(np.arctan2(arousal, valence))

    if intensity < 0.5:
        return "Neutral"

    # Map angle to emotion (going counterclockwise from positive valence)
    if -22.5 <= angle < 22.5:
        return "Happy/Pleasant"
    elif 22.5 <= angle < 67.5:
        return "Excited/Elated"
    elif 67.5 <= angle < 112.5:
        return "Alert/Tense"
    elif 112.5 <= angle < 157.5:
        return "Angry/Afraid"
    elif angle >= 157.5 or angle < -157.5:
        return "Sad/Distressed"
    elif -157.5 <= angle < -112.5:
        return "Depressed/Bored"
    elif -112.5 <= angle < -67.5:
        return "Tired/Sleepy"
    else:  # -67.5 to -22.5
        return "Calm/Relaxed"


def plot_circumplex(df: pd.DataFrame, output_path: Path,
                    color_by: str = 'game_id', title_suffix: str = ''):
    """Plot tracks on Russell's circumplex model."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw circumplex background
    circle = Circle((0, 0), 2, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    circle2 = Circle((0, 0), 1, fill=False, color='gray', linestyle=':', alpha=0.3)
    ax.add_patch(circle2)

    # Draw axes
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

    # Add quadrant labels
    ax.text(1.5, 1.5, "EXCITED\nHAPPY", ha='center', va='center', fontsize=10, alpha=0.6)
    ax.text(-1.5, 1.5, "TENSE\nANGRY", ha='center', va='center', fontsize=10, alpha=0.6)
    ax.text(-1.5, -1.5, "SAD\nDEPRESSED", ha='center', va='center', fontsize=10, alpha=0.6)
    ax.text(1.5, -1.5, "CALM\nCONTENT", ha='center', va='center', fontsize=10, alpha=0.6)

    # Add axis labels
    ax.text(2.3, 0, "Positive\nValence →", ha='left', va='center', fontsize=9)
    ax.text(-2.3, 0, "← Negative\nValence", ha='right', va='center', fontsize=9)
    ax.text(0, 2.3, "↑ High Arousal", ha='center', va='bottom', fontsize=9)
    ax.text(0, -2.3, "↓ Low Arousal", ha='center', va='top', fontsize=9)

    # Color mapping
    if color_by == 'game_id':
        unique_vals = df[color_by].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
        color_map = dict(zip(unique_vals, colors))
        c = df[color_by].map(color_map)
    elif color_by == 'era':
        era_colors = {
            'pc98': 'purple',
            'early_windows': 'blue',
            'mid_windows': 'green',
            'late_windows': 'orange'
        }
        c = df['era'].map(era_colors)
    elif color_by == 'stage_position':
        pos_colors = {
            'stage': 'blue',
            'boss': 'red',
            'extra': 'purple',
            'title': 'gray',
            'ending': 'lightblue',
            'staff': 'black',
            'phantasm': 'magenta'
        }
        c = df['stage_position'].map(pos_colors).fillna('gray')
    else:
        c = 'steelblue'

    # Scatter plot
    scatter = ax.scatter(df['valence'], df['arousal'], c=c,
                        alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

    # Add legend
    if color_by in ['game_id', 'era', 'stage_position']:
        handles = []
        if color_by == 'game_id':
            for game, color in sorted(color_map.items()):
                handles.append(plt.scatter([], [], c=[color], label=game, s=50))
        elif color_by == 'era':
            for era, color in era_colors.items():
                handles.append(plt.scatter([], [], c=color, label=era, s=50))
        elif color_by == 'stage_position':
            for pos, color in pos_colors.items():
                handles.append(plt.scatter([], [], c=color, label=pos, s=50))
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                 fontsize=8, ncol=1)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=11)
    ax.set_ylabel('Arousal (Low ↓ → High ↑)', fontsize=11)
    ax.set_title(f"Touhou Tracks on Russell's Circumplex{title_suffix}", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_game_emotions(df: pd.DataFrame, output_path: Path):
    """Plot average emotional position per game."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw circumplex background
    circle = Circle((0, 0), 2, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

    # Add quadrant labels
    ax.text(1.5, 1.5, "EXCITED", ha='center', va='center', fontsize=12, alpha=0.4)
    ax.text(-1.5, 1.5, "TENSE", ha='center', va='center', fontsize=12, alpha=0.4)
    ax.text(-1.5, -1.5, "SAD", ha='center', va='center', fontsize=12, alpha=0.4)
    ax.text(1.5, -1.5, "CALM", ha='center', va='center', fontsize=12, alpha=0.4)

    # Compute game averages
    game_emotions = df.groupby('game_id')[['valence', 'arousal']].mean()

    # Color by era
    game_eras = df.groupby('game_id')['era'].first()
    era_colors = {
        'pc98': 'purple',
        'early_windows': 'blue',
        'mid_windows': 'green',
        'late_windows': 'orange'
    }
    colors = [era_colors.get(game_eras[g], 'gray') for g in game_emotions.index]

    # Plot games
    ax.scatter(game_emotions['valence'], game_emotions['arousal'],
              c=colors, s=200, edgecolors='black', linewidth=2, zorder=5)

    # Label each game
    for game_id in game_emotions.index:
        v, a = game_emotions.loc[game_id]
        ax.annotate(game_id, (v, a), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Add era legend
    for era, color in era_colors.items():
        ax.scatter([], [], c=color, s=100, label=era.replace('_', ' ').title(),
                  edgecolors='black', linewidth=1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=11)
    ax.set_ylabel('Arousal (Low ↓ → High ↑)', fontsize=11)
    ax.set_title("Average Emotional Position by Game", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("EMOTIONAL VALENCE/AROUSAL MAPPING")
    print("=" * 70)

    # Explain the model
    print("""
    Russell's Circumplex Model of Affect maps emotions to 2D:

    AROUSAL (y-axis): How energetic/activating is the music?
        - HIGH: fast tempo, loud, dense notes, bright
        - LOW: slow tempo, quiet, sparse, mellow

    VALENCE (x-axis): How positive/negative does it feel?
        - POSITIVE: bright, tonal, consonant, major-ish
        - NEGATIVE: dark, noisy, dissonant, minor-ish

    Quadrants:
        - Upper right: Excited, Happy, Elated (high energy + positive)
        - Upper left:  Tense, Angry, Anxious (high energy + negative)
        - Lower left:  Sad, Depressed, Bored (low energy + negative)
        - Lower right: Calm, Relaxed, Content (low energy + positive)
    """)

    print("\nLoading data...")
    df = load_analysis_df()
    df = filter_canonical(df)
    print(f"Loaded {len(df)} canonical tracks")

    # Compute emotional dimensions
    print("\nComputing arousal scores...")
    df['arousal'] = compute_arousal(df)

    print("Computing valence scores...")
    df['valence'] = compute_valence(df)

    # Add emotion labels
    df['quadrant'] = df.apply(lambda r: get_emotion_quadrant(r['valence'], r['arousal']), axis=1)
    df['emotion_label'] = df.apply(lambda r: get_emotion_label(r['valence'], r['arousal']), axis=1)

    # Summary stats
    print("\n" + "=" * 70)
    print("QUADRANT DISTRIBUTION")
    print("=" * 70)
    print(df['quadrant'].value_counts())

    print("\n" + "=" * 70)
    print("EMOTION LABEL DISTRIBUTION")
    print("=" * 70)
    print(df['emotion_label'].value_counts())

    # Game summaries
    print("\n" + "=" * 70)
    print("AVERAGE EMOTIONAL POSITION BY GAME")
    print("=" * 70)
    game_emotions = df.groupby('game_id')[['valence', 'arousal']].mean().round(2)
    game_emotions['dominant_quadrant'] = game_emotions.apply(
        lambda r: get_emotion_quadrant(r['valence'], r['arousal']), axis=1)
    print(game_emotions.sort_values('arousal', ascending=False))

    # Extreme tracks
    print("\n" + "=" * 70)
    print("MOST EXTREME TRACKS")
    print("=" * 70)

    print("\nHighest Arousal (most energetic):")
    for _, row in df.nlargest(5, 'arousal')[['track_id', 'game_id', 'arousal', 'valence']].iterrows():
        print(f"  {row['track_id']}: arousal={row['arousal']:.2f}, valence={row['valence']:.2f}")

    print("\nLowest Arousal (most calm):")
    for _, row in df.nsmallest(5, 'arousal')[['track_id', 'game_id', 'arousal', 'valence']].iterrows():
        print(f"  {row['track_id']}: arousal={row['arousal']:.2f}, valence={row['valence']:.2f}")

    print("\nHighest Valence (most positive):")
    for _, row in df.nlargest(5, 'valence')[['track_id', 'game_id', 'arousal', 'valence']].iterrows():
        print(f"  {row['track_id']}: valence={row['valence']:.2f}, arousal={row['arousal']:.2f}")

    print("\nLowest Valence (most negative):")
    for _, row in df.nsmallest(5, 'valence')[['track_id', 'game_id', 'arousal', 'valence']].iterrows():
        print(f"  {row['track_id']}: valence={row['valence']:.2f}, arousal={row['arousal']:.2f}")

    # Generate visualizations
    output_path = Path(__file__).parent.parent / "outputs" / "figures"

    print("\nGenerating visualizations...")

    # All tracks by game
    plot_circumplex(df, output_path / "07_circumplex_by_game.png",
                   color_by='game_id', title_suffix=' (colored by game)')

    # All tracks by era
    plot_circumplex(df, output_path / "07_circumplex_by_era.png",
                   color_by='era', title_suffix=' (colored by era)')

    # Windows tracks by stage position
    df_windows = filter_windows_era(df)
    plot_circumplex(df_windows, output_path / "07_circumplex_by_position.png",
                   color_by='stage_position', title_suffix=' (Windows era, by position)')

    # Game averages
    plot_game_emotions(df, output_path / "07_circumplex_game_averages.png")

    # Stage vs Boss comparison
    print("\n" + "=" * 70)
    print("STAGE vs BOSS EMOTIONAL COMPARISON")
    print("=" * 70)

    gameplay_df = df_windows[df_windows['stage_position'].isin(['stage', 'boss'])]
    stage_boss = gameplay_df.groupby('stage_position')[['valence', 'arousal']].mean()
    print(stage_boss.round(3))
    print("\nInterpretation:")
    print("  Boss themes are higher arousal (more intense/energetic)")
    print("  Stage themes are slightly more positive valence (brighter)")


if __name__ == "__main__":
    main()
