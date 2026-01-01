"""
Character profiling and tonal descriptor analysis.

Creates sonic signatures for characters and maps features to human-readable descriptors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from analysis.dataloader import load_analysis_df, filter_canonical, filter_windows_era

# Tonal descriptor mappings
# Maps feature ranges to human-readable adjectives
TONAL_DESCRIPTORS = {
    'spectral_centroid_mean': {
        'low': ('dark', 'murky', 'heavy'),
        'mid': ('warm', 'balanced', 'full'),
        'high': ('bright', 'piercing', 'crystalline')
    },
    'rms_energy_mean': {
        'low': ('delicate', 'intimate', 'whispered'),
        'mid': ('moderate', 'steady', 'grounded'),
        'high': ('powerful', 'intense', 'overwhelming')
    },
    'tempo': {
        'low': ('meditative', 'ponderous', 'atmospheric'),
        'mid': ('driving', 'steady', 'flowing'),
        'high': ('frantic', 'urgent', 'relentless')
    },
    'onset_rate': {
        'low': ('sparse', 'spacious', 'breathing'),
        'mid': ('rhythmic', 'pulsing', 'active'),
        'high': ('dense', 'chaotic', 'overwhelming')
    },
    'spectral_entropy': {
        'low': ('pure', 'clean', 'focused'),
        'mid': ('textured', 'layered', 'rich'),
        'high': ('complex', 'dense', 'overwhelming')
    },
    'chroma_entropy': {
        'low': ('tonal', 'melodic', 'centered'),
        'mid': ('harmonically rich', 'colorful', 'varied'),
        'high': ('chromatic', 'restless', 'ambiguous')
    },
    'spectral_flatness_mean': {
        'low': ('tonal', 'melodic', 'pitched'),
        'mid': ('textured', 'mixed', 'hybrid'),
        'high': ('noisy', 'percussive', 'harsh')
    },
    'spectral_bandwidth_mean': {
        'low': ('narrow', 'focused', 'thin'),
        'mid': ('balanced', 'full', 'rounded'),
        'high': ('wide', 'expansive', 'enveloping')
    }
}


def get_percentile_category(value: float, series: pd.Series) -> str:
    """Categorize value as low/mid/high based on percentiles."""
    p33 = series.quantile(0.33)
    p66 = series.quantile(0.66)
    if value < p33:
        return 'low'
    elif value < p66:
        return 'mid'
    return 'high'


def describe_track_tone(row: pd.Series, df: pd.DataFrame) -> dict:
    """Generate tonal descriptors for a single track."""
    descriptors = {}
    for feature, categories in TONAL_DESCRIPTORS.items():
        if feature in row and feature in df.columns:
            cat = get_percentile_category(row[feature], df[feature])
            descriptors[feature] = {
                'category': cat,
                'adjectives': categories[cat]
            }
    return descriptors


def generate_tone_summary(descriptors: dict) -> str:
    """Generate a human-readable tone summary from descriptors."""
    # Pick one adjective from each category for a coherent description
    parts = []

    # Energy/intensity
    if 'rms_energy_mean' in descriptors:
        parts.append(descriptors['rms_energy_mean']['adjectives'][0])

    # Brightness
    if 'spectral_centroid_mean' in descriptors:
        parts.append(descriptors['spectral_centroid_mean']['adjectives'][0])

    # Tempo feel
    if 'tempo' in descriptors:
        parts.append(descriptors['tempo']['adjectives'][0])

    # Complexity
    if 'spectral_entropy' in descriptors:
        parts.append(descriptors['spectral_entropy']['adjectives'][0])

    return ', '.join(parts)


def analyze_characters(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze audio profiles for each character."""
    # Filter to boss themes with character data
    char_df = df[df['character'].notna() & (df['character'] != '')].copy()

    features = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'tempo',
                'rms_energy_mean', 'onset_rate', 'spectral_entropy',
                'chroma_entropy', 'spectral_flatness_mean']

    # Group by character
    char_profiles = char_df.groupby('character')[features].mean()

    # Calculate z-scores relative to all tracks
    char_zscore = pd.DataFrame(index=char_profiles.index)
    for feat in features:
        char_zscore[feat] = (char_profiles[feat] - df[feat].mean()) / df[feat].std()

    return char_profiles, char_zscore


def analyze_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze audio profiles for each location."""
    loc_df = df[df['location'].notna() & (df['location'] != '')].copy()

    features = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'tempo',
                'rms_energy_mean', 'onset_rate', 'spectral_entropy',
                'chroma_entropy', 'spectral_flatness_mean']

    # Group by location
    loc_profiles = loc_df.groupby('location')[features].mean()
    loc_counts = loc_df.groupby('location').size()

    # Filter to locations with 4+ tracks
    significant_locs = loc_counts[loc_counts >= 4].index
    loc_profiles = loc_profiles.loc[significant_locs]

    # Calculate z-scores
    loc_zscore = pd.DataFrame(index=loc_profiles.index)
    for feat in features:
        loc_zscore[feat] = (loc_profiles[feat] - df[feat].mean()) / df[feat].std()

    return loc_profiles, loc_zscore, loc_counts


def plot_character_profiles(char_zscore: pd.DataFrame, output_path: Path):
    """Plot character audio signatures."""
    # Filter to characters with enough data
    char_zscore = char_zscore.dropna()
    if len(char_zscore) < 3:
        print("Not enough character data for visualization")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Heatmap
    im = ax.imshow(char_zscore.values, cmap='RdBu_r', aspect='auto',
                   vmin=-2, vmax=2)

    ax.set_xticks(range(len(char_zscore.columns)))
    ax.set_xticklabels([c.replace('_', '\n') for c in char_zscore.columns],
                       rotation=45, ha='right')
    ax.set_yticks(range(len(char_zscore.index)))
    ax.set_yticklabels(char_zscore.index)

    plt.colorbar(im, ax=ax, label='Z-score')
    ax.set_title('Character Audio Signatures (Z-scores vs all tracks)')

    # Add values
    for i in range(len(char_zscore.index)):
        for j in range(len(char_zscore.columns)):
            val = char_zscore.iloc[i, j]
            color = 'white' if abs(val) > 1 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_location_profiles(loc_zscore: pd.DataFrame, loc_counts: pd.Series,
                          output_path: Path):
    """Plot location audio atmospheres."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Sort by first principal component for visual coherence
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    order = pca.fit_transform(loc_zscore.values).flatten().argsort()
    loc_zscore_sorted = loc_zscore.iloc[order]

    im = ax.imshow(loc_zscore_sorted.values, cmap='RdBu_r', aspect='auto',
                   vmin=-2, vmax=2)

    ax.set_xticks(range(len(loc_zscore_sorted.columns)))
    ax.set_xticklabels([c.replace('_', '\n') for c in loc_zscore_sorted.columns],
                       rotation=45, ha='right')

    # Add track counts to y labels
    ylabels = [f"{loc} (n={loc_counts[loc]})"
               for loc in loc_zscore_sorted.index]
    ax.set_yticks(range(len(loc_zscore_sorted.index)))
    ax.set_yticklabels([l.replace('_', ' ').title() for l in ylabels])

    plt.colorbar(im, ax=ax, label='Z-score')
    ax.set_title('Location Atmospheres (Z-scores vs all tracks)')

    # Add values
    for i in range(len(loc_zscore_sorted.index)):
        for j in range(len(loc_zscore_sorted.columns)):
            val = loc_zscore_sorted.iloc[i, j]
            color = 'white' if abs(val) > 1 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_location_descriptions(loc_zscore: pd.DataFrame, df: pd.DataFrame):
    """Print human-readable descriptions for each location."""
    print("\n" + "=" * 70)
    print("LOCATION ATMOSPHERE DESCRIPTIONS")
    print("=" * 70)

    for location in loc_zscore.index:
        print(f"\n{location.replace('_', ' ').upper()}")
        print("-" * 40)

        # Get extreme features (|z| > 0.5)
        row = loc_zscore.loc[location]
        extremes = row[abs(row) > 0.5].sort_values(key=abs, ascending=False)

        descriptors = []
        for feat, zscore in extremes.items():
            if feat in TONAL_DESCRIPTORS:
                cat = 'high' if zscore > 0 else 'low'
                adj = TONAL_DESCRIPTORS[feat][cat][0]
                descriptors.append(f"{adj} ({feat.replace('_', ' ')})")

        if descriptors:
            print("  Sonic character: " + ", ".join(descriptors[:4]))
        else:
            print("  Sonic character: neutral/average across features")


def print_character_descriptions(char_zscore: pd.DataFrame, df: pd.DataFrame):
    """Print human-readable descriptions for each character."""
    print("\n" + "=" * 70)
    print("CHARACTER THEME DESCRIPTIONS")
    print("=" * 70)

    for character in char_zscore.index:
        print(f"\n{character.upper()}")
        print("-" * 40)

        row = char_zscore.loc[character]
        extremes = row[abs(row) > 0.5].sort_values(key=abs, ascending=False)

        descriptors = []
        for feat, zscore in extremes.items():
            if feat in TONAL_DESCRIPTORS:
                cat = 'high' if zscore > 0 else 'low'
                adj = TONAL_DESCRIPTORS[feat][cat][0]
                descriptors.append(f"{adj} ({feat.replace('_', ' ')})")

        if descriptors:
            print("  Sonic character: " + ", ".join(descriptors[:4]))
        else:
            print("  Sonic character: neutral/average across features")


def main():
    print("Loading data...")
    df = load_analysis_df()
    df = filter_canonical(df)
    df_windows = filter_windows_era(df)

    print(f"Total canonical tracks: {len(df)}")
    print(f"Windows-era tracks: {len(df_windows)}")

    # Character analysis
    print("\n" + "=" * 70)
    print("CHARACTER ANALYSIS")
    print("=" * 70)

    char_profiles, char_zscore = analyze_characters(df_windows)
    print(f"\nCharacters with theme data: {len(char_profiles)}")

    # Filter to characters with notable profiles
    char_zscore_notable = char_zscore[char_zscore.abs().max(axis=1) > 0.3]

    if len(char_zscore_notable) >= 3:
        output_path = Path(__file__).parent.parent / "outputs" / "figures"
        plot_character_profiles(char_zscore_notable,
                               output_path / "05_character_profiles.png")
        print_character_descriptions(char_zscore_notable, df_windows)
    else:
        print("Insufficient character data for detailed analysis")
        print("\nCharacter profiles (raw z-scores):")
        print(char_zscore.round(2))

    # Location analysis
    print("\n" + "=" * 70)
    print("LOCATION ANALYSIS")
    print("=" * 70)

    loc_profiles, loc_zscore, loc_counts = analyze_locations(df_windows)
    print(f"\nLocations with 4+ tracks: {len(loc_profiles)}")

    output_path = Path(__file__).parent.parent / "outputs" / "figures"
    plot_location_profiles(loc_zscore, loc_counts,
                          output_path / "05_location_atmospheres.png")
    print_location_descriptions(loc_zscore, df_windows)

    # Sample track descriptions
    print("\n" + "=" * 70)
    print("SAMPLE TRACK TONE DESCRIPTIONS")
    print("=" * 70)

    # Pick some iconic tracks
    iconic_tracks = [
        'TH06_06',  # Scarlet Devil Mansion
        'TH07_14',  # Yuyuko's theme
        'TH08_15',  # Eirin's theme
        'TH11_12',  # Utsuho's theme
        'TH15_14',  # Junko's theme
    ]

    for track_id in iconic_tracks:
        track = df[df['track_id'] == track_id]
        if len(track) == 0:
            continue
        row = track.iloc[0]

        print(f"\n{track_id}: {row.get('title', 'Unknown')}")
        print(f"  Game: {row['game_id']}, Position: {row.get('stage_position', 'unknown')}")

        descriptors = describe_track_tone(row, df)
        summary = generate_tone_summary(descriptors)
        print(f"  Tone: {summary}")


if __name__ == "__main__":
    main()
