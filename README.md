# Touhou Compositional Analysis

Computational musicology project analyzing ZUN's Touhou Project soundtracks. Extracted 110+ audio features from 379 tracks across 19 games (TH01-TH19) to empirically measure compositional evolution, game atmospheres, and stage vs boss theme differences.

**Live demo**: [touhou-web.vercel.app/explorer](https://touhou-web.vercel.app/explorer)

## Key Findings

### Era Evolution (20 Years of ZUN)

| Era | Games | Tempo | Onset Rate | Character |
|-----|-------|-------|------------|-----------|
| PC-98 | TH01-05 | ~150 BPM | 5.5/s | Bright, dense FM synthesis |
| Early Windows | TH06-09 | ~150 BPM | 3.5/s | Classic sound, MIDI origins |
| Mid Windows | TH10-14 | ~140 BPM | 3.5/s | Maturing, darker |
| Late Windows | TH15-19 | ~120 BPM | 3.0/s | Modern, melancholic |

**Key insight**: ZUN's music has gotten slower and moodier over 20 years. The data confirms what fans intuit.

### Stage vs Boss Themes

| Feature | Stage | Boss | Interpretation |
|---------|-------|------|----------------|
| Tempo | 138 BPM | 125 BPM | Stage drives forward |
| Spectral Centroid | 2503 Hz | 2705 Hz | Boss is brighter/piercing |
| Onset Rate | 3.55/s | 2.84/s | Stage is busier |
| Spectral Entropy | 8.35 | 8.57 | Boss is denser |

**Key insight**: Boss themes emphasize *weight over speed*.

### Game Atmosphere Profiles

| Game | Key Stats | Validated Fan Intuition |
|------|-----------|------------------------|
| TH06 (EoSD) | 142.4 BPM, 2781 Hz centroid | Gothic, piercing (brightest Windows OST) |
| TH07 (PCB) | 0.235 RMS energy | Powerful (loudest soundtrack) |
| TH08 (IN) | 148.2 BPM | Frantic, urgent (2nd fastest) |
| **TH11 (SA)** | **2104 Hz centroid** | **Oppressive (darkest in series)** |
| TH12 (UFO) | 2228 Hz centroid, 128 BPM | Dark undertones beneath spring cheer |
| TH15 (LoLK) | 112.6 BPM | Slow, deliberate (2nd slowest) |

### Emotional Valence (Russell's Circumplex)

- **PC-98**: Happy/Excited quadrant (bright FM synthesis)
- **Early Windows**: High arousal, spans positive-negative
- **Mid Windows**: Shifted toward Tense (darker, energetic)
- **Late Windows**: Drifted to Sad/Calm (melancholic)

## Data Pipeline

```
379 tracks × 19 games
        ↓
   [catalog.py] Scan OST folders
        ↓
   [enrich_catalog.py] Add metadata (era, stage position, character)
        ↓
   [features.py] Extract 110+ audio features via librosa
        ↓
   features.json (379 tracks × 110+ features)
        ↓
   Analysis notebooks (7 experiments)
```

## Features Extracted (110+ dimensions)

### Spectral (brightness, timbre)
- Spectral centroid, bandwidth, rolloff, flatness
- Zero-crossing rate

### Temporal
- Tempo (BPM), beat strength, onset rate

### Harmonic
- Chroma (12 pitch classes), chroma entropy

### Timbral
- MFCCs (13 coefficients)
- Spectral entropy

### Aggregation
Each time-varying feature gets: mean, std, min, max, p10, p50, p90, skew, range

## Analysis Notebooks

| Notebook | Purpose | Key Finding |
|----------|---------|-------------|
| 01_era_analysis.py | Era comparison | PC-98 to Windows: onset rate collapses from 5.5 to 3.5/s |
| 02_stage_position_analysis.py | Stage vs boss | Boss = weight over speed |
| 03_game_vibes.py | Per-game profiles | SA (TH11) is measurably the darkest |
| 04_clustering.py | UMAP + K-means | Era is the strongest cluster factor |
| 05_character_and_tone.py | Location/character | Blazing Hell tracks are spectrally oppressive |
| 06_leitmotif_experiment.py | Melodic similarity | SA tracks are melodically isolated |
| 07_emotional_valence.py | Russell's circumplex | Late ZUN has drifted melancholic |

## Setup

```bash
conda create -n touhou-analysis python=3.11
conda activate touhou-analysis
pip install -r requirements.txt

# Run pipeline
python src/extraction/catalog.py        # Scan OST folders
python src/extraction/enrich_catalog.py # Add metadata
python src/extraction/features.py       # Extract features (~17 min)

# Run analyses
python notebooks/01_era_analysis.py
python notebooks/02_stage_position_analysis.py
# ... etc
```

## Project Structure

```
touhou-composition-analysis/
├── data/
│   ├── raw/                    # Symlink to OST collection (FLAC)
│   ├── processed/
│   │   ├── features.json       # 379 tracks × 110+ features
│   │   └── umap_coords.json    # 2D projections for visualization
│   └── metadata/
│       └── catalog.json        # Enriched track metadata
├── src/
│   ├── extraction/             # Data pipeline
│   ├── analysis/               # DataFrame utilities
│   └── visualization/          # Plotting helpers
├── notebooks/                  # 7 analysis scripts
├── outputs/figures/            # Generated visualizations
└── DECISIONS.md                # Technical decision log
```

## Technical Decisions

See [DECISIONS.md](DECISIONS.md) for detailed rationale on:
- Why handcrafted features over deep learning
- Era boundary definitions
- Stage position classification
- PC-98 handling strategy
- Statistical methodology

## Related Projects

- [touhou-style-classifier](https://github.com/TheApexWu/touhou-style-classifier): ML classifier for doujin circle arrangement style (89.5% accuracy on 5 circles)
- [touhou-web](https://github.com/TheApexWu/touhou-web): Interactive web demo with UMAP track explorer and personal game commentary

## License

MIT
