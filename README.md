# Touhou Compositional Analysis

Computational musicology project analyzing ZUN's Touhou Project soundtracks to empirically explain game atmospheres, trace compositional evolution, and surface patterns across 19 games.

## Goals

1. Validate community intuitions about game atmospheres through spectral evidence
2. Track ZUN's evolution from PC-98 FM synthesis to modern Windows production
3. Surface patterns fans haven't explicitly articulated

## Data

- FLAC audio from Internet Archive Touhou OST collection
- Pre-generated mel spectrograms
- Custom metadata catalog with stage positions, eras, characters

## Status

In development

## Project Structure

```
touhou-composition-analysis/
├── data/
│   ├── raw/                    # Symlink to OST collection
│   ├── processed/              # Extracted features
│   └── metadata/               # Track catalog
├── src/
│   ├── extraction/             # Feature extraction pipeline
│   ├── analysis/               # Classification, clustering
│   └── visualization/          # Plotting
├── notebooks/                  # Exploratory analysis
└── outputs/
    ├── figures/
    └── models/
```

## Setup

```bash
conda create -n touhou-analysis python=3.11
conda activate touhou-analysis
pip install -r requirements.txt
```

## License

MIT
