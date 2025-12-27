"""
Build track catalog from OST collection.

Scans the OST folders and creates structured metadata for each track.
Manual curation required for stage positions, characters, and locations.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


# Game metadata
GAMES = {
    "TH01": {
        "title": "Highly Responsive to Prayers",
        "era": "pc98",
        "year": 1996,
        "folder_pattern": "touhou-1-",
    },
    "TH02": {
        "title": "Story of Eastern Wonderland",
        "era": "pc98",
        "year": 1997,
        "folder_pattern": "touhou-2-",
    },
    "TH03": {
        "title": "Phantasmagoria of Dim.Dream",
        "era": "pc98",
        "year": 1997,
        "folder_pattern": "touhou-3-",
    },
    "TH04": {
        "title": "Lotus Land Story",
        "era": "pc98",
        "year": 1998,
        "folder_pattern": "touhou-4-",
    },
    "TH05": {
        "title": "Mystic Square",
        "era": "pc98",
        "year": 1998,
        "folder_pattern": "touhou-5-",
    },
    "TH06": {
        "title": "Embodiment of Scarlet Devil",
        "era": "early_windows",
        "year": 2002,
        "folder_pattern": "touhou-6-",
    },
    "TH07": {
        "title": "Perfect Cherry Blossom",
        "era": "early_windows",
        "year": 2003,
        "folder_pattern": "touhou-7-",
    },
    "TH08": {
        "title": "Imperishable Night",
        "era": "early_windows",
        "year": 2004,
        "folder_pattern": "touhou-8-",
    },
    "TH09": {
        "title": "Phantasmagoria of Flower View",
        "era": "early_windows",
        "year": 2005,
        "folder_pattern": "touhou-9-",
    },
    "TH10": {
        "title": "Mountain of Faith",
        "era": "mid_windows",
        "year": 2007,
        "folder_pattern": "touhou-10-",
    },
    "TH11": {
        "title": "Subterranean Animism",
        "era": "mid_windows",
        "year": 2008,
        "folder_pattern": "touhou-11-",
    },
    "TH12": {
        "title": "Undefined Fantastic Object",
        "era": "mid_windows",
        "year": 2009,
        "folder_pattern": "touhou-12-",
    },
    "TH13": {
        "title": "Ten Desires",
        "era": "mid_windows",
        "year": 2011,
        "folder_pattern": "touhou-13-",
    },
    "TH14": {
        "title": "Double Dealing Character",
        "era": "mid_windows",
        "year": 2013,
        "folder_pattern": "touhou-14-",
    },
    "TH15": {
        "title": "Legacy of Lunatic Kingdom",
        "era": "late_windows",
        "year": 2015,
        "folder_pattern": "touhou-15-",
    },
    "TH16": {
        "title": "Hidden Star in Four Seasons",
        "era": "late_windows",
        "year": 2017,
        "folder_pattern": "touhou-16-",
    },
    "TH17": {
        "title": "Wily Beast and Weakest Creature",
        "era": "late_windows",
        "year": 2019,
        "folder_pattern": "touhou-17-",
    },
    "TH18": {
        "title": "Unconnected Marketeers",
        "era": "late_windows",
        "year": 2021,
        "folder_pattern": "touhou-18-",
    },
    "TH19": {
        "title": "Unfinished Dream of All Living Ghost",
        "era": "late_windows",
        "year": 2023,
        "folder_pattern": "touhou-19-",
    },
}


def find_game_folder(ost_path: Path, game_id: str) -> Optional[Path]:
    """Find the folder for a specific game."""
    pattern = GAMES[game_id]["folder_pattern"]
    for folder in ost_path.iterdir():
        if folder.is_dir() and pattern in folder.name.lower():
            return folder
    return None


def extract_track_number(filename: str) -> Optional[int]:
    """Extract track number from filename like '01. Track Name.flac'."""
    match = re.match(r"^(\d+)\.", filename)
    if match:
        return int(match.group(1))
    return None


def extract_track_title(filename: str) -> str:
    """Extract track title from filename."""
    # Remove track number prefix
    title = re.sub(r"^\d+\.\s*", "", filename)
    # Remove file extension
    title = re.sub(r"\.(flac|mp3|wav)$", "", title, flags=re.IGNORECASE)
    return title.strip()


def scan_ost_folder(folder: Path) -> list[dict]:
    """Scan an OST folder and return track info."""
    tracks = []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() == ".flac":
            track_num = extract_track_number(file.name)
            title = extract_track_title(file.name)

            # Check for corresponding spectrogram
            spec_name = file.stem + "_spectrogram.png"
            has_spectrogram = (folder / spec_name).exists()

            tracks.append({
                "track_number": track_num,
                "title_en": title,
                "filename": file.name,
                "has_spectrogram": has_spectrogram,
                "flac_path": str(file),
            })

    return tracks


def build_catalog(ost_path: Path) -> dict:
    """Build full catalog from OST collection."""
    catalog = {
        "version": "1.0",
        "games": {},
    }

    for game_id, game_info in GAMES.items():
        folder = find_game_folder(ost_path, game_id)

        if folder is None:
            print(f"Warning: No folder found for {game_id}")
            continue

        tracks = scan_ost_folder(folder)

        catalog["games"][game_id] = {
            "game_id": game_id,
            "title": game_info["title"],
            "era": game_info["era"],
            "year": game_info["year"],
            "folder": str(folder),
            "track_count": len(tracks),
            "tracks": tracks,
        }

        print(f"{game_id}: {game_info['title']} - {len(tracks)} tracks")

    return catalog


def main():
    # Path to OST collection (adjust as needed)
    ost_path = Path(__file__).parent.parent.parent / "data" / "raw" / "ost"

    if not ost_path.exists():
        print(f"OST path not found: {ost_path}")
        print("Expected symlink at data/raw/ost pointing to OST collection")
        return

    catalog = build_catalog(ost_path)

    # Save catalog
    output_path = Path(__file__).parent.parent.parent / "data" / "metadata" / "catalog_raw.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nCatalog saved to: {output_path}")
    print(f"Total games: {len(catalog['games'])}")
    total_tracks = sum(g["track_count"] for g in catalog["games"].values())
    print(f"Total tracks: {total_tracks}")


if __name__ == "__main__":
    main()
