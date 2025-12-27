"""
Enrich catalog with stage positions, characters, and analysis flags.

This is the curated metadata layer. Run after catalog.py.
"""

import json
from pathlib import Path


# Stage position mappings by game
# Format: track_number -> (stage_position, stage_number, character, location, is_canonical, notes)
# stage_position: "title" | "stage" | "boss" | "extra" | "phantasm" | "ending" | "staff"

TRACK_METADATA = {
    "TH06": {
        1: ("title", None, None, "scarlet_devil_mansion", True, None),
        2: ("stage", 1, None, "misty_lake", True, None),
        3: ("boss", 1, "Rumia", "misty_lake", True, None),
        4: ("stage", 2, None, "misty_lake", True, None),
        5: ("boss", 2, "Cirno", "misty_lake", True, None),
        6: ("stage", 3, None, "scarlet_devil_mansion", True, None),
        7: ("boss", 3, "Meiling", "scarlet_devil_mansion", True, None),
        8: ("stage", 4, None, "voile_library", True, None),
        9: ("boss", 4, "Patchouli", "voile_library", True, None),
        10: ("stage", 5, None, "scarlet_devil_mansion", True, None),
        11: ("boss", 5, "Sakuya", "scarlet_devil_mansion", True, None),
        12: ("stage", 6, None, "clock_tower", True, None),
        13: ("boss", 6, "Remilia", "clock_tower", True, None),
        14: ("extra", None, None, "scarlet_devil_mansion", True, "extra stage"),
        15: ("boss", None, "Flandre", "scarlet_devil_mansion", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
    },
    "TH07": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "hakurei_shrine", True, None),
        3: ("boss", 1, "Letty", "hakurei_shrine", True, None),
        4: ("stage", 2, None, "forest", True, None),
        5: ("boss", 2, "Chen", "forest", True, None),
        6: ("stage", 3, None, "human_village", True, None),
        7: ("boss", 3, "Alice", "human_village", True, None),
        8: ("stage", 4, None, "netherworld", True, None),
        9: ("boss", 4, "Prismriver", "netherworld", True, "Prismriver Sisters"),
        10: ("stage", 5, None, "netherworld", True, None),
        11: ("boss", 5, "Youmu", "netherworld", True, None),
        12: ("stage", 6, None, "hakugyokurou", True, None),
        13: ("boss", 6, "Yuyuko", "hakugyokurou", True, None),
        14: ("extra", None, None, "mayohiga", True, "extra stage"),
        15: ("boss", None, "Ran", "mayohiga", True, "extra boss"),
        16: ("phantasm", None, None, "boundary", True, "phantasm stage"),
        17: ("boss", None, "Yukari", "boundary", True, "phantasm boss"),
        18: ("ending", None, None, None, True, None),
        19: ("staff", None, None, None, True, None),
        20: ("ending", None, None, None, True, "player score"),
    },
    "TH08": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "bamboo_forest", True, None),
        3: ("boss", 1, "Wriggle", "bamboo_forest", True, None),
        4: ("stage", 2, None, "bamboo_forest", True, None),
        5: ("boss", 2, "Mystia", "bamboo_forest", True, None),
        6: ("stage", 3, None, "human_village", True, None),
        7: ("boss", 3, "Keine", "human_village", True, None),
        8: ("stage", 4, None, "eientei", True, "4A variant exists"),
        9: ("boss", 4, "Marisa/Reimu", "eientei", True, "stage 4 uncanny"),
        10: ("stage", 5, None, "eientei", True, None),
        11: ("boss", 5, "Reisen", "eientei", True, None),
        12: ("stage", 6, None, "eientei", True, "6A variant exists"),
        13: ("boss", 6, "Eirin", "eientei", True, None),
        14: ("boss", 6, "Kaguya", "eientei", True, "final boss"),
        15: ("extra", None, None, "bamboo_forest", True, "extra stage"),
        16: ("boss", None, "Mokou", "bamboo_forest", True, "extra boss"),
        17: ("ending", None, None, None, True, None),
        18: ("ending", None, None, None, True, "bad ending"),
        19: ("ending", None, None, None, True, "good ending"),
        20: ("staff", None, None, None, True, None),
        21: ("ending", None, None, None, True, "in ending select"),
    },
    "TH10": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "youkai_mountain", True, None),
        3: ("boss", 1, "Shizuha/Minoriko", "youkai_mountain", True, None),
        4: ("stage", 2, None, "youkai_mountain", True, None),
        5: ("boss", 2, "Hina", "youkai_mountain", True, None),
        6: ("stage", 3, None, "youkai_mountain", True, None),
        7: ("boss", 3, "Nitori", "youkai_mountain", True, None),
        8: ("stage", 4, None, "moriya_shrine", True, None),
        9: ("boss", 4, "Momiji/Aya", "moriya_shrine", True, None),
        10: ("stage", 5, None, "moriya_shrine", True, None),
        11: ("boss", 5, "Sanae", "moriya_shrine", True, None),
        12: ("stage", 6, None, "moriya_shrine", True, None),
        13: ("boss", 6, "Kanako", "moriya_shrine", True, None),
        14: ("extra", None, None, "moriya_shrine", True, "extra stage"),
        15: ("boss", None, "Suwako", "moriya_shrine", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH11": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "underground", True, None),
        3: ("boss", 1, "Kisume/Yamame", "underground", True, None),
        4: ("stage", 2, None, "underground", True, None),
        5: ("boss", 2, "Parsee", "underground", True, None),
        6: ("stage", 3, None, "ancient_city", True, None),
        7: ("boss", 3, "Yuugi", "ancient_city", True, None),
        8: ("stage", 4, None, "blazing_hell", True, None),
        9: ("boss", 4, "Satori", "blazing_hell", True, None),
        10: ("stage", 5, None, "blazing_hell", True, None),
        11: ("boss", 5, "Rin", "blazing_hell", True, None),
        12: ("stage", 6, None, "blazing_hell", True, None),
        13: ("boss", 6, "Utsuho", "blazing_hell", True, None),
        14: ("extra", None, None, "underground", True, "extra stage"),
        15: ("boss", None, "Koishi", "underground", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH12": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "sky", True, None),
        3: ("boss", 1, "Nazrin", "sky", True, None),
        4: ("stage", 2, None, "sky", True, None),
        5: ("boss", 2, "Kogasa", "sky", True, None),
        6: ("stage", 3, None, "myouren_temple", True, None),
        7: ("boss", 3, "Ichirin", "myouren_temple", True, None),
        8: ("stage", 4, None, "myouren_temple", True, None),
        9: ("boss", 4, "Murasa", "myouren_temple", True, None),
        10: ("stage", 5, None, "makai", True, None),
        11: ("boss", 5, "Shou", "makai", True, None),
        12: ("stage", 6, None, "makai", True, None),
        13: ("boss", 6, "Byakuren", "makai", True, None),
        14: ("extra", None, None, "myouren_temple", True, "extra stage"),
        15: ("boss", None, "Nue", "myouren_temple", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH15": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "sky", True, None),
        3: ("boss", 1, "Seiran", "sky", True, None),
        4: ("stage", 2, None, "forest", True, None),
        5: ("boss", 2, "Ringo", "forest", True, None),
        6: ("stage", 3, None, "lunar_capital", True, None),
        7: ("boss", 3, "Doremy", "lunar_capital", True, None),
        8: ("stage", 4, None, "lunar_capital", True, None),
        9: ("boss", 4, "Sagume", "lunar_capital", True, None),
        10: ("stage", 5, None, "lunar_capital", True, None),
        11: ("boss", 5, "Clownpiece", "lunar_capital", True, None),
        12: ("stage", 6, None, "lunar_capital", True, None),
        13: ("boss", 6, "Junko", "lunar_capital", True, None),
        14: ("extra", None, None, "lunar_capital", True, "extra stage"),
        15: ("boss", None, "Hecatia", "lunar_capital", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH09": {
        # PoFV has different structure - match-based, not traditional stages
        1: ("title", None, None, None, True, None),
        2: ("stage", None, "Reimu", None, True, "match theme"),
        3: ("stage", None, "Marisa", None, True, "match theme"),
        4: ("stage", None, "Sakuya", None, True, "match theme"),
        5: ("stage", None, "Youmu", None, True, "match theme"),
        6: ("stage", None, "Reisen", None, True, "match theme"),
        7: ("stage", None, "Cirno", None, True, "match theme"),
        8: ("stage", None, "Lyrica", None, True, "match theme"),
        9: ("stage", None, "Mystia", None, True, "match theme"),
        10: ("stage", None, "Tewi", None, True, "match theme"),
        11: ("stage", None, "Aya", None, True, "match theme"),
        12: ("stage", None, "Medicine", None, True, "match theme"),
        13: ("stage", None, "Yuuka", None, True, "match theme"),
        14: ("stage", None, "Komachi", None, True, "match theme"),
        15: ("boss", None, "Eiki", None, True, "final boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "extra ending"),
        19: ("ending", None, None, None, True, "player score"),
    },
    "TH13": {
        # TD has trance remixes - flag as non-canonical
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "graveyard", True, None),
        3: ("boss", 1, "Yuyuko", "graveyard", True, None),
        4: ("stage", 2, None, "graveyard", True, None),
        5: ("boss", 2, "Kogasa", "graveyard", True, None),
        6: ("stage", 3, None, "myouren_temple", True, None),
        7: ("boss", 3, "Yoshika", "myouren_temple", True, None),
        8: ("stage", 4, None, "divine_spirit_mausoleum", True, None),
        9: ("boss", 4, "Seiga", "divine_spirit_mausoleum", True, None),
        10: ("stage", 5, None, "divine_spirit_mausoleum", True, None),
        11: ("boss", 5, "Futo", "divine_spirit_mausoleum", True, None),
        12: ("stage", 6, None, "divine_spirit_mausoleum", True, None),
        13: ("boss", 6, "Miko", "divine_spirit_mausoleum", True, None),
        14: ("extra", None, None, "myouren_temple", True, "extra stage"),
        15: ("boss", None, "Mamizou", "myouren_temple", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("stage", 1, None, "graveyard", False, "trance remix"),
        19: ("stage", 2, None, "graveyard", False, "trance remix"),
        20: ("stage", 3, None, "myouren_temple", False, "trance remix"),
        21: ("stage", 4, None, "divine_spirit_mausoleum", False, "trance remix"),
        22: ("stage", 5, None, "divine_spirit_mausoleum", False, "trance remix"),
        23: ("stage", 6, None, "divine_spirit_mausoleum", False, "trance remix"),
        24: ("extra", None, None, "myouren_temple", False, "trance remix"),
        25: ("boss", 1, "Yuyuko", "graveyard", False, "trance remix"),
        26: ("boss", 2, "Kogasa", "graveyard", False, "trance remix"),
        27: ("boss", 3, "Yoshika", "myouren_temple", False, "trance remix"),
        28: ("boss", 4, "Seiga", "divine_spirit_mausoleum", False, "trance remix"),
        29: ("boss", 5, "Futo", "divine_spirit_mausoleum", False, "trance remix"),
        30: ("boss", 6, "Miko", "divine_spirit_mausoleum", False, "trance remix"),
        31: ("boss", None, "Mamizou", "myouren_temple", False, "trance remix"),
    },
    "TH14": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "shining_needle_castle", True, None),
        3: ("boss", 1, "Wakasagihime", "misty_lake", True, None),
        4: ("stage", 2, None, "forest", True, None),
        5: ("boss", 2, "Sekibanki", "human_village", True, None),
        6: ("stage", 3, None, "human_village", True, None),
        7: ("boss", 3, "Kagerou", "bamboo_forest", True, None),
        8: ("stage", 4, None, "bamboo_forest", True, None),
        9: ("boss", 4, "Benben/Yatsuhashi", "shining_needle_castle", True, None),
        10: ("stage", 5, None, "shining_needle_castle", True, None),
        11: ("boss", 5, "Seija", "shining_needle_castle", True, None),
        12: ("stage", 6, None, "shining_needle_castle", True, None),
        13: ("boss", 6, "Shinmyoumaru", "shining_needle_castle", True, None),
        14: ("extra", None, None, "shining_needle_castle", True, "extra stage"),
        15: ("boss", None, "Raiko", "shining_needle_castle", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH16": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "sky", True, None),
        3: ("boss", 1, "Eternity", "sky", True, None),
        4: ("stage", 2, None, "forest", True, None),
        5: ("boss", 2, "Nemuno", "forest", True, None),
        6: ("stage", 3, None, "youkai_mountain", True, None),
        7: ("boss", 3, "Aunn", "youkai_mountain", True, None),
        8: ("stage", 4, None, "land_of_back_doors", True, None),
        9: ("boss", 4, "Narumi", "land_of_back_doors", True, None),
        10: ("stage", 5, None, "land_of_back_doors", True, None),
        11: ("boss", 5, "Satono/Mai", "land_of_back_doors", True, None),
        12: ("stage", 6, None, "land_of_back_doors", True, None),
        13: ("boss", 6, "Okina", "land_of_back_doors", True, None),
        14: ("extra", None, None, "land_of_back_doors", True, "extra stage"),
        15: ("boss", None, "Okina", "land_of_back_doors", True, "extra boss - true form"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH17": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "hell", True, None),
        3: ("boss", 1, "Eika", "hell", True, None),
        4: ("stage", 2, None, "hell", True, None),
        5: ("boss", 2, "Urumi", "hell", True, None),
        6: ("stage", 3, None, "hell", True, None),
        7: ("boss", 3, "Kutaka", "hell", True, None),
        8: ("stage", 4, None, "beast_realm", True, None),
        9: ("boss", 4, "Yachie", "beast_realm", True, None),
        10: ("stage", 5, None, "beast_realm", True, None),
        11: ("boss", 5, "Mayumi", "beast_realm", True, None),
        12: ("stage", 6, None, "beast_realm", True, None),
        13: ("boss", 6, "Keiki", "beast_realm", True, None),
        14: ("extra", None, None, "primate_garden", True, "extra stage"),
        15: ("boss", None, "Saki", "primate_garden", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH18": {
        1: ("title", None, None, None, True, None),
        2: ("stage", 1, None, "rainbow_dragon_cave", True, None),
        3: ("boss", 1, "Mike", "rainbow_dragon_cave", True, None),
        4: ("stage", 2, None, "forest_of_magic", True, None),
        5: ("boss", 2, "Takane", "forest_of_magic", True, None),
        6: ("stage", 3, None, "youkai_mountain", True, None),
        7: ("boss", 3, "Sannyo", "youkai_mountain", True, None),
        8: ("stage", 4, None, "genbu_ravine", True, None),
        9: ("boss", 4, "Misumaru", "genbu_ravine", True, None),
        10: ("stage", 5, None, "genbu_ravine", True, None),
        11: ("boss", 5, "Megumu", "genbu_ravine", True, None),
        12: ("stage", 6, None, "rainbow_dragon_cave", True, None),
        13: ("boss", 6, "Chimata", "rainbow_dragon_cave", True, None),
        14: ("extra", None, None, "ability_card_forest", True, "extra stage"),
        15: ("boss", None, "Momoyo", "ability_card_forest", True, "extra boss"),
        16: ("ending", None, None, None, True, None),
        17: ("staff", None, None, None, True, None),
        18: ("ending", None, None, None, True, "player score"),
    },
    "TH19": {
        # UDoALG is match-based like PoFV
        1: ("title", None, None, None, True, None),
        2: ("stage", None, "Reimu", None, True, "match theme"),
        3: ("stage", None, "Marisa", None, True, "match theme"),
        4: ("stage", None, "Sanae", None, True, "match theme"),
        5: ("stage", None, "Ran", None, True, "match theme"),
        6: ("stage", None, "Aunn", None, True, "match theme"),
        7: ("stage", None, "Nazrin", None, True, "match theme"),
        8: ("stage", None, "Seiran", None, True, "match theme"),
        9: ("stage", None, "Rin", None, True, "match theme"),
        10: ("stage", None, "Tsukasa", None, True, "match theme"),
        11: ("stage", None, "Mamizou", None, True, "match theme"),
        12: ("stage", None, "Yachie", None, True, "match theme"),
        13: ("stage", None, "Saki", None, True, "match theme"),
        14: ("stage", None, "Yuuma", None, True, "match theme"),
        15: ("stage", None, "Suika", None, True, "match theme"),
        16: ("stage", None, "Zanmu", None, True, "match theme"),
        17: ("stage", None, "Hisami", None, True, "match theme"),
        18: ("boss", None, "Enoko", None, True, None),
        19: ("boss", None, "Chiyari", None, True, None),
        20: ("boss", None, "Biten", None, True, None),
        21: ("boss", None, "Zanmu", None, True, "final boss"),
        22: ("ending", None, None, None, True, None),
        23: ("staff", None, None, None, True, None),
        24: ("ending", None, None, None, True, "player score"),
    },
}


def enrich_catalog(catalog: dict) -> dict:
    """Add stage positions, characters, and locations to catalog."""
    enriched = catalog.copy()

    for game_id, game_data in enriched["games"].items():
        if game_id not in TRACK_METADATA:
            # Mark PC-98 and unmapped games for separate handling
            for track in game_data["tracks"]:
                track["stage_position"] = None
                track["stage_number"] = None
                track["character"] = None
                track["location"] = None
                track["is_canonical"] = True
                track["notes"] = "unmapped - requires manual curation"
            continue

        metadata = TRACK_METADATA[game_id]

        for track in game_data["tracks"]:
            track_num = track["track_number"]
            if track_num in metadata:
                pos, stage, char, loc, canonical, notes = metadata[track_num]
                track["stage_position"] = pos
                track["stage_number"] = stage
                track["character"] = char
                track["location"] = loc
                track["is_canonical"] = canonical
                track["notes"] = notes
            else:
                # Track exists but not in mapping
                track["stage_position"] = None
                track["stage_number"] = None
                track["character"] = None
                track["location"] = None
                track["is_canonical"] = True
                track["notes"] = "not in mapping"

    return enriched


def generate_track_id(game_id: str, track_number: int | None, index: int = 0) -> str:
    """Generate unique track ID."""
    if track_number is None:
        return f"{game_id}_U{index:02d}"  # U for unknown/unmapped
    return f"{game_id}_{track_number:02d}"


def add_track_ids(catalog: dict) -> dict:
    """Add unique track IDs to all tracks."""
    for game_id, game_data in catalog["games"].items():
        for i, track in enumerate(game_data["tracks"]):
            track["track_id"] = generate_track_id(game_id, track["track_number"], i)
    return catalog


def main():
    # Load raw catalog
    raw_path = Path(__file__).parent.parent.parent / "data" / "metadata" / "catalog_raw.json"

    with open(raw_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Enrich
    catalog = enrich_catalog(catalog)
    catalog = add_track_ids(catalog)
    catalog["version"] = "1.1"

    # Save enriched catalog
    output_path = Path(__file__).parent.parent.parent / "data" / "metadata" / "catalog.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    # Stats
    mapped_games = [g for g in catalog["games"].keys() if g in TRACK_METADATA]
    unmapped_games = [g for g in catalog["games"].keys() if g not in TRACK_METADATA]

    print(f"Enriched catalog saved to: {output_path}")
    print(f"Games with full mapping: {len(mapped_games)} ({', '.join(mapped_games)})")
    print(f"Games needing curation: {len(unmapped_games)} ({', '.join(unmapped_games)})")

    # Count by stage position
    positions = {}
    for game_data in catalog["games"].values():
        for track in game_data["tracks"]:
            pos = track.get("stage_position") or "unmapped"
            positions[pos] = positions.get(pos, 0) + 1

    print("\nTracks by stage position:")
    for pos, count in sorted(positions.items()):
        print(f"  {pos}: {count}")


if __name__ == "__main__":
    main()
