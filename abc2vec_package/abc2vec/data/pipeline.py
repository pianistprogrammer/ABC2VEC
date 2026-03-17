"""Data processing pipeline for ABC2Vec."""

import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# Meter to tune type mapping for Irish/British folk music
METER_TO_TYPE = {
    "4/4": "reel",
    "2/2": "reel",
    "C": "reel",
    "C|": "reel",
    "6/8": "jig",
    "12/8": "double jig",
    "9/8": "slip jig",
    "3/8": "slip jig",
    "2/4": "polka",
    "3/4": "waltz",
    "6/4": "march",
    "4/8": "hornpipe",
}

# Ordered longest-first for proper matching
TUNE_TYPE_KEYWORDS = [
    "double jig",
    "slip jig",
    "single jig",
    "jig",
    "hornpipe",
    "reel",
    "polka",
    "waltz",
    "march",
    "air",
    "strathspey",
    "slide",
    "mazurka",
    "barndance",
    "schottische",
]


def infer_tune_type(abc_text: str) -> str:
    """
    Infer tune type from ABC notation.

    Priority: R: field > T: title keywords > meter-based fallback

    Args:
        abc_text: ABC notation text

    Returns:
        Inferred tune type string (empty if unknown)
    """
    meter = ""

    for line in abc_text.split("\n"):
        line = line.strip()

        # Direct R: rhythm field
        if line.startswith("R:"):
            return line[2:].strip().lower()

        # Extract meter for fallback
        if line.startswith("M:"):
            meter = line[2:].strip()

    # Scan title for type keywords
    for line in abc_text.split("\n"):
        if line.strip().startswith("T:"):
            line_lower = line.lower()
            for keyword in TUNE_TYPE_KEYWORDS:
                if keyword in line_lower:
                    return keyword

    # Fallback: infer from meter
    return METER_TO_TYPE.get(meter, "")


def parse_key_signature(key_str: str) -> Dict[str, str]:
    """
    Parse ABC key signature into root note and mode.

    Examples:
        - 'D' → {'root': 'D', 'mode': 'major'}
        - 'Ador' → {'root': 'A', 'mode': 'dorian'}
        - 'Gmix' → {'root': 'G', 'mode': 'mixolydian'}

    Args:
        key_str: Key signature string from ABC notation

    Returns:
        Dictionary with 'root' and 'mode' keys
    """
    key_str = key_str.strip()
    if not key_str:
        return {"root": "", "mode": ""}

    # Pattern: root note (with optional # or b) + optional mode
    match = re.match(
        r"^([A-Ga-g][#b]?)\s*"
        r"(maj|min|mix|dor|phr|lyd|loc|m|major|minor|mixolydian|"
        r"dorian|phrygian|lydian|locrian)?",
        key_str,
        re.IGNORECASE,
    )

    if not match:
        return {"root": key_str[0].upper() if key_str else "", "mode": "major"}

    root = match.group(1).upper() if match.group(1) else ""
    mode_str = (match.group(2) or "").lower()

    mode_map = {
        "": "major",
        "maj": "major",
        "major": "major",
        "m": "minor",
        "min": "minor",
        "minor": "minor",
        "mix": "mixolydian",
        "mixolydian": "mixolydian",
        "dor": "dorian",
        "dorian": "dorian",
        "phr": "phrygian",
        "phrygian": "phrygian",
        "lyd": "lydian",
        "lydian": "lydian",
        "loc": "locrian",
        "locrian": "locrian",
    }

    mode = mode_map.get(mode_str, "major")
    return {"root": root, "mode": mode}


def normalize_abc(abc_text: str) -> Optional[Dict]:
    """
    Normalize ABC notation and extract metadata.

    Process:
        1. Strip control codes (S:, B:, E: from TunesFormer)
        2. Separate headers from body
        3. Extract key, meter, tune type, etc.
        4. Validate structure (must have K: field and bars)
        5. Count bars and sections

    Args:
        abc_text: Raw ABC notation text

    Returns:
        Dictionary with normalized ABC and metadata, or None if invalid
    """
    if not abc_text or not isinstance(abc_text, str):
        return None

    lines = abc_text.strip().split("\n")

    # Strip control codes (from sander-wood/abc_cc format)
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip control code lines (S:number, B:number, E:number)
        if re.match(r"^[SBE]:\d+$", stripped):
            continue
        cleaned_lines.append(stripped)

    # Separate header and body
    header_lines = []
    body_lines = []
    in_body = False

    header_info = {
        "key": "",
        "meter": "",
        "tune_type": "",
        "unit_length": "",
        "title": "",
    }

    for line in cleaned_lines:
        if not line:
            continue

        # ABC header fields start with letter:
        if not in_body and re.match(r"^[A-Za-z]:", line):
            header_lines.append(line)
            field = line[0].upper()
            value = line[2:].strip()

            if field == "K":
                header_info["key"] = value
                in_body = True  # K: is always last header field
            elif field == "M":
                header_info["meter"] = value
            elif field == "R":
                header_info["tune_type"] = value.lower()
            elif field == "L":
                header_info["unit_length"] = value
            elif field == "T":
                if not header_info["title"]:
                    header_info["title"] = value
        else:
            in_body = True
            body_lines.append(line)

    # Infer tune type if not present
    if not header_info["tune_type"]:
        # Check title for keywords
        title_lower = header_info["title"].lower()
        for keyword in TUNE_TYPE_KEYWORDS:
            if keyword in title_lower:
                header_info["tune_type"] = keyword
                break

        # Fallback to meter
        if not header_info["tune_type"]:
            header_info["tune_type"] = METER_TO_TYPE.get(
                header_info["meter"], ""
            )

    # Validate: must have key and body
    if not header_info["key"] or not body_lines:
        return None

    body = " ".join(body_lines)

    # Count bars
    num_bars = body.count("|")
    if num_bars < 2:
        return None  # Too short

    # Detect sections (delimited by repeat markers)
    section_delimiters = re.findall(r"\|:|:\||\|\||\[\||\|\]|::", body)
    num_sections = len(section_delimiters) + 1

    # Reconstruct clean ABC
    abc_clean = "\n".join(header_lines) + "\n" + "\n".join(body_lines)

    # Normalize whitespace in body
    body_normalized = re.sub(r"\s+", " ", body).strip()

    return {
        "abc_clean": abc_clean,
        "abc_body": body_normalized,
        "abc_header": "\n".join(header_lines),
        "key": header_info["key"],
        "meter": header_info["meter"],
        "tune_type": header_info["tune_type"],
        "unit_length": header_info["unit_length"],
        "title": header_info["title"],
        "num_bars": num_bars,
        "num_sections": num_sections,
        "char_length": len(abc_clean),
    }


def deduplicate_tunes(records: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Deduplicate tunes based on body hash.

    Catches exact duplicates (same notes, different headers).

    Args:
        records: List of normalized tune dictionaries

    Returns:
        Tuple of (unique_records, duplicate_count)
    """
    seen_hashes = set()
    unique_records = []
    dup_count = 0

    for rec in records:
        # Hash the body only (ignore headers/titles)
        body_hash = hashlib.md5(
            rec["abc_body"].encode("utf-8")
        ).hexdigest()

        if body_hash not in seen_hashes:
            seen_hashes.add(body_hash)
            rec["body_hash"] = body_hash
            unique_records.append(rec)
        else:
            dup_count += 1

    return unique_records, dup_count


def extract_sections(abc_body: str) -> List[str]:
    """
    Extract sections from ABC body.

    Splits on section delimiters: |:, :|, ::, ||, [|, |]

    Args:
        abc_body: ABC notation body string

    Returns:
        List of section strings
    """
    sections = re.split(r"\|:|:\||::||\|||\[\||\|\]", abc_body)
    # Clean and filter empty sections
    sections = [
        s.strip() for s in sections if s.strip() and len(s.strip()) > 5
    ]
    return sections


def download_and_process_dataset(
    output_dir: str,
    test_size: float = 0.05,
    min_char_freq: int = 5,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Complete data pipeline: download, normalize, deduplicate, and split.

    Process:
        1. Download IrishMAN dataset from HuggingFace
        2. Normalize ABC notation and extract metadata
        3. Deduplicate based on body hash
        4. Extract sections for Section Contrastive Loss
        5. Create train/validation/test splits
        6. Save processed data to disk

    Args:
        output_dir: Directory to save processed data
        test_size: Fraction of data for validation split
        min_char_freq: Minimum character frequency for vocabulary
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress information

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download IrishMAN dataset
    if verbose:
        print("Downloading IrishMAN dataset...")

    dataset = load_dataset("sander-wood/irishman")
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if verbose:
        print(f"  Train: {len(train_ds):,} tunes")
        print(f"  Validation: {len(val_ds):,} tunes")

    # Normalize training data
    if verbose:
        print("\nNormalizing ABC notation...")

    text_col = "abc notation"
    normalized_records = []
    failed_count = 0

    iterator = tqdm(train_ds, desc="Normalizing") if verbose else train_ds
    for i, example in enumerate(iterator):
        abc_text = example[text_col]
        result = normalize_abc(abc_text)

        if result is not None:
            result["source"] = "irishman"
            result["original_idx"] = i
            normalized_records.append(result)
        else:
            failed_count += 1

    if verbose:
        print(
            f"  Normalized: {len(normalized_records):,} tunes "
            f"({failed_count} failed)"
        )

    # Normalize validation set
    val_records = []
    for i, example in enumerate(val_ds):
        result = normalize_abc(example[text_col])
        if result is not None:
            result["source"] = "irishman_val"
            result["original_idx"] = i
            val_records.append(result)

    # Deduplicate
    if verbose:
        print("\nDeduplicating...")

    unique_records, dup_count = deduplicate_tunes(normalized_records)
    unique_val, val_dup = deduplicate_tunes(val_records)

    if verbose:
        print(f"  Removed {dup_count:,} duplicates from train")
        print(f"  Removed {val_dup:,} duplicates from validation")

    # Extract metadata (key parsing)
    for rec in unique_records + unique_val:
        key_info = parse_key_signature(rec["key"])
        rec["root_note"] = key_info["root"]
        rec["mode"] = key_info["mode"]

    # Extract sections
    if verbose:
        print("\nExtracting sections...")

    for rec in unique_records:
        sections = extract_sections(rec["abc_body"])
        rec["sections_list"] = sections
        rec["extracted_num_sections"] = len(sections)

    # Create train/val split
    np.random.seed(random_seed)
    train_records, val_records_split = train_test_split(
        unique_records, test_size=test_size, random_state=random_seed
    )
    test_records = unique_val

    # Assign tune IDs
    for i, rec in enumerate(train_records):
        rec["tune_id"] = f"train_{i:06d}"
    for i, rec in enumerate(val_records_split):
        rec["tune_id"] = f"val_{i:06d}"
    for i, rec in enumerate(test_records):
        rec["tune_id"] = f"test_{i:06d}"

    if verbose:
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_records):,} tunes")
        print(f"  Val:   {len(val_records_split):,} tunes")
        print(f"  Test:  {len(test_records):,} tunes")

    # Save processed data
    if verbose:
        print(f"\nSaving to {output_dir}...")

    def save_records(records, name):
        # Remove sections_list for JSON serialization
        serializable = [
            {k: v for k, v in r.items() if k != "sections_list"}
            for r in records
        ]

        # Save JSON
        json_path = output_path / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(serializable, f)

        # Save Parquet
        df = pd.DataFrame(serializable)
        parquet_path = output_path / f"{name}.parquet"
        df.to_parquet(parquet_path, index=False)

        return df

    train_df = save_records(train_records, "train")
    val_df = save_records(val_records_split, "val")
    test_df = save_records(test_records, "test")

    # Save combined metadata CSV
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    meta_cols = [
        "tune_id", "key", "meter", "tune_type", "root_note",
        "mode", "num_bars", "num_sections", "char_length", "source"
    ]
    meta_df = all_df[[c for c in meta_cols if c in all_df.columns]]
    meta_path = output_path / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    # Extract section pairs for Section Contrastive Loss
    if verbose:
        print("\nExtracting section pairs...")

    section_pairs = []
    for rec in train_records:
        if "sections_list" in rec and len(rec["sections_list"]) >= 2:
            sections = rec["sections_list"]
            # Create consecutive pairs
            for j in range(len(sections) - 1):
                section_pairs.append(
                    {
                        "tune_id": rec["tune_id"],
                        "section_a": sections[j],
                        "section_b": sections[j + 1],
                        "key": rec["key"],
                        "tune_type": rec["tune_type"],
                    }
                )

    section_pairs_path = output_path / "section_pairs.json"
    with open(section_pairs_path, "w") as f:
        json.dump(section_pairs, f)

    if verbose:
        print(f"  Extracted {len(section_pairs):,} section pairs")
        print("\n✓ Data pipeline complete!")

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
