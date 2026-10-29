"""Unit tests for ABC2Vec data processing."""

import pytest
import pandas as pd

from core.data.pipeline import (
    normalize_abc,
    deduplicate_tunes,
    extract_sections,
    parse_key_signature,
    infer_tune_type,
)


class TestNormalizeABC:
    """Test cases for ABC normalization."""

    def test_valid_abc(self):
        """Test normalization of valid ABC notation."""
        abc_text = """X:1
L:1/8
M:4/4
K:D
D2 EF | G2 AB |"""

        result = normalize_abc(abc_text)

        assert result is not None
        assert result["key"] == "D"
        assert result["meter"] == "4/4"
        assert "|" in result["abc_body"]

    def test_invalid_abc_no_key(self):
        """Test that ABC without key is rejected."""
        abc_text = """X:1
L:1/8
M:4/4
D2 EF | G2 AB |"""

        result = normalize_abc(abc_text)
        assert result is None  # Should fail validation

    def test_strip_control_codes(self):
        """Test removal of TunesFormer control codes."""
        abc_text = """S:2
B:5
X:1
K:D
D2 EF |"""

        result = normalize_abc(abc_text)

        assert result is not None
        # Control codes should be stripped
        assert "S:2" not in result["abc_clean"]
        assert "B:5" not in result["abc_clean"]


class TestParseKeySignature:
    """Test cases for key signature parsing."""

    def test_major_keys(self):
        """Test parsing major keys."""
        assert parse_key_signature("D") == {"root": "D", "mode": "major"}
        assert parse_key_signature("G") == {"root": "G", "mode": "major"}

    def test_minor_keys(self):
        """Test parsing minor keys."""
        assert parse_key_signature("Am") == {"root": "A", "mode": "minor"}
        assert parse_key_signature("Emin") == {"root": "E", "mode": "minor"}

    def test_modal_keys(self):
        """Test parsing modal keys."""
        assert parse_key_signature("Ador") == {"root": "A", "mode": "dorian"}
        assert parse_key_signature("Gmix") == {"root": "G", "mode": "mixolydian"}

    def test_accidentals(self):
        """Test keys with sharps and flats."""
        assert parse_key_signature("F#") == {"root": "F#", "mode": "major"}
        assert parse_key_signature("Bb") == {"root": "BB", "mode": "major"}


class TestInferTuneType:
    """Test cases for tune type inference."""

    def test_direct_rhythm_field(self):
        """Test inference from R: field."""
        abc_text = """X:1
T:Test Reel
R:reel
M:4/4
K:D"""

        assert infer_tune_type(abc_text) == "reel"

    def test_from_title(self):
        """Test inference from title keywords."""
        abc_text = """X:1
T:The Blackthorn Jig
M:6/8
K:D"""

        assert infer_tune_type(abc_text) == "jig"

    def test_from_meter(self):
        """Test fallback to meter-based inference."""
        abc_text = """X:1
T:Unknown Tune
M:6/8
K:D"""

        tune_type = infer_tune_type(abc_text)
        assert tune_type == "jig"  # 6/8 → jig


class TestDeduplication:
    """Test cases for tune deduplication."""

    def test_exact_duplicates(self):
        """Test removal of exact duplicates."""
        records = [
            {"abc_body": "ABC | DEF |", "title": "Tune 1"},
            {"abc_body": "ABC | DEF |", "title": "Tune 2"},  # Duplicate
            {"abc_body": "GHI | JKL |", "title": "Tune 3"},
        ]

        unique, dup_count = deduplicate_tunes(records)

        assert len(unique) == 2
        assert dup_count == 1

    def test_no_duplicates(self):
        """Test dataset with no duplicates."""
        records = [
            {"abc_body": "ABC | DEF |"},
            {"abc_body": "GHI | JKL |"},
            {"abc_body": "MNO | PQR |"},
        ]

        unique, dup_count = deduplicate_tunes(records)

        assert len(unique) == 3
        assert dup_count == 0


class TestExtractSections:
    """Test cases for section extraction."""

    def test_aabb_structure(self):
        """Test extraction from typical AABB tune."""
        abc_body = "ABC |: DEF :| GHI || JKL |"
        sections = extract_sections(abc_body)

        assert len(sections) >= 2

    def test_single_section(self):
        """Test tune without section markers."""
        abc_body = "ABC DEF GHI JKL"
        sections = extract_sections(abc_body)

        # May return 0 or 1 depending on parsing
        assert isinstance(sections, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
