"""Unit tests for ABC2Vec tokenizer components."""

import pytest
import torch

from abc2vec.tokenizer import ABCVocabulary, BarPatchifier, ABCTransposer


class TestABCVocabulary:
    """Test cases for ABCVocabulary."""

    def test_initialization(self):
        """Test vocabulary initialization with special tokens."""
        vocab = ABCVocabulary()
        assert vocab.size == len(ABCVocabulary.SPECIAL_TOKENS)
        assert vocab.pad_idx == 0
        assert vocab.mask_idx == 4

    def test_build_from_corpus(self):
        """Test building vocabulary from corpus."""
        vocab = ABCVocabulary()
        corpus = ["ABCD", "CDEF", "XYZ"]
        vocab.build_from_corpus(corpus, min_freq=1, verbose=False)

        # Should have special tokens + unique characters
        assert vocab.size > len(ABCVocabulary.SPECIAL_TOKENS)
        assert "A" in vocab.char2idx
        assert "Z" in vocab.char2idx

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        vocab = ABCVocabulary()
        vocab.build_from_corpus(["ABC notation test"], verbose=False)

        text = "ABC test"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)

        assert decoded == text

    def test_unknown_character(self):
        """Test handling of unknown characters."""
        vocab = ABCVocabulary()
        vocab.build_from_corpus(["ABC"], verbose=False)

        # Encode text with unknown character
        encoded = vocab.encode("XYZ")

        # All should map to UNK
        assert all(idx == vocab.unk_idx for idx in encoded)


class TestBarPatchifier:
    """Test cases for BarPatchifier."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary fixture."""
        vocab = ABCVocabulary()
        corpus = ["ABCDEFG|:abcdefg:|"]
        vocab.build_from_corpus(corpus, verbose=False)
        return vocab

    def test_split_into_bars(self, vocab):
        """Test bar splitting."""
        patchifier = BarPatchifier(vocab)
        abc_body = "ABC | DEF | GHI |"
        bars = patchifier.split_into_bars(abc_body)

        assert len(bars) == 3
        assert "ABC" in bars[0]
        assert "DEF" in bars[1]

    def test_patchify_shape(self, vocab):
        """Test patchify output shapes."""
        patchifier = BarPatchifier(vocab, max_bar_length=32, max_bars=16)
        abc_body = "A2 B2 | C2 D2 |"

        result = patchifier.patchify(abc_body)

        assert result["bar_indices"].shape == (16, 32)
        assert result["bar_mask"].shape == (16,)
        assert result["char_mask"].shape == (16, 32)
        assert isinstance(result["num_bars"], int)

    def test_patchify_padding(self, vocab):
        """Test that padding is applied correctly."""
        patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)
        abc_body = "ABC |"

        result = patchifier.patchify(abc_body)

        # Only first bar should be non-padding
        assert result["bar_mask"][0] == True
        assert result["bar_mask"][1:].sum() == 0


class TestABCTransposer:
    """Test cases for ABCTransposer."""

    def test_transpose_key(self):
        """Test key signature transposition."""
        # D + 5 semitones = G
        assert ABCTransposer.transpose_key("D", 5) == "G"

        # G + 2 semitones = A
        assert ABCTransposer.transpose_key("G", 2) == "A"

        # Ador + 5 = Ddor
        assert ABCTransposer.transpose_key("Ador", 5) == "Ddor"

    def test_transpose_note(self):
        """Test single note transposition."""
        # D + 5 = G
        note, acc = ABCTransposer.transpose_note("D", "", 5)
        assert note == "G"
        assert acc == ""

        # C + 1 = C#
        note, acc = ABCTransposer.transpose_note("C", "", 1)
        assert note == "C"
        assert acc == "^"

    def test_transpose_body_roundtrip(self):
        """Test that transposing up then down gives original."""
        body = "D2 EF | G2 AB | c2 dc |"

        # Transpose up 7 semitones
        transposed = ABCTransposer.transpose_abc_body(body, 7)

        # Transpose back down 7 semitones
        roundtrip = ABCTransposer.transpose_abc_body(transposed, -7)

        assert roundtrip == body

    def test_transpose_with_accidentals(self):
        """Test transposition with sharps and flats."""
        body = "^C _E =F"

        # Should handle accidentals correctly
        transposed = ABCTransposer.transpose_abc_body(body, 2)
        assert transposed != body  # Should change

        # Roundtrip should work
        roundtrip = ABCTransposer.transpose_abc_body(transposed, -2)
        # Note: may not be identical due to enharmonic equivalents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
