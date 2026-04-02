"""Character-level vocabulary for ABC notation."""

import collections
import json
from typing import Dict, List, Optional


class ABCVocabulary:
    """
    Character-level vocabulary for ABC notation.

    Manages the mapping between characters and indices, including
    special tokens for padding, masking, and sequence markers.

    Attributes:
        SPECIAL_TOKENS: List of special tokens [PAD], [UNK], [CLS], [SEP], [MASK]
        char2idx: Dictionary mapping characters to integer indices
        idx2char: Dictionary mapping integer indices to characters
        char_freq: Counter of character frequencies in the corpus
    """

    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self) -> None:
        """Initialize vocabulary with special tokens."""
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.char_freq = collections.Counter()

        # Initialize with special tokens
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.char2idx[token] = i
            self.idx2char[i] = token

    def build_from_corpus(
        self,
        texts: List[str],
        min_freq: int = 1,
        verbose: bool = True
    ) -> None:
        """
        Build vocabulary from a list of ABC text strings.

        Args:
            texts: List of ABC notation strings to build vocabulary from
            min_freq: Minimum frequency threshold for including a character
            verbose: Whether to print progress information
        """
        # Count all characters in the corpus
        for text in texts:
            for char in text:
                self.char_freq[char] += 1

        # Add characters that meet minimum frequency threshold
        idx = len(self.SPECIAL_TOKENS)
        for char, freq in sorted(self.char_freq.items()):
            if freq >= min_freq and char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

        if verbose:
            print(
                f"Vocabulary size: {len(self.char2idx)} "
                f"({len(self.SPECIAL_TOKENS)} special + "
                f"{len(self.char2idx) - len(self.SPECIAL_TOKENS)} chars)"
            )

    def encode(self, text: str) -> List[int]:
        """
        Encode text string to list of character indices.

        Args:
            text: ABC notation string to encode

        Returns:
            List of integer indices corresponding to characters
        """
        unk_idx = self.char2idx["[UNK]"]
        return [self.char2idx.get(char, unk_idx) for char in text]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Decode list of indices back to text string.

        Args:
            indices: List of integer indices to decode
            skip_special: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, "?")
            if skip_special and char in self.SPECIAL_TOKENS:
                continue
            chars.append(char)
        return "".join(chars)

    @property
    def size(self) -> int:
        """Return vocabulary size."""
        return len(self.char2idx)

    @property
    def pad_idx(self) -> int:
        """Return padding token index."""
        return self.char2idx["[PAD]"]

    @property
    def mask_idx(self) -> int:
        """Return mask token index."""
        return self.char2idx["[MASK]"]

    @property
    def cls_idx(self) -> int:
        """Return CLS token index."""
        return self.char2idx["[CLS]"]

    @property
    def sep_idx(self) -> int:
        """Return SEP token index."""
        return self.char2idx["[SEP]"]

    @property
    def unk_idx(self) -> int:
        """Return unknown token index."""
        return self.char2idx["[UNK]"]

    def save(self, path: str) -> None:
        """
        Save vocabulary to JSON file.

        Args:
            path: Path to save vocabulary file
        """
        with open(path, "w") as f:
            json.dump(
                {
                    "char2idx": self.char2idx,
                    "char_freq": dict(self.char_freq),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "ABCVocabulary":
        """
        Load vocabulary from JSON file.

        Args:
            path: Path to vocabulary JSON file

        Returns:
            Loaded ABCVocabulary instance
        """
        vocab = cls()
        with open(path) as f:
            data = json.load(f)

        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(v): k for k, v in vocab.char2idx.items()}
        vocab.char_freq = collections.Counter(data.get("char_freq", {}))

        return vocab

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.size

    def __repr__(self) -> str:
        """Return string representation of vocabulary."""
        return (
            f"ABCVocabulary(size={self.size}, "
            f"special_tokens={len(self.SPECIAL_TOKENS)})"
        )
