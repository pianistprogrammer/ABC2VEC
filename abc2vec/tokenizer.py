
"""ABC Tokenizer and Bar Patchifier for ABC2Vec."""

import re, json, collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ABCVocabulary:
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.char_freq = collections.Counter()
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.char2idx[tok] = i
            self.idx2char[i] = tok

    def build_from_corpus(self, texts, min_freq=1):
        for text in texts:
            for ch in text:
                self.char_freq[ch] += 1
        idx = len(self.SPECIAL_TOKENS)
        for ch, freq in sorted(self.char_freq.items()):
            if freq >= min_freq and ch not in self.char2idx:
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch
                idx += 1

    def encode(self, text):
        unk = self.char2idx["[UNK]"]
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, indices):
        return "".join(self.idx2char.get(i, "?") for i in indices
                       if self.idx2char.get(i, "") not in self.SPECIAL_TOKENS)

    @property
    def size(self):
        return len(self.char2idx)

    @property
    def pad_idx(self):
        return self.char2idx["[PAD]"]

    @property
    def mask_idx(self):
        return self.char2idx["[MASK]"]

    @property
    def cls_idx(self):
        return self.char2idx["[CLS]"]

    @property
    def sep_idx(self):
        return self.char2idx["[SEP]"]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"char2idx": self.char2idx, "char_freq": dict(self.char_freq)}, f, indent=2)

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(v): k for k, v in vocab.char2idx.items()}
        vocab.char_freq = collections.Counter(data.get("char_freq", {}))
        return vocab


class BarPatchifier:
    def __init__(self, vocab, max_bar_length=64, max_bars=64):
        self.vocab = vocab
        self.max_bar_length = max_bar_length
        self.max_bars = max_bars

    def split_into_bars(self, abc_body):
        body = abc_body.strip()
        bars = []
        current_bar = []
        i = 0
        while i < len(body):
            ch = body[i]
            if ch == "|":
                bar_str = "".join(current_bar).strip()
                if bar_str:
                    bars.append(bar_str)
                current_bar = []
                if i + 1 < len(body) and body[i + 1] in ":|]":
                    i += 2
                    continue
            elif ch == ":" and i + 1 < len(body) and body[i + 1] == "|":
                bar_str = "".join(current_bar).strip()
                if bar_str:
                    bars.append(bar_str)
                current_bar = []
                i += 2
                continue
            else:
                current_bar.append(ch)
            i += 1
        bar_str = "".join(current_bar).strip()
        if bar_str:
            bars.append(bar_str)
        return bars

    def patchify(self, abc_body):
        bars = self.split_into_bars(abc_body)[:self.max_bars]
        num_bars = len(bars)
        bar_indices = torch.full((self.max_bars, self.max_bar_length), self.vocab.pad_idx, dtype=torch.long)
        char_mask = torch.zeros(self.max_bars, self.max_bar_length, dtype=torch.bool)
        bar_mask = torch.zeros(self.max_bars, dtype=torch.bool)
        for i, bar in enumerate(bars):
            encoded = self.vocab.encode(bar)[:self.max_bar_length]
            bar_indices[i, :len(encoded)] = torch.tensor(encoded, dtype=torch.long)
            char_mask[i, :len(encoded)] = True
            bar_mask[i] = True
        return {"bar_indices": bar_indices, "bar_mask": bar_mask, "char_mask": char_mask,
                "num_bars": num_bars, "bars": bars}


class ABCTransposer:
    CHROMATIC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    KEY_TO_SEMITONE = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
    }

    @classmethod
    def transpose_note(cls, note_char, accidental, semitones):
        base = note_char.upper()
        if base not in cls.NOTE_TO_SEMITONE:
            return note_char, accidental
        current = cls.NOTE_TO_SEMITONE[base]
        if accidental == "^":
            current += 1
        elif accidental == "_":
            current -= 1
        new_semitone = (current + semitones) % 12
        sharp_name = cls.CHROMATIC_SHARP[new_semitone]
        if len(sharp_name) == 1:
            new_note, new_acc = sharp_name, ""
        else:
            new_note, new_acc = sharp_name[0], "^"
        if note_char.islower():
            new_note = new_note.lower()
        return new_note, new_acc

    @classmethod
    def transpose_key(cls, key_str, semitones):
        match = re.match(r"^([A-G][#b]?)(.*)$", key_str.strip())
        if not match:
            return key_str
        root, mode = match.group(1), match.group(2)
        if root in cls.KEY_TO_SEMITONE:
            new_semitone = (cls.KEY_TO_SEMITONE[root] + semitones) % 12
            new_root = cls.CHROMATIC_SHARP[new_semitone]
            return new_root + mode
        return key_str

    @classmethod
    def transpose_abc_body(cls, abc_body, semitones):
        if semitones == 0:
            return abc_body
        result = []
        i = 0
        while i < len(abc_body):
            ch = abc_body[i]
            if ch in "^_=" and i + 1 < len(abc_body) and abc_body[i + 1].upper() in "ABCDEFG":
                accidental = ch if ch != "=" else ""
                note = abc_body[i + 1]
                new_note, new_acc = cls.transpose_note(note, accidental, semitones)
                if new_acc:
                    result.append(new_acc)
                result.append(new_note)
                i += 2
            elif ch.upper() in "ABCDEFG":
                new_note, new_acc = cls.transpose_note(ch, "", semitones)
                if new_acc:
                    result.append(new_acc)
                result.append(new_note)
                i += 1
            else:
                result.append(ch)
                i += 1
        return "".join(result)


class PatchEmbedding(nn.Module):
    def __init__(self, vocab_size, d_char=64, d_model=256, max_bar_length=64, max_bars=64, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=pad_idx)
        self.projection = nn.Linear(d_char, d_model)
        self.pos_embed = nn.Embedding(max_bars + 1, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, bar_indices, char_mask, bar_mask):
        char_embeds = self.char_embed(bar_indices)
        mask_expanded = char_mask.unsqueeze(-1).float()
        char_sum = (char_embeds * mask_expanded).sum(dim=2)
        char_count = char_mask.sum(dim=2, keepdim=True).float().clamp(min=1)
        bar_embeds = self.projection(char_sum / char_count)
        max_bars = bar_indices.shape[1]
        positions = torch.arange(max_bars, device=bar_indices.device).unsqueeze(0)
        bar_embeds = bar_embeds + self.pos_embed(positions)
        return self.layer_norm(self.dropout(bar_embeds)), bar_mask


class ABC2VecDataset(Dataset):
    def __init__(self, df, patchifier, augment_transpose=True, body_col="abc_body"):
        self.df = df.reset_index(drop=True)
        self.patchifier = patchifier
        self.augment_transpose = augment_transpose
        self.body_col = body_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        abc_body = self.df.iloc[idx][self.body_col]
        patches = self.patchifier.patchify(abc_body)
        item = {"bar_indices": patches["bar_indices"], "char_mask": patches["char_mask"],
                "bar_mask": patches["bar_mask"], "num_bars": patches["num_bars"]}
        if self.augment_transpose:
            semitones = np.random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])
            transposed = ABCTransposer.transpose_abc_body(abc_body, semitones)
            tp = self.patchifier.patchify(transposed)
            item["trans_bar_indices"] = tp["bar_indices"]
            item["trans_char_mask"] = tp["char_mask"]
            item["trans_bar_mask"] = tp["bar_mask"]
        return item


class SectionPairDataset(Dataset):
    def __init__(self, section_pairs_path, patchifier):
        with open(section_pairs_path) as f:
            self.pairs = json.load(f)
        self.patchifier = patchifier

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        pa = self.patchifier.patchify(pair["section_a"])
        pb = self.patchifier.patchify(pair["section_b"])
        return {"a_bar_indices": pa["bar_indices"], "a_char_mask": pa["char_mask"],
                "a_bar_mask": pa["bar_mask"], "b_bar_indices": pb["bar_indices"],
                "b_char_mask": pb["char_mask"], "b_bar_mask": pb["bar_mask"],
                "tune_id": pair["tune_id"]}
