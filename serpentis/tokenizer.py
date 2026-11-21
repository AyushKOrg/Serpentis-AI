"""SERPENTIS byte-level BPE tokenizer, trained from scratch on supplied text.

No pretrained vocabulary or external tokenizer dependency. The base vocabulary is
all 256 byte values plus EOS and PAD. Learned merge rules extend that vocabulary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

EOS_ID = 256
PAD_ID = 257
BASE_VOCAB_SIZE = 258


class SerpentisTokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 4096,
              min_pair_frequency: int = 2) -> "SerpentisTokenizer":
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_VOCAB_SIZE}")
        sequences = [list(text.encode("utf-8")) for text in texts if text]
        target_merges = vocab_size - BASE_VOCAB_SIZE
        merges: list[tuple[int, int]] = []
        for _ in range(target_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            # Deterministic tie-breaking: highest count, then lowest IDs.
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_pair_frequency:
                break
            new_id = BASE_VOCAB_SIZE + len(merges)
            merges.append(pair)
            replaced = []
            for seq in sequences:
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                replaced.append(out)
            sequences = replaced
        return cls(merges)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            merged_id = BASE_VOCAB_SIZE + rank
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    out.append(merged_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        token_bytes = {i: bytes([i]) for i in range(256)}
        for i, (left, right) in enumerate(self.merges, start=BASE_VOCAB_SIZE):
            token_bytes[i] = token_bytes[left] + token_bytes[right]
        data = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id in (EOS_ID, PAD_ID):
                if skip_special:
                    continue
                continue
            piece = token_bytes.get(token_id)
            if piece is None:
                raise ValueError(f"Unknown token id: {token_id}")
            data.extend(piece)
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {
            "format": "serpentis-byte-bpe-v1",
            "base_vocab_size": BASE_VOCAB_SIZE,
            "eos_id": EOS_ID,
            "pad_id": PAD_ID,
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SerpentisTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "serpentis-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer file format")
        if payload.get("base_vocab_size") != BASE_VOCAB_SIZE:
            raise ValueError("Tokenizer base vocabulary mismatch")
        merges = [tuple(map(int, pair)) for pair in payload["merges"]]
        tok = cls(merges)
        if tok.vocab_size != int(payload.get("vocab_size", -1)):
            raise ValueError("Tokenizer vocabulary size is inconsistent")
        return tok
