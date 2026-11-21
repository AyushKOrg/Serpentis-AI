"""Train SERPENTIS BPE tokenizer from plain-text files."""
from __future__ import annotations
import argparse
from pathlib import Path
from serpentis.tokenizer import SerpentisTokenizer


def read_text_files(paths: list[str]) -> list[str]:
    texts = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files = sorted(path.rglob("*.txt"))
        else:
            files = [path]
        for file in files:
            texts.append(file.read_text(encoding="utf-8", errors="replace"))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SERPENTIS byte-BPE tokenizer from scratch.")
    parser.add_argument("corpus", nargs="+", help="Text file(s) or directories recursively containing .txt files")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    args = parser.parse_args()
    corpus = read_text_files(args.corpus)
    if not corpus:
        parser.error("No text was loaded. Provide text files or a directory containing .txt files.")
    tokenizer = SerpentisTokenizer.train(corpus, args.vocab_size, args.min_frequency)
    tokenizer.save(args.output)
    print(f"Tokenizer trained from scratch.")
    print(f"Documents: {len(corpus)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Merge rules learned: {len(tokenizer.merges)}")
    print(f"Saved: {args.output}")
    sample = corpus[0][:300]
    ids = tokenizer.encode(sample)
    print(f"Round-trip sample: {tokenizer.decode(ids)[:200]!r}")


if __name__ == "__main__":
    main()
