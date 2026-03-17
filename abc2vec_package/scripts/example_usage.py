#!/usr/bin/env python
"""
Quick example script demonstrating ABC2Vec usage.

Shows how to:
1. Load a pre-trained model
2. Encode ABC tunes to embeddings
3. Compute similarity between tunes
"""

import argparse

import torch
import torch.nn.functional as F

from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier


def main():
    """Run example encoding and similarity computation."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Example - Encode tunes and compute similarity"
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./data/processed/vocab.json",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (optional, uses random init if not provided)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Example Usage")
    print("=" * 80)

    # Load vocabulary
    print("\n1. Loading vocabulary...")
    vocab = ABCVocabulary.load(args.vocab_path)
    print(f"   Vocabulary size: {vocab.size}")

    # Create patchifier
    print("\n2. Creating patchifier...")
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)
    print(f"   Max bars: {patchifier.max_bars}")

    # Load or create model
    print("\n3. Loading model...")
    if args.checkpoint:
        model = ABC2VecModel.load_pretrained(args.checkpoint)
        print(f"   ✓ Loaded from {args.checkpoint}")
    else:
        from abc2vec.model import ABC2VecConfig

        config = ABC2VecConfig(vocab_size=vocab.size)
        model = ABC2VecModel(config)
        print("   ⚠ Using randomly initialized model (no checkpoint provided)")

    model.eval()

    # Example tunes (simple Irish reels)
    print("\n4. Example tunes:")
    print()

    tunes = {
        "Tune A (D reel)": "D2 EF | G2 AB | c2 dc | BAGF |",
        "Tune B (D reel, similar)": "D2 FE | G2 BA | c2 cd | BGAF |",
        "Tune C (G jig)": "G2 B | d2 d | e2 c | BAG |",
    }

    embeddings = {}

    for name, abc_body in tunes.items():
        print(f"   {name}")
        print(f"   ABC: {abc_body}")

        # Patchify
        patches = patchifier.patchify(abc_body)

        # Add batch dimension
        bar_indices = patches["bar_indices"].unsqueeze(0)
        char_mask = patches["char_mask"].unsqueeze(0)
        bar_mask = patches["bar_mask"].unsqueeze(0)

        # Encode
        with torch.no_grad():
            embedding = model.get_embedding(bar_indices, char_mask, bar_mask)

        embeddings[name] = embedding.squeeze(0)
        print(f"   Embedding shape: {embedding.shape}")
        print()

    # Compute similarities
    print("5. Pairwise similarities:")
    print()

    tune_names = list(embeddings.keys())
    for i, name_i in enumerate(tune_names):
        for j, name_j in enumerate(tune_names):
            if i < j:
                emb_i = embeddings[name_i]
                emb_j = embeddings[name_j]

                similarity = F.cosine_similarity(
                    emb_i.unsqueeze(0), emb_j.unsqueeze(0)
                ).item()

                print(f"   {name_i} <-> {name_j}")
                print(f"   Cosine similarity: {similarity:.4f}")
                print()

    print("=" * 80)
    print("✓ Example complete!")
    print()
    print("Expected behavior:")
    print("  - Tune A and B (both D reels) should have HIGH similarity")
    print("  - Tune C (G jig) should have LOWER similarity to A and B")
    print("=" * 80)


if __name__ == "__main__":
    main()
