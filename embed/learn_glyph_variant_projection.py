from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed.glyph_variant_codec import (
    GLYPH_VARIANT_STRATEGY,
    load_glyph_variant_sequences,
    resolve_anchor_token_index,
    resolve_anchor_vocab_key,
)
from embed.learn_projection_to_transformer_space import (
    DEFAULT_ANCHORS_FILE,
    DEFAULT_OUTPUT_DIR,
    ProjectionModel,
    build_vocab,
    context_pairs,
    gloss_embedding,
    load_gloss_anchors,
    load_transformer_embeddings,
    save_outputs,
    train_projection,
)

DEFAULT_TOKENIZED_DIR = ROOT / "rr_tablets" / "transliterated" / "complete" / "tokenized" / GLYPH_VARIANT_STRATEGY


def build_glyph_variant_anchor_targets(
    anchors: dict[str, list[str]],
    token_to_idx: dict[str, int],
    vocab: list[str],
    tokenizer: AutoTokenizer,
    embedding_matrix: np.ndarray,
) -> list[tuple[int, np.ndarray, list[str]]]:
    anchor_targets: list[tuple[int, np.ndarray, list[str]]] = []
    for glyph, glosses in anchors.items():
        token_idx = resolve_anchor_token_index(glyph, token_to_idx)
        if token_idx is None:
            print(f"warning: anchor glyph {glyph!r} not in vocabulary; skipping")
            continue
        vocab_key = resolve_anchor_vocab_key(glyph, vocab)
        anchor_targets.append((token_idx, gloss_embedding(glosses, tokenizer, embedding_matrix), glosses))
    return anchor_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn glyph+variant structured Rongorongo embeddings projected into a frozen transformer space."
    )
    parser.add_argument("--tokenized-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--transformer", default="intfloat/e5-small-v2")
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchors-file", type=Path, default=DEFAULT_ANCHORS_FILE)
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--anchor-weight", type=float, default=8.0)
    parser.add_argument("--projected-cooccurrence-weight", type=float, default=0.5)
    parser.add_argument("--collapse-weight", type=float, default=0.15)
    parser.add_argument("--anchor-passes", type=int, default=12)
    parser.add_argument("--collapse-sample-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = load_glyph_variant_sequences(args.tokenized_dir)
    if not sequences:
        raise ValueError(
            f"No glyph_variants sequences found in {args.tokenized_dir}. "
            "Run: python utils/tokenize_transliterated.py --strategy glyph_variants"
        )
    vocab, token_to_idx = build_vocab(sequences)
    pairs = context_pairs(sequences, token_to_idx, window=args.window, mask_separators=True)
    if not pairs:
        raise ValueError("No context pairs could be generated; increase data or context window.")
    print(f"glyph_variants vocab={len(vocab)} pairs={len(pairs)} (structured tokens flattened to vocab keys)")

    tokenizer, transformer_embeddings = load_transformer_embeddings(args.transformer)

    anchor_targets: list[tuple[int, np.ndarray, list[str]]] = []
    if not args.no_anchors and args.anchors_file.exists():
        anchor_targets = build_glyph_variant_anchor_targets(
            load_gloss_anchors(args.anchors_file),
            token_to_idx,
            vocab,
            tokenizer,
            transformer_embeddings,
        )
        print(f"Loaded {len(anchor_targets)} gloss anchors from {args.anchors_file}")

    model = ProjectionModel.create(
        vocab_size=len(vocab),
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=transformer_embeddings.shape[1],
        seed=args.seed,
    )
    training_config = {
        "mode": "hybrid_glyph_variants",
        "mask_separators": True,
        "english_context_targets": False,
        "anchor_weight": args.anchor_weight,
        "projected_cooccurrence_weight": args.projected_cooccurrence_weight,
        "collapse_weight": args.collapse_weight,
        "anchor_passes": args.anchor_passes,
        "vocab_key_format": "glyph[:variant[+modifiers]]",
    }
    losses = train_projection(
        model=model,
        pairs=pairs,
        anchor_targets=anchor_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negatives=args.negatives,
        learning_rate=args.learning_rate,
        anchor_weight=args.anchor_weight,
        projected_cooccurrence_weight=args.projected_cooccurrence_weight,
        collapse_weight=args.collapse_weight,
        anchor_passes=args.anchor_passes,
        collapse_sample_size=args.collapse_sample_size,
        seed=args.seed,
    )
    anchor_records = [(token_idx, glosses) for token_idx, _, glosses in anchor_targets]
    save_outputs(
        model,
        vocab,
        args.output_dir,
        args.transformer,
        GLYPH_VARIANT_STRATEGY,
        losses,
        anchor_records,
        training_config,
    )
    print(f"Saved glyph_variants projection artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
