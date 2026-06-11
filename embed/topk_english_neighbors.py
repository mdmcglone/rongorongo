from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoTokenizer


import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PROJECTION = ROOT / "outputs" / "projection" / "barthel_projection.npz"
DEFAULT_METADATA = ROOT / "outputs" / "projection" / "barthel_projection_metadata.json"

from embed.glyph_variant_codec import GLYPH_VARIANT_STRATEGY, resolve_anchor_vocab_key


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def load_transformer_embeddings(model_name: str) -> tuple[AutoTokenizer, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        weight_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    except Exception:
        weight_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.safetensors")
    weights = load_file(weight_path)
    candidates: list[np.ndarray] = []
    for key, value in weights.items():
        if value.ndim != 2:
            continue
        if key.endswith(
            (
                "embeddings.word_embeddings.weight",
                "word_embeddings.weight",
                "embed_tokens.weight",
                "shared.weight",
                "wte.weight",
            )
        ):
            candidates.append(value)
    if candidates:
        matrix = max(candidates, key=lambda array: array.shape[0])
        return tokenizer, matrix.astype(np.float32)
    raise KeyError("Could not find transformer token embedding matrix in safetensors file.")


def english_vocab_rows(tokenizer: AutoTokenizer, embeddings: np.ndarray, max_words: int) -> tuple[list[str], np.ndarray]:
    words: list[str] = []
    rows: list[int] = []
    for token, idx in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        clean = token.replace("##", "").strip()
        if not re.fullmatch(r"[a-z]+", clean):
            continue
        if len(clean) < 3:
            continue
        words.append(clean)
        rows.append(int(idx))
        if len(words) >= max_words:
            break
    return words, embeddings[np.asarray(rows, dtype=np.int64)]


def topk_english_neighbors(
    rongo_vector: np.ndarray,
    english_words: list[str],
    english_vectors: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    rv = rongo_vector / max(np.linalg.norm(rongo_vector), 1e-8)
    ev = normalize(english_vectors)
    sims = ev @ rv
    order = np.argsort(-sims)[:k]
    return [(english_words[i], float(sims[i])) for i in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show top-k English neighbors to a Rongorongo vector.")
    parser.add_argument("--projection-file", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--token", help="Rongorongo token label (e.g. 004, 120a).")
    parser.add_argument("--index", type=int, help="Alternative to --token: use token index in vocab.")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--english-word-count", type=int, default=5000)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.token is None and args.index is None:
        raise ValueError("Provide --token or --index.")
    metadata = json.loads(args.metadata_file.read_text())
    model_name = str(metadata["transformer_model"])
    projection = np.load(args.projection_file, allow_pickle=False)
    vocab = projection["vocab"].astype(str)
    projected = projection["projected_embeddings"].astype(np.float32)
    projected = normalize(projected)

    if args.index is not None:
        token_idx = int(args.index)
    else:
        token = str(args.token)
        if str(metadata.get("strategy", "")) == GLYPH_VARIANT_STRATEGY:
            token = resolve_anchor_vocab_key(token, vocab) or token
        token_idx = int(np.where(vocab == token)[0][0])
    if token_idx < 0 or token_idx >= len(vocab):
        raise IndexError(f"Token index {token_idx} out of range for vocab size {len(vocab)}")

    tokenizer, transformer_embeddings = load_transformer_embeddings(model_name)
    english_words, english_vectors = english_vocab_rows(tokenizer, transformer_embeddings, args.english_word_count)
    neighbors = topk_english_neighbors(projected[token_idx], english_words, english_vectors.astype(np.float32), args.k)

    payload = {"token": str(vocab[token_idx]), "index": token_idx, "neighbors": neighbors}
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"token={payload['token']} index={payload['index']}")
    for word, score in payload["neighbors"]:
        print(f"  {word:>18s}  {score:.6f}")


if __name__ == "__main__":
    main()
