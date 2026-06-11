from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
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
DEFAULT_GRAPHS_ROOT = ROOT / "embed" / "graphs"
GLOSS_ANCHORS_FILE = Path(__file__).resolve().parent / "gloss_anchors.json"

from embed.glyph_variant_codec import GLYPH_VARIANT_STRATEGY, resolve_anchor_vocab_key


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def sanitize_name(value: str) -> str:
    return value.replace("/", "_")


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


def text_embedding(text: str, tokenizer: AutoTokenizer, embedding_matrix: np.ndarray) -> np.ndarray:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return np.zeros(embedding_matrix.shape[1], dtype=np.float32)
    return embedding_matrix[np.asarray(token_ids, dtype=np.int64)].mean(axis=0)


def load_gloss_english_words() -> list[str]:
    if not GLOSS_ANCHORS_FILE.exists():
        return []
    data: dict[str, list[str]] = json.loads(GLOSS_ANCHORS_FILE.read_text())
    return sorted({str(word).lower() for glosses in data.values() for word in glosses})


def english_vocab_rows(
    tokenizer: AutoTokenizer,
    embeddings: np.ndarray,
    max_words: int,
    extra_words: Iterable[str] | None = None,
    include_gloss_words: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Build English neighbor pool from early vocab ids plus explicit gloss/extra words.

    Early tokenizer ids omit many rare but semantically important words (e.g. penis).
    Gloss words are always appended via mean subtoken embedding so anchors can appear as top-1.
    """
    words: list[str] = []
    rows: list[int] = []
    seen: set[str] = set()
    for token, idx in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        clean = token.replace("##", "").strip().lower()
        if clean in seen:
            continue
        if not re.fullmatch(r"[a-z]+", clean):
            continue
        if len(clean) < 3:
            continue
        seen.add(clean)
        words.append(clean)
        rows.append(int(idx))
        if len(words) >= max_words:
            break
    matrix = embeddings[np.asarray(rows, dtype=np.int64)].astype(np.float32)
    extras: set[str] = set()
    if include_gloss_words:
        extras.update(load_gloss_english_words())
    if extra_words:
        extras.update(str(word).lower() for word in extra_words)
    extra_vectors: list[np.ndarray] = []
    for word in sorted(extras):
        if word in seen:
            continue
        if not re.fullmatch(r"[a-z]+", word):
            continue
        seen.add(word)
        words.append(word)
        extra_vectors.append(text_embedding(word, tokenizer, embeddings).astype(np.float32))
    if extra_vectors:
        matrix = np.vstack([matrix, np.asarray(extra_vectors, dtype=np.float32)])
    return words, matrix


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


def evaluate_neighbor_quality(neighbors: dict[str, list[tuple[str, float]]]) -> dict[str, float | int]:
    top_words = [items[0][0] for items in neighbors.values() if items]
    top_scores = [items[0][1] for items in neighbors.values() if items]
    margins = [items[0][1] - items[1][1] for items in neighbors.values() if len(items) > 1]
    counts = Counter(top_words)
    probs = np.asarray([count / len(top_words) for count in counts.values()], dtype=np.float64) if top_words else np.asarray([])
    entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0)))) if probs.size else 0.0
    return {
        "token_count": len(neighbors),
        "unique_top1_count": len(counts),
        "top1_entropy_bits": entropy,
        "top1_mean_cosine": float(np.mean(top_scores)) if top_scores else 0.0,
        "top1_std_cosine": float(np.std(top_scores)) if top_scores else 0.0,
        "top1_min_cosine": float(np.min(top_scores)) if top_scores else 0.0,
        "top1_max_cosine": float(np.max(top_scores)) if top_scores else 0.0,
        "top1_top2_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "top1_top2_margin_std": float(np.std(margins)) if margins else 0.0,
    }


def plot_heatmap(similarity: np.ndarray, labels: np.ndarray, path: Path, max_labels: int) -> None:
    count = min(max_labels, similarity.shape[0])
    plt.figure(figsize=(9, 7))
    plt.imshow(similarity[:count, :count], cmap="viridis", aspect="auto", interpolation="nearest")
    ticks = np.arange(count)
    short_labels = [str(label) for label in labels[:count]]
    plt.xticks(ticks=ticks, labels=short_labels, rotation=90, fontsize=6)
    plt.yticks(ticks=ticks, labels=short_labels, fontsize=6)
    plt.colorbar(label="Cosine similarity")
    plt.title("Rongorongo token-to-token cosine similarity")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_joint_scatter(rongo_2d: np.ndarray, english_2d: np.ndarray, labels: np.ndarray, path: Path, annotate_n: int) -> None:
    plt.figure(figsize=(10, 8))
    plt.scatter(english_2d[:, 0], english_2d[:, 1], s=8, alpha=0.25, label="English tokens")
    plt.scatter(rongo_2d[:, 0], rongo_2d[:, 1], s=20, alpha=0.9, label="Rongorongo tokens")
    for i in range(min(annotate_n, len(labels))):
        plt.annotate(str(labels[i]), (rongo_2d[i, 0], rongo_2d[i, 1]), fontsize=7, alpha=0.9)
    plt.legend()
    plt.title("Projected Rongorongo vectors vs English vectors (PCA)")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Rongorongo embedding graphs and English comparisons.")
    parser.add_argument("--projection-file", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--graphs-root", type=Path, default=DEFAULT_GRAPHS_ROOT)
    parser.add_argument("--english-word-count", type=int, default=5000)
    parser.add_argument("--heatmap-max-labels", type=int, default=120)
    parser.add_argument("--annotate-rongo-count", type=int, default=40)
    parser.add_argument("--neighbor-k", type=int, default=12)
    return parser.parse_args()


def render_projection_graphs(
    projection_file: Path,
    metadata_file: Path,
    graph_dir: Path,
    *,
    english_word_count: int = 5000,
    heatmap_max_labels: int = 120,
    annotate_rongo_count: int = 40,
    neighbor_k: int = 12,
) -> dict[str, Any]:
    metadata: dict[str, Any] = json.loads(metadata_file.read_text())
    model_name = str(metadata["transformer_model"])
    graph_dir.mkdir(parents=True, exist_ok=True)

    projection = np.load(projection_file, allow_pickle=False)
    vocab = projection["vocab"].astype(str)
    projected = projection["projected_embeddings"].astype(np.float32)
    projected = normalize(projected)

    tokenizer, transformer_embeddings = load_transformer_embeddings(model_name)
    gloss_words = [str(gloss) for entry in metadata.get("gloss_anchors", []) for gloss in entry.get("glosses", [])]
    english_words, english_vectors = english_vocab_rows(
        tokenizer,
        transformer_embeddings,
        english_word_count,
        extra_words=gloss_words,
    )
    english_vectors = normalize(english_vectors.astype(np.float32))

    similarity = projected @ projected.T
    plot_heatmap(similarity, vocab, graph_dir / "rongo_similarity_heatmap.png", heatmap_max_labels)

    joint = np.vstack([projected, english_vectors])
    joint_2d = pca_2d(joint)
    rongo_2d = joint_2d[: projected.shape[0]]
    english_2d = joint_2d[projected.shape[0] :]
    plot_joint_scatter(rongo_2d, english_2d, vocab, graph_dir / "rongo_vs_english_pca.png", annotate_rongo_count)

    neighbors = {
        str(token): topk_english_neighbors(projected[idx], english_words, english_vectors, neighbor_k)
        for idx, token in enumerate(vocab)
    }
    (graph_dir / "rongo_topk_english_neighbors.json").write_text(json.dumps(neighbors, indent=2) + "\n")
    metrics = evaluate_neighbor_quality(neighbors)
    anchor_checks = metadata.get("gloss_anchors", [])
    anchor_top1: dict[str, list] = {}
    anchor_gloss_cosine: dict[str, float] = {}
    strategy = str(metadata.get("strategy", ""))
    for entry in anchor_checks:
        anchor_glyph = str(entry["token"])
        vocab_token = (
            resolve_anchor_vocab_key(anchor_glyph, vocab)
            if strategy == GLYPH_VARIANT_STRATEGY
            else anchor_glyph
        )
        if not vocab_token or vocab_token not in neighbors:
            continue
        anchor_top1[anchor_glyph] = neighbors[vocab_token][:3]
        gloss_words = [str(gloss) for gloss in entry.get("glosses", [])]
        if gloss_words:
            gloss_vec = normalize(
                np.asarray(
                    [text_embedding(gloss, tokenizer, transformer_embeddings) for gloss in gloss_words],
                    dtype=np.float32,
                ).mean(axis=0, keepdims=True)
            )[0]
            token_idx = int(np.where(vocab == vocab_token)[0][0])
            anchor_gloss_cosine[anchor_glyph] = float(np.dot(projected[token_idx], gloss_vec))
    metrics["anchor_top1_hits"] = anchor_top1
    metrics["anchor_gloss_cosine"] = anchor_gloss_cosine
    (graph_dir / "rongo_eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    args = parse_args()
    metadata: dict[str, Any] = json.loads(args.metadata_file.read_text())
    model_name = str(metadata["transformer_model"])
    tokenizer_name = str(metadata["strategy"])
    graph_dir = args.graphs_root / sanitize_name(model_name) / tokenizer_name
    metrics = render_projection_graphs(
        args.projection_file,
        args.metadata_file,
        graph_dir,
        english_word_count=args.english_word_count,
        heatmap_max_labels=args.heatmap_max_labels,
        annotate_rongo_count=args.annotate_rongo_count,
        neighbor_k=args.neighbor_k,
    )
    print(f"Saved graphs and neighbors to {graph_dir}")
    _ = metrics


if __name__ == "__main__":
    main()
