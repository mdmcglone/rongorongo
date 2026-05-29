from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZED_DIR = ROOT / "rr_tablets" / "transliterated" / "complete" / "tokenized" / "barthel"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "projection"
DEFAULT_ANCHORS_FILE = Path(__file__).resolve().parent / "gloss_anchors.json"
SEPARATORS = frozenset({"-", ".", ":"})
TEMPERATURE = 0.07


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.clip(norm, 1e-8, None)


def relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, 0.0)


def encode_token_for_transformer(token: str) -> str:
    return f"<R{token}>"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_sequences(tokenized_dir: Path) -> list[list[str]]:
    sequences: list[list[str]] = []
    for path in sorted(tokenized_dir.glob("*.json")):
        lines: dict[str, Sequence[str]] = json.loads(path.read_text())
        for tokens in lines.values():
            if isinstance(tokens, list) and tokens:
                sequences.append([str(token) for token in tokens if str(token).strip()])
    return sequences


def build_vocab(sequences: Sequence[Sequence[str]]) -> tuple[list[str], dict[str, int]]:
    vocab = sorted({token for sequence in sequences for token in sequence})
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    return vocab, token_to_idx


def is_separator_token(token: str) -> bool:
    return token in SEPARATORS


def context_pairs(
    sequences: Sequence[Sequence[str]],
    token_to_idx: dict[str, int],
    window: int,
    mask_separators: bool = True,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    for sequence in sequences:
        seq_ids = [token_to_idx[token] for token in sequence]
        for i, center in enumerate(seq_ids):
            left = max(0, i - window)
            right = min(len(seq_ids), i + window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context = seq_ids[j]
                if mask_separators and (
                    is_separator_token(idx_to_token[center])
                    or is_separator_token(idx_to_token[context])
                ):
                    continue
                pairs.append((center, context))
    return pairs


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


def gloss_embedding(glosses: Sequence[str], tokenizer: AutoTokenizer, embedding_matrix: np.ndarray) -> np.ndarray:
    vectors = [text_embedding(gloss, tokenizer, embedding_matrix) for gloss in glosses]
    return normalize(np.asarray(vectors, dtype=np.float32).mean(axis=0, keepdims=True))[0]


def resolve_token_index(token: str, token_to_idx: dict[str, int]) -> int | None:
    if token in token_to_idx:
        return token_to_idx[token]
    padded = token.zfill(3)
    if padded in token_to_idx:
        return token_to_idx[padded]
    return None


def load_gloss_anchors(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    return {str(token): [str(gloss) for gloss in glosses] for token, glosses in payload.items()}


def build_anchor_targets(
    anchors: dict[str, list[str]],
    token_to_idx: dict[str, int],
    tokenizer: AutoTokenizer,
    embedding_matrix: np.ndarray,
) -> list[tuple[int, np.ndarray, list[str]]]:
    anchor_targets: list[tuple[int, np.ndarray, list[str]]] = []
    for token, glosses in anchors.items():
        token_idx = resolve_token_index(token, token_to_idx)
        if token_idx is None:
            print(f"warning: anchor token {token!r} not in vocabulary; skipping")
            continue
        anchor_targets.append((token_idx, gloss_embedding(glosses, tokenizer, embedding_matrix), glosses))
    return anchor_targets


@dataclass
class ProjectionModel:
    token_embeddings: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def create(cls, vocab_size: int, input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> "ProjectionModel":
        rng = np.random.default_rng(seed)
        token_embeddings = (rng.normal(0.0, 0.02, (vocab_size, input_dim))).astype(np.float32)
        w1 = (rng.normal(0.0, 0.02, (input_dim, hidden_dim))).astype(np.float32)
        b1 = np.zeros((hidden_dim,), dtype=np.float32)
        w2 = (rng.normal(0.0, 0.02, (hidden_dim, output_dim))).astype(np.float32)
        b2 = np.zeros((output_dim,), dtype=np.float32)
        return cls(token_embeddings=token_embeddings, w1=w1, b1=b1, w2=w2, b2=b2)

    def project(self, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.token_embeddings[token_ids]
        h_pre = x @ self.w1 + self.b1
        h = relu(h_pre)
        y = h @ self.w2 + self.b2
        return x, h, y


def contrastive_loss_and_grad(
    vectors: np.ndarray,
    pos_targets: np.ndarray,
    neg_targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    pos_logits = np.sum(vectors * pos_targets, axis=1, keepdims=True)
    neg_logits = np.sum(vectors[:, None, :] * neg_targets, axis=2)
    logits = np.concatenate([pos_logits, neg_logits], axis=1) / TEMPERATURE
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    loss = float(-np.log(np.clip(probs[:, 0], 1e-8, 1.0)).mean())
    grad_logits = probs.copy()
    grad_logits[:, 0] -= 1.0
    grad_logits /= len(vectors)
    grad_logits /= TEMPERATURE
    grad_vectors = grad_logits[:, :1] * pos_targets + np.sum(grad_logits[:, 1:, None] * neg_targets, axis=1)
    return loss, grad_vectors


def spherical_grad(vectors: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return grad_vectors / np.clip(norms, 1e-8, None) - vectors * np.sum(vectors * grad_vectors, axis=1, keepdims=True)


def apply_embedding_gradients(
    model: ProjectionModel,
    token_ids: np.ndarray,
    vectors: np.ndarray,
    grad_vectors: np.ndarray,
    learning_rate: float,
) -> None:
    grad = spherical_grad(vectors, grad_vectors)
    np.add.at(model.token_embeddings, token_ids, -learning_rate * grad.astype(np.float32))


def apply_projection_gradients(
    model: ProjectionModel,
    token_ids: np.ndarray,
    hidden: np.ndarray,
    projected: np.ndarray,
    grad_projected: np.ndarray,
    learning_rate: float,
) -> None:
    grad_y = spherical_grad(projected, grad_projected)
    grad_w2 = hidden.T @ grad_y
    grad_b2 = grad_y.sum(axis=0)
    grad_hidden = grad_y @ model.w2.T
    grad_hidden[hidden <= 0.0] = 0.0
    center_vectors = model.token_embeddings[token_ids]
    grad_w1 = center_vectors.T @ grad_hidden
    grad_b1 = grad_hidden.sum(axis=0)
    grad_centers = grad_hidden @ model.w1.T
    model.w2 -= learning_rate * grad_w2.astype(np.float32)
    model.b2 -= learning_rate * grad_b2.astype(np.float32)
    model.w1 -= learning_rate * grad_w1.astype(np.float32)
    model.b1 -= learning_rate * grad_b1.astype(np.float32)
    np.add.at(model.token_embeddings, token_ids, -learning_rate * grad_centers.astype(np.float32))


def train_rr_context_batch(
    model: ProjectionModel,
    centers: np.ndarray,
    contexts: np.ndarray,
    learning_rate: float,
    negatives: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> float:
    center_vectors = normalize(model.token_embeddings[centers])
    pos_targets = normalize(model.token_embeddings[contexts])
    neg_contexts = rng.integers(0, vocab_size, size=(len(centers), negatives), endpoint=False)
    neg_targets = normalize(model.token_embeddings[neg_contexts])
    loss, grad_vectors = contrastive_loss_and_grad(center_vectors, pos_targets, neg_targets)
    apply_embedding_gradients(model, centers, center_vectors, grad_vectors, learning_rate)
    return loss


def train_projected_cooccurrence_batch(
    model: ProjectionModel,
    centers: np.ndarray,
    contexts: np.ndarray,
    learning_rate: float,
    negatives: int,
    vocab_size: int,
    cooccurrence_weight: float,
    rng: np.random.Generator,
) -> float:
    _, hidden, projected = model.project(centers)
    projected = normalize(projected)
    pos_targets = normalize(model.project(contexts)[2])
    neg_contexts = rng.integers(0, vocab_size, size=(len(centers), negatives), endpoint=False)
    neg_targets = normalize(model.project(neg_contexts)[2])
    loss, grad_projected = contrastive_loss_and_grad(projected, pos_targets, neg_targets)
    grad_projected *= cooccurrence_weight
    apply_projection_gradients(model, centers, hidden, projected, grad_projected, learning_rate)
    return cooccurrence_weight * loss


def train_anchor_step(
    model: ProjectionModel,
    anchor_targets: Sequence[tuple[int, np.ndarray, list[str]]],
    learning_rate: float,
    anchor_weight: float,
) -> float:
    if not anchor_targets:
        return 0.0
    token_ids = np.asarray([item[0] for item in anchor_targets], dtype=np.int64)
    targets = np.asarray([item[1] for item in anchor_targets], dtype=np.float32)
    _, hidden, projected = model.project(token_ids)
    projected = normalize(projected)
    cosine = np.sum(projected * targets, axis=1)
    loss = anchor_weight * float(np.mean(1.0 - cosine))
    grad_projected = -anchor_weight * (targets - projected * cosine[:, None]) / len(token_ids)
    apply_projection_gradients(model, token_ids, hidden, projected, grad_projected, learning_rate)
    return loss


def train_anti_collapse_step(
    model: ProjectionModel,
    learning_rate: float,
    collapse_weight: float,
    sample_size: int,
    rng: np.random.Generator,
) -> float:
    vocab_size = model.token_embeddings.shape[0]
    if sample_size <= 0 or sample_size >= vocab_size:
        token_ids = np.arange(vocab_size, dtype=np.int64)
    else:
        token_ids = rng.choice(vocab_size, size=sample_size, replace=False).astype(np.int64)
    _, hidden, projected = model.project(token_ids)
    projected = normalize(projected)
    std = projected.std(axis=0)
    variance_loss = float(np.mean(np.maximum(0.0, 1.0 - std) ** 2))
    gram = projected @ projected.T
    off_diag = gram[~np.eye(len(token_ids), dtype=bool)]
    uniformity_loss = float(np.mean(off_diag**2))
    loss = collapse_weight * (variance_loss + uniformity_loss)
    grad_projected = np.zeros_like(projected)
    grad_projected -= collapse_weight * 2.0 * (projected - projected.mean(axis=0, keepdims=True)) / len(token_ids)
    grad_projected += collapse_weight * 2.0 * (gram @ projected) / len(token_ids)
    apply_projection_gradients(model, token_ids, hidden, projected, grad_projected, learning_rate)
    return loss


def train_projection(
    model: ProjectionModel,
    pairs: Sequence[tuple[int, int]],
    anchor_targets: Sequence[tuple[int, np.ndarray, list[str]]],
    epochs: int,
    batch_size: int,
    negatives: int,
    learning_rate: float,
    anchor_weight: float,
    projected_cooccurrence_weight: float,
    collapse_weight: float,
    anchor_passes: int,
    collapse_sample_size: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    vocab_size = model.token_embeddings.shape[0]
    batch_size = min(batch_size, len(pairs))
    all_indices = np.arange(len(pairs))

    for epoch in range(epochs):
        rng.shuffle(all_indices)
        epoch_rr_losses: list[float] = []
        epoch_projected_losses: list[float] = []

        for start in range(0, len(all_indices), batch_size):
            batch_idx = all_indices[start:start + batch_size]
            centers = np.asarray([pairs[i][0] for i in batch_idx], dtype=np.int64)
            contexts = np.asarray([pairs[i][1] for i in batch_idx], dtype=np.int64)
            rr_loss = train_rr_context_batch(
                model=model,
                centers=centers,
                contexts=contexts,
                learning_rate=learning_rate,
                negatives=negatives,
                vocab_size=vocab_size,
                rng=rng,
            )
            projected_loss = train_projected_cooccurrence_batch(
                model=model,
                centers=centers,
                contexts=contexts,
                learning_rate=learning_rate,
                negatives=negatives,
                vocab_size=vocab_size,
                cooccurrence_weight=projected_cooccurrence_weight,
                rng=rng,
            )
            epoch_rr_losses.append(rr_loss)
            epoch_projected_losses.append(projected_loss)

        anchor_losses = [
            train_anchor_step(model, anchor_targets, learning_rate, anchor_weight)
            for _ in range(anchor_passes)
        ]
        collapse_loss = train_anti_collapse_step(
            model=model,
            learning_rate=learning_rate,
            collapse_weight=collapse_weight,
            sample_size=collapse_sample_size,
            rng=rng,
        )
        mean_rr = float(np.mean(epoch_rr_losses)) if epoch_rr_losses else 0.0
        mean_projected = float(np.mean(epoch_projected_losses)) if epoch_projected_losses else 0.0
        anchor_loss = float(np.mean(anchor_losses)) if anchor_losses else 0.0
        losses.append(mean_rr + mean_projected)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"rr_context_loss={mean_rr:.6f} projected_cooc_loss={mean_projected:.6f} "
            f"anchor_loss={anchor_loss:.6f} collapse_loss={collapse_loss:.6f}"
        )

    return losses


def save_outputs(
    model: ProjectionModel,
    vocab: Sequence[str],
    output_dir: Path,
    model_name: str,
    strategy_name: str,
    losses: Sequence[float],
    anchor_records: Sequence[tuple[int, list[str]]],
    training_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    projected = normalize(relu(model.token_embeddings @ model.w1 + model.b1) @ model.w2 + model.b2)
    wrapped_vocab = [encode_token_for_transformer(token) for token in vocab]
    model_slug = sanitize_model_name(model_name)
    np.savez_compressed(
        output_dir / f"{model_slug}_{strategy_name}_projection.npz",
        vocab=np.asarray(vocab),
        wrapped_vocab=np.asarray(wrapped_vocab),
        token_embeddings=model.token_embeddings,
        projected_embeddings=projected,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
    )
    metadata = {
        "transformer_model": model_name,
        "strategy": strategy_name,
        "vocab_size": len(vocab),
        "losses": list(losses),
        "gloss_anchors": [
            {"token": vocab[token_idx], "glosses": glosses}
            for token_idx, glosses in anchor_records
        ],
        "training_config": training_config,
    }
    (output_dir / f"{model_slug}_{strategy_name}_projection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn Rongorongo embeddings projected into a frozen transformer space.")
    parser.add_argument("--tokenized-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--transformer", default="distilbert-base-uncased")
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchors-file", type=Path, default=DEFAULT_ANCHORS_FILE)
    parser.add_argument("--no-anchors", action="store_true", help="Disable gloss-anchor supervision.")
    parser.add_argument("--anchor-weight", type=float, default=2.5, help="Weight for gloss-anchor cosine loss.")
    parser.add_argument(
        "--projected-cooccurrence-weight",
        type=float,
        default=0.5,
        help="Weight for projected co-occurrence contrastive loss.",
    )
    parser.add_argument("--collapse-weight", type=float, default=0.15, help="Weight for anti-collapse regularization.")
    parser.add_argument("--anchor-passes", type=int, default=8, help="Anchor optimization passes per epoch.")
    parser.add_argument(
        "--collapse-sample-size",
        type=int,
        default=512,
        help="Projected tokens sampled for anti-collapse step (0 means full vocab).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = load_sequences(args.tokenized_dir)
    if not sequences:
        raise ValueError(f"No token sequences found in {args.tokenized_dir}")
    vocab, token_to_idx = build_vocab(sequences)
    pairs = context_pairs(sequences, token_to_idx, window=args.window, mask_separators=True)
    if not pairs:
        raise ValueError("No context pairs could be generated; increase data or context window.")
    print(f"Using {len(pairs)} RR context pairs (separator-masked, no English context targets)")

    tokenizer, transformer_embeddings = load_transformer_embeddings(args.transformer)

    anchor_targets: list[tuple[int, np.ndarray, list[str]]] = []
    if not args.no_anchors and args.anchors_file.exists():
        anchor_targets = build_anchor_targets(
            load_gloss_anchors(args.anchors_file),
            token_to_idx,
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
        "mode": "hybrid",
        "mask_separators": True,
        "english_context_targets": False,
        "anchor_weight": args.anchor_weight,
        "projected_cooccurrence_weight": args.projected_cooccurrence_weight,
        "collapse_weight": args.collapse_weight,
        "anchor_passes": args.anchor_passes,
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
    strategy_name = args.tokenized_dir.name
    anchor_records = [(token_idx, glosses) for token_idx, _, glosses in anchor_targets]
    save_outputs(
        model,
        vocab,
        args.output_dir,
        args.transformer,
        strategy_name,
        losses,
        anchor_records,
        training_config,
    )
    print(f"Saved projection artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
