from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZED_DIR = ROOT / "rr_tablets" / "transliterated" / "complete" / "tokenized" / "barthel"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "projection"
TRANSFORMER_PRESETS = {
    "distilbert": "distilbert-base-uncased",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small": "intfloat/e5-small-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def run_command(command: list[str], cwd: Path) -> None:
    print(">", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rongorongo projection learning and graph generation pipeline.")
    parser.add_argument("--tokenized-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--transformer", default="distilbert-base-uncased")
    parser.add_argument(
        "--transformer-preset",
        choices=tuple(TRANSFORMER_PRESETS),
        help="Preset alias for recommended embedding models.",
    )
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-learn", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument("--topk-token", help="Optional Rongorongo token label for neighbor lookup (e.g. 004).")
    parser.add_argument("--topk-index", type=int, help="Optional token index for neighbor lookup.")
    parser.add_argument("--topk-k", type=int, default=12, help="Top-k neighbors for lookup step.")
    parser.add_argument("--topk-json", action="store_true", help="Print top-k output as JSON.")
    parser.add_argument("--anchors-file", type=Path, default=ROOT / "embed" / "gloss_anchors.json")
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--anchor-weight", type=float, default=2.5)
    parser.add_argument("--projected-cooccurrence-weight", type=float, default=0.5)
    parser.add_argument("--collapse-weight", type=float, default=0.15)
    parser.add_argument("--anchor-passes", type=int, default=8)
    parser.add_argument("--collapse-sample-size", type=int, default=512)
    parser.add_argument(
        "--english-word-count",
        type=int,
        default=5000,
        help="English vocabulary size for neighbor eval and graphs (match tuning default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.transformer_preset:
        args.transformer = TRANSFORMER_PRESETS[args.transformer_preset]
    strategy = args.tokenized_dir.name
    model_slug = sanitize_model_name(args.transformer)
    projection_file = args.output_dir / f"{model_slug}_{strategy}_projection.npz"
    metadata_file = args.output_dir / f"{model_slug}_{strategy}_projection_metadata.json"

    if not args.skip_learn:
        run_command(
            [
                sys.executable,
                str(ROOT / "embed" / "learn_projection_to_transformer_space.py"),
                "--tokenized-dir",
                str(args.tokenized_dir),
                "--transformer",
                args.transformer,
                "--input-dim",
                str(args.input_dim),
                "--hidden-dim",
                str(args.hidden_dim),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--window",
                str(args.window),
                "--negatives",
                str(args.negatives),
                "--learning-rate",
                str(args.learning_rate),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(args.output_dir),
                "--anchors-file",
                str(args.anchors_file),
                "--anchor-weight",
                str(args.anchor_weight),
                "--projected-cooccurrence-weight",
                str(args.projected_cooccurrence_weight),
                "--collapse-weight",
                str(args.collapse_weight),
                "--anchor-passes",
                str(args.anchor_passes),
                "--collapse-sample-size",
                str(args.collapse_sample_size),
            ]
            + (["--no-anchors"] if args.no_anchors else []),
            ROOT,
        )

    if not args.skip_graphs:
        run_command(
            [
                sys.executable,
                str(ROOT / "embed" / "plot_projection_graphs.py"),
                "--projection-file",
                str(projection_file),
                "--metadata-file",
                str(metadata_file),
                "--english-word-count",
                str(args.english_word_count),
            ],
            ROOT,
        )

    if args.topk_token is not None or args.topk_index is not None:
        topk_command = [
            sys.executable,
            str(ROOT / "embed" / "topk_english_neighbors.py"),
            "--projection-file",
            str(projection_file),
            "--metadata-file",
            str(metadata_file),
            "--k",
            str(args.topk_k),
        ]
        if args.topk_token is not None:
            topk_command.extend(["--token", args.topk_token])
        if args.topk_index is not None:
            topk_command.extend(["--index", str(args.topk_index)])
        if args.topk_json:
            topk_command.append("--json")
        run_command(topk_command, ROOT)

    print("Pipeline complete.")
    print(f"Projection: {projection_file}")
    print(f"Metadata: {metadata_file}")
    print(f"Graphs root: {ROOT / 'embed' / 'graphs'}")


if __name__ == "__main__":
    main()
