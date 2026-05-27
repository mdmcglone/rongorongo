from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = ROOT / "rr_tablets" / "compressed" / "Aa" / "1.webp"
DEFAULT_DICTIONARY = ROOT / "references" / "dictionary" / "split"
DEFAULT_SCORES_DIR = ROOT / "outputs" / "scores" / "internvl"
DEFAULT_VARIANT = "8b"
INTERNVL_VARIANTS: dict[str, tuple[str, Path, Path]] = {
    "8b": (
        "mlx-community/InternVL3-8B-MLX-4bit",
        Path("/Users/emdmcglone/Desktop/mareep/internvl3_8b/util.py"),
        Path("/Users/emdmcglone/Desktop/mareep/internvl3_8b/models/mlx-community--InternVL3-8B-MLX-4bit"),
    ),
    "14b": (
        "mlx-community/InternVL3-14B-4bit",
        Path("/Users/emdmcglone/Desktop/mareep/internvl3_14b/util.py"),
        Path("/Users/emdmcglone/Desktop/mareep/internvl3_14b/models/mlx-community--InternVL3-14B-4bit"),
    ),
    "38b": (
        "mlx-community/InternVL3-38B-4bit",
        Path("/Users/emdmcglone/Desktop/mareep/internvl3_38b/util.py"),
        Path("/Users/emdmcglone/Desktop/mareep/internvl3_38b/models/mlx-community--InternVL3-38B-4bit"),
    ),
}
DEFAULT_MAX_TOKENS = 256
SUPPORTED_SUFFIXES = frozenset({".gif", ".jpg", ".jpeg", ".png", ".webp"})
SCORE_PATTERN = re.compile(r"(?<!\d)0\.[0-9]{3}(?!\d)")
DIGIT_PATTERN = re.compile(
    r"(?P<label>\d{3})\s*:\s*overall\s*=\s*(?P<overall>[0-9])\s*,\s*"
    r"structure\s*=\s*(?P<structure>[0-9])\s*,\s*shape\s*=\s*(?P<shape>[0-9])",
    re.IGNORECASE,
)
T = TypeVar("T")


@dataclass(frozen=True)
class DictionaryScore:
    path: str
    label: str
    score: str
    raw: str


@dataclass(frozen=True)
class ScoreRun:
    query: str
    checked: int
    best_score: DictionaryScore | None
    scores: list[DictionaryScore]


class LocalVisionModel:
    def __init__(self, *, model_id: str, local_dir: Path | None) -> None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_ref = str(Path(local_dir)) if local_dir else model_id
        print(f"Loading InternVL model once from {self.model_ref}", file=sys.stderr, flush=True)
        self.model, self.processor = load(self.model_ref)
        self.config = load_config(self.model_ref)
        print("InternVL model loaded; subsequent Prefill lines are per image/prompt, not reloads.", file=sys.stderr, flush=True)

    def generate(self, prompt: str, image_path: Path, *, max_tokens: int, temperature: float, verbose: bool) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        images = [str(Path(image_path))]
        formatted_prompt = apply_chat_template(self.processor, self.config, prompt, num_images=len(images))
        response = generate(
            self.model,
            self.processor,
            formatted_prompt,
            images,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        return str(response).strip()


def numeric_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def dictionary_paths(dictionary_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in dictionary_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=numeric_sort_key,
    )


def chunks(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def progress(items: list[T], *, enabled: bool, description: str) -> list[T]:
    if not enabled or tqdm is None:
        return items
    return tqdm(items, desc=description, unit="batch")


def print_iteration_result(result: DictionaryScore, *, checked: int, total: int, show_progress: bool) -> None:
    message = f"[{checked}/{total}] {result.label}: {result.score}"
    if show_progress and tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    fitted = image.convert("RGBA")
    fitted.thumbnail(max_size, Image.Resampling.LANCZOS)
    return fitted


def draw_centered_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = left + (right - left - text_width) // 2
    y = top + (bottom - top - text_height) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def make_comparison_sheet(query_path: Path, dictionary_batch: list[Path], output_path: Path) -> Path:
    cell_width, cell_height = 150, 120
    label_height = 26
    padding = 16
    query_height = 140
    columns = max(1, len(dictionary_batch))
    width = columns * cell_width + padding * 2
    height = query_height + cell_height + padding * 3
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(16)
    title_font = load_font(18)

    draw_centered_text(draw, "QUERY", (0, padding, width, padding + label_height), title_font)
    query = fit_image(Image.open(query_path), (width - padding * 2, query_height - label_height - 8))
    canvas.paste(query.convert("RGB"), ((width - query.width) // 2, padding + label_height + 4))

    top = padding * 2 + query_height
    for index, path in enumerate(dictionary_batch):
        left = padding + index * cell_width
        right = left + cell_width
        bottom = top + cell_height
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(220, 220, 220))
        draw_centered_text(draw, path.stem, (left, top, right, top + label_height), label_font)
        image = fit_image(Image.open(path), (cell_width - 12, cell_height - label_height - 12))
        x = left + (cell_width - image.width) // 2
        y = top + label_height + (cell_height - label_height - image.height) // 2
        canvas.paste(image.convert("RGB"), (x, y))

    canvas.save(output_path, optimize=True)
    return output_path


def score_guidance() -> str:
    return (
        "Do not output percentages or decimal similarity scores. Choose three separate "
        "integer digits from 0 to 9 for each label. overall is the coarse overall similarity. "
        "structure refines the overall score: lower when counts or arrangements of edges, "
        "corners, strokes, appendages, or junctions are bad, higher when they support the "
        "overall match. shape refines the overall score: lower when roundness, pointiness, "
        "curvature, or contour style along structures is bad, higher when those qualities "
        "support the overall match. Tolerate differences in line width, spacing, handwriting, "
        "stroke looseness, small alignment shifts, progressive simplification or laziness from "
        "repeated writing, and query degradation, expansion, or condensation. Do not require "
        "the same exact outline or shape. Penalize missing major features, added major "
        "features, changed counts of important strokes or appendages, and different character "
        "families."
    )


def batch_score_prompt(labels: list[str]) -> str:
    label_list = ", ".join(labels)
    return (
        "The image contains one query rongorongo glyph at the top and labeled dictionary "
        "entries below it. Each dictionary entry is labeled above its glyph; multiple "
        "alternates inside one entry are separated by whitespace. Score the query against "
        f"each labeled dictionary entry independently. Labels to score: {label_list}. "
        f"{score_guidance()} "
        "Answer with exactly one line per label in this format:\n"
        "<label>: overall=<0-9>, structure=<0-9>, shape=<0-9>"
    )


def parse_batch_scores(raw: str, labels: list[str]) -> dict[str, str]:
    if not raw.strip():
        raise ValueError("Expected one score per label, got an empty response.")

    scores: dict[str, str] = {}
    for line in raw.splitlines():
        digit_match = DIGIT_PATTERN.search(line)
        if digit_match:
            label = digit_match.group("label")
            if label in labels:
                scores[label] = (
                    f"0.{digit_match.group('overall')}"
                    f"{digit_match.group('structure')}"
                    f"{digit_match.group('shape')}"
                )
            continue

        # Backward-compatible fallback if the model still emits an encoded score.
        label_match = re.search(r"(?<!\d)(\d{3})(?!\d)", line)
        score_match = SCORE_PATTERN.search(line)
        if label_match and score_match and label_match.group(1) in labels:
            scores[label_match.group(1)] = score_match.group(0)

    missing = [label for label in labels if label not in scores]
    if missing:
        raise ValueError(f"Expected scores for labels {missing}, got: {raw!r}")
    return scores


def score_dictionary_batch(
    vision_model: LocalVisionModel,
    query_path: Path,
    dictionary_batch: list[Path],
    temp_dir: Path,
    *,
    max_tokens: int,
    temperature: float,
    verbose: bool,
) -> list[DictionaryScore]:
    labels = [path.stem for path in dictionary_batch]
    sheet_path = make_comparison_sheet(query_path, dictionary_batch, temp_dir / f"batch_{labels[0]}_{labels[-1]}.png")
    raw = vision_model.generate(
        batch_score_prompt(labels),
        sheet_path,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
    )
    scores = parse_batch_scores(raw, labels)
    return [
        DictionaryScore(path=str(path), label=path.stem, score=scores[path.stem], raw=raw)
        for path in dictionary_batch
    ]


def score_query_against_dictionary_internvl(
    query_path: Path,
    dictionary_dir: Path,
    *,
    model_id: str,
    local_dir: Path | None,
    max_tokens: int,
    temperature: float,
    batch_size: int,
    show_progress: bool,
    debug: bool,
    verbose: bool,
) -> ScoreRun:
    candidates = dictionary_paths(dictionary_dir)
    if not candidates:
        raise FileNotFoundError(f"No supported dictionary images found in {dictionary_dir}")

    vision_model = LocalVisionModel(model_id=model_id, local_dir=local_dir)
    scores: list[DictionaryScore] = []
    total = len(candidates)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        checked = 0
        for batch in progress(chunks(candidates, batch_size), enabled=show_progress, description=f"Scoring {query_path.stem}"):
            results = score_dictionary_batch(
                vision_model,
                query_path,
                batch,
                temp_dir,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
            )
            for result in results:
                checked += 1
                scores.append(result)
                if debug:
                    print_iteration_result(result, checked=checked, total=total, show_progress=show_progress)

    ranked = sorted(scores, key=lambda result: float(result.score), reverse=True)
    return ScoreRun(
        query=str(query_path),
        checked=len(scores),
        best_score=ranked[0] if ranked else None,
        scores=scores,
    )


def default_scores_output(query_path: Path, variant: str) -> Path:
    return DEFAULT_SCORES_DIR / f"{query_path.parent.name}_{query_path.stem}_{variant}.json"


def write_scores_json(run: ScoreRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(run), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one query glyph against dictionary glyphs using local InternVL3."
    )
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY, help=f"Query glyph. Default: {DEFAULT_QUERY}")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY, help=f"Dictionary split folder. Default: {DEFAULT_DICTIONARY}")
    parser.add_argument("--variant", choices=tuple(INTERNVL_VARIANTS), default=DEFAULT_VARIANT, help=f"InternVL variant to use. Default: {DEFAULT_VARIANT}.")
    parser.add_argument("--internvl-util", type=Path, help="Deprecated; kept for CLI compatibility. Defaults from --variant.")
    parser.add_argument("--model-id", help="InternVL model id. Defaults from --variant.")
    parser.add_argument("--local-dir", type=Path, help="Local model directory. Defaults from --variant.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Maximum tokens to generate. Default: {DEFAULT_MAX_TOKENS}.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model sampling temperature.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of dictionary glyphs to score per local model call. Default: 1.")
    parser.add_argument("--scores-output", type=Path, help=f"Path to write all scores as JSON. Default: {DEFAULT_SCORES_DIR}/<query>_<variant>.json")
    parser.add_argument("--no-progress", action="store_true", help="Disable the per-batch progress bar.")
    parser.add_argument("--debug", action="store_true", help="Print each score while running.")
    parser.add_argument("--verbose", action="store_true", help="Pass verbose=True to mlx_vlm.generate.")
    args = parser.parse_args()
    variant_model_id, variant_util, variant_local_dir = INTERNVL_VARIANTS[args.variant]
    args.model_id = args.model_id or variant_model_id
    args.internvl_util = args.internvl_util or variant_util
    args.local_dir = args.local_dir or variant_local_dir
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_tokens < 16:
        parser.error("--max-tokens must be >= 16")
    return args


def main() -> None:
    args = parse_args()
    run = score_query_against_dictionary_internvl(
        query_path=args.query,
        dictionary_dir=args.dictionary,
        model_id=args.model_id,
        local_dir=args.local_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        debug=args.debug,
        verbose=args.verbose,
    )
    scores_output = args.scores_output or default_scores_output(args.query, args.variant)
    write_scores_json(run, scores_output)
    output = asdict(run)
    del output["scores"]
    output["scores_output"] = str(scores_output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
