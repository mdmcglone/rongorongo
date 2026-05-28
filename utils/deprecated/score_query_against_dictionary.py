from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = ROOT / "rr_tablets" / "compressed" / "Aa" / "1.webp"
DEFAULT_DICTIONARY = ROOT / "references" / "dictionary" / "split"
DEFAULT_ENV = ROOT / ".env"
DEFAULT_SCORES_DIR = ROOT / "outputs" / "scores" / "openai"
DEFAULT_MODEL = "o3"
MIN_OUTPUT_TOKENS = 16
O_SERIES_OUTPUT_TOKENS = 512
SUPPORTED_SUFFIXES = frozenset({".gif", ".jpg", ".jpeg", ".png", ".webp"})
SCORE_PATTERN = re.compile(r"(?<!\d)0\.[0-9]{3}(?!\d)")
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


def data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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


def progress(items: list[T], *, enabled: bool, description: str) -> list[T]:
    if not enabled or tqdm is None:
        return items
    return tqdm(items, desc=description, unit="glyph")


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


def make_batch_sheet(dictionary_paths: list[Path], output_path: Path) -> Path:
    cell_width, cell_height = 150, 120
    label_height = 26
    padding = 16
    columns = len(dictionary_paths)
    width = columns * cell_width + padding * 2
    height = cell_height + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(16)

    for index, path in enumerate(dictionary_paths):
        left = padding + index * cell_width
        top = padding
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
        "Return one encoded score with exactly three digits after the decimal, formatted as "
        "0.xyz. The tenths digit x is the coarse overall similarity score from 0 to 9. The "
        "hundredths digit y is a refinement relative to that overall score: make y lower "
        "when counts or arrangements of edges, corners, strokes, appendages, or junctions "
        "are bad, and higher when they support the overall match. The thousandths digit z "
        "is a further refinement relative to the overall score: make z lower when roundness, "
        "pointiness, curvature, or contour style along the structures is bad, and higher "
        "when those shape qualities support the overall match. Tolerate differences in line "
        "width, spacing, handwriting, stroke looseness, small alignment shifts, progressive "
        "simplification or laziness from repeated writing, and query degradation, expansion, "
        "or condensation. Do not require the same exact outline or shape. Penalize missing "
        "major features, added major features, changed counts of important strokes or "
        "appendages, and different character families. "
    )


def score_prompt(label: str) -> str:
    return (
        "Compare the first image, the query rongorongo glyph, with the second image, "
        f"dictionary entry {label}. The dictionary entry may contain one or more glyphs. "
        "When there are multiple alternates in the dictionary image, they are separated by "
        "whitespace; compare the query to each alternate independently. "
        "Score the query against the best complete glyph inside the dictionary entry. "
        f"{score_guidance()}"
        "Answer in exactly one line: score: 0.xyz"
    )


def batch_score_prompt(labels: list[str]) -> str:
    label_list = ", ".join(labels)
    return (
        "Compare the first image, the query rongorongo glyph, with the second image, a "
        "labeled row of dictionary entries. Each dictionary entry is labeled above its glyph; "
        "multiple alternates inside one entry are separated by whitespace. Score the query "
        f"against each labeled dictionary entry independently. Labels to score: {label_list}. "
        f"{score_guidance()}"
        "Answer with exactly one line per label in this format:\n"
        "<label>: 0.xyz"
    )


def parse_score(raw: str) -> str:
    if not raw.strip():
        raise ValueError(
            "Expected score formatted as 0.xyz, got an empty response. "
            "For o-series models, increase --max-output-tokens."
        )
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "score":
            match = SCORE_PATTERN.search(value)
            if match:
                return match.group(0)
    match = SCORE_PATTERN.search(raw)
    if not match:
        raise ValueError(f"Expected score formatted as 0.xyz, got: {raw!r}")
    return match.group(0)


def parse_batch_scores(raw: str, labels: list[str]) -> dict[str, str]:
    if not raw.strip():
        raise ValueError(
            "Expected one score per label, got an empty response. "
            "For o-series models, increase --max-output-tokens."
        )

    scores: dict[str, str] = {}
    for line in raw.splitlines():
        label_match = re.search(r"(?<!\d)(\d{3})(?!\d)", line)
        score_match = SCORE_PATTERN.search(line)
        if label_match and score_match:
            label = label_match.group(1)
            if label in labels:
                scores[label] = score_match.group(0)

    missing = [label for label in labels if label not in scores]
    if missing:
        raise ValueError(f"Expected scores for labels {missing}, got: {raw!r}")
    return scores


def env_file_value(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    return None


def openai_client_from_env() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or env_file_value(DEFAULT_ENV, "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY is not set in the environment or {DEFAULT_ENV}.")
    return OpenAI(api_key=api_key)


def supports_temperature(model: str) -> bool:
    return not model.lower().startswith("o")


def default_max_output_tokens(model: str, batch_size: int) -> int:
    if model.lower().startswith("o"):
        return max(O_SERIES_OUTPUT_TOKENS, batch_size * 64)
    return max(MIN_OUTPUT_TOKENS, batch_size * 12)


def call_score_model(
    client: OpenAI,
    query_path: Path,
    dictionary_path: Path,
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    request: dict[str, object] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": score_prompt(dictionary_path.stem)},
                    {"type": "input_image", "image_url": data_uri(query_path), "detail": detail},
                    {"type": "input_image", "image_url": data_uri(dictionary_path), "detail": detail},
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    if supports_temperature(model):
        request["temperature"] = temperature
    response = client.responses.create(**request)
    return response.output_text


def call_batch_score_model(
    client: OpenAI,
    query_path: Path,
    sheet_path: Path,
    labels: list[str],
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    request: dict[str, object] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": batch_score_prompt(labels)},
                    {"type": "input_image", "image_url": data_uri(query_path), "detail": detail},
                    {"type": "input_image", "image_url": data_uri(sheet_path), "detail": detail},
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    if supports_temperature(model):
        request["temperature"] = temperature
    response = client.responses.create(**request)
    return response.output_text


def score_dictionary_entry(
    client: OpenAI,
    query_path: Path,
    dictionary_path: Path,
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
) -> DictionaryScore:
    raw = call_score_model(
        client,
        query_path,
        dictionary_path,
        model=model,
        detail=detail,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return DictionaryScore(
        path=str(dictionary_path),
        label=dictionary_path.stem,
        score=parse_score(raw),
        raw=raw.strip(),
    )


def score_dictionary_batch(
    client: OpenAI,
    query_path: Path,
    dictionary_batch: list[Path],
    temp_dir: Path,
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
) -> list[DictionaryScore]:
    labels = [path.stem for path in dictionary_batch]
    sheet_path = make_batch_sheet(dictionary_batch, temp_dir / f"batch_{labels[0]}_{labels[-1]}.png")
    raw = call_batch_score_model(
        client,
        query_path,
        sheet_path,
        labels,
        model=model,
        detail=detail,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    scores = parse_batch_scores(raw, labels)
    return [
        DictionaryScore(path=str(path), label=path.stem, score=scores[path.stem], raw=raw.strip())
        for path in dictionary_batch
    ]


def chunks(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def score_query_against_dictionary(
    query_path: Path,
    dictionary_dir: Path,
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
    show_progress: bool,
    debug: bool,
    batch_size: int,
) -> ScoreRun:
    candidates = dictionary_paths(dictionary_dir)
    if not candidates:
        raise FileNotFoundError(f"No supported dictionary images found in {dictionary_dir}")

    client = openai_client_from_env()
    scores: list[DictionaryScore] = []
    total = len(candidates)

    if batch_size == 1:
        for checked, dictionary_path in enumerate(progress(candidates, enabled=show_progress, description=f"Scoring {query_path.stem}"), start=1):
            result = score_dictionary_entry(
                client,
                query_path,
                dictionary_path,
                model=model,
                detail=detail,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            scores.append(result)
            if debug:
                print_iteration_result(result, checked=checked, total=total, show_progress=show_progress)
    else:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            checked = 0
            for batch in progress(chunks(candidates, batch_size), enabled=show_progress, description=f"Scoring {query_path.stem}"):
                results = score_dictionary_batch(
                    client,
                    query_path,
                    batch,
                    temp_dir,
                    model=model,
                    detail=detail,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
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


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def default_scores_output(query_path: Path, model: str) -> Path:
    return DEFAULT_SCORES_DIR / f"{query_path.parent.name}_{query_path.stem}_{safe_filename_part(model)}.json"


def write_scores_json(run: ScoreRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(run), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one query glyph against dictionary split glyphs with OpenAI vision."
    )
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY, help=f"Query glyph. Default: {DEFAULT_QUERY}")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY, help=f"Dictionary split folder. Default: {DEFAULT_DICTIONARY}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--detail", choices=("low", "high", "auto"), default="low", help="Vision detail level. Default: low.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help=f"Maximum tokens to generate per request. Minimum: {MIN_OUTPUT_TOKENS}. Defaults scale with --batch-size, or at least {O_SERIES_OUTPUT_TOKENS} for o-series models.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model sampling temperature.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of dictionary glyphs to score per model request. Default: 1.")
    parser.add_argument("--scores-output", type=Path, help=f"Path to write all scores as JSON. Default: {DEFAULT_SCORES_DIR}/<query>_<model>.json")
    parser.add_argument("--no-progress", action="store_true", help="Disable the per-dictionary-glyph progress bar.")
    parser.add_argument("--debug", action="store_true", help="Print each score while running.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    args.max_output_tokens = args.max_output_tokens or default_max_output_tokens(args.model, args.batch_size)
    if args.max_output_tokens < MIN_OUTPUT_TOKENS:
        parser.error(f"--max-output-tokens must be >= {MIN_OUTPUT_TOKENS}")
    return args


def main() -> None:
    args = parse_args()
    run = score_query_against_dictionary(
        query_path=args.query,
        dictionary_dir=args.dictionary,
        model=args.model,
        detail=args.detail,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        show_progress=not args.no_progress,
        debug=args.debug,
        batch_size=args.batch_size,
    )
    scores_output = args.scores_output or default_scores_output(args.query, args.model)
    write_scores_json(run, scores_output)
    output = asdict(run)
    del output["scores"]
    output["scores_output"] = str(scores_output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
