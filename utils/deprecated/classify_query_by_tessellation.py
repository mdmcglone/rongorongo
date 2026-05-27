from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI, RateLimitError
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = ROOT / "rr_tablets" / "compressed" / "Aa" / "1.webp"
DEFAULT_DICTIONARY = ROOT / "references" / "dictionary" / "split"
DEFAULT_TESSELLATIONS = ROOT / "references" / "dictionary" / "tessellations"
DEFAULT_ENV = ROOT / ".env"
DEFAULT_MODEL = "o3"
BATCH_SIZE = 100
BATCH_STARTS = tuple(range(0, 800, BATCH_SIZE))
SUPPORTED_SUFFIXES = frozenset({".gif", ".jpg", ".jpeg", ".png", ".webp"})
MIN_OUTPUT_TOKENS = 16
DEFAULT_OUTPUT_TOKENS = 48
O_SERIES_OUTPUT_TOKENS = 1024
DEFAULT_LABEL_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class LabelChoice:
    label: str
    raw: str


@dataclass(frozen=True)
class TessellationRun:
    query: str
    label: LabelChoice


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


def data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def dictionary_image_by_label(dictionary_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in dictionary_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and path.stem.isdigit():
            paths[int(path.stem)] = path
    return paths


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


def tessellation_path(output_dir: Path, batch_start: int) -> Path:
    return output_dir / f"{batch_start:03d}-{batch_start + BATCH_SIZE - 1:03d}.png"


def build_tessellation(
    dictionary_images: dict[int, Path],
    output_path: Path,
    *,
    batch_start: int,
    columns: int,
    cell_size: tuple[int, int],
    padding: int,
) -> None:
    rows = BATCH_SIZE // columns
    cell_width, cell_height = cell_size
    label_height = 26
    title_height = 42
    width = columns * cell_width + padding * 2
    height = rows * cell_height + title_height + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    label_font = load_font(16)
    draw_centered_text(
        draw,
        f"Dictionary batch {batch_start:03d}-{batch_start + BATCH_SIZE - 1:03d}",
        (0, padding // 2, width, title_height),
        title_font,
    )

    for offset in range(BATCH_SIZE):
        label = batch_start + offset
        row, column = divmod(offset, columns)
        left = padding + column * cell_width
        top = padding + title_height + row * cell_height
        right = left + cell_width
        bottom = top + cell_height
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(220, 220, 220))
        draw_centered_text(draw, f"{label:03d}", (left, top, right, top + label_height), label_font)

        image_path = dictionary_images.get(label)
        if image_path is None:
            draw_centered_text(draw, "missing", (left, top + label_height, right, bottom), label_font)
            continue

        image = fit_image(Image.open(image_path), (cell_width - 12, cell_height - label_height - 12))
        x = left + (cell_width - image.width) // 2
        y = top + label_height + (cell_height - label_height - image.height) // 2
        canvas.paste(image.convert("RGB"), (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def build_tessellations(
    dictionary_dir: Path,
    output_dir: Path,
    *,
    rebuild: bool,
    columns: int,
    cell_size: tuple[int, int],
    padding: int,
) -> list[Path]:
    if rebuild and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_images = dictionary_image_by_label(dictionary_dir)
    if not dictionary_images:
        raise FileNotFoundError(f"No supported dictionary images found in {dictionary_dir}")

    paths: list[Path] = []
    for batch_start in BATCH_STARTS:
        path = tessellation_path(output_dir, batch_start)
        if rebuild or not path.exists():
            build_tessellation(
                dictionary_images,
                path,
                batch_start=batch_start,
                columns=columns,
                cell_size=cell_size,
                padding=padding,
            )
        paths.append(path)
    return paths


def label_prompt() -> str:
    return (
        "You will receive one query rongorongo glyph followed by eight labeled dictionary "
        "batch sheets. Each sheet covers one range: 000-099, 100-199, ..., 700-799. "
        "Dictionary entries may contain multiple alternates separated by whitespace. Compare "
        "the query to each labeled entry independently across all eight sheets and choose the "
        "single most likely label. Tolerate line width, spacing, handwriting variation, "
        "degradation, expansion, and condensation, but prefer the same structural feature set. "
        "Answer exactly one line: label: <three-digit label from 000 to 799>"
    )


def supports_temperature(model: str) -> bool:
    return not model.lower().startswith("o")


def default_max_output_tokens(model: str) -> int:
    return O_SERIES_OUTPUT_TOKENS if model.lower().startswith("o") else DEFAULT_OUTPUT_TOKENS


def call_openai_images(
    client: OpenAI,
    prompt: str,
    image_paths: list[Path],
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
                    {"type": "input_text", "text": prompt},
                    *({"type": "input_image", "image_url": data_uri(path), "detail": detail} for path in image_paths),
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    if supports_temperature(model):
        request["temperature"] = temperature
    response = client.responses.create(**request)
    return response.output_text


def retry_delay_from_error(error: RateLimitError, fallback: float) -> float:
    response = getattr(error, "response", None)
    retry_after = response.headers.get("retry-after") if response else None
    if retry_after:
        try:
            return max(float(retry_after), fallback)
        except ValueError:
            return fallback

    match = re.search(r"try again in (\d+)ms", str(error), re.IGNORECASE)
    if match:
        return max(int(match.group(1)) / 1000, fallback)
    return fallback


def call_openai_images_with_rate_limit_retries(
    client: OpenAI,
    prompt: str,
    image_paths: list[Path],
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
    retries: int,
) -> str:
    delay = DEFAULT_RETRY_DELAY_SECONDS
    for attempt in range(retries + 1):
        try:
            return call_openai_images(
                client,
                prompt,
                image_paths,
                model=model,
                detail=detail,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except RateLimitError as error:
            if attempt >= retries:
                raise
            sleep_for = retry_delay_from_error(error, delay)
            print(
                f"Rate limit during tessellation label query; retrying in {sleep_for:.2f}s "
                f"({attempt + 1}/{retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_for)
            delay *= 2
    raise RuntimeError("unreachable")


def parse_label(raw: str) -> LabelChoice:
    if not raw.strip():
        raise ValueError(
            "Expected three-digit label, got an empty response. "
            "For o-series models, increase --max-output-tokens."
        )
    match = re.search(r"(?<!\d)(\d{3})(?!\d)", raw)
    if not match:
        raise ValueError(f"Expected three-digit label, got: {raw!r}")
    label = match.group(1)
    value = int(label)
    if not 0 <= value <= 799:
        raise ValueError(f"Label {label} is outside dictionary range 000-799")
    return LabelChoice(label=label, raw=raw.strip())


def classify_query_by_tessellation(
    query_path: Path,
    dictionary_dir: Path,
    tessellation_dir: Path,
    *,
    rebuild_tessellations: bool,
    columns: int,
    cell_size: tuple[int, int],
    padding: int,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
    label_retries: int,
) -> TessellationRun:
    sheets = build_tessellations(
        dictionary_dir,
        tessellation_dir,
        rebuild=rebuild_tessellations,
        columns=columns,
        cell_size=cell_size,
        padding=padding,
    )
    client = openai_client_from_env()
    label_raw = call_openai_images_with_rate_limit_retries(
        client,
        label_prompt(),
        [query_path, *sheets],
        model=model,
        detail=detail,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        retries=label_retries,
    )
    label = parse_label(label_raw)
    return TessellationRun(query=str(query_path), label=label)


def parse_cell_size(value: str) -> tuple[int, int]:
    width, _, height = value.lower().partition("x")
    if not width or not height:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, e.g. 140x120")
    return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one query glyph by choosing directly from labeled dictionary tessellations."
    )
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY, help=f"Query glyph. Default: {DEFAULT_QUERY}")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY, help=f"Dictionary split folder. Default: {DEFAULT_DICTIONARY}")
    parser.add_argument("--tessellations", type=Path, default=DEFAULT_TESSELLATIONS, help=f"Tessellation output/cache folder. Default: {DEFAULT_TESSELLATIONS}")
    parser.add_argument("--rebuild-tessellations", action="store_true", help="Rebuild tessellation sheets before classifying.")
    parser.add_argument("--build-only", action="store_true", help="Build tessellation sheets and exit without OpenAI calls.")
    parser.add_argument("--columns", type=int, default=10, help="Grid columns per tessellation sheet.")
    parser.add_argument("--cell-size", type=parse_cell_size, default=(140, 120), help="Cell size as WIDTHxHEIGHT. Default: 140x120.")
    parser.add_argument("--padding", type=int, default=24, help="Outer sheet padding in pixels.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--detail", choices=("low", "high", "auto"), default="high", help="Vision detail level. Default: high, so labels stay readable.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help=f"Maximum tokens per classification step. Minimum: {MIN_OUTPUT_TOKENS}. Defaults to {DEFAULT_OUTPUT_TOKENS}, or {O_SERIES_OUTPUT_TOKENS} for o-series models.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model sampling temperature.")
    parser.add_argument("--label-retries", type=int, default=DEFAULT_LABEL_RETRIES, help="Rate-limit retries for the label-selection query.")
    args = parser.parse_args()
    args.max_output_tokens = args.max_output_tokens or default_max_output_tokens(args.model)
    if args.max_output_tokens < MIN_OUTPUT_TOKENS:
        parser.error(f"--max-output-tokens must be >= {MIN_OUTPUT_TOKENS}")
    if args.label_retries < 0:
        parser.error("--label-retries must be >= 0")
    if BATCH_SIZE % args.columns != 0:
        parser.error(f"--columns must evenly divide {BATCH_SIZE}")
    return args


def main() -> None:
    args = parse_args()
    if args.build_only:
        sheets = build_tessellations(
            args.dictionary,
            args.tessellations,
            rebuild=True,
            columns=args.columns,
            cell_size=args.cell_size,
            padding=args.padding,
        )
        print(json.dumps({"tessellations": [str(path) for path in sheets]}, indent=2))
        return

    run = classify_query_by_tessellation(
        query_path=args.query,
        dictionary_dir=args.dictionary,
        tessellation_dir=args.tessellations,
        rebuild_tessellations=args.rebuild_tessellations,
        columns=args.columns,
        cell_size=args.cell_size,
        padding=args.padding,
        model=args.model,
        detail=args.detail,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        label_retries=args.label_retries,
    )
    print(json.dumps(asdict(run), indent=2))


if __name__ == "__main__":
    main()
