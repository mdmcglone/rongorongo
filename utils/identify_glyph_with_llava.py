from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MAREEP_ROOT = Path("/Users/emdmcglone/Desktop/mareep")
LLAVA_UTIL_DIR = Path("/Users/emdmcglone/Desktop/mareep/llava_vision")
DEFAULT_GLYPH = ROOT / "rr_tablets" / "split" / "Aa" / "1.png"
DEFAULT_KEY = ROOT / "references" / "rr_key.png"
DEFAULT_LABELS = (
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "14",
    "15",
    "16",
    "22",
    "25",
    "27a",
    "28",
    "34",
    "38",
    "41",
    "44",
    "46",
    "47",
    "50",
    "52",
    "53",
    "59",
    "60",
    "61",
    "62",
    "63",
    "66",
    "67",
    "69",
    "70",
    "71",
    "74",
    "76",
    "91",
    "95",
    "99",
    "200",
    "240",
    "280",
    "380",
    "400",
    "530",
    "660",
    "700",
    "720",
    "730",
    "901",
)


@dataclass(frozen=True)
class GlyphMatch:
    label: str
    reasoning: str

    def format(self) -> str:
        return f"label: {self.label}"


@dataclass(frozen=True)
class Candidate:
    label: str
    glyph: Image.Image
    score: float


def import_simple_call():
    sys.path.insert(0, str(MAREEP_ROOT))
    sys.path.insert(0, str(LLAVA_UTIL_DIR))
    try:
        from util import simple_call
    except ModuleNotFoundError as error:
        missing = error.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Install this repo's requirements first:\n"
            "  venv/bin/python -m pip install -r requirements.txt"
        ) from error

    return simple_call


def scale_to_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def crop_ink(image: Image.Image, threshold: int = 210) -> Image.Image:
    rgba = image.convert("RGBA")
    gray = ImageOps.grayscale(rgba)
    mask = gray.point(lambda pixel: 255 if pixel < threshold else 0)
    bbox = mask.getbbox()
    return rgba.crop(bbox) if bbox else rgba


def glyph_mask(image: Image.Image, size: int = 72) -> np.ndarray:
    gray = ImageOps.grayscale(image)
    binary = gray.point(lambda pixel: 255 if pixel < 180 else 0)
    bbox = binary.getbbox()
    if bbox:
        gray = gray.crop(bbox)

    gray = ImageOps.contain(gray, (size - 8, size - 8), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(gray, ((size - gray.width) // 2, (size - gray.height) // 2))
    return np.array(canvas) < 200


def key_candidates(glyph_path: Path, key_path: Path, top_k: int) -> list[Candidate]:
    key = Image.open(key_path).convert("RGB")
    query_mask = glyph_mask(Image.open(glyph_path).convert("RGB"))
    width, height = key.size
    scored: list[Candidate] = []

    for index, label in enumerate(DEFAULT_LABELS):
        row = index // 13
        column = index % 13
        left = round(column * width / 13)
        right = round((column + 1) * width / 13)
        top = round(row * height / 4)
        bottom = round((row + 1) * height / 4)
        glyph_bottom = top + round((bottom - top) * 0.56)
        glyph_area = key.crop((left + 8, top + 6, right - 8, glyph_bottom))
        candidate_mask = glyph_mask(glyph_area)
        overlap = np.logical_and(query_mask, candidate_mask).sum()
        union = np.logical_or(query_mask, candidate_mask).sum() or 1
        disagreement = np.logical_xor(query_mask, candidate_mask).sum()
        score = float(overlap / union - disagreement * 0.0001)
        scored.append(Candidate(label=label, glyph=crop_ink(glyph_area), score=score))

    return sorted(scored, key=lambda candidate: candidate.score, reverse=True)[:top_k]


def make_comparison_image(glyph_path: Path, key_path: Path, output_path: Path, top_k: int = 8) -> list[Candidate]:
    glyph = ImageOps.contain(crop_ink(Image.open(glyph_path)), (420, 360), Image.Resampling.NEAREST)
    candidates = key_candidates(glyph_path, key_path, top_k)
    padding = 32
    cell_width = 240
    cell_height = 240
    columns = 4
    rows = (len(candidates) + columns - 1) // columns
    grid_width = columns * cell_width
    grid_height = rows * cell_height
    width = glyph.width + grid_width + padding * 3
    height = max(glyph.height + 120, grid_height) + padding * 2
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((padding, padding), "QUERY GLYPH", fill=(0, 0, 0, 255))
    canvas.paste(glyph, (padding, padding + 40), glyph)
    draw.text((glyph.width + padding * 2, padding), "PREFILTERED REFERENCE CANDIDATES", fill=(0, 0, 0, 255))

    for index, candidate in enumerate(candidates):
        row = index // columns
        column = index % columns
        x = glyph.width + padding * 2 + column * cell_width
        y = padding + 40 + row * cell_height
        draw.rectangle((x, y, x + cell_width - 12, y + cell_height - 12), outline=(190, 190, 190, 255), width=2)
        draw.text((x + 12, y + 10), f"label: {candidate.label}", fill=(0, 0, 0, 255))
        candidate_glyph = ImageOps.contain(candidate.glyph, (cell_width - 52, cell_height - 72), Image.Resampling.NEAREST)
        canvas.paste(candidate_glyph, (x + (cell_width - candidate_glyph.width) // 2, y + 52), candidate_glyph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return candidates


def prompt(labels: list[str]) -> str:
    return (
        "The image contains one query rongorongo glyph on the left and a small set of prefiltered "
        "reference candidates on the right. Each reference candidate has its label printed below it. "
        "Compare the query glyph against the reference glyph drawings, ignoring size, stroke thickness, "
        "crop padding, and small handwriting variation. If the query is a ligature or composite made from "
        "multiple whole or partial reference glyphs, return every constituent label joined by '.', such as "
        "08.200.38. If it is a single glyph, return one label. You must choose the closest available "
        "reference glyph or glyph combination even if the match is imperfect. Do not answer UNK or unknown. "
        f"The only allowed labels are: {', '.join(labels)}. "
        "Do not list alternatives. Do not mention any label except your chosen label or chosen '.' combination. "
        "Return exactly one line in this format:\n"
        "label: <one allowed label or . separated allowed labels>"
    )


def normalize_label_candidate(candidate: str, labels: tuple[str, ...]) -> str | None:
    normalized_parts: list[str] = []

    for part in re.split(r"\s*[+.]\s*", candidate.strip()):
        if not part:
            return None
        match = next((label for label in labels if part.lower() == label.lower()), None)
        if match is None:
            return None
        normalized_parts.append(match)

    return ".".join(normalized_parts)


def normalize_label(answer: str, labels: tuple[str, ...]) -> str:
    stripped = answer.strip()
    label_match = re.search(r"^\s*label\s*:\s*([A-Za-z0-9+. ]+)\s*$", stripped, re.IGNORECASE | re.MULTILINE)
    if label_match:
        normalized = normalize_label_candidate(label_match.group(1), labels)
        if normalized:
            return normalized

    for label in sorted(labels, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", stripped, re.IGNORECASE):
            return label

    return labels[0]


def parse_match(answer: str, labels: tuple[str, ...]) -> GlyphMatch:
    label = normalize_label(answer, labels)
    reasoning_match = re.search(
        r"reasoning\s*:\s*(.+?)(?:\n\s*label\s*:|\n\s*raw\s*:|\Z)",
        answer.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else answer.strip()
    return GlyphMatch(label=label, reasoning=reasoning)


def identify_glyph(
    glyph_path: Path,
    key_path: Path,
    *,
    keep_comparison: Path | None,
    raw: bool,
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    simple_call = import_simple_call()

    with tempfile.TemporaryDirectory() as temp_dir:
        comparison_path = keep_comparison or Path(temp_dir) / "glyph_key_comparison.png"
        candidates = make_comparison_image(glyph_path, key_path, comparison_path, top_k)
        candidate_labels = [candidate.label for candidate in candidates]
        answer = simple_call(
            prompt(candidate_labels),
            image_path=comparison_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    match = parse_match(str(answer), tuple(candidate_labels))
    return f"{match.format()}\nraw: {answer}" if raw else match.format()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify one split rongorongo glyph with LLaVA and rr_key.png.")
    parser.add_argument("glyph", nargs="?", type=Path, default=DEFAULT_GLYPH, help=f"Glyph image. Default: {DEFAULT_GLYPH}")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY, help=f"Reference key image. Default: {DEFAULT_KEY}")
    parser.add_argument("--keep-comparison", type=Path, help="Optional path to save the combined query/key image.")
    parser.add_argument("--raw", action="store_true", help="Print the raw model answer after the normalized output.")
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model sampling temperature.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of visual prefilter candidates to show LLaVA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        identify_glyph(
            args.glyph,
            args.key,
            keep_comparison=args.keep_comparison,
            raw=args.raw,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
