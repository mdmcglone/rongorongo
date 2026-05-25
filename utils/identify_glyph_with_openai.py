from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageOps

try:
    from identify_glyph_with_llava import DEFAULT_GLYPH, DEFAULT_KEY, DEFAULT_LABELS, crop_ink, normalize_label_candidate, parse_match
except ModuleNotFoundError:
    from .identify_glyph_with_llava import DEFAULT_GLYPH, DEFAULT_KEY, DEFAULT_LABELS, crop_ink, normalize_label_candidate, parse_match


DEFAULT_MODEL = "gpt-4.1"
DEFAULT_LIGATURE_REFERENCES = (
    DEFAULT_KEY.parent / "lig1.png",
    DEFAULT_KEY.parent / "lig2.png",
    DEFAULT_KEY.parent / "lig3.png",
)


@dataclass(frozen=True)
class OpenAIGlyphMatch:
    label: str
    alternates: tuple[str, ...]
    reasoning: str

    def format(self) -> str:
        alternates = ", ".join(self.alternates) if self.alternates else "none"
        return f"label: {self.label}\nalternates: {alternates}"


def data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def make_comparison_image(glyph_path: Path, key_path: Path, output_path: Path, ligature_paths: tuple[Path, ...]) -> None:
    glyph = ImageOps.contain(crop_ink(Image.open(glyph_path)), (420, 360), Image.Resampling.NEAREST)
    key = ImageOps.contain(Image.open(key_path).convert("RGBA"), (1800, 700), Image.Resampling.LANCZOS)
    ligatures = [ImageOps.contain(Image.open(path).convert("RGBA"), (560, 360), Image.Resampling.LANCZOS) for path in ligature_paths if path.exists()]
    padding = 32
    label_band = 48
    width = max(key.width, glyph.width) + padding * 2
    ligature_height = max((image.height for image in ligatures), default=0)
    height = glyph.height + key.height + ligature_height + padding * (4 if ligatures else 3) + label_band * (2 if ligatures else 1)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((padding, padding), "QUERY GLYPH", fill=(0, 0, 0, 255))
    canvas.paste(glyph, ((width - glyph.width) // 2, padding + 28), glyph)
    y = glyph.height + padding * 2

    if ligatures:
        draw.text((padding, y), "LIGATURE IDENTIFICATION EXAMPLES", fill=(0, 0, 0, 255))
        x = padding
        y += label_band
        for index, image in enumerate(ligatures, start=1):
            draw.text((x, y), f"lig{index}", fill=(0, 0, 0, 255))
            canvas.paste(image, (x, y + 22), image)
            x += image.width + padding
        y += ligature_height + padding

    draw.text((padding, y), "FULL REFERENCE KEY", fill=(0, 0, 0, 255))
    canvas.paste(key, ((width - key.width) // 2, y + label_band), key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def full_key_prompt(labels: tuple[str, ...]) -> str:
    return (
        "You will receive multiple images. The first image is one query rongorongo glyph. "
        "The second image is the full reference key table. Each reference glyph has its label printed below it. "
        "The remaining images are examples showing how ligatures and composite glyphs should be identified. "
        "Compare the query glyph against all glyphs in the reference key, ignoring size, stroke thickness, "
        "crop padding, and small handwriting variation. If the query is a ligature or composite made from "
        "multiple whole or partial reference glyphs, return every constituent label joined by '.', such as "
        "08.200.38. If it is a single glyph, return one label. You must choose the closest available "
        "reference glyph or glyph combination even if the match is imperfect. "
        "Return UNK only if it shares little to no visual similarity to any reference glyph."
        "Note that humanoid forms such as 240 or 380 are often composite via their legs are arms. "
        "Note that 06 includes a simple four or five fingered hand, but not all hands are 06 "
        "Note that 03 applies whenever another glyph has its spikes "
        "Be aware of the difference betweened the 08, the six spiked star, and 09, the horned arrow with a stem"
        f"The only allowed labels are: {', '.join(labels)}. "
        "Provide a comma-separated list of plausible alternate labels or '.' combinations. "
        "Return exactly two lines in this format:\n"
        "label: <one allowed label or . separated allowed labels>\n"
        "alternates: <comma-separated alternate labels or . separated label combinations, or none>"
    )


def parse_alternates(answer: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    match = re.search(r"^\s*alternates\s*:\s*(.+?)\s*$", answer.strip(), re.IGNORECASE | re.MULTILINE)
    if not match:
        return ()

    raw_alternates = match.group(1).strip()
    if raw_alternates.lower() in {"", "none", "n/a", "na"}:
        return ()

    alternates: list[str] = []
    for item in raw_alternates.split(","):
        normalized = normalize_label_candidate(item, labels)
        if normalized and normalized not in alternates:
            alternates.append(normalized)

    return tuple(alternates)


def parse_openai_match(answer: str, labels: tuple[str, ...]) -> OpenAIGlyphMatch:
    match = parse_match(answer, labels)
    return OpenAIGlyphMatch(
        label=match.label,
        alternates=parse_alternates(answer, labels),
        reasoning=match.reasoning,
    )


def call_openai(
    glyph_path: Path,
    key_path: Path,
    ligature_paths: tuple[Path, ...],
    *,
    model: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    client = OpenAI()
    content = [
        {"type": "input_text", "text": full_key_prompt(DEFAULT_LABELS)},
        {"type": "input_image", "image_url": data_uri(glyph_path)},
        {"type": "input_image", "image_url": data_uri(key_path)},
    ]
    content.extend({"type": "input_image", "image_url": data_uri(path)} for path in ligature_paths if path.exists())

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return response.output_text


def identify_glyph(
    glyph_path: Path,
    key_path: Path,
    *,
    ligature_paths: tuple[Path, ...],
    keep_comparison: Path | None,
    raw: bool,
    model: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    if keep_comparison:
        make_comparison_image(glyph_path, key_path, keep_comparison, ligature_paths)

    answer = call_openai(
        glyph_path,
        key_path,
        ligature_paths,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    match = parse_openai_match(answer, DEFAULT_LABELS)
    return f"{match.format()}\nraw: {answer}" if raw else match.format()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify one split rongorongo glyph with OpenAI vision.")
    parser.add_argument("glyph", nargs="?", type=Path, default=DEFAULT_GLYPH, help=f"Glyph image. Default: {DEFAULT_GLYPH}")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY, help=f"Reference key image. Default: {DEFAULT_KEY}")
    parser.add_argument("--ligature-reference", action="append", type=Path, dest="ligature_references", help="Ligature example image. Can be passed multiple times.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--keep-comparison", type=Path, help="Optional path to save a combined query/full-key debug image.")
    parser.add_argument("--raw", action="store_true", help="Print the raw model answer after the normalized output.")
    parser.add_argument("--max-output-tokens", type=int, default=80, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model sampling temperature.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        identify_glyph(
            args.glyph,
            args.key,
            ligature_paths=tuple(args.ligature_references or DEFAULT_LIGATURE_REFERENCES),
            keep_comparison=args.keep_comparison,
            raw=args.raw,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
