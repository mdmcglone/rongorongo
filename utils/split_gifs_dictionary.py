from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "references" / "dictionary" / "gifs"
DEFAULT_OUTPUT = ROOT / "references" / "dictionary" / "split"
STRIPS_PER_GIF = 5
DEFAULT_STRIP_FRACTION = 1 / STRIPS_PER_GIF
CUSTOM_STRIP_FRACTION: dict[int, float] = {93: 0.18, 94: 0.22, 468: 0.18, 469: 0.22}
OUTPUT_SUFFIX = ".webp"
SAVE_FORMAT = "WEBP"
SAVE_OPTIONS = {"lossless": True, "method": 6}
MAX_TINY_BYTES = 54
REMOVE_STEMS = frozenset({"000", "114"})
TRIM_RIGHT_FRACTION: dict[int, float] = {639: 0.05}
BORDER_TRIM_RANGE = range(535, 540)
BOTTOM_WHITE_BAND = (32, 16)
GIF_NUMBER = re.compile(r"^(\d+)$", re.IGNORECASE)


def compress_for_save(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    return gray.point(lambda value: 0 if value < 128 else 255, mode="1")


def strip_fraction(output_number: int) -> float:
    return CUSTOM_STRIP_FRACTION.get(output_number, DEFAULT_STRIP_FRACTION)


def strip_crop(image: Image.Image, strip_index: int, output_number: int) -> Image.Image:
    width, height = image.size
    fraction = strip_fraction(output_number)
    center_x = (strip_index + 0.5) * width / STRIPS_PER_GIF
    crop_width = int(round(fraction * width))
    left = int(round(center_x - crop_width / 2))
    right = left + crop_width
    if right > width:
        right = width
        left = max(0, right - crop_width)
    if left < 0:
        left = 0
        right = min(width, crop_width)
    return image.crop((left, 0, right, height))


def trim_right_fraction(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    keep_width = max(1, int(round(width * (1 - fraction))))
    return image.crop((0, 0, keep_width, height))


def trim_border_fraction(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    inset_x = int(round(width * fraction))
    inset_y = int(round(height * fraction))
    return image.crop((inset_x, inset_y, width - inset_x, height - inset_y))


def whiten_bottom_left(image: Image.Image, band_width: int, band_height: int) -> Image.Image:
    gray = image.convert("L")
    width, height = gray.size
    right = min(width, band_width) - 1
    top = max(0, height - band_height)
    ImageDraw.Draw(gray).rectangle((0, top, right, height - 1), fill=255)
    return gray


def fix_strip_artifacts(strip: Image.Image, output_number: int) -> Image.Image:
    if output_number in TRIM_RIGHT_FRACTION:
        strip = trim_right_fraction(strip, TRIM_RIGHT_FRACTION[output_number])
    if output_number in BORDER_TRIM_RANGE:
        strip = trim_border_fraction(strip, 0.01)
        strip = whiten_bottom_left(strip, *BOTTOM_WHITE_BAND)
    return strip


def parse_gif_number(path: Path) -> int:
    match = GIF_NUMBER.match(path.stem)
    if not match:
        raise ValueError(f"Expected a numeric GIF name like 000.GIF, got {path.name}")
    return int(match.group(1))


def save_strip(strip: Image.Image, output_path: Path) -> None:
    compress_for_save(strip).save(output_path, format=SAVE_FORMAT, **SAVE_OPTIONS)


def should_remove_split_file(path: Path) -> bool:
    return path.stem in REMOVE_STEMS or path.stat().st_size <= MAX_TINY_BYTES


def cleanup_split_output(output_dir: Path) -> int:
    removed = 0
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != OUTPUT_SUFFIX:
            continue
        if should_remove_split_file(path):
            path.unlink()
            removed += 1
    return removed


def split_gifs_dictionary(
    input_dir: Path,
    output_dir: Path,
    clear_output: bool,
) -> tuple[int, int]:
    gif_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".gif"
    )
    if not gif_paths:
        raise FileNotFoundError(f"No GIF files found in {input_dir}")

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for gif_path in gif_paths:
        base_number = parse_gif_number(gif_path)
        image = Image.open(gif_path)
        for index in range(STRIPS_PER_GIF):
            output_number = base_number + index
            output_path = output_dir / f"{output_number:03d}{OUTPUT_SUFFIX}"
            strip = fix_strip_artifacts(strip_crop(image, index, output_number), output_number)
            save_strip(strip, output_path)
            written += 1

    removed = cleanup_split_output(output_dir)
    return written, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dictionary GIF strips into five compressed glyph images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Folder of dictionary GIFs. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output folder. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the output folder before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written, removed = split_gifs_dictionary(
        input_dir=args.input,
        output_dir=args.output,
        clear_output=not args.keep_existing,
    )
    print(
        f"Wrote {written} strips to {args.output} ({SAVE_FORMAT} lossless{OUTPUT_SUFFIX}); "
        f"removed {removed} (≤{MAX_TINY_BYTES} bytes or {', '.join(sorted(REMOVE_STEMS))})"
    )


if __name__ == "__main__":
    main()
