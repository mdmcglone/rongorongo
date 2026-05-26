from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "rr_tablets" / "split"
DEFAULT_OUTPUT = ROOT / "rr_tablets" / "compressed"
SUPPORTED_SUFFIXES = frozenset({".gif", ".jpg", ".jpeg", ".png", ".webp"})
SAVE_CANDIDATES = (
    ("png", "PNG", {"optimize": True, "compress_level": 9}),
    ("webp", "WEBP", {"lossless": True, "method": 6}),
    ("gif", "GIF", {"optimize": True}),
)


@dataclass(frozen=True)
class EncodedImage:
    suffix: str
    data: bytes


def compressed_glyph(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    gray = background.convert("L")
    return gray.point(lambda value: 0 if value < 128 else 255, mode="1")


def encode_candidate(image: Image.Image, suffix: str, image_format: str, options: dict[str, object]) -> EncodedImage:
    buffer = BytesIO()
    image.save(buffer, format=image_format, **options)
    return EncodedImage(suffix=suffix, data=buffer.getvalue())


def encode_smallest(image: Image.Image) -> EncodedImage:
    prepared = compressed_glyph(image)
    return min(
        (
            encode_candidate(prepared, suffix, image_format, options)
            for suffix, image_format, options in SAVE_CANDIDATES
        ),
        key=lambda encoded: len(encoded.data),
    )


def image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def output_path(input_dir: Path, output_dir: Path, source_path: Path, suffix: str) -> Path:
    relative = source_path.relative_to(input_dir).with_suffix(f".{suffix}")
    return output_dir / relative


def compress_split_glyphs(input_dir: Path, output_dir: Path, clear_output: bool) -> tuple[int, int]:
    sources = image_paths(input_dir)
    if not sources:
        raise FileNotFoundError(f"No supported image files found in {input_dir}")

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for source_path in sources:
        encoded = encode_smallest(Image.open(source_path))
        destination = output_path(input_dir, output_dir, source_path, encoded.suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded.data)
        total_bytes += len(encoded.data)

    return len(sources), total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maximally compress split tablet glyphs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Split glyph folder. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output folder. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear the output folder before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, total_bytes = compress_split_glyphs(
        input_dir=args.input,
        output_dir=args.output,
        clear_output=not args.keep_existing,
    )
    print(f"Wrote {count} compressed glyphs to {args.output} ({total_bytes:,} bytes total)")


if __name__ == "__main__":
    main()
