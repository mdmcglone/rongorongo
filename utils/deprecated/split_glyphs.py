from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "rr_tablets" / "raw" / "Aa.png"
DEFAULT_OUTPUT = ROOT / "rr_tablets" / "split" / "Aa"


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    def padded(self, width: int, height: int, padding: int) -> "Box":
        return Box(
            max(0, self.left - padding),
            max(0, self.top - padding),
            min(width, self.right + padding),
            min(height, self.bottom + padding),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def is_dark(pixel: int, threshold: int) -> bool:
    return pixel < threshold


def runs_from_counts(counts: list[int], min_count: int, max_gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_ink: int | None = None

    for index, count in enumerate(counts):
        if count >= min_count:
            start = index if start is None else start
            last_ink = index
        elif start is not None and last_ink is not None and index - last_ink > max_gap:
            runs.append((start, last_ink + 1))
            start = None
            last_ink = None

    if start is not None and last_ink is not None:
        runs.append((start, last_ink + 1))

    return runs


def row_runs(image: Image.Image, threshold: int, min_row_ink: int, row_gap: int) -> list[tuple[int, int]]:
    width, height = image.size
    pixels = image.load()
    counts = [sum(is_dark(pixels[x, y], threshold) for x in range(width)) for y in range(height)]
    return runs_from_counts(counts, min_row_ink, row_gap)


def glyph_boxes(
    image: Image.Image,
    rows: list[tuple[int, int]],
    threshold: int,
    min_column_ink: int,
    glyph_gap: int,
    padding: int,
) -> list[Box]:
    width, height = image.size
    pixels = image.load()
    boxes: list[Box] = []

    for top, bottom in rows:
        counts = [
            sum(is_dark(pixels[x, y], threshold) for y in range(top, bottom))
            for x in range(width)
        ]

        for left, right in runs_from_counts(counts, min_column_ink, glyph_gap):
            dark_points = [
                (x, y)
                for x in range(left, right)
                for y in range(top, bottom)
                if is_dark(pixels[x, y], threshold)
            ]
            if not dark_points:
                continue

            xs, ys = zip(*dark_points)
            boxes.append(Box(min(xs), min(ys), max(xs) + 1, max(ys) + 1).padded(width, height, padding))

    return boxes


def split_glyphs(
    input_path: Path,
    output_dir: Path,
    threshold: int,
    min_row_ink: int,
    min_column_ink: int,
    row_gap: int,
    glyph_gap: int,
    padding: int,
    clear_output: bool,
) -> int:
    image = Image.open(input_path).convert("L")
    rows = row_runs(image, threshold, min_row_ink, row_gap)
    boxes = glyph_boxes(image, rows, threshold, min_column_ink, glyph_gap, padding)

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(input_path).convert("RGBA")
    for index, box in enumerate(boxes, start=1):
        source.crop(box.as_tuple()).save(output_dir / f"{index}.png")

    return len(boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a rongorongo tablet image into ordered glyph crops.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Source tablet image. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output folder. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--threshold", type=int, default=190, help="Pixels darker than this are treated as glyph ink.")
    parser.add_argument("--min-row-ink", type=int, default=6, help="Minimum dark pixels needed for a row to count as ink.")
    parser.add_argument("--min-column-ink", type=int, default=2, help="Minimum dark pixels needed for a column to count as ink.")
    parser.add_argument("--row-gap", type=int, default=5, help="Maximum blank-pixel row gap to keep within one text row.")
    parser.add_argument("--glyph-gap", type=int, default=1, help="Maximum blank-pixel column gap to keep within one glyph.")
    parser.add_argument("--padding", type=int, default=2, help="Transparent-image crop padding in pixels.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear the output folder before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = split_glyphs(
        input_path=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        min_row_ink=args.min_row_ink,
        min_column_ink=args.min_column_ink,
        row_gap=args.row_gap,
        glyph_gap=args.glyph_gap,
        padding=args.padding,
        clear_output=not args.keep_existing,
    )
    print(f"Wrote {count} glyphs to {args.output}")


if __name__ == "__main__":
    main()
