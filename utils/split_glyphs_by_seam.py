from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "rr_tablets" / "raw" / "Aa.png"
DEFAULT_OUTPUT = ROOT / "rr_tablets" / "split" / "Aa"
Point = tuple[int, int]


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    def width(self) -> int:
        return self.right - self.left

    def height(self) -> int:
        return self.bottom - self.top

    def padded(self, image_width: int, image_height: int, padding: int) -> "Box":
        return Box(
            max(0, self.left - padding),
            max(0, self.top - padding),
            min(image_width, self.right + padding),
            min(image_height, self.bottom + padding),
        )


@dataclass(frozen=True)
class Seam:
    path: tuple[int, ...]
    ink_hits: int


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


def bounding_box(points: Iterable[Point]) -> Box:
    xs, ys = zip(*points)
    return Box(min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def row_runs(image: Image.Image, threshold: int, min_row_ink: int, row_gap: int) -> list[tuple[int, int]]:
    width, height = image.size
    pixels = image.load()
    counts = [sum(is_dark(pixels[x, y], threshold) for x in range(width)) for y in range(height)]
    return runs_from_counts(counts, min_row_ink, row_gap)


def coarse_regions(
    image: Image.Image,
    rows: list[tuple[int, int]],
    threshold: int,
    min_column_ink: int,
    glyph_gap: int,
) -> list[set[Point]]:
    width, _ = image.size
    pixels = image.load()
    regions: list[set[Point]] = []

    for top, bottom in rows:
        counts = [
            sum(is_dark(pixels[x, y], threshold) for y in range(top, bottom))
            for x in range(width)
        ]

        for left, right in runs_from_counts(counts, min_column_ink, glyph_gap):
            points = {
                (x, y)
                for x in range(left, right)
                for y in range(top, bottom)
                if is_dark(pixels[x, y], threshold)
            }
            if points:
                regions.append(points)

    return regions


def best_whitespace_seam(points: set[Point], min_glyph_width: int, max_step: int) -> Seam | None:
    box = bounding_box(points)
    width = box.width()
    height = box.height()
    first_x = min_glyph_width
    last_x = width - min_glyph_width - 1

    if first_x > last_x:
        return None

    def pixel_cost(local_x: int, local_y: int) -> int:
        return 1000 if (box.left + local_x, box.top + local_y) in points else 0

    inf = 10**12
    previous = [inf] * width
    parents: list[list[int]] = []

    for x in range(first_x, last_x + 1):
        previous[x] = pixel_cost(x, 0)

    for y in range(1, height):
        current = [inf] * width
        parent_row = [-1] * width

        for x in range(first_x, last_x + 1):
            left = max(first_x, x - max_step)
            right = min(last_x, x + max_step)
            parent = min(range(left, right + 1), key=previous.__getitem__)
            current[x] = previous[parent] + pixel_cost(x, y)
            parent_row[x] = parent

        parents.append(parent_row)
        previous = current

    end_x = min(range(first_x, last_x + 1), key=previous.__getitem__)
    path = [end_x]

    for parent_row in reversed(parents):
        path.append(parent_row[path[-1]])

    path.reverse()
    ink_hits = sum((box.left + x, box.top + y) in points for y, x in enumerate(path))
    return Seam(tuple(box.left + x for x in path), ink_hits)


def split_region(
    points: set[Point],
    min_glyph_width: int,
    min_split_width: int,
    max_seam_ink: int,
    max_step: int,
) -> list[set[Point]]:
    box = bounding_box(points)

    if box.width() < min_split_width or box.width() < min_glyph_width * 2:
        return [points]

    seam = best_whitespace_seam(points, min_glyph_width, max_step)
    if seam is None or seam.ink_hits > max_seam_ink:
        return [points]

    left_points = {(x, y) for x, y in points if x < seam.path[y - box.top]}
    right_points = points - left_points

    if not left_points or not right_points:
        return [points]

    return [
        *split_region(left_points, min_glyph_width, min_split_width, max_seam_ink, max_step),
        *split_region(right_points, min_glyph_width, min_split_width, max_seam_ink, max_step),
    ]


def render_region(source: Image.Image, points: set[Point], padding: int) -> Image.Image:
    image_width, image_height = source.size
    box = bounding_box(points).padded(image_width, image_height, padding)
    output = Image.new("RGBA", (box.width(), box.height()), (255, 255, 255, 255))
    source_rgba = source.convert("RGBA")
    source_pixels = source_rgba.load()
    output_pixels = output.load()

    for x, y in points:
        if box.left <= x < box.right and box.top <= y < box.bottom:
            output_pixels[x - box.left, y - box.top] = source_pixels[x, y]

    return output


def split_glyphs_by_seam(
    input_path: Path,
    output_dir: Path,
    threshold: int,
    min_row_ink: int,
    min_column_ink: int,
    row_gap: int,
    coarse_glyph_gap: int,
    min_glyph_width: int,
    min_split_width: int,
    max_seam_ink: int,
    max_step: int,
    padding: int,
    clear_output: bool,
) -> int:
    image = Image.open(input_path).convert("L")
    source = Image.open(input_path)
    rows = row_runs(image, threshold, min_row_ink, row_gap)
    regions = coarse_regions(image, rows, threshold, min_column_ink, coarse_glyph_gap)
    glyphs = [
        glyph
        for region in regions
        for glyph in split_region(region, min_glyph_width, min_split_width, max_seam_ink, max_step)
    ]

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, glyph in enumerate(glyphs, start=1):
        render_region(source, glyph, padding).save(output_dir / f"{index}.png")

    return len(glyphs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split rongorongo glyphs with diagonal whitespace seams.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Source tablet image. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output folder. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--threshold", type=int, default=190, help="Pixels darker than this are treated as glyph ink.")
    parser.add_argument("--min-row-ink", type=int, default=6, help="Minimum dark pixels needed for a row to count as ink.")
    parser.add_argument("--min-column-ink", type=int, default=2, help="Minimum dark pixels needed for a column to count as ink.")
    parser.add_argument("--row-gap", type=int, default=5, help="Maximum blank-pixel row gap to keep within one text row.")
    parser.add_argument("--coarse-glyph-gap", type=int, default=1, help="Initial column-gap split before seam refinement.")
    parser.add_argument("--min-glyph-width", type=int, default=10, help="Smallest allowed width on either side of a seam.")
    parser.add_argument("--min-split-width", type=int, default=35, help="Only regions at least this wide are considered for seam splitting.")
    parser.add_argument("--max-seam-ink", type=int, default=0, help="Maximum dark pixels a valid seam may cross.")
    parser.add_argument("--max-step", type=int, default=1, help="Maximum horizontal movement per row for diagonal seams.")
    parser.add_argument("--padding", type=int, default=2, help="Crop padding in pixels.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear the output folder before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = split_glyphs_by_seam(
        input_path=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        min_row_ink=args.min_row_ink,
        min_column_ink=args.min_column_ink,
        row_gap=args.row_gap,
        coarse_glyph_gap=args.coarse_glyph_gap,
        min_glyph_width=args.min_glyph_width,
        min_split_width=args.min_split_width,
        max_seam_ink=args.max_seam_ink,
        max_step=args.max_step,
        padding=args.padding,
        clear_output=not args.keep_existing,
    )
    print(f"Wrote {count} glyphs to {args.output}")


if __name__ == "__main__":
    main()
