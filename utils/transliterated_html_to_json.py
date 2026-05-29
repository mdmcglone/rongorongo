from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "rr_tablets" / "transliterated"
CORPORA = ("complete", "broken")


class LineListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: dict[str, str] = {}
        self._in_li = False
        self._in_anchor = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "li":
            self._in_li = True
            self._title_parts = []
            self._text_parts = []
        elif self._in_li and tag == "a":
            self._in_anchor = True
        elif self._in_li and tag == "br":
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._in_li and tag == "a":
            self._in_anchor = False
        elif tag == "li" and self._in_li:
            title = normalize_text("".join(self._title_parts))
            text = normalize_text("".join(self._text_parts))
            if title:
                self.lines[title] = text
            self._in_li = False
            self._in_anchor = False

    def handle_data(self, data: str) -> None:
        if not self._in_li:
            return
        if self._in_anchor:
            self._title_parts.append(data)
        else:
            self._text_parts.append(data)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_html_lines(html: str) -> dict[str, str]:
    parser = LineListParser()
    parser.feed(html)
    parser.close()
    return parser.lines


def convert_file(input_path: Path, output_path: Path) -> int:
    lines = parse_html_lines(input_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lines, indent=2, ensure_ascii=True) + "\n")
    return len(lines)


def convert_transliterated(root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for corpus in CORPORA:
        raw_dir = root / corpus / "raw"
        text_dir = root / corpus / "text"
        corpus_counts: dict[str, int] = {}
        for input_path in sorted(raw_dir.glob("*.html")):
            output_path = text_dir / f"{input_path.stem}.json"
            corpus_counts[input_path.name] = convert_file(input_path, output_path)
        counts[corpus] = corpus_counts
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw transliteration HTML list files into title-to-text JSON."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"Transliterated root. Default: {DEFAULT_ROOT}")
    return parser.parse_args()


def main() -> None:
    counts = convert_transliterated(parse_args().root)
    total_files = sum(len(files) for files in counts.values())
    total_lines = sum(sum(files.values()) for files in counts.values())
    print(f"Wrote {total_files} JSON files with {total_lines} lines")


if __name__ == "__main__":
    main()
