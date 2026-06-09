from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "rr_tablets" / "transliterated"
CORPORA = ("complete", "broken")
TOKEN_PATTERN = re.compile(r"\([^)]+\)[A-Za-z?!*]*|\d+[A-Za-z]*[?!*]*")
TOKEN_OR_SEPARATOR_PATTERN = re.compile(r"\([^)]+\)[A-Za-z?!*]*|\d+[A-Za-z]*[?!*]*|[-.:]")
NUMERIC_TOKEN_PATTERN = re.compile(r"\d+")
NUMERIC_TOKEN_OR_SEPARATOR_PATTERN = re.compile(r"\d+[A-Za-z]*[?!*]*|[-.:]")
GLYPH = r"(?:\([^)]+\)[A-Za-z]*|\d+[A-Za-z]*)"
LIGATURE_PATTERN = re.compile(rf"{GLYPH}(?:[.:]{GLYPH})*")
NUMERIC_LIGATURE_PATTERN = re.compile(r"\d+[A-Za-z]*(?:[.:]\d+[A-Za-z]*)*")
NUMERIC_COMPONENT_PATTERN = re.compile(r"\d+")
TEXT_MARKERS = str.maketrans("", "", "!*?()")
NOAG_MARKERS = str.maketrans("", "", "abcdefgV")


class GlyphVariantToken(TypedDict):
    glyph: str
    variants: list[str]


class TransliterationTokenizer:
    def __init__(self) -> None:
        self.strategies: dict[str, Callable[[str], Any]] = {
            "barthel": self.tokenize_glyphs,
            "barthel_separators": self.tokenize_glyphs_with_separators,
            "barthel_ligatures": self.tokenize_ligatures,
            "barthel_noag": self.tokenize_barthel_noag,
            "barthel_separators_noag": self.tokenize_barthel_separators_noag,
            "barthel_ligatures_noag": self.tokenize_barthel_ligatures_noag,
            "suffix": self.tokenize_suffix,
            "suffix_separators": self.tokenize_suffix_separators,
            "suffix_ligatures": self.tokenize_suffix_ligatures,
            "suffix_noag": self.tokenize_suffix_noag,
            "simple": self.tokenize_simple,
            "simple_separators": self.tokenize_simple_with_separators,
            "simple_ligatures": self.tokenize_simple_ligatures,
            "glyph_variants": self.tokenize_glyph_variants,
            "normalized_text": self.tokenize_normalized_text,
            "characters": self.tokenize_characters,
        }

    @property
    def strategy_names(self) -> tuple[str, ...]:
        return tuple(self.strategies)

    def tokenize(self, text: str, strategy: str) -> Any:
        try:
            tokenizer = self.strategies[strategy]
        except KeyError as error:
            raise ValueError(f"Unknown tokenizer strategy: {strategy}") from error
        return tokenizer(clean_text(text))

    def tokenize_glyphs(self, text: str) -> list[str]:
        return [normalize_numbers_in_token(token) for token in TOKEN_PATTERN.findall(text)]

    def tokenize_glyphs_with_separators(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in TOKEN_OR_SEPARATOR_PATTERN.findall(text):
            tokens.append(token if token in "-.:" else normalize_numbers_in_token(token))
        return tokens

    def tokenize_ligatures(self, text: str) -> list[str]:
        return [normalize_numbers_in_token(token) for token in LIGATURE_PATTERN.findall(text)]

    def tokenize_barthel_noag(self, text: str) -> list[str]:
        return self.tokenize_glyphs(strip_noag(text))

    def tokenize_barthel_separators_noag(self, text: str) -> list[str]:
        return self.tokenize_glyphs_with_separators(strip_noag(text))

    def tokenize_barthel_ligatures_noag(self, text: str) -> list[str]:
        return self.tokenize_ligatures(strip_noag(text))

    def tokenize_suffix(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in TOKEN_PATTERN.findall(text):
            tokens.extend(split_suffix_token(token))
        return tokens

    def tokenize_suffix_separators(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in TOKEN_OR_SEPARATOR_PATTERN.findall(text):
            if token in "-.:":
                tokens.append(token)
            else:
                tokens.extend(split_suffix_token(token))
        return tokens

    def tokenize_suffix_ligatures(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in LIGATURE_PATTERN.findall(text):
            tokens.extend(split_suffix_token(token))
        return tokens

    def tokenize_suffix_noag(self, text: str) -> list[str]:
        return self.tokenize_suffix(strip_noag(text))

    def tokenize_simple(self, text: str) -> list[str]:
        return [normalize_code(match) for match in NUMERIC_TOKEN_PATTERN.findall(text)]

    def tokenize_simple_with_separators(self, text: str) -> list[str]:
        """Simple numeric codes with `.`/`:` markers; hyphen is a boundary only (not a token)."""
        tokens: list[str] = []
        for token in NUMERIC_TOKEN_OR_SEPARATOR_PATTERN.findall(text):
            if token == "-":
                continue
            tokens.append(token if token in ".:" else normalize_code(token))
        return tokens

    def tokenize_simple_ligatures(self, text: str) -> list[str]:
        tokens: list[str] = []
        for match in NUMERIC_LIGATURE_PATTERN.finditer(text):
            token = match.group(0)
            parts = re.split(r"([.:])", token)
            tokens.append("".join(normalize_code(part) if part not in ".:" else part for part in parts))
        return tokens

    def tokenize_glyph_variants(self, text: str) -> list[GlyphVariantToken]:
        """One token per glyph occurrence: 3-digit code plus individual suffix letters as variants."""
        return [parse_glyph_variants_token(token) for token in TOKEN_PATTERN.findall(text)]

    def tokenize_normalized_text(self, text: str) -> str:
        return normalize_text(text)

    def tokenize_characters(self, text: str) -> list[str]:
        return list(normalize_text(text))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_text(value: str) -> str:
    return value.translate(TEXT_MARKERS)


def strip_noag(value: str) -> str:
    return value.translate(NOAG_MARKERS)


def normalize_code(value: str) -> str:
    match = NUMERIC_COMPONENT_PATTERN.search(value)
    if not match:
        raise ValueError(f"Expected numeric glyph code in {value!r}")
    return match.group(0).zfill(3)


def normalize_numbers_in_token(value: str) -> str:
    return re.sub(r"\d+", lambda match: match.group(0).zfill(3), value)


def split_suffix_token(value: str) -> list[str]:
    base = normalize_numbers_in_token(re.sub(r"[A-Za-z]+", "", value))
    suffix = re.findall(r"[A-Za-z]", value)
    return [base, *suffix] if base else suffix


def parse_glyph_variants_token(value: str) -> GlyphVariantToken:
    """Split a raw transliteration token into a zero-padded glyph code and suffix letter variants."""
    letters = re.findall(r"[A-Za-z]", value)
    match = NUMERIC_COMPONENT_PATTERN.search(value)
    glyph = match.group(0).zfill(3) if match else ""
    return {"glyph": glyph, "variants": letters}


def tokenize_file(input_path: Path, output_path: Path, tokenizer: TransliterationTokenizer, strategy: str) -> int:
    lines = json.loads(input_path.read_text())
    tokenized = {
        title: tokenizer.tokenize(text, strategy)
        for title, text in lines.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tokenized, indent=2, ensure_ascii=True) + "\n")
    return len(tokenized)


def tokenize_transliterated(root: Path, strategy: str) -> dict[str, dict[str, int]]:
    tokenizer = TransliterationTokenizer()
    counts: dict[str, dict[str, int]] = {}
    for corpus in CORPORA:
        text_dir = root / corpus / "text"
        output_dir = (
            text_dir / strategy
            if strategy in {"normalized_text", "characters"}
            else root / corpus / "tokenized" / strategy
        )
        corpus_counts: dict[str, int] = {}
        for input_path in sorted(text_dir.glob("*.json")):
            output_path = output_dir / input_path.name
            corpus_counts[input_path.name] = tokenize_file(input_path, output_path, tokenizer, strategy)
        counts[corpus] = corpus_counts
    return counts


def parse_args() -> argparse.Namespace:
    tokenizer = TransliterationTokenizer()
    parser = argparse.ArgumentParser(
        description="Tokenize transliterated rongorongo text JSON files with a selected strategy."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"Transliterated root. Default: {DEFAULT_ROOT}")
    parser.add_argument("--strategy", choices=tokenizer.strategy_names, default="barthel", help="Tokenizer strategy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = tokenize_transliterated(args.root, args.strategy)
    total_files = sum(len(files) for files in counts.values())
    total_lines = sum(sum(files.values()) for files in counts.values())
    print(f"Wrote {total_files} {args.strategy!r} tokenized JSON files with {total_lines} lines")


if __name__ == "__main__":
    main()
