from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = ROOT / "rr_tablets" / "compressed" / "Aa" / "1.webp"
DEFAULT_DICTIONARY = ROOT / "references" / "dictionary" / "split"
DEFAULT_ENV = ROOT / ".env"
DEFAULT_MODEL = "gpt-4o-mini"
MIN_OUTPUT_TOKENS = 16
DEBUG_OUTPUT_TOKENS = 64
SUPPORTED_SUFFIXES = frozenset({".gif", ".jpg", ".jpeg", ".png", ".webp"})
MODES = ("shortcircuit", "tryall")
T = TypeVar("T")


@dataclass(frozen=True)
class DictionaryMatch:
    path: str
    label: str
    matched: bool
    raw: str
    reasoning: str | None = None


@dataclass(frozen=True)
class MatchRun:
    query: str
    mode: str
    checked: int
    first_match: DictionaryMatch | None
    matches: list[DictionaryMatch]


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


def print_iteration_result(result: DictionaryMatch, *, checked: int, total: int, debug: bool, show_progress: bool) -> None:
    status = "true" if result.matched else "false"
    message = f"[{checked}/{total}] {result.label}: {status}"
    if debug and result.reasoning:
        message = f"{message} | {result.reasoning}"
    if show_progress and tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def match_prompt(label: str, *, debug: bool) -> str:
    prompt = (
        "Compare the first image, the query rongorongo glyph, with the second image, "
        f"dictionary entry {label}. The dictionary entry may contain one or more glyphs. "
        "Return true if the query has the same structural feature set as any complete glyph "
        "in the dictionary entry. Tolerate differences in line width, spacing, handwriting, "
        "stroke looseness, small alignment shifts, and the progressive simplification or "
        "laziness that comes from writing the same character repeatedly. The query may also "
        "be partially degraded, expanded, or condensed. Do not require the same exact outline "
        "or shape. Return false if the closest visible glyph lacks a major "
        "feature, adds a major feature, changes the count of important strokes or appendages, "
        "or represents a different character family. "
    )
    if debug:
        return prompt + (
            "Answer in exactly two lines:\n"
            "match: <true or false>\n"
            "reasoning: <one concise sentence explaining the acceptance or rejection, describing the feature difference>"
        )
    return prompt + "Answer with exactly one lowercase word: true or false."


def parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"true", "yes"}:
        return True
    if normalized in {"false", "no"}:
        return False
    raise ValueError(f"Expected true or false, got: {raw!r}")


def parse_debug_response(raw: str) -> tuple[bool, str]:
    matched: bool | None = None
    reasoning = ""
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        normalized_key = key.strip().lower()
        if normalized_key == "match":
            matched = parse_bool(value)
        elif normalized_key == "reasoning":
            reasoning = value.strip()
    if matched is None:
        matched = parse_bool(raw)
    return matched, reasoning


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


def call_match_model(
    client: OpenAI,
    query_path: Path,
    dictionary_path: Path,
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
    debug: bool,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": match_prompt(dictionary_path.stem, debug=debug)},
                    {"type": "input_image", "image_url": data_uri(query_path), "detail": detail},
                    {"type": "input_image", "image_url": data_uri(dictionary_path), "detail": detail},
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return response.output_text


def check_dictionary_entry(
    client: OpenAI,
    query_path: Path,
    dictionary_path: Path,
    *,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
    debug: bool,
) -> DictionaryMatch:
    raw = call_match_model(
        client,
        query_path,
        dictionary_path,
        model=model,
        detail=detail,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        debug=debug,
    )
    matched, reasoning = parse_debug_response(raw) if debug else (parse_bool(raw), None)
    return DictionaryMatch(
        path=str(dictionary_path),
        label=dictionary_path.stem,
        matched=matched,
        raw=raw.strip(),
        reasoning=reasoning,
    )


def match_query_against_dictionary(
    query_path: Path,
    dictionary_dir: Path,
    *,
    mode: str,
    model: str,
    detail: str,
    max_output_tokens: int,
    temperature: float,
    show_progress: bool,
    debug: bool,
) -> MatchRun:
    candidates = dictionary_paths(dictionary_dir)
    if not candidates:
        raise FileNotFoundError(f"No supported dictionary images found in {dictionary_dir}")

    client = openai_client_from_env()
    matches: list[DictionaryMatch] = []
    checked = 0

    total = len(candidates)
    for dictionary_path in progress(candidates, enabled=show_progress, description=f"Matching {query_path.stem}"):
        checked += 1
        result = check_dictionary_entry(
            client,
            query_path,
            dictionary_path,
            model=model,
            detail=detail,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            debug=debug,
        )
        print_iteration_result(result, checked=checked, total=total, debug=debug, show_progress=show_progress)
        if result.matched:
            matches.append(result)
            if mode == "shortcircuit":
                break

    return MatchRun(
        query=str(query_path),
        mode=mode,
        checked=checked,
        first_match=matches[0] if matches else None,
        matches=matches,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one query glyph against dictionary split glyphs with OpenAI vision."
    )
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY, help=f"Query glyph. Default: {DEFAULT_QUERY}")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY, help=f"Dictionary split folder. Default: {DEFAULT_DICTIONARY}")
    parser.add_argument("--mode", choices=MODES, default="shortcircuit", help="Stop on first true match or check all entries.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--detail", choices=("low", "high", "auto"), default="low", help="Vision detail level. Default: low.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help=f"Maximum tokens to generate per comparison. Minimum: {MIN_OUTPUT_TOKENS}. Defaults to {MIN_OUTPUT_TOKENS}, or {DEBUG_OUTPUT_TOKENS} in debug mode.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model sampling temperature.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the per-dictionary-glyph progress bar.")
    parser.add_argument("--debug", action="store_true", help="Ask for and print short reasoning with each true/false decision.")
    args = parser.parse_args()
    args.max_output_tokens = args.max_output_tokens or (DEBUG_OUTPUT_TOKENS if args.debug else MIN_OUTPUT_TOKENS)
    if args.max_output_tokens < MIN_OUTPUT_TOKENS:
        parser.error(f"--max-output-tokens must be >= {MIN_OUTPUT_TOKENS}")
    return args


def main() -> None:
    args = parse_args()
    run = match_query_against_dictionary(
        query_path=args.query,
        dictionary_dir=args.dictionary,
        mode=args.mode,
        model=args.model,
        detail=args.detail,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        show_progress=not args.no_progress,
        debug=args.debug,
    )
    print(json.dumps(asdict(run), indent=2))


if __name__ == "__main__":
    main()
