from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, TypedDict


class GlyphVariantToken(TypedDict):
    glyph: str
    variant: str
    modifiers: list[str]


GLYPH_VARIANT_STRATEGY = "glyph_variants"


def glyph_variant_vocab_key(glyph: str, variant: str = "", modifiers: Sequence[str] | None = None) -> str:
    """Stable flat vocab id: 076, 076:a, 076:a+x,y, 076:+h,x."""
    normalized_glyph = glyph.zfill(3) if glyph.isdigit() else glyph
    mods = list(modifiers or [])
    if not variant and not mods:
        return normalized_glyph
    body = variant
    if mods:
        mod = ",".join(mods)
        body = f"{body}+{mod}" if body else f"+{mod}"
    return f"{normalized_glyph}:{body}"


def parse_glyph_variant_vocab_key(key: str) -> GlyphVariantToken:
    if ":" not in key:
        return {"glyph": key.zfill(3) if key.isdigit() else key, "variant": "", "modifiers": []}
    glyph, body = key.split(":", 1)
    glyph = glyph.zfill(3) if glyph.isdigit() else glyph
    if body.startswith("+"):
        return {"glyph": glyph, "variant": "", "modifiers": body[1:].split(",") if body[1:] else []}
    if "+" in body:
        variant, mod = body.split("+", 1)
        return {"glyph": glyph, "variant": variant, "modifiers": mod.split(",") if mod else []}
    return {"glyph": glyph, "variant": body, "modifiers": []}


def glyph_variant_token_to_key(token: GlyphVariantToken | dict[str, Any]) -> str:
    return glyph_variant_vocab_key(
        str(token["glyph"]),
        str(token.get("variant", "")),
        list(token.get("modifiers", [])),
    )


def load_glyph_variant_sequences(tokenized_dir: Path) -> list[list[str]]:
    sequences: list[list[str]] = []
    for path in sorted(tokenized_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        lines: dict[str, list[GlyphVariantToken]] = json.loads(path.read_text())
        for tokens in lines.values():
            if not isinstance(tokens, list) or not tokens:
                continue
            sequences.append([glyph_variant_token_to_key(token) for token in tokens])
    return sequences


def resolve_anchor_vocab_key(glyph: str, vocab: Sequence[str]) -> str | None:
    """Map gloss anchor glyph (e.g. 076) to a vocab row, preferring the plain unmodified form."""
    glyph = glyph.zfill(3)
    vocab_set = set(vocab)
    if glyph in vocab_set:
        return glyph
    matches = sorted(v for v in vocab if v == glyph or v.startswith(f"{glyph}:"))
    return matches[0] if matches else None


def resolve_anchor_token_index(glyph: str, token_to_idx: dict[str, int]) -> int | None:
    key = resolve_anchor_vocab_key(glyph, list(token_to_idx))
    return token_to_idx.get(key) if key else None
