from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EMBED_DIR = Path(__file__).resolve().parent
TUNE_DIR = EMBED_DIR / "tuning"
NEIGHBORS_DIR = TUNE_DIR / "trial_neighbors"
TRIAL_OUTPUTS_DIR = TUNE_DIR / "trial_outputs"
RESULTS_PATH = TUNE_DIR / "trials.jsonl"
SUMMARY_PATH = TUNE_DIR / "best_summary.json"
TOKENIZED_ROOT = ROOT / "rr_tablets" / "transliterated" / "complete" / "tokenized"
ANCHORS_FILE = EMBED_DIR / "gloss_anchors.json"

MODEL_PRESETS: dict[str, str] = {
    "e5-small": "intfloat/e5-small-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert": "distilbert-base-uncased",
}

ANCHOR_EXPECTED: dict[str, set[str]] = {
    "008": {"sun", "day", "light", "star"},
    "200": {"king", "chief", "queen", "prince", "ruler"},
    "076": {"father", "fathers", "copulated", "copulate", "copulation", "penis", "phallus"},
    "040": {"night", "dark", "evening", "moon"},
}


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    strategy: str
    model_preset: str
    epochs: int
    anchor_weight: float
    projected_cooccurrence_weight: float
    collapse_weight: float
    anchor_passes: int
    learning_rate: float = 0.01
    batch_size: int = 512
    window: int = 4
    negatives: int = 16
    collapse_sample_size: int = 512
    seed: int = 7


def load_learn_module():
    path = EMBED_DIR / "learn_projection_to_transformer_space.py"
    spec = importlib.util.spec_from_file_location("learn_projection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["learn_projection"] = module
    spec.loader.exec_module(module)
    return module


def load_plot_helpers():
    path = EMBED_DIR / "plot_projection_graphs.py"
    spec = importlib.util.spec_from_file_location("plot_projection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_trial_grid() -> list[TrialConfig]:
    hypers: list[tuple[float, float, float, int]] = [
        (8.0, 0.35, 0.10, 16),
        (6.0, 0.35, 0.10, 12),
        (6.0, 0.50, 0.15, 12),
        (5.0, 0.45, 0.12, 12),
        (4.0, 0.50, 0.10, 10),
        (7.0, 0.30, 0.08, 14),
        (6.0, 0.25, 0.05, 12),
        (8.0, 0.50, 0.15, 12),
    ]
    trials: list[TrialConfig] = []
    counter = 0
    for strategy in ("barthel", "simple"):
        for model_preset in ("e5-small", "bge-small", "minilm"):
            for anchor_w, cooc_w, collapse_w, passes in hypers:
                counter += 1
                epochs = 16 if strategy == "barthel" and counter % 3 == 0 else 12
                trials.append(
                    TrialConfig(
                        trial_id=f"t{counter:03d}",
                        strategy=strategy,
                        model_preset=model_preset,
                        epochs=epochs,
                        anchor_weight=anchor_w,
                        projected_cooccurrence_weight=cooc_w,
                        collapse_weight=collapse_w,
                        anchor_passes=passes,
                    )
                )
    trials.append(
        TrialConfig(
            trial_id="long_e5_barthel",
            strategy="barthel",
            model_preset="e5-small",
            epochs=24,
            anchor_weight=6.0,
            projected_cooccurrence_weight=0.35,
            collapse_weight=0.10,
            anchor_passes=14,
        )
    )
    trials.append(
        TrialConfig(
            trial_id="long_bge_barthel",
            strategy="barthel",
            model_preset="bge-small",
            epochs=24,
            anchor_weight=6.0,
            projected_cooccurrence_weight=0.35,
            collapse_weight=0.10,
            anchor_passes=14,
        )
    )
    return trials


SIMPLE_VARIANT_STRATEGIES = ("simple", "simple_separators", "simple_ligatures")
SUFFIX_VARIANT_STRATEGIES = ("suffix", "suffix_separators", "suffix_ligatures", "suffix_noag")
BARTHEL_VARIANT_STRATEGIES = (
    "barthel",
    "barthel_separators",
    "barthel_ligatures",
    "barthel_noag",
    "barthel_separators_noag",
    "barthel_ligatures_noag",
)


def _enumerate_hyper_candidates() -> list[tuple[float, float, float, int, int]]:
    """Anchor/cooc/collapse/passes/epochs tuples around prior best (e5 + simple)."""
    aw, co, cw, ap, ep = 8.0, 0.5, 0.15, 12, 12
    seen: set[tuple[float, float, float, int, int]] = set()
    out: list[tuple[float, float, float, int, int]] = []

    def add(anchor_weight: float, cooc: float, collapse: float, passes: int, epochs: int) -> None:
        key = (anchor_weight, cooc, collapse, passes, epochs)
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    add(aw, co, cw, ap, ep)
    for anchor_weight in (3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0, 14.0):
        add(anchor_weight, co, cw, ap, ep)
    for cooc in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70):
        add(aw, cooc, cw, ap, ep)
    for collapse in (0.03, 0.05, 0.08, 0.10, 0.12, 0.18, 0.20, 0.22, 0.25):
        add(aw, co, collapse, ap, ep)
    for passes in (6, 8, 10, 14, 16, 20):
        add(aw, co, cw, passes, ep)
    for epochs in (9, 10, 11, 13, 14, 15, 16):
        add(aw, co, cw, ap, epochs)
    for anchor_weight, cooc, collapse, passes, epochs in (
        (6.0, 0.35, 0.10, 12, 12),
        (7.0, 0.30, 0.08, 14, 12),
        (5.0, 0.45, 0.12, 12, 12),
        (4.0, 0.50, 0.10, 10, 12),
        (10.0, 0.40, 0.12, 14, 11),
        (8.0, 0.35, 0.10, 16, 12),
        (6.0, 0.25, 0.05, 12, 12),
        (9.0, 0.55, 0.18, 10, 13),
        (7.0, 0.50, 0.15, 8, 12),
        (8.0, 0.60, 0.08, 12, 14),
        (5.0, 0.60, 0.20, 14, 10),
        (12.0, 0.45, 0.12, 12, 12),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 11),
        (8.0, 0.50, 0.15, 12, 13),
        (4.0, 0.70, 0.08, 12, 12),
        (12.0, 0.25, 0.10, 12, 12),
        (8.0, 0.50, 0.05, 16, 12),
        (8.0, 0.50, 0.25, 8, 12),
        (7.0, 0.45, 0.12, 14, 14),
        (9.0, 0.35, 0.15, 10, 12),
    ):
        add(anchor_weight, cooc, collapse, passes, epochs)
    return out


def _append_trials(
    trials: list[TrialConfig],
    seen: set[tuple[Any, ...]],
    counter: list[int],
    *,
    strategy: str,
    model_preset: str,
    hypers: list[tuple[float, float, float, int, int]],
    learning_rate: float = 0.01,
    prefix: str,
) -> None:
    for anchor_weight, cooc, collapse, passes, epochs in hypers:
        key = (strategy, model_preset, anchor_weight, cooc, collapse, passes, epochs, learning_rate)
        if key in seen:
            continue
        seen.add(key)
        counter[0] += 1
        trials.append(
            TrialConfig(
                trial_id=f"{prefix}{counter[0]:04d}",
                strategy=strategy,
                model_preset=model_preset,
                epochs=epochs,
                anchor_weight=anchor_weight,
                projected_cooccurrence_weight=cooc,
                collapse_weight=collapse,
                anchor_passes=passes,
                learning_rate=learning_rate,
            )
        )


def build_simple_variants_grid() -> list[TrialConfig]:
    """Overnight grid: simple / simple_separators / simple_ligatures × e5 + minilm + bge."""
    hypers = _enumerate_hyper_candidates()
    minilm_hypers = hypers[::2]  # every other tuple
    bge_hypers = hypers[::4] + hypers[1::4][:8]  # ~half, capped spread
    lr_variants = (0.005, 0.015, 0.02)
    trials: list[TrialConfig] = []
    seen: set[tuple[Any, ...]] = set()
    counter = [0]

    for strategy in SIMPLE_VARIANT_STRATEGIES:
        _append_trials(trials, seen, counter, strategy=strategy, model_preset="e5-small", hypers=hypers, prefix="sv_e5_")
        _append_trials(trials, seen, counter, strategy=strategy, model_preset="minilm", hypers=minilm_hypers, prefix="sv_mn_")
        _append_trials(trials, seen, counter, strategy=strategy, model_preset="bge-small", hypers=bge_hypers, prefix="sv_bg_")
        for learning_rate in lr_variants:
            _append_trials(
                trials,
                seen,
                counter,
                strategy=strategy,
                model_preset="e5-small",
                hypers=[(8.0, 0.5, 0.15, 12, 12)],
                learning_rate=learning_rate,
                prefix="sv_lr_",
            )
    return trials


def build_simple_variants_focused_grid() -> list[TrialConfig]:
    """Focused grid from partial simple_variants winners; saves neighbor JSONs per trial."""
    simple_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.12, 12, 12),
        (7.0, 0.50, 0.15, 12, 12),
        (9.0, 0.50, 0.15, 12, 12),
        (8.0, 0.55, 0.15, 12, 12),
        (8.0, 0.45, 0.15, 12, 12),
        (8.0, 0.50, 0.10, 12, 12),
        (8.0, 0.50, 0.18, 12, 12),
        (8.0, 0.50, 0.15, 10, 12),
        (8.0, 0.50, 0.15, 14, 12),
        (8.0, 0.50, 0.15, 12, 11),
        (8.0, 0.50, 0.15, 12, 13),
        (6.0, 0.55, 0.15, 12, 12),
        (8.0, 0.60, 0.12, 12, 12),
        (7.0, 0.55, 0.12, 12, 12),
    ]
    separators_hypers = [
        (8.0, 0.50, 0.15, 12, 15),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.15, 12, 16),
        (7.0, 0.50, 0.15, 12, 15),
        (9.0, 0.50, 0.15, 12, 15),
        (8.0, 0.45, 0.15, 12, 15),
        (8.0, 0.55, 0.15, 12, 15),
        (8.0, 0.50, 0.12, 12, 15),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.50, 0.10, 12, 15),
        (6.0, 0.50, 0.15, 12, 15),
        (8.0, 0.50, 0.15, 14, 15),
    ]
    ligatures_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.55, 0.15, 12, 12),
        (8.0, 0.45, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 10, 12),
        (8.0, 0.50, 0.15, 14, 12),
        (8.0, 0.50, 0.15, 12, 13),
        (7.0, 0.50, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 15),
    ]
    minilm_simple_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (8.0, 0.65, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 10),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
    ]
    trials: list[TrialConfig] = []
    seen: set[tuple[Any, ...]] = set()
    counter = [0]
    _append_trials(trials, seen, counter, strategy="simple", model_preset="e5-small", hypers=simple_hypers, prefix="svf_e5_")
    _append_trials(
        trials, seen, counter, strategy="simple_separators", model_preset="e5-small", hypers=separators_hypers, prefix="svf_sep_"
    )
    _append_trials(
        trials, seen, counter, strategy="simple_ligatures", model_preset="e5-small", hypers=ligatures_hypers, prefix="svf_lig_"
    )
    _append_trials(trials, seen, counter, strategy="simple", model_preset="minilm", hypers=minilm_simple_hypers, prefix="svf_mn_")
    return trials


def build_suffix_variants_focused_grid() -> list[TrialConfig]:
    """Same hyperparameter blocks as simple_variants_focused, with suffix tokenizers."""
    simple_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.12, 12, 12),
        (7.0, 0.50, 0.15, 12, 12),
        (9.0, 0.50, 0.15, 12, 12),
        (8.0, 0.55, 0.15, 12, 12),
        (8.0, 0.45, 0.15, 12, 12),
        (8.0, 0.50, 0.10, 12, 12),
        (8.0, 0.50, 0.18, 12, 12),
        (8.0, 0.50, 0.15, 10, 12),
        (8.0, 0.50, 0.15, 14, 12),
        (8.0, 0.50, 0.15, 12, 11),
        (8.0, 0.50, 0.15, 12, 13),
        (6.0, 0.55, 0.15, 12, 12),
        (8.0, 0.60, 0.12, 12, 12),
        (7.0, 0.55, 0.12, 12, 12),
    ]
    separators_hypers = [
        (8.0, 0.50, 0.15, 12, 15),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.15, 12, 16),
        (7.0, 0.50, 0.15, 12, 15),
        (9.0, 0.50, 0.15, 12, 15),
        (8.0, 0.45, 0.15, 12, 15),
        (8.0, 0.55, 0.15, 12, 15),
        (8.0, 0.50, 0.12, 12, 15),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.50, 0.10, 12, 15),
        (6.0, 0.50, 0.15, 12, 15),
        (8.0, 0.50, 0.15, 14, 15),
    ]
    ligatures_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.55, 0.15, 12, 12),
        (8.0, 0.45, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 10, 12),
        (8.0, 0.50, 0.15, 14, 12),
        (8.0, 0.50, 0.15, 12, 13),
        (7.0, 0.50, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 15),
    ]
    noag_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (8.0, 0.65, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 10),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
    ]
    trials: list[TrialConfig] = []
    seen: set[tuple[Any, ...]] = set()
    counter = [0]
    _append_trials(trials, seen, counter, strategy="suffix", model_preset="e5-small", hypers=simple_hypers, prefix="svfx_e5_")
    _append_trials(
        trials, seen, counter, strategy="suffix_separators", model_preset="e5-small", hypers=separators_hypers, prefix="svfx_sep_"
    )
    _append_trials(
        trials, seen, counter, strategy="suffix_ligatures", model_preset="e5-small", hypers=ligatures_hypers, prefix="svfx_lig_"
    )
    _append_trials(
        trials, seen, counter, strategy="suffix_noag", model_preset="e5-small", hypers=noag_hypers, prefix="svfx_ng_"
    )
    return trials


def build_barthel_variants_focused_grid() -> list[TrialConfig]:
    """Focused grid across all six Barthel tokenizer variants (48 trials, e5-small)."""
    base_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.12, 12, 12),
        (7.0, 0.50, 0.15, 12, 12),
        (9.0, 0.50, 0.15, 12, 12),
        (8.0, 0.55, 0.15, 12, 12),
        (8.0, 0.45, 0.15, 12, 12),
        (8.0, 0.50, 0.10, 12, 12),
        (8.0, 0.50, 0.18, 12, 12),
        (8.0, 0.50, 0.15, 10, 12),
    ]
    separators_hypers = [
        (8.0, 0.50, 0.15, 12, 15),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.15, 12, 16),
        (7.0, 0.50, 0.15, 12, 15),
        (9.0, 0.50, 0.15, 12, 15),
        (8.0, 0.45, 0.15, 12, 15),
        (8.0, 0.55, 0.15, 12, 15),
        (8.0, 0.50, 0.12, 12, 15),
    ]
    ligatures_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 14),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.55, 0.15, 12, 12),
        (8.0, 0.45, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 10, 12),
    ]
    noag_hypers = [
        (8.0, 0.50, 0.15, 12, 12),
        (8.0, 0.65, 0.15, 12, 12),
        (8.0, 0.50, 0.15, 12, 10),
        (6.0, 0.50, 0.15, 12, 12),
        (8.0, 0.50, 0.12, 12, 12),
        (8.0, 0.60, 0.15, 12, 12),
    ]
    trials: list[TrialConfig] = []
    seen: set[tuple[Any, ...]] = set()
    counter = [0]
    _append_trials(trials, seen, counter, strategy="barthel", model_preset="e5-small", hypers=base_hypers, prefix="svfb_bh_")
    _append_trials(
        trials,
        seen,
        counter,
        strategy="barthel_separators",
        model_preset="e5-small",
        hypers=separators_hypers,
        prefix="svfb_sep_",
    )
    _append_trials(
        trials,
        seen,
        counter,
        strategy="barthel_ligatures",
        model_preset="e5-small",
        hypers=ligatures_hypers,
        prefix="svfb_lig_",
    )
    _append_trials(
        trials, seen, counter, strategy="barthel_noag", model_preset="e5-small", hypers=noag_hypers, prefix="svfb_ng_"
    )
    _append_trials(
        trials,
        seen,
        counter,
        strategy="barthel_separators_noag",
        model_preset="e5-small",
        hypers=separators_hypers,
        prefix="svfb_sng_",
    )
    _append_trials(
        trials,
        seen,
        counter,
        strategy="barthel_ligatures_noag",
        model_preset="e5-small",
        hypers=noag_hypers,
        prefix="svfb_lng_",
    )
    return trials


def build_simple_focused_grid() -> list[TrialConfig]:
    """Focused search around best simple+e5-small config (t032)."""
    center = {
        "epochs": 12,
        "anchor_weight": 8.0,
        "projected_cooccurrence_weight": 0.5,
        "collapse_weight": 0.15,
        "anchor_passes": 12,
        "learning_rate": 0.01,
    }
    seen: set[tuple[Any, ...]] = set()
    trials: list[TrialConfig] = []

    def add(trial_id: str, model_preset: str, **overrides: Any) -> None:
        params = {**center, **overrides}
        key = (
            model_preset,
            params["epochs"],
            params["anchor_weight"],
            params["projected_cooccurrence_weight"],
            params["collapse_weight"],
            params["anchor_passes"],
            params["learning_rate"],
        )
        if key in seen:
            return
        seen.add(key)
        trials.append(
            TrialConfig(
                trial_id=trial_id,
                strategy="simple",
                model_preset=model_preset,
                epochs=int(params["epochs"]),
                anchor_weight=float(params["anchor_weight"]),
                projected_cooccurrence_weight=float(params["projected_cooccurrence_weight"]),
                collapse_weight=float(params["collapse_weight"]),
                anchor_passes=int(params["anchor_passes"]),
                learning_rate=float(params["learning_rate"]),
            )
        )

    add("s_center_e5", "e5-small")
    for anchor_weight in (3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0, 14.0):
        add(f"s_aw_{anchor_weight:g}", "e5-small", anchor_weight=anchor_weight)
    for cooc in (0.20, 0.30, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        add(f"s_co_{cooc:g}", "e5-small", projected_cooccurrence_weight=cooc)
    for collapse in (0.03, 0.05, 0.08, 0.10, 0.12, 0.18, 0.22, 0.28):
        add(f"s_cw_{collapse:g}", "e5-small", collapse_weight=collapse)
    for passes in (4, 6, 8, 10, 14, 16, 20, 24):
        add(f"s_ap_{passes}", "e5-small", anchor_passes=passes)
    for epochs in (8, 10, 11, 13, 14, 15, 16, 18, 20):
        add(f"s_ep_{epochs}", "e5-small", epochs=epochs)
    for learning_rate in (0.005, 0.015, 0.02):
        add(f"s_lr_{learning_rate:g}", "e5-small", learning_rate=learning_rate)

    # Corner / boundary probes around the winning basin.
    corners: list[tuple[str, dict[str, Any]]] = [
        ("s_lo_anchor_hi_cooc", {"anchor_weight": 4.0, "projected_cooccurrence_weight": 0.7, "collapse_weight": 0.08}),
        ("s_hi_anchor_lo_cooc", {"anchor_weight": 12.0, "projected_cooccurrence_weight": 0.25, "collapse_weight": 0.10}),
        ("s_tight_collapse", {"anchor_weight": 8.0, "projected_cooccurrence_weight": 0.5, "collapse_weight": 0.05, "anchor_passes": 16}),
        ("s_loose_collapse", {"anchor_weight": 8.0, "projected_cooccurrence_weight": 0.5, "collapse_weight": 0.25, "anchor_passes": 8}),
        ("s_fast_lr", {"learning_rate": 0.02, "epochs": 10}),
        ("s_slow_lr", {"learning_rate": 0.005, "epochs": 14}),
        ("s_long_train", {"epochs": 18, "anchor_weight": 7.0, "anchor_passes": 14}),
        ("s_short_train", {"epochs": 8, "anchor_weight": 9.0, "anchor_passes": 8}),
        ("s_max_explore", {"anchor_weight": 14.0, "projected_cooccurrence_weight": 0.85, "collapse_weight": 0.03, "epochs": 16}),
        ("s_min_explore", {"anchor_weight": 3.0, "projected_cooccurrence_weight": 0.20, "collapse_weight": 0.28, "epochs": 10}),
    ]
    for trial_id, overrides in corners:
        add(trial_id, "e5-small", **overrides)

    # Cross-check best basin on minilm (strong anchor semantics in prior search).
    add("s_center_minilm", "minilm")
    for anchor_weight in (6.0, 8.0, 10.0):
        add(f"s_minilm_aw_{anchor_weight:g}", "minilm", anchor_weight=anchor_weight)
    for cooc in (0.35, 0.5):
        add(f"s_minilm_co_{cooc:g}", "minilm", projected_cooccurrence_weight=cooc, anchor_weight=8.0)
    return trials


def anchor_hit_rate(anchor_top1: dict[str, list[tuple[str, float]]]) -> float:
    hits = 0
    total = 0
    for token, expected in ANCHOR_EXPECTED.items():
        if token not in anchor_top1 or not anchor_top1[token]:
            continue
        total += 1
        word = anchor_top1[token][0][0].lower()
        if word in expected:
            hits += 1
    return hits / total if total else 0.0


def score_run(metrics: dict[str, Any], anchor_gloss_cosine: dict[str, float], anchor_top1: dict[str, list]) -> float:
    anchor_values = list(anchor_gloss_cosine.values())
    anchor_cos_mean = float(np.mean(anchor_values)) if anchor_values else -1.0
    unique_top1 = float(metrics["unique_top1_count"])
    token_count = max(float(metrics["token_count"]), 1.0)
    entropy = float(metrics["top1_entropy_bits"])
    diversity = min(unique_top1 / 80.0, 1.5)
    entropy_norm = min(entropy / 4.0, 1.0)
    hit_rate = anchor_hit_rate(anchor_top1)
    margin = float(metrics["top1_top2_margin_mean"])

    score = (
        3.0 * anchor_cos_mean
        + 2.0 * hit_rate
        + 1.2 * diversity
        + 0.6 * entropy_norm
        + 0.4 * margin
    )
    if anchor_cos_mean < 0.0:
        score -= 2.5
    if unique_top1 <= 5:
        score -= 1.5
    if hit_rate < 0.5:
        score -= 1.0
    return float(score)


def save_trial_pipeline_outputs(
    trial: TrialConfig,
    learn,
    plot,
    model,
    vocab: list[str],
    transformer_name: str,
    losses: list[float],
    anchor_records: list[tuple[int, list[str]]],
    outputs_dir: Path,
    english_word_count: int,
) -> dict[str, str]:
    trial_dir = outputs_dir / trial.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    training_config: dict[str, Any] = {
        "trial_id": trial.trial_id,
        "epochs": trial.epochs,
        "anchor_weight": trial.anchor_weight,
        "projected_cooccurrence_weight": trial.projected_cooccurrence_weight,
        "collapse_weight": trial.collapse_weight,
        "anchor_passes": trial.anchor_passes,
        "learning_rate": trial.learning_rate,
        "batch_size": trial.batch_size,
        "window": trial.window,
        "negatives": trial.negatives,
        "collapse_sample_size": trial.collapse_sample_size,
        "seed": trial.seed,
    }
    learn.save_outputs(
        model,
        vocab,
        trial_dir,
        transformer_name,
        trial.strategy,
        losses,
        anchor_records,
        training_config,
    )
    model_slug = learn.sanitize_model_name(transformer_name)
    projection_file = trial_dir / f"{model_slug}_{trial.strategy}_projection.npz"
    metadata_file = trial_dir / f"{model_slug}_{trial.strategy}_projection_metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["trial_id"] = trial.trial_id
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    graph_dir = trial_dir / "graphs"
    plot.render_projection_graphs(
        projection_file,
        metadata_file,
        graph_dir,
        english_word_count=english_word_count,
    )
    return {
        "trial_output_dir": str(trial_dir),
        "projection_file": str(projection_file),
        "metadata_file": str(metadata_file),
        "graphs_dir": str(graph_dir),
        "neighbors_path": str(graph_dir / "rongo_topk_english_neighbors.json"),
        "eval_metrics_path": str(graph_dir / "rongo_eval_metrics.json"),
    }


def run_trial(
    trial: TrialConfig,
    learn,
    plot,
    *,
    save_artifacts: bool = True,
    outputs_dir: Path = TRIAL_OUTPUTS_DIR,
    english_word_count: int = 5000,
) -> dict[str, Any]:
    tokenized_dir = TOKENIZED_ROOT / trial.strategy
    transformer_name = MODEL_PRESETS[trial.model_preset]
    sequences = learn.load_sequences(tokenized_dir)
    vocab, token_to_idx = learn.build_vocab(sequences)
    pairs = learn.context_pairs(sequences, token_to_idx, window=trial.window, mask_separators=True)
    tokenizer, transformer_embeddings = learn.load_transformer_embeddings(transformer_name)
    anchor_targets = learn.build_anchor_targets(
        learn.load_gloss_anchors(ANCHORS_FILE),
        token_to_idx,
        tokenizer,
        transformer_embeddings,
    )
    model = learn.ProjectionModel.create(
        vocab_size=len(vocab),
        input_dim=128,
        hidden_dim=256,
        output_dim=transformer_embeddings.shape[1],
        seed=trial.seed,
    )
    started = time.time()
    losses = learn.train_projection(
        model=model,
        pairs=pairs,
        anchor_targets=anchor_targets,
        epochs=trial.epochs,
        batch_size=trial.batch_size,
        negatives=trial.negatives,
        learning_rate=trial.learning_rate,
        anchor_weight=trial.anchor_weight,
        projected_cooccurrence_weight=trial.projected_cooccurrence_weight,
        collapse_weight=trial.collapse_weight,
        anchor_passes=trial.anchor_passes,
        collapse_sample_size=trial.collapse_sample_size,
        seed=trial.seed,
    )
    anchor_records = [(token_idx, glosses) for token_idx, _, glosses in anchor_targets]
    artifact_paths: dict[str, str] = {}
    if save_artifacts:
        artifact_paths = save_trial_pipeline_outputs(
            trial,
            learn,
            plot,
            model,
            vocab,
            transformer_name,
            losses,
            anchor_records,
            outputs_dir,
            english_word_count,
        )
        metrics = json.loads(Path(artifact_paths["eval_metrics_path"]).read_text())
    else:
        projected = learn.normalize(learn.relu(model.token_embeddings @ model.w1 + model.b1) @ model.w2 + model.b2)
        projected = learn.normalize(projected)
        tokenizer, transformer_embeddings = plot.load_transformer_embeddings(transformer_name)
        english_words, english_vectors = plot.english_vocab_rows(tokenizer, transformer_embeddings, english_word_count)
        english_vectors = plot.normalize(english_vectors.astype(np.float32))
        neighbors = {
            str(token): plot.topk_english_neighbors(projected[idx], english_words, english_vectors, 12)
            for idx, token in enumerate(vocab)
        }
        metrics = plot.evaluate_neighbor_quality(neighbors)
        metrics["anchor_gloss_cosine"] = {}
        metrics["anchor_top1_hits"] = {}
    elapsed = time.time() - started
    score = score_run(metrics, metrics["anchor_gloss_cosine"], metrics["anchor_top1_hits"])
    return {
        "trial_id": trial.trial_id,
        "strategy": trial.strategy,
        "model_preset": trial.model_preset,
        "transformer": transformer_name,
        "epochs": trial.epochs,
        "anchor_weight": trial.anchor_weight,
        "projected_cooccurrence_weight": trial.projected_cooccurrence_weight,
        "collapse_weight": trial.collapse_weight,
        "anchor_passes": trial.anchor_passes,
        "learning_rate": trial.learning_rate,
        "pair_count": len(pairs),
        "vocab_size": len(vocab),
        "final_loss": losses[-1] if losses else None,
        "elapsed_sec": round(elapsed, 1),
        "score": score,
        **artifact_paths,
        "metrics": metrics,
    }


GRID_OUTPUT: dict[str, tuple[str, str, Any]] = {
    "broad": ("trials.jsonl", "best_summary.json", build_trial_grid),
    "simple": ("simple_grid_trials.jsonl", "simple_grid_best_summary.json", build_simple_focused_grid),
    "simple_variants": (
        "simple_variants_trials.jsonl",
        "simple_variants_best_summary.json",
        build_simple_variants_grid,
    ),
    "simple_variants_focused": (
        "simple_variants_focused_trials.jsonl",
        "simple_variants_focused_best_summary.json",
        build_simple_variants_focused_grid,
    ),
    "suffix_variants_focused": (
        "suffix_variants_focused_trials.jsonl",
        "suffix_variants_focused_best_summary.json",
        build_suffix_variants_focused_grid,
    ),
    "barthel_variants_focused": (
        "barthel_variants_focused_trials.jsonl",
        "barthel_variants_focused_best_summary.json",
        build_barthel_variants_focused_grid,
    ),
}


def load_completed_trial_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    completed: set[str] = set()
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        trial_id = record.get("trial_id")
        if trial_id:
            completed.add(str(trial_id))
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune projection hyperparameters.")
    parser.add_argument(
        "--grid",
        choices=tuple(GRID_OUTPUT),
        default="simple_variants_focused",
        help="simple_variants_focused = honed grid with per-trial neighbor JSONs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trial_ids already present in the results jsonl.",
    )
    parser.add_argument(
        "--save-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full pipeline outputs per trial under embed/tuning/trial_outputs/{trial_id}/.",
    )
    parser.add_argument("--english-word-count", type=int, default=5000)
    parser.add_argument(
        "--analyze-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run consistent non-gloss analysis after focused variant grids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_artifacts:
        TRIAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_name, summary_name, grid_builder = GRID_OUTPUT[args.grid]
    results_path = TUNE_DIR / results_name
    summary_path = TUNE_DIR / summary_name
    if not args.resume:
        results_path.write_text("")
    completed_ids = load_completed_trial_ids(results_path) if args.resume else set()
    learn = load_learn_module()
    plot = load_plot_helpers()
    trials = grid_builder()
    if completed_ids:
        trials = [trial for trial in trials if trial.trial_id not in completed_ids]
    print(
        f"Running {len(trials)} tuning trials ({args.grid} grid)"
        + (f", skipping {len(completed_ids)} completed" if completed_ids else "")
        + "..."
    )
    results: list[dict[str, Any]] = []
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
    for index, trial in enumerate(trials, start=1):
        print(f"\n[{index}/{len(trials)}] {trial.trial_id} {trial.model_preset}/{trial.strategy} ...", flush=True)
        try:
            result = run_trial(
                trial,
                learn,
                plot,
                save_artifacts=args.save_artifacts,
                outputs_dir=TRIAL_OUTPUTS_DIR,
                english_word_count=args.english_word_count,
            )
        except Exception as error:
            result = {
                "trial_id": trial.trial_id,
                "strategy": trial.strategy,
                "model_preset": trial.model_preset,
                "error": str(error),
                "score": -999.0,
            }
        results.append(result)
        with results_path.open("a") as handle:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
        if "error" not in result:
            m = result["metrics"]
            print(
                f"  score={result['score']:.3f} unique_top1={m['unique_top1_count']} "
                f"entropy={m['top1_entropy_bits']:.2f} anchor_cos={m['anchor_gloss_cosine']}"
            )

    ranked = sorted(results, key=lambda item: item.get("score", -999.0), reverse=True)
    best = [item for item in ranked if "error" not in item][:5]
    summary = {
        "grid": args.grid,
        "trial_count": len(trials) + len(completed_ids),
        "completed_this_run": len(trials),
        "best_trials": [
            {
                "trial_id": item["trial_id"],
                "score": item["score"],
                "strategy": item["strategy"],
                "model_preset": item["model_preset"],
                "epochs": item["epochs"],
                "anchor_weight": item["anchor_weight"],
                "projected_cooccurrence_weight": item["projected_cooccurrence_weight"],
                "collapse_weight": item["collapse_weight"],
                "anchor_passes": item["anchor_passes"],
                "learning_rate": item.get("learning_rate"),
                "metrics": {
                    "unique_top1_count": item["metrics"]["unique_top1_count"],
                    "top1_entropy_bits": item["metrics"]["top1_entropy_bits"],
                    "anchor_gloss_cosine": item["metrics"]["anchor_gloss_cosine"],
                    "anchor_top1_hits": {
                        token: hits[:3]
                        for token, hits in item["metrics"]["anchor_top1_hits"].items()
                    },
                },
            }
            for item in best
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\nTop trials:")
    for item in summary["best_trials"]:
        print(json.dumps(item, indent=2))

    focused_grids = ("simple_variants_focused", "suffix_variants_focused", "barthel_variants_focused")
    if args.analyze_after and args.grid in focused_grids and args.save_artifacts:
        analyze_script = EMBED_DIR / "analyze_consistent_non_gloss_neighbors.py"
        if analyze_script.exists():
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    str(analyze_script),
                    "--trials-jsonl",
                    str(results_path),
                    "--neighbors-root",
                    str(TRIAL_OUTPUTS_DIR),
                ],
                check=False,
                cwd=ROOT,
            )


if __name__ == "__main__":
    main()
