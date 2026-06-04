from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEIGHBORS_ROOT = ROOT / "embed" / "tuning" / "trial_outputs"
DEFAULT_TRIALS = ROOT / "embed" / "tuning" / "simple_variants_focused_trials.jsonl"
ANCHORS_FILE = ROOT / "embed" / "gloss_anchors.json"

ANCHOR_LEAK_WORDS = {
    "sun", "day", "days", "light", "star", "today", "morning", "daily",
    "king", "chief", "queen", "prince", "ruler", "leader", "chairman", "kingdom",
    "night", "nights", "moon", "evening", "dark",
    "father", "fathers", "copulated", "copulate", "copulation",
    "pulled", "officer", "officers", "police",
    "penis", "phallus",
}


def load_glossed_tokens() -> set[str]:
    return set(json.loads(ANCHORS_FILE.read_text()).keys())


def load_trial_records(trials_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not trials_path.exists():
        return records
    for line in trials_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error"):
            continue
        records[str(record["trial_id"])] = record
    return records


def trial_passes_quality(
    record: dict[str, Any],
    *,
    min_score: float,
    min_unique_top1: int,
    min_entropy: float,
) -> bool:
    if record.get("score", -999) < min_score:
        return False
    metrics = record.get("metrics") or {}
    if int(metrics.get("unique_top1_count", 0)) < min_unique_top1:
        return False
    if float(metrics.get("top1_entropy_bits", 0.0)) < min_entropy:
        return False
    return True


def global_top1_words(
    neighbors: dict[str, list],
    glossed: set[str],
    max_fraction: float,
) -> set[str]:
    """Words that are top-1 for too many non-gloss glyphs in one trial (collapse attractors)."""
    counts: Counter[str] = Counter()
    total = 0
    for token, hits in neighbors.items():
        if token in glossed or not hits:
            continue
        total += 1
        counts[str(hits[0][0]).lower()] += 1
    if total == 0:
        return set()
    threshold = max(1, math.ceil(max_fraction * total))
    return {word for word, count in counts.items() if count > threshold}


def analyze(
    neighbors_root: Path,
    trials_path: Path,
    *,
    min_score: float,
    min_unique_top1: int,
    min_entropy: float,
    min_trials: int,
    min_trial_fraction: float,
    min_margin: float,
    max_global_top1_fraction: float,
    top_n: int,
) -> dict[str, Any]:
    glossed = load_glossed_tokens()
    trial_records = load_trial_records(trials_path)
    token_word_trials: dict[str, Counter[str]] = defaultdict(Counter)
    token_word_margin: dict[tuple[str, str], list[float]] = defaultdict(list)
    token_word_score: dict[tuple[str, str], list[float]] = defaultdict(list)
    trials_used: list[str] = []
    trials_skipped: list[dict[str, Any]] = []

    for trial_dir in sorted(neighbors_root.iterdir()):
        if not trial_dir.is_dir():
            continue
        trial_id = trial_dir.name
        record = trial_records.get(trial_id, {})
        if trial_records and trial_id not in trial_records:
            continue
        if record and not trial_passes_quality(
            record,
            min_score=min_score,
            min_unique_top1=min_unique_top1,
            min_entropy=min_entropy,
        ):
            metrics = record.get("metrics") or {}
            trials_skipped.append(
                {
                    "trial_id": trial_id,
                    "score": record.get("score"),
                    "unique_top1_count": metrics.get("unique_top1_count"),
                    "top1_entropy_bits": metrics.get("top1_entropy_bits"),
                }
            )
            continue
        neighbors_path = trial_dir / "graphs" / "rongo_topk_english_neighbors.json"
        if not neighbors_path.exists():
            neighbors_path = trial_dir / "rongo_topk_english_neighbors.json"
        if not neighbors_path.exists():
            continue
        neighbors = json.loads(neighbors_path.read_text())
        banned_words = global_top1_words(neighbors, glossed, max_global_top1_fraction)
        trials_used.append(trial_id)
        for token, hits in neighbors.items():
            if token in glossed or not hits:
                continue
            word = str(hits[0][0]).lower()
            score = float(hits[0][1])
            margin = score - float(hits[1][1]) if len(hits) > 1 else 0.0
            if word in ANCHOR_LEAK_WORDS or word in banned_words:
                continue
            if margin < min_margin:
                continue
            token_word_trials[token][word] += 1
            token_word_margin[(token, word)].append(margin)
            token_word_score[(token, word)].append(score)

    required_trials = max(min_trials, math.ceil(min_trial_fraction * len(trials_used))) if trials_used else min_trials
    consistent: list[dict[str, Any]] = []
    for token, word_counts in token_word_trials.items():
        word, count = word_counts.most_common(1)[0]
        if count < required_trials:
            continue
        margins = token_word_margin[(token, word)]
        scores = token_word_score[(token, word)]
        consistent.append(
            {
                "token": token,
                "top1_english": word,
                "trial_count": count,
                "trials_required": required_trials,
                "trials_fraction": round(count / max(len(trials_used), 1), 4),
                "mean_margin": round(float(sum(margins) / len(margins)), 4),
                "mean_cosine": round(float(sum(scores) / len(scores)), 4),
                "alternate_top1": [
                    {"word": w, "trials": c}
                    for w, c in word_counts.most_common(5)
                    if w != word
                ],
            }
        )

    consistent.sort(
        key=lambda row: (row["trial_count"], row["mean_margin"], row["mean_cosine"]),
        reverse=True,
    )
    return {
        "trials_used": len(trials_used),
        "trial_ids": trials_used,
        "trials_skipped_low_quality": trials_skipped,
        "filters": {
            "min_score": min_score,
            "min_unique_top1": min_unique_top1,
            "min_entropy": min_entropy,
            "min_trials": min_trials,
            "min_trial_fraction": min_trial_fraction,
            "required_trials": required_trials,
            "min_margin": min_margin,
            "max_global_top1_fraction": max_global_top1_fraction,
        },
        "consistent_non_gloss": consistent[:top_n],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find glyphs with consistent non-gloss English top-1 across high-quality tuning trials."
    )
    parser.add_argument("--neighbors-root", type=Path, default=DEFAULT_NEIGHBORS_ROOT)
    parser.add_argument("--trials-jsonl", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--min-score", type=float, default=3.0, help="Minimum tuning score for a trial to count.")
    parser.add_argument(
        "--min-unique-top1",
        type=int,
        default=50,
        help="Minimum unique English top-1 labels (filters hard collapse).",
    )
    parser.add_argument(
        "--min-entropy",
        type=float,
        default=3.0,
        help="Minimum top-1 entropy in bits across glyphs.",
    )
    parser.add_argument("--min-trials", type=int, default=2, help="Absolute minimum trial agreement count.")
    parser.add_argument(
        "--min-trial-fraction",
        type=float,
        default=0.35,
        help="Fraction of quality trials that must agree (used with min-trials, whichever is stricter).",
    )
    parser.add_argument("--min-margin", type=float, default=0.03, help="Minimum top1-top2 cosine margin per vote.")
    parser.add_argument(
        "--max-global-top1-fraction",
        type=float,
        default=0.08,
        help="Ignore top-1 words that appear for more than this fraction of non-gloss glyphs within a trial.",
    )
    parser.add_argument(
        "--preset",
        choices=("default", "strict", "relaxed"),
        default="default",
        help="strict = old tight filters; relaxed = more trials and lower agreement bar.",
    )
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--output", type=Path, default=ROOT / "embed" / "tuning" / "consistent_non_gloss.json")
    return parser.parse_args()


PRESETS: dict[str, dict[str, float | int]] = {
    "strict": {
        "min_score": 4.0,
        "min_unique_top1": 100,
        "min_entropy": 4.0,
        "min_trials": 2,
        "min_trial_fraction": 0.5,
        "min_margin": 0.05,
        "max_global_top1_fraction": 0.05,
    },
    "default": {
        "min_score": 3.0,
        "min_unique_top1": 50,
        "min_entropy": 3.0,
        "min_trials": 2,
        "min_trial_fraction": 0.35,
        "min_margin": 0.03,
        "max_global_top1_fraction": 0.08,
    },
    "relaxed": {
        "min_score": 2.0,
        "min_unique_top1": 25,
        "min_entropy": 2.0,
        "min_trials": 2,
        "min_trial_fraction": 0.25,
        "min_margin": 0.02,
        "max_global_top1_fraction": 0.12,
    },
}


def main() -> None:
    args = parse_args()
    if args.preset != "default":
        for key, value in PRESETS[args.preset].items():
            setattr(args, key, value)
    report = analyze(
        args.neighbors_root,
        args.trials_jsonl,
        min_score=args.min_score,
        min_unique_top1=args.min_unique_top1,
        min_entropy=args.min_entropy,
        min_trials=args.min_trials,
        min_trial_fraction=args.min_trial_fraction,
        min_margin=args.min_margin,
        max_global_top1_fraction=args.max_global_top1_fraction,
        top_n=args.top_n,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    filters = report["filters"]
    print(f"Quality trials used: {report['trials_used']} (skipped {len(report['trials_skipped_low_quality'])} low-quality)")
    print(f"Required agreement: {filters['required_trials']} trials (fraction>={filters['min_trial_fraction']})")
    print(f"Consistent non-gloss glyphs: {len(report['consistent_non_gloss'])}")
    print(f"Wrote {args.output}")
    if report["consistent_non_gloss"]:
        print("\nTop 25:")
        for row in report["consistent_non_gloss"][:25]:
            print(
                f"  {row['token']} -> {row['top1_english']}  "
                f"trials={row['trial_count']}/{report['trials_used']}  "
                f"margin={row['mean_margin']:.3f}  cosine={row['mean_cosine']:.3f}"
            )
    else:
        print("\nNo glyphs passed filters (try --preset relaxed or lower --min-trial-fraction).")


if __name__ == "__main__":
    main()
