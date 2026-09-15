"""
scripts/generate_golden_set.py
──────────────────────────────
Generates the golden evaluation set (data/golden_set.csv).

Strategy:
  1. Load sample.csv (already K-Means labelled)
  2. Stratify-sample ~200 messages across the 8 intent clusters
  3. Run the SupportAgent on each to get: intent, expected_action, suggested reply
  4. Write to data/golden_set.csv with ALL agent suggestions as candidates

⚠️  YOU MUST REVIEW AND CORRECT THESE LABELS BEFORE RUNNING EVALUATION.
    Antigravity drafts the candidates; you supply the final "hand-labelled" judgement.
    Budget: 30-45 minutes.

Run:
    python scripts/generate_golden_set.py
    python scripts/generate_golden_set.py --n 200  # default
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path
from src.agent import SupportAgent

SEED = cfg["sampling"]["seed"]
BRAND = cfg["brand"]["name"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=cfg["sampling"]["golden_set_size"],
                   help="Target number of golden set rows (150-250)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing golden_set.csv")
    return p.parse_args()


def stratified_sample(df: pd.DataFrame, n: int, label_col: str) -> pd.DataFrame:
    """Sample n rows stratified by label_col, as evenly as possible."""
    labels = df[label_col].unique()
    n_per_label = max(1, n // len(labels))
    remainder = n - n_per_label * len(labels)

    parts = []
    for i, lbl in enumerate(labels):
        sub = df[df[label_col] == lbl]
        k = n_per_label + (1 if i < remainder else 0)
        k = min(k, len(sub))
        parts.append(sub.sample(k, random_state=SEED))

    result = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return result


def main():
    args = parse_args()

    golden_path = resolve_path(cfg["paths"]["golden_set"])
    sample_path = resolve_path(cfg["paths"]["sample_csv"])

    if golden_path.exists() and not args.force:
        print(f"golden_set.csv already exists at {golden_path}")
        print("Use --force to regenerate.")
        return

    if not sample_path.exists():
        print(f"ERROR: {sample_path} not found. Run scripts/sample_data.py and "
              "scripts/derive_intents.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sample_path, encoding="utf-8")
    df = df.dropna(subset=["customer_message"])
    df = df[df["customer_message"].str.strip() != ""]

    # Ensure intent column exists
    if "intent" not in df.columns:
        print("WARNING: sample.csv has no 'intent' column. "
              "Run scripts/derive_intents.py first.", file=sys.stderr)
        df["intent"] = "general_inquiry"

    print(f"Loaded {len(df):,} rows from sample.csv")
    print(f"Intent distribution:\n{df['intent'].value_counts().to_string()}\n")

    # Stratified sample
    sampled = stratified_sample(df, args.n, "intent")
    print(f"Stratified sample: {len(sampled)} rows")

    # Run agent on each message to generate candidate labels
    print("\nRunning SupportAgent to draft golden set candidates...")
    print("(This uses K-Means labels as seed; you will correct them.)\n")

    agent = SupportAgent()

    rows = []
    for idx, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Generating"):
        msg = str(row["customer_message"])
        result = agent.run(msg)

        # Map decision → expected_action vocabulary
        exp_action = result["decision"]  # "auto_handle" or "escalate"

        # Use brand_reply from the thread as reference if available
        ref_reply = row.get("brand_reply", "")
        if pd.isna(ref_reply):
            ref_reply = result["reply"]

        rows.append({
            "id": f"GS_{idx:04d}",
            "customer_message": msg,
            # ↓↓↓ REVIEW THESE — agent suggestions, not ground truth ↓↓↓
            "intent": result["intent"],
            "expected_action": exp_action,
            "reference_reply_or_resolution": str(ref_reply)[:500],
            "top_retrieval_score": result["top_retrieval_score"],
            "intent_confidence": result["intent_confidence"],
            # Notes column for your corrections
            "notes": "",
        })

    golden_df = pd.DataFrame(rows)

    # Save
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_df.to_csv(golden_path, index=False, encoding="utf-8")

    print(f"\n✓ Wrote {len(golden_df)} rows → {golden_path}")
    print(f"\n{'='*70}")
    print("⚠️  ACTION REQUIRED — YOU MUST REVIEW THESE LABELS BEFORE EVALUATION")
    print(f"{'='*70}")
    print(f"File: {golden_path}")
    print()
    print("Columns to review and correct:")
    print("  • 'intent'          — is the labelled intent actually correct?")
    print("  • 'expected_action' — should this be auto_handle or escalate?")
    print("  • 'notes'           — add any comments or corrections")
    print()
    print("Intent distribution in the draft golden set:")
    print(golden_df["intent"].value_counts().to_string())
    print()
    print("Expected_action distribution:")
    print(golden_df["expected_action"].value_counts().to_string())
    print()
    print(f"Budget: 30-45 minutes.  Target: correct all obvious errors,")
    print(f"        spot-check at least 50% of rows carefully.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
