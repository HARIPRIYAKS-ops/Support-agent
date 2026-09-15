"""
evaluation/judge_agreement_sheet.py
────────────────────────────────────
Generates the human-vs-judge agreement scoring sheet.

Samples 40 replies from agent_predictions.csv, scores them with the
LLM judge (or local fallback), and writes a CSV you fill in manually.

Output: evaluation/judge_agreement_sheet.csv
  Columns: id, customer_message, agent_reply, judge_overall,
            judge_correctness, judge_relevance, judge_groundedness,
            judge_helpfulness, judge_tone,
            your_score (← FILL THIS IN, 1-5),
            notes

After filling in your_score, run:
    python evaluation/compute_agreement.py

Run:
    python evaluation/judge_agreement_sheet.py
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path
from evaluation.llm_judge import run_judge

OUTPUTS_DIR = resolve_path(cfg["paths"]["evaluation_outputs"])
SEED = cfg["sampling"]["seed"]
N_SAMPLE = cfg["llm_judge"]["judge_agreement_sample_size"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_SAMPLE,
                   help=f"Number of replies to sample (default: {N_SAMPLE})")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing sheet")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUTS_DIR / "judge_agreement_sheet.csv"
    if out_path.exists() and not args.force:
        print(f"Sheet already exists: {out_path}")
        print("Use --force to regenerate.")
        return

    pred_path = OUTPUTS_DIR / "agent_predictions.csv"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found. Run evaluation/evaluate.py first.",
              file=sys.stderr)
        sys.exit(1)

    # Score with judge
    scored_df = run_judge(pred_path)

    # Sample N rows
    n = min(args.n, len(scored_df))
    sampled = scored_df.sample(n, random_state=SEED).reset_index(drop=True)

    # Build agreement sheet
    sheet = pd.DataFrame({
        "id": sampled.index.map(lambda i: f"AGR_{i:03d}"),
        "customer_message": sampled["customer_message"].fillna("").str[:300],
        "agent_reply": sampled["reply"].fillna("").str[:400],
        "judge_overall": sampled["judge_overall"].round(2),
        "judge_correctness": sampled["judge_correctness"].round(2),
        "judge_relevance": sampled["judge_relevance"].round(2),
        "judge_groundedness": sampled["judge_groundedness"].round(2),
        "judge_helpfulness": sampled["judge_helpfulness"].round(2),
        "judge_tone": sampled["judge_tone"].round(2),
        "your_score": "",   # ← FILL THIS IN (1-5)
        "notes": "",
    })

    sheet.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n✓ Wrote {len(sheet)}-row agreement sheet → {out_path}")
    print(f"\n{'='*65}")
    print("⚠️  ACTION REQUIRED — FILL IN YOUR SCORES BEFORE COMPUTING AGREEMENT")
    print(f"{'='*65}")
    print(f"File: {out_path}")
    print()
    print("Instructions:")
    print("  1. Open the CSV in Excel / LibreOffice Calc")
    print("  2. For each row, read the customer_message and agent_reply")
    print("  3. Fill in 'your_score' with an integer 1-5:")
    print("       1 = Very poor reply")
    print("       2 = Poor")
    print("       3 = OK / acceptable")
    print("       4 = Good")
    print("       5 = Excellent")
    print("  4. Save the CSV (keep UTF-8 encoding)")
    print("  5. Run: python evaluation/compute_agreement.py")
    print()
    print("Budget: ~15 minutes (about 20 seconds per reply)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
