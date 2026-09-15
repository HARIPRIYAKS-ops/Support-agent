"""
evaluation/compute_agreement.py
────────────────────────────────
Computes human-vs-judge agreement after you fill in your_score column.

Metrics:
  - Spearman rank correlation (ρ) between your_score and judge_overall
  - Mean absolute difference
  - Agreement-within-1 rate (proportion of rows where |yours - judge| <= 1)

Run:
    python evaluation/compute_agreement.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path

OUTPUTS_DIR = resolve_path(cfg["paths"]["evaluation_outputs"])


def main():
    sheet_path = OUTPUTS_DIR / "judge_agreement_sheet.csv"
    if not sheet_path.exists():
        print(f"ERROR: {sheet_path} not found.", file=sys.stderr)
        print("Run: python evaluation/judge_agreement_sheet.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sheet_path, encoding="utf-8")

    # Check your_score is filled
    if "your_score" not in df.columns:
        print("ERROR: 'your_score' column missing from sheet.", file=sys.stderr)
        sys.exit(1)

    df["your_score"] = pd.to_numeric(df["your_score"], errors="coerce")
    missing = df["your_score"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have missing your_score — they will be excluded.")
    df = df.dropna(subset=["your_score", "judge_overall"])

    if len(df) < 5:
        print(f"ERROR: Only {len(df)} rows have scores — need at least 5.", file=sys.stderr)
        sys.exit(1)

    human = df["your_score"].tolist()
    judge = df["judge_overall"].tolist()

    # Spearman correlation
    rho, p_value = stats.spearmanr(human, judge)

    # Mean absolute difference
    mad = np.mean(np.abs(np.array(human) - np.array(judge)))

    # Agreement within 1
    within1 = np.mean(np.abs(np.array(human) - np.array(judge)) <= 1.0)

    print(f"\n{'='*55}")
    print("  Human-vs-Judge Agreement Report")
    print(f"{'='*55}")
    print(f"  Rows evaluated:            {len(df)}")
    print(f"  Spearman ρ:                {rho:.4f}  (p={p_value:.4f})")
    print(f"  Mean absolute difference:  {mad:.4f}")
    print(f"  Agreement within 1 point:  {within1:.4f} = {within1*100:.1f}%")
    print(f"{'='*55}")

    # Interpretation
    if rho >= 0.7:
        interp = "Strong agreement — judge is a reliable proxy for human judgement."
    elif rho >= 0.4:
        interp = "Moderate agreement — judge captures the signal but with noise."
    else:
        interp = "Weak agreement — judge and human diverge; interpret LLM scores cautiously."
    print(f"\n  Interpretation: {interp}")

    # Save
    result = {
        "n_rows": len(df),
        "spearman_rho": round(rho, 4),
        "p_value": round(p_value, 4),
        "mean_absolute_difference": round(mad, 4),
        "agreement_within_1_rate": round(within1, 4),
        "interpretation": interp,
    }
    import json
    out_path = OUTPUTS_DIR / "agreement_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
