"""
run_pipeline.py
───────────────
End-to-end pipeline orchestrator.

Usage:
    python run_pipeline.py                  # full pipeline (uses sample.csv if raw unavailable)
    python run_pipeline.py --sample-only    # skip raw data scan, use existing sample.csv
    python run_pipeline.py --skip-sampling  # alias for --sample-only

Steps executed:
  1. sample_data        — extract SpotifyCares threads from raw CSV → sample.csv
  2. build_index        — TF-IDF index over brand threads
  3. derive_intents     — K-Means clustering, intent label assignment
  4. generate_golden_set — draft 200 golden set candidates
  5. evaluate           — baselines + agent metrics
  6. llm_judge          — reply quality scores
  7. judge_agreement    — generate human scoring sheet

⚠️  After step 4 you MUST review data/golden_set.csv before evaluation results
    are meaningful.  The pipeline will pause and remind you.
"""

import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config_loader import cfg, resolve_path, get_raw_data_path

PYTHON = sys.executable


def run(cmd: list, desc: str):
    """Run a subprocess command, printing a clear header."""
    print(f"\n{'─'*65}")
    print(f"  STEP: {desc}")
    print(f"{'─'*65}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n⚠️  Step '{desc}' exited with code {result.returncode}.")
        print("   Review the error above, fix if needed, then re-run.")
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(description="Hiver project end-to-end pipeline")
    p.add_argument("--sample-only", "--skip-sampling", action="store_true",
                   help="Skip raw CSV scan; assume sample.csv already exists")
    p.add_argument("--no-golden", action="store_true",
                   help="Skip golden set generation (use existing)")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip evaluation (just build index + intents)")
    return p.parse_args()


def main():
    args = parse_args()

    sample_path = resolve_path(cfg["paths"]["sample_csv"])
    golden_path = resolve_path(cfg["paths"]["golden_set"])
    index_path  = resolve_path(cfg["paths"]["tfidf_index"])

    print("=" * 65)
    print("  Hiver SDE Intern Project — Pipeline Runner")
    print("=" * 65)

    # ── Step 1: Sample data ───────────────────────────────────────────────────
    if args.sample_only:
        if not sample_path.exists():
            print(f"\nERROR: --sample-only specified but {sample_path} not found.")
            print("  Either run without --sample-only (needs raw CSV),")
            print("  or provide a pre-built sample.csv at that path.")
            sys.exit(1)
        print(f"\n[Step 1] Skipped (using existing {sample_path})")
    else:
        raw_path = get_raw_data_path()
        if not raw_path.exists():
            print(f"\nERROR: Raw CSV not found at {raw_path}")
            print("  Set RAW_DATA_PATH in .env, or use --sample-only if sample.csv exists.")
            sys.exit(1)
        run([PYTHON, "scripts/sample_data.py"], "Extract SpotifyCares sample from raw CSV")

    # ── Step 2: Build index ───────────────────────────────────────────────────
    if not index_path.exists():
        run([PYTHON, "scripts/build_index.py"], "Build TF-IDF retrieval index")
    else:
        print(f"\n[Step 2] TF-IDF index already exists, skipping rebuild.")

    # ── Step 3: Derive intents ────────────────────────────────────────────────
    run([PYTHON, "scripts/derive_intents.py"], "Derive intent taxonomy via K-Means")

    # ── Step 4: Generate golden set ───────────────────────────────────────────
    if not args.no_golden:
        if golden_path.exists():
            print(f"\n[Step 4] golden_set.csv already exists ({golden_path}).")
            print("         Delete it and re-run to regenerate, or use --no-golden.")
        else:
            run([PYTHON, "scripts/generate_golden_set.py"],
                "Draft golden set candidates (200 rows)")

            # Pause for human review
            print("\n" + "!" * 65)
            print("!  HUMAN REVIEW REQUIRED BEFORE EVALUATION               !")
            print("!" * 65)
            print(f"  File: {golden_path}")
            print()
            print("  Open data/golden_set.csv and:")
            print("    • Correct any wrong 'intent' labels")
            print("    • Correct any wrong 'expected_action' values")
            print("    • Add notes where helpful")
            print("    Budget: 30-45 minutes")
            print()
            answer = input("  Press ENTER when you've reviewed the golden set, "
                           "or type 'skip' to continue without reviewing: ")
            if answer.strip().lower() == "skip":
                print("  ⚠️  Skipping review — evaluation results will be less reliable.")
            else:
                print("  ✓ Review acknowledged. Continuing to evaluation.")

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    if not args.no_eval:
        run([PYTHON, "evaluation/evaluate.py"], "Run evaluation harness (baselines + agent)")

        # ── Step 6: LLM Judge ─────────────────────────────────────────────────
        run([PYTHON, "evaluation/llm_judge.py"], "Score reply quality (LLM or local judge)")

        # ── Step 7: Agreement sheet ───────────────────────────────────────────
        run([PYTHON, "evaluation/judge_agreement_sheet.py"],
            "Generate human-vs-judge agreement scoring sheet")

    # ── Done ──────────────────────────────────────────────────────────────────
    outputs = resolve_path(cfg["paths"]["evaluation_outputs"])
    print("\n" + "=" * 65)
    print("  Pipeline complete!")
    print("=" * 65)
    print(f"\n  Key outputs:")
    print(f"    data/golden_set.csv           — golden evaluation set")
    print(f"    data/processed/sample.csv      — working dataset")
    print(f"    {outputs}/")
    print(f"      comparison_table.csv          — baseline vs agent metrics")
    print(f"      intent_report_agent.txt        — per-intent P/R/F1")
    print(f"      confusion_matrix_agent.csv     — confusion matrix")
    print(f"      escalation_report.txt          — escalation metrics")
    print(f"      judge_scores.csv               — reply quality scores")
    print(f"      judge_agreement_sheet.csv      — fill this in! (15 min)")
    print(f"      summary.json                   — all headline numbers")
    print()
    print("  Next steps (mandatory before submission):")
    print("  1. Review data/golden_set.csv labels if not done")
    print("  2. Fill in evaluation/outputs/judge_agreement_sheet.csv")
    print("     then run: python evaluation/compute_agreement.py")
    print("  3. Re-run evaluation after correcting golden set:")
    print("     python evaluation/evaluate.py")
    print("  4. Read REPORT.md end-to-end")


if __name__ == "__main__":
    main()
