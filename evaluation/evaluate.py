"""
evaluation/evaluate.py
──────────────────────
Evaluation harness.  One command, all metrics.

Produces:
  1. Intent: accuracy, macro-F1, per-intent P/R/F1, confusion matrix
  2. Escalation: accuracy, P/R/F1, missed-escalation rate
  3. Retrieval coverage: % of replies backed by retrieval score > threshold
  4. Comparison table: Baseline A vs Baseline B vs SupportAgent
  5. Files in evaluation/outputs/:
       intent_report.txt, confusion_matrix.csv,
       escalation_report.txt, comparison_table.csv,
       agent_predictions.csv

Run:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --data data/golden_set.csv  # explicit path
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path
from src.baselines import BaselineA, BaselineB, evaluate_baselines
from src.agent import SupportAgent

SEED = cfg["sampling"]["seed"]
OUTPUTS_DIR = resolve_path(cfg["paths"]["evaluation_outputs"])


def parse_args():
    p = argparse.ArgumentParser(description="Run full evaluation harness")
    p.add_argument("--data", default=None,
                   help="Path to labelled CSV (default: golden_set.csv → sample.csv)")
    p.add_argument("--no-agent", action="store_true",
                   help="Skip agent evaluation (faster, just baselines)")
    return p.parse_args()


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding="utf-8")
    df = df.dropna(subset=["customer_message", "intent"])
    df = df[df["customer_message"].str.strip() != ""]
    df["intent"] = df["intent"].str.strip()
    # Normalise expected_action
    if "expected_action" in df.columns:
        df["expected_action"] = df["expected_action"].str.strip().str.lower()
    return df.reset_index(drop=True)


def evaluate_intent(y_true, y_pred, labels=None) -> dict:
    """Return dict of intent classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm_labels = labels if labels else sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_intent": {k: v for k, v in report.items()
                       if k not in ("accuracy", "macro avg", "weighted avg")},
        "confusion_matrix": cm,
        "cm_labels": cm_labels,
        "report_str": classification_report(y_true, y_pred, zero_division=0),
    }


def evaluate_escalation(y_true_esc, y_pred_esc) -> dict:
    """
    y_true_esc / y_pred_esc: lists of "escalate" or "auto_handle".
    Returns dict including missed_escalation_rate.
    """
    acc = accuracy_score(y_true_esc, y_pred_esc)
    # Treat "escalate" as the positive class
    prec = precision_score(y_true_esc, y_pred_esc, pos_label="escalate",
                           zero_division=0)
    rec  = recall_score(y_true_esc, y_pred_esc, pos_label="escalate",
                        zero_division=0)
    f1   = f1_score(y_true_esc, y_pred_esc, pos_label="escalate",
                    zero_division=0)

    # Missed escalation: true=escalate but predicted=auto_handle
    true_esc = [t == "escalate" for t in y_true_esc]
    pred_esc = [p == "escalate" for p in y_pred_esc]
    n_true_esc = sum(true_esc)
    n_missed = sum(t and not p for t, p in zip(true_esc, pred_esc))
    missed_rate = n_missed / n_true_esc if n_true_esc > 0 else 0.0

    report_str = (
        f"Escalation Evaluation\n"
        f"  True escalations:   {n_true_esc}\n"
        f"  Missed escalations: {n_missed} "
        f"(missed_rate={missed_rate:.4f} = {missed_rate*100:.1f}%)\n"
        f"  Accuracy:   {acc:.4f}\n"
        f"  Precision:  {prec:.4f}  (of predicted escalations, how many were correct)\n"
        f"  Recall:     {rec:.4f}  (of true escalations, how many we caught)\n"
        f"  F1:         {f1:.4f}\n"
    )
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "missed_escalation_rate": missed_rate,
        "n_true_escalations": n_true_esc,
        "n_missed_escalations": n_missed,
        "report_str": report_str,
    }


def retrieval_coverage(scores, threshold=None):
    """% of replies backed by retrieval score > threshold."""
    if threshold is None:
        threshold = cfg["retrieval"]["low_similarity_threshold"]
    covered = sum(s >= threshold for s in scores)
    total = len(scores)
    return covered / total if total > 0 else 0.0


def print_banner(text: str):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data:
        data_path = Path(args.data)
    else:
        golden_path = resolve_path(cfg["paths"]["golden_set"])
        sample_path = resolve_path(cfg["paths"]["sample_csv"])
        data_path = golden_path if golden_path.exists() else sample_path

    print(f"Loading evaluation data: {data_path}")
    df = load_data(data_path)
    print(f"  {len(df)} rows, {df['intent'].nunique()} intent classes")
    print(f"  Intent distribution:")
    for intent, cnt in df["intent"].value_counts().items():
        print(f"    {intent}: {cnt} ({cnt/len(df)*100:.1f}%)")

    texts  = df["customer_message"].tolist()
    labels = df["intent"].tolist()
    has_esc = "expected_action" in df.columns and df["expected_action"].notna().any()

    test_size = 1.0 - cfg["sampling"]["train_test_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=SEED,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Baseline A ────────────────────────────────────────────────────────────
    print_banner("BASELINE A — Most-Frequent-Intent")
    bl_a = BaselineA()
    bl_a.fit(X_train, y_train)
    pred_a = bl_a.predict(X_test)
    metrics_a = evaluate_intent(y_test, pred_a)
    print(f"Accuracy:   {metrics_a['accuracy']:.4f}")
    print(f"Macro-F1:   {metrics_a['macro_f1']:.4f}")
    print(metrics_a["report_str"])

    # ── Baseline B ────────────────────────────────────────────────────────────
    print_banner("BASELINE B — TF-IDF + LinearSVC")
    bl_b = BaselineB()
    bl_b.fit(X_train, y_train)
    pred_b = bl_b.predict(X_test)
    metrics_b = evaluate_intent(y_test, pred_b)
    print(f"Accuracy:   {metrics_b['accuracy']:.4f}")
    print(f"Macro-F1:   {metrics_b['macro_f1']:.4f}")
    print(metrics_b["report_str"])

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_rows = []
    metrics_agent = None

    if not args.no_agent:
        print_banner("SUPPORT AGENT (TF-IDF Retrieval + LinearSVC Intent)")
        agent = SupportAgent()
        # IMPORTANT: Re-fit the intent classifier on TRAIN SPLIT ONLY to avoid
        # data leakage. By default, SupportAgent trains on the full golden_set.
        # We override that here so evaluation uses a clean held-out test set.
        agent.classifier._fit(X_train, y_train)
        agent.classifier._source = "golden_set_train_split_only"
        print(f"[Agent] Classifier retrained on train split ({len(X_train)} rows).")

        pred_agent = []
        esc_pred   = []
        ret_scores = []

        for msg in tqdm(X_test, desc="Agent predictions"):
            result = agent.run(msg)
            pred_agent.append(result["intent"])
            esc_pred.append(result["decision"])
            ret_scores.append(result["top_retrieval_score"])
            agent_rows.append({
                "customer_message": msg[:300],
                "true_intent": None,  # filled below
                "pred_intent": result["intent"],
                "intent_confidence": result["intent_confidence"],
                "decision": result["decision"],
                "reason": result["reason"],
                "top_retrieval_score": result["top_retrieval_score"],
                "reply": result["reply"][:400],
            })

        # Fill true_intent
        for row, true in zip(agent_rows, y_test):
            row["true_intent"] = true

        metrics_agent = evaluate_intent(y_test, pred_agent)
        print(f"Accuracy:   {metrics_agent['accuracy']:.4f}")
        print(f"Macro-F1:   {metrics_agent['macro_f1']:.4f}")
        print(metrics_agent["report_str"])

        # Retrieval coverage
        cov = retrieval_coverage(ret_scores)
        print(f"Retrieval coverage (score >= {cfg['retrieval']['low_similarity_threshold']}): "
              f"{cov:.4f} = {cov*100:.1f}%")

        # Escalation evaluation (if golden set has expected_action)
        if has_esc:
            print_banner("ESCALATION EVALUATION")
            # Get test indices
            test_df = df.iloc[-len(X_test):]  # approximate — proper index below

            # Build proper test esc labels by aligning on message text
            msg_to_esc = dict(zip(df["customer_message"], df.get("expected_action", "")))
            esc_true = [str(msg_to_esc.get(m, "auto_handle")) for m in X_test]
            esc_true = [e if e in ("escalate", "auto_handle") else "auto_handle"
                        for e in esc_true]

            esc_metrics = evaluate_escalation(esc_true, esc_pred)
            print(esc_metrics["report_str"])

            esc_out = OUTPUTS_DIR / "escalation_report.txt"
            esc_out.write_text(esc_metrics["report_str"], encoding="utf-8")
            print(f"  → {esc_out}")
        else:
            esc_metrics = None
            print("\n[Escalation] No 'expected_action' column found — skipping "
                  "escalation evaluation. Add this column to golden_set.csv.")

    # ── Comparison table ──────────────────────────────────────────────────────
    print_banner("COMPARISON TABLE")
    rows = [
        {
            "System": "Baseline A (majority class)",
            "Accuracy": f"{metrics_a['accuracy']:.4f}",
            "Macro-F1": f"{metrics_a['macro_f1']:.4f}",
        },
        {
            "System": "Baseline B (TF-IDF + LinearSVC)",
            "Accuracy": f"{metrics_b['accuracy']:.4f}",
            "Macro-F1": f"{metrics_b['macro_f1']:.4f}",
        },
    ]
    if metrics_agent:
        rows.append({
            "System": "Support Agent (retrieval-grounded)",
            "Accuracy": f"{metrics_agent['accuracy']:.4f}",
            "Macro-F1": f"{metrics_agent['macro_f1']:.4f}",
        })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────────
    print_banner("SAVING OUTPUTS")

    # Intent reports
    (OUTPUTS_DIR / "intent_report_baseline_a.txt").write_text(
        f"Baseline A — Most-Frequent-Intent\n"
        f"Accuracy: {metrics_a['accuracy']:.4f}\n"
        f"Macro-F1: {metrics_a['macro_f1']:.4f}\n\n"
        + metrics_a["report_str"], encoding="utf-8")

    (OUTPUTS_DIR / "intent_report_baseline_b.txt").write_text(
        f"Baseline B — TF-IDF + LinearSVC\n"
        f"Accuracy: {metrics_b['accuracy']:.4f}\n"
        f"Macro-F1: {metrics_b['macro_f1']:.4f}\n\n"
        + metrics_b["report_str"], encoding="utf-8")

    if metrics_agent:
        (OUTPUTS_DIR / "intent_report_agent.txt").write_text(
            f"Support Agent\n"
            f"Accuracy: {metrics_agent['accuracy']:.4f}\n"
            f"Macro-F1: {metrics_agent['macro_f1']:.4f}\n\n"
            + metrics_agent["report_str"], encoding="utf-8")

    # Confusion matrix
    cm_df = pd.DataFrame(
        metrics_b["confusion_matrix"],
        index=metrics_b["cm_labels"],
        columns=metrics_b["cm_labels"],
    )
    cm_df.to_csv(OUTPUTS_DIR / "confusion_matrix_baseline_b.csv", encoding="utf-8")

    if metrics_agent:
        cm_agent_df = pd.DataFrame(
            metrics_agent["confusion_matrix"],
            index=metrics_agent["cm_labels"],
            columns=metrics_agent["cm_labels"],
        )
        cm_agent_df.to_csv(OUTPUTS_DIR / "confusion_matrix_agent.csv", encoding="utf-8")

    # Agent predictions
    if agent_rows:
        agent_pred_df = pd.DataFrame(agent_rows)
        agent_pred_df.to_csv(OUTPUTS_DIR / "agent_predictions.csv",
                              index=False, encoding="utf-8")
        print(f"  agent_predictions.csv ({len(agent_pred_df)} rows)")

    # Comparison table
    comp_df.to_csv(OUTPUTS_DIR / "comparison_table.csv", index=False, encoding="utf-8")

    # Summary JSON (for REPORT.md)
    summary = {
        "data_file": str(data_path),
        "n_test": len(X_test),
        "n_train": len(X_train),
        "baseline_a": {
            "accuracy": round(metrics_a["accuracy"], 4),
            "macro_f1": round(metrics_a["macro_f1"], 4),
        },
        "baseline_b": {
            "accuracy": round(metrics_b["accuracy"], 4),
            "macro_f1": round(metrics_b["macro_f1"], 4),
        },
    }
    if metrics_agent:
        summary["agent"] = {
            "accuracy": round(metrics_agent["accuracy"], 4),
            "macro_f1": round(metrics_agent["macro_f1"], 4),
            "retrieval_coverage": round(retrieval_coverage(ret_scores), 4),
        }
    if has_esc and metrics_agent:
        summary["escalation"] = {
            "accuracy": round(esc_metrics["accuracy"], 4),
            "precision": round(esc_metrics["precision"], 4),
            "recall": round(esc_metrics["recall"], 4),
            "f1": round(esc_metrics["f1"], 4),
            "missed_escalation_rate": round(esc_metrics["missed_escalation_rate"], 4),
        }

    (OUTPUTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nAll outputs saved to: {OUTPUTS_DIR}")
    print("\nEvaluation complete. ✓")
    return summary


if __name__ == "__main__":
    main()
