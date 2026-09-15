"""
evaluation/llm_judge.py
───────────────────────
LLM-as-judge for reply quality evaluation.

Scores each reply on 5 dimensions (1-5 each):
  1. Correctness    — is the reply factually accurate?
  2. Relevance      — does it address the customer's actual issue?
  3. Groundedness   — does it draw from real historical examples?
  4. Helpfulness    — would this reply actually help the customer?
  5. Tone           — empathetic and professional?

Two modes (auto-selected):
  LLM mode    : GPT-4o-mini via OpenAI API (if OPENAI_API_KEY is set)
  Local mode  : Deterministic fallback using ROUGE-1 + retrieval similarity

Run:
    python evaluation/llm_judge.py
    python evaluation/llm_judge.py --input evaluation/outputs/agent_predictions.csv
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path

OUTPUTS_DIR = resolve_path(cfg["paths"]["evaluation_outputs"])
SEED = cfg["sampling"]["seed"]


# ──────────────────────────────────────────────────────────────────────────────
#  Deterministic local fallback scorer
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Simple whitespace tokenizer for ROUGE-1."""
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return set(text.split())


def _rouge1(candidate: str, reference: str) -> float:
    """ROUGE-1 F1 between candidate and reference."""
    c_tokens = _tokenize(candidate)
    r_tokens = _tokenize(reference)
    if not c_tokens or not r_tokens:
        return 0.0
    overlap = len(c_tokens & r_tokens)
    precision = overlap / len(c_tokens)
    recall    = overlap / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _scale_to_5(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Map a [lo, hi] value linearly to [1, 5]."""
    if hi == lo:
        return 3.0
    return 1.0 + 4.0 * (value - lo) / (hi - lo)


def local_score(
    customer_message: str,
    agent_reply: str,
    reference_reply: str,
    retrieval_score: float,
) -> Dict[str, float]:
    """
    Deterministic scoring heuristic (no API key needed).
    Returns a dict of five scores in [1, 5].
    """
    rouge = _rouge1(agent_reply, reference_reply)

    # Correctness: proxy = ROUGE overlap with reference
    correctness = _scale_to_5(rouge)

    # Relevance: does the reply contain key terms from the customer message?
    cust_tokens = _tokenize(customer_message)
    reply_tokens = _tokenize(agent_reply)
    relevance_raw = len(cust_tokens & reply_tokens) / max(len(cust_tokens), 1)
    relevance = _scale_to_5(min(relevance_raw * 3, 1.0))  # scale up, cap at 1

    # Groundedness: proxy = retrieval similarity score
    groundedness = _scale_to_5(retrieval_score, lo=0.0, hi=0.5)

    # Helpfulness: combination of relevance + non-emptiness + length
    reply_len = len(agent_reply.split())
    length_score = min(reply_len / 30, 1.0)  # 30+ words → full score
    helpfulness = _scale_to_5((relevance_raw + length_score) / 2)

    # Tone: keyword presence heuristic
    positive_words = {"sorry", "happy", "help", "assist", "understand",
                      "thanks", "glad", "hope", "resolve", "fix", "support"}
    tone_overlap = len(reply_tokens & positive_words) / max(len(reply_tokens), 1)
    tone = _scale_to_5(min(tone_overlap * 20, 1.0))

    # Clamp all to [1, 5]
    scores = {k: round(max(1.0, min(5.0, v)), 2) for k, v in {
        "correctness": correctness,
        "relevance": relevance,
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "tone": tone,
    }.items()}
    scores["overall"] = round(sum(scores.values()) / len(scores), 2)
    scores["mode"] = "local_deterministic"
    return scores


# ──────────────────────────────────────────────────────────────────────────────
#  LLM scorer (OpenAI)
# ──────────────────────────────────────────────────────────────────────────────

_LLM_PROMPT_SYSTEM = """You are an expert evaluator of customer support reply quality.
Score the following reply on five dimensions using integers 1-5:
  1=Very poor, 2=Poor, 3=OK, 4=Good, 5=Excellent

Dimensions:
  correctness  — factually accurate (no wrong info)?
  relevance    — addresses the customer's actual issue?
  groundedness — grounded in real support knowledge (not made-up)?
  helpfulness  — would this actually help the customer?
  tone         — empathetic, professional, not robotic?

Return ONLY valid JSON: {"correctness": X, "relevance": X, "groundedness": X, "helpfulness": X, "tone": X}
No explanation, no extra text."""

_LLM_PROMPT_USER = """Customer message: {customer_message}

Agent reply: {agent_reply}

Historical reference reply (if available): {reference_reply}

Score the agent reply:"""


def llm_score(
    customer_message: str,
    agent_reply: str,
    reference_reply: str,
    api_key: str,
) -> Dict[str, float]:
    """Call GPT-4o-mini to score a reply. Falls back to local on error."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = _LLM_PROMPT_USER.format(
            customer_message=customer_message[:300],
            agent_reply=agent_reply[:400],
            reference_reply=str(reference_reply)[:300] if reference_reply else "N/A",
        )
        resp = client.chat.completions.create(
            model=cfg["llm_judge"]["model"],
            messages=[
                {"role": "system", "content": _LLM_PROMPT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        scores = json.loads(raw)
        scores = {k: float(v) for k, v in scores.items()}
        scores["overall"] = round(sum(scores.values()) / len(scores), 2)
        scores["mode"] = "llm_gpt4o_mini"
        return scores
    except Exception as e:
        print(f"[llm_judge] LLM scoring failed ({e}), using local fallback.",
              file=sys.stderr)
        return None  # caller will fall back


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(OUTPUTS_DIR / "agent_predictions.csv"),
                   help="Agent predictions CSV (from evaluate.py)")
    p.add_argument("--n", type=int, default=None,
                   help="Score only N rows (default: all)")
    return p.parse_args()


def run_judge(predictions_path: Path, n: int = None) -> pd.DataFrame:
    """Score all rows in predictions CSV. Returns scored DataFrame."""
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{predictions_path} not found. Run evaluation/evaluate.py first."
        )

    df = pd.read_csv(predictions_path, encoding="utf-8")
    if n:
        df = df.head(n)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    mode = "LLM (GPT-4o-mini)" if api_key else "local deterministic"
    print(f"Scoring {len(df)} replies using {mode} judge...")

    scored_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        msg   = str(row.get("customer_message", ""))
        reply = str(row.get("reply", ""))
        ref   = str(row.get("reference_reply_or_resolution", ""))
        sim   = float(row.get("top_retrieval_score", 0.0))

        if api_key:
            scores = llm_score(msg, reply, ref, api_key)
            if scores is None:
                scores = local_score(msg, reply, ref, sim)
        else:
            scores = local_score(msg, reply, ref, sim)

        entry = row.to_dict()
        entry.update({f"judge_{k}": v for k, v in scores.items()})
        scored_rows.append(entry)

    return pd.DataFrame(scored_rows)


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.input)
    scored_df = run_judge(pred_path, args.n)

    # Print summary
    judge_cols = [c for c in scored_df.columns if c.startswith("judge_")
                  and c != "judge_mode"]
    print("\nJudge score summary (mean ± std across all replies):")
    for col in judge_cols:
        vals = scored_df[col].dropna()
        if vals.dtype in (float, int):
            print(f"  {col.replace('judge_', ''):15s}: "
                  f"{vals.mean():.2f} ± {vals.std():.2f}  "
                  f"(min={vals.min():.1f}, max={vals.max():.1f})")

    out_path = OUTPUTS_DIR / "judge_scores.csv"
    scored_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved judge scores → {out_path}")
    return scored_df


if __name__ == "__main__":
    main()
