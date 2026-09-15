"""
scripts/test_agent.py
─────────────────────
Interactive terminal tester for the SupportAgent.

Usage:
    python scripts/test_agent.py                  # REPL loop (type 'quit' to exit)
    python scripts/test_agent.py "your message"   # single-shot from command line

Output includes the full agent result:
    - Intent + confidence
    - Escalation decision + reason
    - Generated reply
    - Top retrieval similarity score
    - Top-3 retrieved historical examples

This script is READ-ONLY with respect to src/agent.py and evaluation/evaluate.py.
"""

import sys
import io
from pathlib import Path

# ── Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError) ──────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Make sure the project root is on the path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import SupportAgent  # noqa: E402  (import after path setup)


def _safe_print(*args, **kwargs):
    """Print, replacing any unencodable characters instead of crashing."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [str(a).encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding)
                     for a in args]
        print(*safe_args, **kwargs)


# ── Formatting helpers ────────────────────────────────────────────────────────

_SEP = "─" * 64

_DECISION_ICONS = {
    "auto_handle": "✅  AUTO-HANDLE",
    "escalate":    "🚨  ESCALATE",
}

_INTENT_MAP = {
    "playback_issue":       "🎵  Playback Issue",
    "app_crash_or_bug":     "🐛  App Crash / Bug",
    "account_access":       "🔐  Account Access",
    "subscription_billing": "💳  Subscription / Billing",
    "feature_request":      "💡  Feature Request",
    "connectivity_device":  "📡  Connectivity / Device",
    "content_catalog":      "🎶  Content / Catalog",
    "general_inquiry":      "❓  General Inquiry",
}


def _fmt_confidence(conf: float) -> str:
    """Return a human-friendly confidence bar."""
    pct = int(conf * 100)
    filled = pct // 5          # 20 blocks max
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {pct}%"


def _print_result(message: str, result: dict) -> None:
    """Pretty-print the full agent result to stdout."""
    intent_label = _INTENT_MAP.get(result["intent"], result["intent"])
    decision_label = _DECISION_ICONS.get(result["decision"], result["decision"].upper())

    _safe_print()
    _safe_print(_SEP)
    _safe_print(f"  [MESSAGE]  CUSTOMER INPUT")
    _safe_print(f"  {message}")
    _safe_print(_SEP)

    # ── Intent ────────────────────────────────────────────────────────────────
    _safe_print(f"\n  INTENT:      {intent_label}")
    _safe_print(f"  CONFIDENCE:  {_fmt_confidence(result['intent_confidence'])}")

    # ── Escalation ────────────────────────────────────────────────────────────
    _safe_print(f"\n  DECISION:    {decision_label}")
    _safe_print(f"  REASON:      {result['reason']}")

    # ── Reply ─────────────────────────────────────────────────────────────────
    _safe_print(f"\n  AGENT REPLY")
    # Word-wrap at ~70 chars for readability
    reply = result["reply"]
    words = reply.split()
    line, lines = [], []
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > 70:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        _safe_print(f"  {ln}")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    _safe_print(f"\n  TOP RETRIEVAL SCORE: {result['top_retrieval_score']:.4f}")
    examples = result.get("retrieved_examples", [])
    if examples:
        _safe_print(f"\n  TOP-3 HISTORICAL EXAMPLES")
        for i, ex in enumerate(examples[:3], 1):
            sim = ex.get("similarity_score", 0.0)
            cust = ex.get("customer_message", "")[:120]
            brand = ex.get("brand_reply", "")[:120]
            _safe_print(f"\n  [{i}]  sim={sim:.4f}")
            _safe_print(f"       Customer: {cust}")
            _safe_print(f"       Reply:    {brand}")

    _safe_print()
    _safe_print(_SEP)


# ── Core runner ───────────────────────────────────────────────────────────────

def run_once(agent: SupportAgent, message: str) -> None:
    """Run the agent on a single message and print the full output."""
    result = agent.run(message)
    _print_result(message, result)


def repl(agent: SupportAgent) -> None:
    """Interactive REPL: accept messages until the user types 'quit' or 'exit'."""
    _safe_print()
    _safe_print("=" * 64)
    _safe_print("       SupportAgent -- Interactive Test Console")
    _safe_print("=" * 64)
    _safe_print("  Type a customer message and press Enter to run the agent.")
    _safe_print("  Type  quit  or  exit  (or press Ctrl-C) to stop.")
    _safe_print("=" * 64)

    while True:
        try:
            print()
            message = input("  Message > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if not message:
            print("  (empty input — please type a message)")
            continue

        if message.lower() in {"quit", "exit", "q"}:
            print("  Goodbye!")
            break

        run_once(agent, message)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Initialise agent once (expensive — loads TF-IDF index + trains classifier)
    print("\n[test_agent] Initialising SupportAgent …")
    agent = SupportAgent()

    # Single-shot mode: message passed as a CLI argument
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        run_once(agent, message)
    else:
        # Interactive REPL mode
        repl(agent)


if __name__ == "__main__":
    main()
