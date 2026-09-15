"""
scripts/test_retrieval_regression.py
──────────────────────────────────────
Regression test: billing query should rank a billing historical example
above the TWICE (K-pop artist) examples.

Run:
    python scripts/test_retrieval_regression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import Retriever  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

QUERY = "I was charged twice for my Spotify subscription"

# Keywords that indicate a billing result (at least one must appear)
BILLING_KEYWORDS = {"charged", "charge", "billed", "billing", "subscription", "payment"}

# Keywords that indicate an artist/catalog result (the false-positive we want demoted)
ARTIST_TWICE_KEYWORDS = {"twice not on spotify", "put twice on spotify",
                          "get twice on spotify", "twice on spotify"}


def _is_billing(msg: str) -> bool:
    m = msg.lower()
    return any(kw in m for kw in BILLING_KEYWORDS)


def _is_artist_twice(msg: str) -> bool:
    m = msg.lower()
    return any(kw in m for kw in ARTIST_TWICE_KEYWORDS)


# ── Test ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Regression test: billing query ranking")
    print(f"Query: {QUERY!r}")
    print("=" * 60)

    retriever = Retriever()
    results = retriever.retrieve(QUERY)

    print(f"\nTop-{len(results)} retrieved examples:")
    for i, r in enumerate(results, 1):
        msg = r["customer_message"]
        score = r["similarity_score"]
        tag = ""
        if _is_billing(msg):
            tag = " <- BILLING"
        elif _is_artist_twice(msg):
            tag = " <- TWICE-ARTIST (false positive)"
        print(f"  Rank {i}  sim={score:.4f}  {msg[:90]}{tag}")

    # Assertion 1: rank-1 must be a billing example
    rank1_msg = results[0]["customer_message"]
    assert _is_billing(rank1_msg), (
        f"FAIL: Rank-1 result is NOT a billing example.\n"
        f"  Got: {rank1_msg!r}\n"
        f"  Expected a message containing one of: {BILLING_KEYWORDS}"
    )
    print("\n  PASS: Rank-1 result is a billing example.")

    # Assertion 2: no TWICE-artist doc outranks the first billing doc
    first_billing_rank = next(
        i for i, r in enumerate(results) if _is_billing(r["customer_message"])
    )
    for i, r in enumerate(results):
        if i >= first_billing_rank:
            break
        msg = r["customer_message"]
        assert not _is_artist_twice(msg), (
            f"FAIL: TWICE-artist example at rank {i+1} outranks first billing example "
            f"(at rank {first_billing_rank+1}).\n"
            f"  Artist msg: {msg!r}"
        )
    print("  PASS: No TWICE-artist example outranks the first billing example.")

    print("\nAll regression assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    run()
