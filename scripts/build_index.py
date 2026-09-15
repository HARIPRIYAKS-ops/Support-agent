"""
scripts/build_index.py
──────────────────────
Builds a TF-IDF index over brand_threads.csv (customer messages side only)
for retrieval-grounded reply generation.

Outputs:
  data/processed/tfidf_index.pkl  – dict with keys:
      "vectorizer"   : fitted TfidfVectorizer
      "matrix"       : sparse TF-IDF matrix (n_threads × n_features)
      "threads"      : list of dicts {tweet_id, customer_message, brand_reply}

Run:
    python scripts/build_index.py
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path


def clean_text(text: str) -> str:
    """Light cleaning: lowercase, strip Twitter handles and URLs."""
    import re
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)        # remove @mentions
    text = re.sub(r"https?://\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s',.!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    threads_path = resolve_path(cfg["paths"]["threads_csv"])
    index_path = resolve_path(cfg["paths"]["tfidf_index"])
    tfidf_cfg = cfg["tfidf"]

    if not threads_path.exists():
        print(f"ERROR: {threads_path} not found. Run scripts/sample_data.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {threads_path} ...")
    df = pd.read_csv(threads_path, encoding="utf-8")
    # Drop rows with missing text
    df = df.dropna(subset=["customer_message", "brand_reply"])
    df = df[df["customer_message"].str.strip() != ""]
    df = df[df["brand_reply"].str.strip() != ""]
    print(f"  {len(df):,} valid threads")

    threads = df.to_dict("records")
    cleaned = [clean_text(t["customer_message"]) for t in threads]

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        strip_accents="unicode",
        analyzer="word",
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(cleaned)
    print(f"  Matrix shape: {matrix.shape}")

    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "threads": threads,
    }
    with open(index_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"Saved TF-IDF index → {index_path}")
    print("Done. ✓")


if __name__ == "__main__":
    main()
