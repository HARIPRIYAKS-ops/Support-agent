"""
scripts/derive_intents.py
─────────────────────────
Derives intent taxonomy from the brand's customer messages using:
  1. TF-IDF + K-Means clustering (n_clusters from config)
  2. Top-N keywords per cluster printed for manual naming
  3. Saves cluster assignments to data/processed/intent_clusters.pkl

The final intent label mapping is defined in config.yaml (intents.labels).
This script lets you validate/adjust that mapping by inspecting real message samples.

Run:
    python scripts/derive_intents.py
    python scripts/derive_intents.py --n-samples 20  # show 20 msgs per cluster
"""

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path


SEED = cfg["sampling"]["seed"]


def clean_text(text: str) -> str:
    import re
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s',.!?-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=10,
                   help="Message samples to print per cluster")
    return p.parse_args()


def main():
    args = parse_args()
    sample_path = resolve_path(cfg["paths"]["sample_csv"])
    cluster_path = resolve_path(cfg["paths"]["intent_clusters"])
    intent_labels = cfg["intents"]["labels"]
    n_clusters = cfg["intents"]["n_clusters"]

    if not sample_path.exists():
        print(f"ERROR: {sample_path} not found. Run scripts/sample_data.py first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sample_path, encoding="utf-8")
    df = df.dropna(subset=["customer_message"])
    df["cleaned"] = df["customer_message"].apply(clean_text)
    df = df[df["cleaned"].str.len() > 10].reset_index(drop=True)

    print(f"Clustering {len(df):,} messages into {n_clusters} intents...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(df["cleaned"])

    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    df["cluster"] = labels

    feature_names = np.array(vectorizer.get_feature_names_out())
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS — use this to validate/adjust config.yaml intents.labels")
    print("=" * 70)
    for c_id in range(n_clusters):
        configured_label = intent_labels.get(c_id, f"cluster_{c_id}")
        top_terms = feature_names[order_centroids[c_id, :15]]
        cluster_msgs = df[df["cluster"] == c_id]["customer_message"]
        count = len(cluster_msgs)

        print(f"\n[Cluster {c_id}] → '{configured_label}'  ({count:,} msgs, "
              f"{count/len(df)*100:.1f}%)")
        print(f"  Top terms: {', '.join(top_terms)}")
        samples = cluster_msgs.sample(min(args.n_samples, count),
                                      random_state=SEED).tolist()
        for i, msg in enumerate(samples, 1):
            print(f"  [{i:02d}] {msg[:120]}")

    print("\n" + "=" * 70)

    # Assign intent labels from config
    df["intent"] = df["cluster"].map(
        {int(k): v for k, v in intent_labels.items()}
    ).fillna("unknown")

    # Save
    cluster_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cluster_path, "wb") as f:
        pickle.dump({
            "df": df,
            "vectorizer": vectorizer,
            "km_model": km,
            "intent_labels": intent_labels,
        }, f, protocol=4)
    print(f"\nSaved cluster assignments → {cluster_path}")

    # Also write intent-labelled sample back to sample.csv
    df[["tweet_id", "customer_message", "brand_reply", "intent"]].to_csv(
        sample_path, index=False, encoding="utf-8"
    )
    print(f"Updated sample.csv with intent column → {sample_path}")
    print("Done. ✓")


if __name__ == "__main__":
    main()
