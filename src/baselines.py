"""
src/baselines.py
────────────────
Two baseline classifiers for intent prediction:

  BaselineA — Most-Frequent-Intent classifier
              Always predicts the majority class label seen in training.
              Upper bound for a trivial system.

  BaselineB — TF-IDF + LinearSVC
              Standard sklearn pipeline.

Both follow the same .fit(texts, labels) / .predict(texts) interface,
making them drop-in replaceable with the SupportAgent classifier.

Run standalone for quick diagnostics:
    python src/baselines.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path

SEED = cfg["sampling"]["seed"]


def _clean(text: str) -> str:
    import re
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s',.!?-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────────────────────────────────────
#  Baseline A: Most-Frequent-Intent
# ──────────────────────────────────────────────────────────────────────────────

class BaselineA:
    """Always predicts the majority class from training data."""

    name = "Baseline A: Most-Frequent-Intent"

    def __init__(self):
        self._majority_label: str = "general_inquiry"

    def fit(self, texts: List[str], labels: List[str]) -> "BaselineA":
        from collections import Counter
        counts = Counter(labels)
        self._majority_label, self._majority_count = counts.most_common(1)[0]
        self._class_dist = counts
        print(f"[{self.name}] majority label = '{self._majority_label}' "
              f"({self._majority_count}/{len(labels)} = "
              f"{self._majority_count/len(labels)*100:.1f}%)")
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return [self._majority_label] * len(texts)

    def predict_single(self, text: str) -> Tuple[str, float]:
        """Returns (label, confidence=1.0) to match IntentClassifier interface."""
        return self._majority_label, 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  Baseline B: TF-IDF + LinearSVC
# ──────────────────────────────────────────────────────────────────────────────

class BaselineB:
    """TF-IDF vectorizer + LinearSVC classifier pipeline."""

    name = "Baseline B: TF-IDF + LinearSVC"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=cfg["tfidf"]["max_features"],
                ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
                sublinear_tf=cfg["tfidf"]["sublinear_tf"],
                stop_words="english",
                strip_accents="unicode",
                preprocessor=_clean,
            )),
            ("clf", LinearSVC(random_state=SEED, max_iter=2000, C=1.0)),
        ])
        self._fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "BaselineB":
        self.pipeline.fit(texts, labels)
        self._fitted = True
        print(f"[{self.name}] Fitted on {len(texts)} samples.")
        return self

    def predict(self, texts: List[str]) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return self.pipeline.predict(texts).tolist()

    def predict_single(self, text: str) -> Tuple[str, float]:
        label = self.predict([text])[0]
        # Derive pseudo-confidence from decision function margin
        margins = self.pipeline.decision_function([text])[0]
        if hasattr(margins, "__len__"):
            exp_m = np.exp(margins - margins.max())
            confidence = float((exp_m / exp_m.sum()).max())
        else:
            confidence = float(1 / (1 + np.exp(-abs(float(margins)))))
        return label, confidence


# ──────────────────────────────────────────────────────────────────────────────
#  Standalone evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_baselines(data_path: Path = None, test_size: float = None):
    """
    Load golden set (or sample.csv), split, fit & evaluate both baselines.
    Returns dict of results for use by evaluation/evaluate.py.
    """
    if data_path is None:
        golden_path = resolve_path(cfg["paths"]["golden_set"])
        sample_path = resolve_path(cfg["paths"]["sample_csv"])
        if golden_path.exists():
            data_path = golden_path
            print(f"Using golden set: {golden_path}")
        elif sample_path.exists():
            data_path = sample_path
            print(f"Golden set not found; using sample: {sample_path}")
        else:
            raise FileNotFoundError("Neither golden_set.csv nor sample.csv found.")

    if test_size is None:
        test_size = 1.0 - cfg["sampling"]["train_test_split"]

    df = pd.read_csv(data_path, encoding="utf-8")
    df = df.dropna(subset=["customer_message", "intent"])
    df = df[df["customer_message"].str.strip() != ""]
    print(f"Loaded {len(df)} labelled rows from {data_path.name}")

    texts = df["customer_message"].tolist()
    labels = df["intent"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=SEED,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    results = {}
    for Baseline in [BaselineA, BaselineB]:
        bl = Baseline()
        bl.fit(X_train, y_train)
        y_pred = bl.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test + y_pred)))

        print(f"\n{'='*55}")
        print(f"{bl.name}")
        print(f"{'='*55}")
        print(f"Accuracy:  {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        results[bl.name] = {
            "model": bl,
            "accuracy": acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "report": report,
            "confusion_matrix": cm,
            "cm_labels": sorted(set(y_test + y_pred)),
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results


if __name__ == "__main__":
    evaluate_baselines()
