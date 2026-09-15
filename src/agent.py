"""
src/agent.py
────────────
Core support-agent pipeline for SpotifyCares.

Components:
  Retriever           – TF-IDF cosine similarity over historical brand threads
  IntentClassifier    – TF-IDF + LinearSVC trained on golden set (falls back
                        to K-Means cluster assignment if golden set absent)
  ReplyGenerator      – Template-based reply using retrieved examples; LLM if
                        OPENAI_API_KEY is set
  EscalationDecider   – Explicit if/else rule set (readable, not a black box)
  SupportAgent        – Orchestrates all four into a single .run() call

Usage:
    from src.agent import SupportAgent
    agent = SupportAgent()
    result = agent.run("my songs keep skipping on bluetooth")
    # result: {"intent": ..., "reply": ..., "decision": ..., "reason": ...}
"""

import os
import re
import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Lowercase, strip @handles and URLs for vectorization."""
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s',.!?-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentiment_score(text: str) -> float:
    """
    Returns a polarity score in [-1, 1].
    Uses TextBlob if available; otherwise falls back to a simple keyword heuristic.
    """
    try:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
    except ImportError:
        angry_kws = ["hate", "angry", "furious", "worst", "terrible", "awful",
                     "disgusting", "useless", "stupid", "crap", "sucks", "wtf"]
        positive_kws = ["thank", "great", "awesome", "love", "perfect", "fixed",
                        "works", "wonderful"]
        t = text.lower()
        score = 0.0
        for kw in angry_kws:
            if kw in t:
                score -= 0.3
        for kw in positive_kws:
            if kw in t:
                score += 0.2
        return max(-1.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────────────────
#  Retriever
# ──────────────────────────────────────────────────────────────────────────────

class Retriever:
    """TF-IDF cosine-similarity retriever over historical SpotifyCares threads."""

    def __init__(self, index_path: Optional[Path] = None):
        if index_path is None:
            index_path = resolve_path(cfg["paths"]["tfidf_index"])

        if not index_path.exists():
            raise FileNotFoundError(
                f"TF-IDF index not found at {index_path}. "
                "Run: python scripts/build_index.py"
            )

        with open(index_path, "rb") as f:
            payload = pickle.load(f)

        self.vectorizer: TfidfVectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]          # sparse (n_docs × n_features)
        self.threads: List[dict] = payload["threads"]
        self.top_k: int = cfg["retrieval"]["top_k"]

    def retrieve(self, query: str) -> List[dict]:
        """
        Return top-k most similar threads.
        Each result is a dict with keys:
          customer_message, brand_reply, similarity_score
        """
        vec = self.vectorizer.transform([_clean(query)])
        sims = cosine_similarity(vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][: self.top_k]

        results = []
        for idx in top_idx:
            t = self.threads[idx].copy()
            t["similarity_score"] = float(sims[idx])
            results.append(t)
        return results


# ──────────────────────────────────────────────────────────────────────────────
#  Intent Classifier
# ──────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    TF-IDF + LinearSVC intent classifier.

    If a golden set exists (data/golden_set.csv), it trains on the labelled
    portion of that set.  Otherwise it falls back to K-Means cluster labels
    from data/processed/sample.csv.
    """

    # Ordered list of known intent labels (must match config.yaml intents.labels)
    INTENT_LABELS: List[str] = list(cfg["intents"]["labels"].values())

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LinearSVC] = None
        self._fitted = False
        self._source = "none"
        self._try_load_or_train()

    # ── Training ──────────────────────────────────────────────────────────────

    def _try_load_or_train(self):
        golden_path = resolve_path(cfg["paths"]["golden_set"])
        sample_path = resolve_path(cfg["paths"]["sample_csv"])

        if golden_path.exists():
            df = pd.read_csv(golden_path, encoding="utf-8")
            if "intent" in df.columns and df["intent"].notna().sum() >= 10:
                self._fit(df["customer_message"].tolist(), df["intent"].tolist())
                self._source = "golden_set"
                print(f"[IntentClassifier] Trained on golden set "
                      f"({len(df)} rows)  source={self._source}")
                return

        if sample_path.exists():
            df = pd.read_csv(sample_path, encoding="utf-8")
            if "intent" in df.columns and df["intent"].notna().sum() >= 50:
                self._fit(df["customer_message"].tolist(), df["intent"].tolist())
                self._source = "sample_kmeans"
                print(f"[IntentClassifier] Trained on K-Means labelled sample "
                      f"({len(df)} rows)  source={self._source}")
                return

        print("[IntentClassifier] WARNING: No training data found; "
              "intent prediction will return 'general_inquiry'.", file=sys.stderr)

    def _fit(self, texts: List[str], labels: List[str]):
        cleaned = [_clean(t) for t in texts]
        self.vectorizer = TfidfVectorizer(
            max_features=cfg["tfidf"]["max_features"],
            ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
            sublinear_tf=cfg["tfidf"]["sublinear_tf"],
            stop_words="english",
            strip_accents="unicode",
        )
        X = self.vectorizer.fit_transform(cleaned)
        self.classifier = LinearSVC(random_state=cfg["sampling"]["seed"],
                                    max_iter=2000, C=1.0)
        self.classifier.fit(X, labels)
        self._fitted = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Returns (intent_label, confidence_score ∈ [0, 1]).
        Confidence is derived from the decision function margin.
        """
        if not self._fitted:
            return "general_inquiry", 0.0

        vec = self.vectorizer.transform([_clean(text)])
        intent = self.classifier.predict(vec)[0]

        # Decision-function margin → pseudo-confidence via sigmoid
        margins = self.classifier.decision_function(vec)[0]
        if hasattr(margins, "__len__"):
            # Multi-class: confidence = softmax of margins
            exp_m = np.exp(margins - margins.max())
            probs = exp_m / exp_m.sum()
            confidence = float(probs.max())
        else:
            # Binary (shouldn't happen with 8 classes, but guard)
            confidence = float(1 / (1 + np.exp(-abs(margins))))

        return intent, confidence


# ──────────────────────────────────────────────────────────────────────────────
#  Escalation Decider
# ──────────────────────────────────────────────────────────────────────────────

class EscalationDecider:
    """
    Explicit, readable rule-set for escalation decisions.
    Decision ∈ {"auto_handle", "escalate"}.

    Rules (applied in order — first match wins):
      1. Account security action (hacked/banned/locked account)
      2. Billing dispute with weak retrieval evidence
      3. Angry / threatening tone
      4. Very low retrieval similarity (novel/ambiguous issue)
      5. Low intent confidence (classifier uncertain)
      → default: auto_handle
    """

    def __init__(self):
        self._esc_cfg = cfg["escalation"]
        self._ret_cfg = cfg["retrieval"]
        self._security_kws = [k.lower() for k in self._esc_cfg["account_security_keywords"]]
        self._threat_kws   = [k.lower() for k in self._esc_cfg["threatening_keywords"]]

    def decide(
        self,
        message: str,
        intent: str,
        retrieval_score: float,
        intent_confidence: float,
    ) -> Tuple[str, str]:
        """
        Returns (decision, reason).
        """
        msg_lower = message.lower()

        # Rule 1 — Account security action
        if intent == "account_access" and any(kw in msg_lower for kw in self._security_kws):
            return "escalate", (
                "Account security keyword detected "
                f"(intent={intent}; requires human review of account data)"
            )

        # Rule 2 — Billing dispute with weak retrieval grounding
        billing_threshold = self._ret_cfg["billing_low_threshold"]
        if intent == "subscription_billing" and retrieval_score < billing_threshold:
            return "escalate", (
                f"Billing/payment dispute with low retrieval confidence "
                f"(score={retrieval_score:.3f} < threshold={billing_threshold})"
            )

        # Rule 3 — Threatening or extremely angry tone
        threat_threshold = self._esc_cfg["angry_sentiment_threshold"]
        sentiment = _sentiment_score(message)
        if sentiment < threat_threshold or any(kw in msg_lower for kw in self._threat_kws):
            return "escalate", (
                f"Threatening or aggressive tone detected "
                f"(sentiment={sentiment:.3f}, threshold={threat_threshold})"
            )

        # Rule 4 — Very low retrieval similarity (novel / ambiguous issue)
        low_sim_threshold = self._ret_cfg["low_similarity_threshold"]
        if retrieval_score < low_sim_threshold:
            return "escalate", (
                f"Low retrieval similarity — likely novel or ambiguous issue "
                f"(score={retrieval_score:.3f} < threshold={low_sim_threshold})"
            )

        # Rule 5 — Low intent confidence
        min_conf = self._esc_cfg["min_intent_confidence"]
        if intent_confidence < min_conf:
            return "escalate", (
                f"Low intent confidence — ambiguous message "
                f"(confidence={intent_confidence:.3f} < threshold={min_conf})"
            )

        # Default
        return "auto_handle", (
            f"Standard issue with sufficient retrieval grounding "
            f"(score={retrieval_score:.3f}, intent={intent}, "
            f"confidence={intent_confidence:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Reply Generator
# ──────────────────────────────────────────────────────────────────────────────

class ReplyGenerator:
    """
    Generates a reply grounded in retrieved historical examples.

    If OPENAI_API_KEY is set → calls GPT-4o-mini with retrieved context.
    Otherwise             → deterministic template using the best-match reply.
    """

    BRAND_NAME = cfg["brand"]["name"].replace("Cares", "")  # "Spotify"

    # Intent-specific acknowledgement phrases
    _ACK = {
        "playback_issue":       "Sorry to hear about the playback issue",
        "app_crash_or_bug":     "We're sorry the app is acting up",
        "account_access":       "We understand this is frustrating",
        "subscription_billing": "We're sorry about the billing concern",
        "feature_request":      "Thanks for the feedback",
        "connectivity_device":  "We can help troubleshoot the connection",
        "content_catalog":      "We understand your frustration about missing content",
        "general_inquiry":      "Happy to help",
    }

    def generate(
        self,
        message: str,
        intent: str,
        retrieved: List[dict],
    ) -> str:
        """Return a reply string, grounded in retrieved examples."""
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            return self._llm_reply(message, intent, retrieved, api_key)
        return self._template_reply(message, intent, retrieved)

    # ── LLM path ──────────────────────────────────────────────────────────────

    def _llm_reply(
        self,
        message: str,
        intent: str,
        retrieved: List[dict],
        api_key: str,
    ) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            examples_text = "\n".join(
                f"[Example {i+1}]\n"
                f"  Customer: {r['customer_message'][:200]}\n"
                f"  {self.BRAND_NAME} reply: {r['brand_reply'][:300]}\n"
                f"  (similarity={r['similarity_score']:.3f})"
                for i, r in enumerate(retrieved[:3])
            )

            system_prompt = (
                f"You are a {self.BRAND_NAME} customer support agent. "
                "Reply concisely, empathetically, and in line with how Spotify Support "
                "actually responds on Twitter. Use the historical examples below as your "
                "grounding — do not invent policies. Keep replies under 280 characters "
                "when possible. Do NOT use placeholders like [name] or [DM link]."
            )
            user_prompt = (
                f"Customer message (intent: {intent}):\n{message}\n\n"
                f"Historical examples:\n{examples_text}\n\n"
                "Write the reply:"
            )

            resp = client.chat.completions.create(
                model=cfg["llm_judge"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # Fall back to template on any LLM error
            print(f"[ReplyGenerator] LLM failed ({e}), using template fallback.",
                  file=sys.stderr)
            return self._template_reply(message, intent, retrieved)

    # ── Deterministic template path ───────────────────────────────────────────

    def _template_reply(
        self,
        message: str,
        intent: str,
        retrieved: List[dict],
    ) -> str:
        """
        Build a reply from the best retrieved example.
        The reply always cites what evidence it drew from.
        """
        ack = self._ACK.get(intent, "Thanks for reaching out")
        best = retrieved[0] if retrieved else None

        if best and best["similarity_score"] > 0.10:
            # Use the historical reply as a grounding base, citing similarity
            base_reply = best["brand_reply"]
            # Strip @handle prefix if present
            base_reply = re.sub(r"^@\w+\s*", "", base_reply).strip()
            # Truncate if very long
            if len(base_reply) > 250:
                base_reply = base_reply[:247] + "..."
            reply = (
                f"{ack}! Here's how we've helped with similar issues before: "
                f"{base_reply} "
                f"[grounded on historical example, sim={best['similarity_score']:.3f}]"
            )
        else:
            reply = (
                f"{ack}! Please send us a DM with more details about your issue "
                f"and we'll look into it right away. "
                f"[no close historical match found, sim="
                f"{best['similarity_score']:.3f if best else 0:.3f}]"
            )

        return reply


# ──────────────────────────────────────────────────────────────────────────────
#  SupportAgent — main entry point
# ──────────────────────────────────────────────────────────────────────────────

class SupportAgent:
    """
    Orchestrates: Retriever → IntentClassifier → EscalationDecider → ReplyGenerator

    Usage:
        agent = SupportAgent()
        result = agent.run("songs keep buffering on my phone")
        # {
        #   "intent": "playback_issue",
        #   "intent_confidence": 0.82,
        #   "decision": "auto_handle",
        #   "reason": "Standard issue with sufficient retrieval grounding ...",
        #   "reply": "Sorry to hear about the playback issue! ...",
        #   "top_retrieval_score": 0.43,
        #   "retrieved_examples": [...],
        # }
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        golden_set_path: Optional[Path] = None,
    ):
        print("[SupportAgent] Initialising components...")
        self.retriever = Retriever(index_path)
        self.classifier = IntentClassifier()
        self.escalation = EscalationDecider()
        self.reply_gen = ReplyGenerator()
        print("[SupportAgent] Ready.")

    def run(self, customer_message: str) -> Dict:
        """
        Process a single customer message end-to-end.

        Returns a dict with keys:
            intent, intent_confidence, decision, reason, reply,
            top_retrieval_score, retrieved_examples
        """
        # 1. Retrieve similar historical threads
        retrieved = self.retriever.retrieve(customer_message)
        top_score = retrieved[0]["similarity_score"] if retrieved else 0.0

        # 2. Classify intent
        intent, confidence = self.classifier.predict(customer_message)

        # 3. Escalation decision
        decision, reason = self.escalation.decide(
            message=customer_message,
            intent=intent,
            retrieval_score=top_score,
            intent_confidence=confidence,
        )

        # 4. Generate reply (only for auto_handle, but generate for both so we
        #    can evaluate reply quality even on escalations)
        reply = self.reply_gen.generate(customer_message, intent, retrieved)

        return {
            "intent": intent,
            "intent_confidence": round(confidence, 4),
            "decision": decision,
            "reason": reason,
            "reply": reply,
            "top_retrieval_score": round(top_score, 4),
            "retrieved_examples": retrieved,
        }


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_messages = [
        "my songs keep skipping and it's driving me crazy",
        "I was charged twice this month, this is fraud",
        "When will Taylor Swift's new album be on Spotify?",
        "I can't log into my account — it says it's been locked or hacked",
        "asdfghjkl random gibberish message xyz",
    ]

    agent = SupportAgent()
    for msg in test_messages:
        print("\n" + "=" * 60)
        print(f"INPUT:  {msg}")
        result = agent.run(msg)
        print(f"INTENT: {result['intent']}  (conf={result['intent_confidence']})")
        print(f"DECISION: {result['decision']}")
        print(f"REASON: {result['reason']}")
        print(f"REPLY:  {result['reply'][:200]}")
        print(f"TOP_SIM: {result['top_retrieval_score']}")
