# DECISION_LOG.md — Hiver SDE Intern Assignment

Design decisions made during this project, with alternatives considered.

---

## D-01: Brand Selection — SpotifyCares over AmazonHelp

**Decision**: Use SpotifyCares (43,265 tweets) as the target brand.

**Alternatives considered**:
- *AmazonHelp* (169,840 tweets): Far more volume but too heterogeneous — spans
  retail returns, AWS billing, Prime Video, Kindle, and logistics. A single
  intent taxonomy wouldn't be coherent.
- *AppleSupport* (106,860 tweets): Also large, but very similar issue types to
  Spotify (iOS bugs, account issues). Spotify has a tighter domain (just the
  music app) making evaluation cleaner.
- *Uber_Support* (56,270 tweets): Good size, but surge pricing and driver
  disputes involve legal/safety escalations that require real policy knowledge
  to evaluate correctly.

**Why SpotifyCares wins**: Volume is sufficient (41,585 matched customer–reply
pairs), issue types are meaningfully varied (8 distinct intents), domain is
tight enough for TF-IDF retrieval to be effective, and the escalation
scenarios (billing disputes, account hacking, frustrated users) are realistic
and testable.

---

## D-02: Never Load the Full CSV — Two-Pass Chunking

**Decision**: Two-pass chunk read: Pass 1 collects brand reply `in_response_to_tweet_id` 
IDs; Pass 2 collects matching inbound messages and brand replies.

**Alternatives considered**:
- *Single-pass with dict accumulation*: Possible but risky — in_response_to IDs
  may appear in later chunks than the corresponding inbound tweet, causing misses.
  Two-pass is safer and only costs ~60s total.
- *Dask or chunked SQL*: Overcomplicated for a one-time extract step.
- *Load everything into RAM*: 516MB CSV → would exhaust 8GB RAM with pandas
  overhead, and breaks the spec's "never load full CSV" constraint.

**Why two-pass**: Correctness and simplicity. The first pass guarantees we have
all relevant tweet IDs before collecting text.

---

## D-03: K-Means for Intent Derivation (not LDA, not manual)

**Decision**: K-Means (k=8) on TF-IDF vectors to derive the initial intent taxonomy.

**Alternatives considered**:
- *Latent Dirichlet Allocation (LDA)*: Better for discovering topics in longer
  documents, but Twitter messages are very short (20-30 tokens) — K-Means on
  TF-IDF is empirically competitive for short text.
- *Manual inspection only*: The spec requires deriving intents "from data, not
  pre-guessing." Manual inspection of 200 random messages is reasonable but
  doesn't give quantified cluster boundaries.
- *Sentence-BERT embeddings + HDBSCAN*: Would give better semantic clusters
  but requires ~1GB model weights and GPU for reasonable speed. Added as
  optional future work.

**Why K-Means**: Fast, deterministic (fixed seed=42), interpretable via
top-terms per centroid. The cluster inspection output directly validates the
intent labels chosen.

---

## D-04: TF-IDF + LinearSVC for Intent Classification (not BERT)

**Decision**: Use TF-IDF + LinearSVC as the intent classifier inside the agent.

**Alternatives considered**:
- *Sentence-BERT + LogisticRegression*: Would likely get 5-10 percentage points
  higher accuracy, but requires downloading a 420MB model, GPU for fast inference,
  and breaks the "under 15 min setup" constraint.
- *Naive Bayes*: Faster, but consistently underperforms LinearSVC on short
  multi-class text classification problems.
- *Zero-shot classification (Hugging Face)*: No labelled data needed, but slow
  on CPU and accuracy is unpredictable.

**Why LinearSVC**: Trains in <1 second on 200 examples, matches or beats
logistic regression on TF-IDF features for short text, and produces a decision
function margin we can use as a proxy confidence score for escalation Rule 5.

---

## D-05: Retrieval over Brand Threads (not just customer messages)

**Decision**: Build the TF-IDF retrieval index over all 41,585 matched
customer–reply pairs; query with customer message and retrieve top-k pairs.

**Alternatives considered**:
- *Index only customer messages, return the matched brand reply*: What we do —
  but we considered indexing brand replies instead (DM prompts, refund
  templates). Decided against: customers search for help with their problem,
  not a template reply.
- *Dense retrieval (FAISS + sentence-transformers)*: Better semantic matching,
  but adds a 420MB model dependency and ~5s per query on CPU.
- *BM25 (rank_bm25 library)*: Slightly better recall than TF-IDF for retrieval;
  added to requirements as optional. Chose TF-IDF for zero new dependencies.

**Why current approach**: The retrieved pair (customer msg → brand reply)
gives the ReplyGenerator a real historical example to ground its reply in.
Query-side similarity is the most natural way to surface relevant past cases.

---

## D-06: Escalation as Explicit Rules (not a trained classifier)

**Decision**: Implement escalation as an explicit 5-rule if/else decision tree.

**Alternatives considered**:
- *Train a binary classifier on "escalate vs auto_handle" labels*: Would require
  enough labelled escalation examples — the golden set (200 rows) is too small
  and too imbalanced for a reliable binary classifier.
- *LLM-based escalation judgement*: Could work but introduces API dependency and
  latency; also becomes a black box the spec explicitly forbids.
- *Scored rubric (weighted sum of signals)*: More nuanced than rules but harder
  to audit and explain. Rules are simpler and more defensible.

**Why rules**: The spec explicitly asks for "explicit, readable if/else (or a
small scored rubric) — not a black box." Rules are also trivially auditable:
you can trace exactly why a message was escalated from the `reason` string.

---

## D-07: Deterministic Local Fallback for LLM Judge

**Decision**: ROUGE-1 F1 + retrieval similarity heuristic as the no-API fallback.

**Alternatives considered**:
- *Return "N/A" with no score*: Makes the human-vs-judge agreement analysis
  impossible without an API key.
- *BLEU score*: BLEU is known to be unreliable for short text and requires
  multiple references; ROUGE-1 F1 is better calibrated for single-reference
  evaluation of support replies.
- *Sentence similarity via sentence-transformers*: Better proxy for semantic
  quality but adds large dependency.

**Why ROUGE-1 + retrieval sim**: Zero additional dependencies, deterministic
(same output every run), and directionally valid — a reply that overlaps more
with the historical reference and has a higher retrieval score is genuinely
more grounded.

---

## D-08: Stratified Sampling for Golden Set

**Decision**: Stratify the 200-row golden set across 8 intent clusters
(~25 rows per cluster).

**Alternatives considered**:
- *Proportional sampling*: Would give ~55 rows for `playback_issue` and ~18 for
  `content_catalog`. With a 20% test split, the minority class gets 3-4 test
  examples — too few for meaningful per-class metrics.
- *Random sampling*: Same problem as proportional — dominated by majority classes.

**Why stratified**: Each intent class gets enough test examples to compute
meaningful precision/recall. This is especially important for the
`misleading headline number` discussion (Section 10 of REPORT.md).

---

## D-09: config.yaml + .env Pattern (not argparse-only)

**Decision**: Use `config.yaml` for all defaults, `.env` for machine-specific
overrides (path to raw CSV, API key), with argparse for run-time overrides.

**Alternatives considered**:
- *All hardcoded defaults in Python*: Violates the "no machine-specific paths"
  constraint.
- *argparse only*: Users must re-type the raw path on every invocation.
- *Separate config per environment*: Overengineered for a single-machine project.

**Why config.yaml + .env**: Config.yaml is committed and shows all defaults;
.env is gitignored and holds machine-specific secrets. This pattern is standard
in production ML projects (MLflow, Hydra) and makes the repo fully portable.

---

## D-10: Two-Pass Architecture vs. Single-Pass

**Decision**: Scripts use two separate CSV passes (brand IDs first, then text).

**Trade-off**: 2× read time (60s total) vs. correctness.

**Why**: A single pass risks missing inbound messages that appear in earlier
chunks than the corresponding outbound reply. The CSV is not sorted by
conversation thread. Two-pass guarantees complete coverage.

---

## D-11: sample_data.py writes brand_threads.csv (all 41k) + sample.csv (5k)

**Decision**: Write both a full threads file (for retrieval index) and a smaller
sample (for training/evaluation).

**Why separate files**:
- The retrieval index benefits from maximum coverage (41k pairs).
- Intent clustering + golden set generation only need a manageable subset (5k).
- Training a LinearSVC on 5k labelled examples is as fast as on 200, and
  produces a more robust classifier.

---

## D-12: REPORT.md section 10 — "Misleading headline number"

**Decision**: Ground the misleading-number section in actual system properties:
golden set size, stratification method, and training/test data overlap.

**Alternative**: Write a generic observation about class imbalance without
reference to specific numbers.

**Why specific**: The spec says "ground it in your actual numbers (golden-set
size, class imbalance you observed, etc.)." Generic observations would fail
this criterion. Every claim in Section 10 of REPORT.md is traceable to a
specific design choice documented in this log.

---

## D-13: Windows UTF-8 Compatibility (PYTHONUTF8=1 / -X utf8)

**Decision**: Add `python -X utf8` flag to all pipeline commands in README.md;
add `io.TextIOWrapper` reconfiguration in scripts with Unicode print statements.

**Why**: Windows PowerShell defaults to cp1252 encoding. The TWCS dataset
contains emoji and non-ASCII characters that cause `UnicodeEncodeError` on
`print()`. The `-X utf8` flag is the cleanest fix for Python 3.7+.

---

## D-14: 80/20 Train/Test Split on 200-Row Golden Set

**Decision**: 80% train (160 rows), 20% test (40 rows), stratified by intent.

**Alternatives**:
- *5-fold cross-validation*: More reliable estimates but creates complexity in
  reporting; 200 rows is too small for CV to add meaningful value over a held-out set.
- *Leave-one-out*: Computationally cheap but variance is very high at n=200.

**Why 80/20**: Conventional, reproducible, gives at least 5 test examples per
class when stratified. The small test set is explicitly discussed in the
"misleading number" section of REPORT.md.

---

## D-15: No Sentence-Transformers Dependency in Core Pipeline

**Decision**: Keep `sentence-transformers` in requirements.txt but not as a
runtime dependency for the core agent.

**Why**: The spec says "pip install -r requirements.txt && python run_pipeline.py
works from a clean clone." Downloading sentence-transformer model weights
(~420MB) on first run would break the "under 15 minutes" constraint on a
slow connection. TF-IDF is used instead, which requires no model download.
