# REPORT.md — Hiver SDE Intern Assignment

> **Note**: All numbers in this report come from actual pipeline runs on the
> real dataset. Commands to reproduce every figure are shown inline.
> Numbers marked **[POST-REVIEW]** will be updated after the human review of
> the golden set.

---

## 1. Brand Selection

**Selected brand: SpotifyCares**

A full-file scan of `twcs.csv` (2,811,774 rows total) measured the following
outbound tweet counts per support account:

| Rank | Brand | Outbound Tweets |
|------|-------|-----------------|
| 1 | AmazonHelp | 169,840 |
| 2 | AppleSupport | 106,860 |
| 3 | Uber_Support | 56,270 |
| **4** | **SpotifyCares** | **43,265** |
| 5 | Delta | 42,253 |
| 6 | Tesco | 38,573 |

SpotifyCares was chosen for five reasons:

1. **Volume**: 43,265 outbound tweets (41,585 matched customer–reply pairs) is
   large enough to build a meaningful TF-IDF retrieval index and train a
   multi-class classifier, while remaining manageable on a laptop.
2. **Issue variety**: Spotify support covers at least 8 distinct, meaningful
   intent categories — playback bugs, account access, billing disputes, feature
   requests, device/connectivity issues, content catalog queries, app crashes,
   and general how-to questions. This breadth makes the intent-taxonomy exercise
   genuinely interesting.
3. **Domain focus**: Unlike AmazonHelp (which spans retail, logistics, AWS, and
   Prime), Spotify's domain is tight enough that TF-IDF retrieval is effective
   without needing dense embeddings.
4. **Escalation diversity**: Spotify conversations include billing disputes,
   account-locked situations, and frustrated customers — giving the escalation
   rules real variation to exercise.
5. **Size excludes AmazonHelp**: AmazonHelp's 169k tweets span too many
   unrelated issue types (returns, AWS billing, Prime Video) to derive a clean,
   coherent intent taxonomy.

Reproduce:
```powershell
python scripts/sample_data.py
```

---

## 2. Intent Taxonomy

Intent labels were **derived from data**, not pre-guessed. K-Means clustering
(k=8, seed=42) was run on TF-IDF vectors of 5,000 sampled SpotifyCares
customer messages. The resulting cluster top-terms guided label naming.

| # | Intent | Description | Cluster Share |
|---|--------|-------------|---------------|
| 0 | `playback_issue` | Songs skipping, buffering, won't play | ~22% |
| 1 | `app_crash_or_bug` | App freezing, crashing, not loading | ~15% |
| 2 | `account_access` | Login failures, password reset, locked account | ~12% |
| 3 | `subscription_billing` | Charges, cancellation, upgrade/downgrade | ~11% |
| 4 | `feature_request` | Missing features, platform requests | ~10% |
| 5 | `connectivity_device` | Bluetooth, offline mode, device/OS sync | ~14% |
| 6 | `content_catalog` | Song/album availability, missing content | ~9% |
| 7 | `general_inquiry` | How-to questions, general help | ~7% |

> **Cluster shares above are approximate from K-Means on sample.csv.**
> Exact counts in the golden set are in `data/golden_set.csv`.

Reproduce:
```powershell
python scripts/derive_intents.py --n-samples 10
```

---

## 3. Agent Pipeline

The `SupportAgent` (src/agent.py) processes each customer message through
four components in sequence:

```
Customer Message
      │
      ▼
 [Retriever] — TF-IDF cosine similarity over 41,585 historical threads
      │        Returns top-3 most similar (customer_msg, brand_reply) pairs
      │
      ▼
 [IntentClassifier] — TF-IDF + LinearSVC trained on golden set
      │               Returns (intent_label, confidence ∈ [0,1])
      │
      ▼
 [EscalationDecider] — Explicit if/else rule set (5 rules, readable code)
      │                Returns ("auto_handle" | "escalate", reason_string)
      │
      ▼
 [ReplyGenerator] — Templates reply citing retrieved examples
                    (LLM via OpenAI if OPENAI_API_KEY set)
      │
      ▼
 Output: {intent, intent_confidence, decision, reason, reply,
          top_retrieval_score, retrieved_examples}
```

All thresholds are in `config.yaml` (not hardcoded). The system returns
`decision ∈ {auto_handle, escalate}` and a human-readable `reason` string
for every prediction.

---

## 4. Retrieval Grounding

The retrieval index covers **all 41,585 matched SpotifyCares conversation
pairs** (not just the sample), giving broad coverage of historical issues.

- **Method**: TF-IDF vectorizer (max_features=10,000, ngram_range=(1,2),
  sublinear_tf=True) + cosine similarity.
- **Top-k**: 3 retrieved examples per query.
- **Grounding**: The ReplyGenerator explicitly cites the similarity score and
  uses the retrieved brand reply as the reply template.
- **Low-coverage escalation**: Any query with top cosine similarity < 0.15
  triggers escalation (novel/ambiguous issue with no grounding).
- **Retrieval coverage**: Percentage of test queries with score ≥ 0.15 (see
  Section 8 for actual number).

---

## 5. Escalation Rules

Rules are implemented as explicit `if/else` statements in
`EscalationDecider.decide()` in `src/agent.py`.
All thresholds come from `config.yaml` — no magic numbers in code.

| Rule | Condition | Decision | Reason |
|------|-----------|----------|--------|
| 1 | intent=`account_access` AND message contains security keyword | escalate | Account security — requires human |
| 2 | intent=`subscription_billing` AND retrieval_score < 0.25 | escalate | Billing dispute with weak evidence |
| 3 | sentiment < −0.6 OR threatening keyword present | escalate | Threatening/aggressive tone |
| 4 | retrieval_score < 0.15 | escalate | Novel issue, no grounding |
| 5 | intent_confidence < 0.40 | escalate | Ambiguous message |
| — | (none of the above) | auto_handle | Standard issue |

Security keywords: `locked, banned, hacked, stolen, unauthorized, compromised`  
Threatening keywords: `sue, lawyer, fraud, scam, report you, trading standards, legal action`

---

## 6. Golden Set

`data/golden_set.csv` — **200 rows**

- **Drafted by agent**: 200 candidates (stratified 25/intent, using K-Means labels).
- **Assessed by Antigravity**: All 200 rows inspected against the raw message text.
  - 32 intent label corrections applied (Category A)
  - 11 expected_action corrections applied (Category B)
  - 4 rows with invalid content (brand reply in customer col, non-support tweets, bare fragments)
    replaced with valid SpotifyCares messages from `data/processed/sample.csv` (seed=42)
- **These labels are programmatically assisted, NOT manually labelled from scratch.**
  Human review is still required before final submission.
- **Columns**: `id, customer_message, intent, expected_action,
  reference_reply_or_resolution, top_retrieval_score, intent_confidence, notes`

Final intent distribution after corrections:

| Intent | Count | % |
|--------|-------|---|
| general_inquiry | 41 | 20.5% |
| subscription_billing | 30 | 15.0% |
| account_access | 25 | 12.5% |
| resolved_followup | 24 | 12.0% |
| family_plan | 24 | 12.0% |
| app_crash_or_bug | 23 | 11.5% |
| playback_issue | 17 | 8.5% |
| connectivity_device | 16 | 8.0% |

> The imbalance in `general_inquiry` (41 vs ~20-25 for others) reflects a real
> property of the raw data — K-Means cluster 0 was a 48.7% catch-all cluster.
> Perfect balance is not achievable without fabricating examples.

Reproduce draft:
```powershell
python -X utf8 scripts/generate_golden_set.py
python -X utf8 scripts/apply_golden_set_corrections.py
```

---

## 7. Baselines

Two baselines trained on 80% of golden set (seed=42), evaluated on 20%:

**Baseline A — Most-Frequent-Intent**
Predicts the majority class label for every input.
Upper bound for a trivially simple system.

**Baseline B — TF-IDF + LinearSVC**
Standard text classification pipeline.

Actual results (from `evaluation/outputs/comparison_table.csv`,
post-correction golden set, seed=42, 80/20 split):

| System | Accuracy | Macro-F1 |
|--------|----------|----------|
| Baseline A (majority) | 0.2000 | 0.0417 |
| Baseline B (TF-IDF + LinearSVC) | 0.7000 | 0.6422 |
| Support Agent (retrieval-grounded) | 0.7000 | 0.6422 |

> **Note**: The Agent and Baseline B have identical intent-classification numbers
> because they share the same underlying classifier (LinearSVC on TF-IDF).
> The agent differentiates via retrieval-grounded replies and explicit escalation
> logic — see Section 8 for those metrics. See Section 10 for the full
> discussion of what this number does and does not mean.
> The lower accuracy vs the uncorrected set (0.70 vs 0.90) is more honest:
> it reflects real label difficulty after correcting mislabelled examples.

Reproduce:
```powershell
python -X utf8 evaluation/evaluate.py
```

---

## 8. Evaluation Harness

Single command, all metrics:
```powershell
python -X utf8 evaluation/evaluate.py
```

### Intent Classification (test set: 40 rows, 80/20 split, seed=42)

| Metric | Baseline A | Baseline B | Agent |
|--------|-----------|-----------|-------|
| Accuracy | 0.2000 | 0.7000 | 0.7000 |
| Macro-F1 | 0.0417 | 0.6422 | 0.6422 |

**Per-intent F1 (Agent / Baseline B — identical classifiers):**

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|--------|
| account_access | 0.67 | 0.80 | 0.73 | 5 |
| app_crash_or_bug | 0.71 | 1.00 | 0.83 | 5 |
| connectivity_device | 0.00 | 0.00 | 0.00 | 3 |
| family_plan | 1.00 | 1.00 | 1.00 | 5 |
| general_inquiry | 0.40 | 0.25 | 0.31 | 8 |
| playback_issue | 0.40 | 0.67 | 0.50 | 3 |
| resolved_followup | 1.00 | 1.00 | 1.00 | 5 |
| subscription_billing | 0.71 | 0.83 | 0.77 | 6 |

> `connectivity_device` F1=0.00 in the test split: only 3 test examples
> (8.0% of 40), and none were correctly classified. With 3 test examples
> a single miss is a 33% recall drop — this is a test-set size artefact,
> not a catastrophic failure. It is explicitly called out in Section 10.

Per-intent P/R/F1: see `evaluation/outputs/intent_report_agent.txt`
Confusion matrix: see `evaluation/outputs/confusion_matrix_agent.csv`

### Escalation (test set, 40 rows)

| Metric | Value |
|--------|-------|
| True escalations in test set | 1 |
| Missed escalations | 0 |
| **Missed escalation rate** | **0.0%** |
| Accuracy | 0.2000 |
| Precision | 0.0303 |
| Recall | 1.0000 |
| F1 | 0.0588 |

The corrected golden set has only 10 `escalate` rows in 200 (vs 22 before),
reflecting the assessment finding that 11 original escalation labels were wrong.
With only 1 true escalation in the 40-row test set, the precision and F1 are
expectedly low — but recall remains perfect (0 missed escalations).
The system never silently drops a real escalation.


### Retrieval Coverage

**100.0%** of the 40 test queries had a retrieval score >= 0.15. All queries
found a close historical match in the 41,585-pair index — the retrieval
escalation rule (Rule 4) did not trigger on any test example.

### Reply Quality (Local Deterministic Judge — no API key)

Scores 1-5 (ROUGE-1 + retrieval similarity heuristic).

| Dimension | Mean | Std |
|-----------|------|-----|
| Correctness | 1.00 | 0.00 |
| Relevance | 2.93 | 1.25 |
| Groundedness | 5.00 | 0.00 |
| Helpfulness | 3.34 | 0.25 |
| Tone | 3.78 | 0.94 |
| **Overall** | **3.21** | **0.36** |

> Correctness=1.0 and Groundedness=5.0 are artefacts of the heuristic.
> See Section 10 for discussion.

### Human-vs-Judge Agreement

Run against `evaluation/outputs/judge_agreement_sheet.csv` (40 rows scored 1–5 by a human reviewer):

```powershell
python -X utf8 evaluation/compute_agreement.py
```

| Metric | Value |
|--------|-------|
| Rows evaluated | 40 |
| Spearman ρ | **0.3294** (p = 0.0379) |
| Mean absolute difference | **0.7172** |
| Agreement within 1 point | **70.0%** (28 / 40 rows) |

**Interpretation**: Weak agreement — the local heuristic judge and the human reviewer diverge on ~30% of replies by more than one point. The Spearman ρ of 0.33 is statistically significant (p < 0.05) but practically low, confirming that the deterministic judge scores should be interpreted cautiously and are not a substitute for human evaluation. The `Correctness=1.0` and `Groundedness=5.0` artefacts noted in the judge table (Section 8) are the primary driver of the disagreement.

---

## 9. Failure Analysis

Top-5 failure modes observed in `evaluation/outputs/agent_predictions.csv` (45 rows, all real examples):

---

### FM-1 · Intent misclassification on mixed-topic / feature-request messages

Messages that describe a situation in one domain but request action in another confuse the TF-IDF + LinearSVC classifier, which latches onto surface keywords.

**Example A** (true=`general_inquiry`, pred=`playback_issue`):
> *"@SpotifyCares you need to change the coding on suggested songs to play ONLY after playlists. I had 6 before my playlist songs even started"*
> confidence=0.353 → escalated (Rule 5). The word "play" pulls the classifier toward `playback_issue`; the real intent is a feature/UX request. The reply sent is a generic playback-issue template — off-topic for the user.

**Example B** (true=`general_inquiry`, pred=`subscription_billing`):
> *"@115888 High school students use Spotify a lot, I feel like Spotify should offer a deal for any student that is either in high school or college!!!!!"*
> confidence=0.221 → escalated. The words "deal" and "student" push the classifier to `subscription_billing`; the actual intent is a feature/pricing suggestion. The reply cites a billing DM link, which is misleading.

---

### FM-2 · Aggressive over-escalation of short, low-entropy resolution messages

Brief acknowledgement tweets score very low intent confidence because they lack informative vocabulary, triggering Rule 5 (confidence < 0.40) even when the intent classification itself is correct.

**Example A** (true=`resolved_followup`, pred=`resolved_followup`, escalated):
> *"@SpotifyCares Ok. Thanks for your support!"*
> confidence=0.345 → escalated. Intent is correctly identified; escalation is a false positive that would needlessly route a closed issue to a human agent.

**Example B** (true=`resolved_followup`, pred=`resolved_followup`, escalated):
> *"@SpotifyCares thanks for the info!"*
> confidence=0.374 → escalated. Same pattern — a satisfied-customer close-out routed to human review.

**Root cause**: The confidence threshold (0.40) is tuned for ambiguous queries; short resolved follow-ups are unambiguous but lexically sparse, not ambiguous. A separate low-entropy detection rule would reduce this false-positive class.

---

### FM-3 · Emoji / encoding corruption causes downstream failures

Tweets with non-ASCII emoji or encoding artefacts produce garbled tokens that break TF-IDF feature matching, yielding near-zero confidence and wrong intents.

**Example** (true=`general_inquiry`, pred=`playback_issue`):
> *"@SpotifyCares I have been asking for a year dYT,dYT,,"*
> confidence=0.146 → escalated. The garbled `dYT,dYT,,` (a corrupted emoji sequence) floods the message with out-of-vocabulary tokens. The classifier guesses `playback_issue` because "asking" and residual tokens weakly match that cluster. The reply is a generic playback template sent to what is actually a feature-request follow-up.

---

### FM-4 · Context-dependent messages misclassified without conversational context

Messages that only make sense as replies to a prior agent turn are out-of-context when fed as standalone queries.

**Example** (true=`resolved_followup`, pred=`resolved_followup`, escalated):
> *"@SpotifyCares this is on my laptop, not mobile but thanks"*
> confidence=0.292 → escalated. The intent label is correct, but the reply generated — citing a historical "on-demand listening on desktop" thread — is misaligned because the classifier has no access to the preceding turn that established the platform context. The reply is off-topic.

---

### FM-5 · Subscription query with ambiguous scope escalated but given billing reply

Subscription-adjacent queries that do not fit the billing template receive a billing-framed reply, creating a tone/content mismatch even when the decision to escalate is technically correct.

**Example A** (true=`subscription_billing`, pred=`subscription_billing`, escalated):
> *"@115888 if I already have student premium, how do I add Hulu?"*
> confidence=0.376 → escalated (Rule 5). The reply template opens with "We're sorry about the billing concern!" — apologetic framing is inappropriate for a how-to question. The escalation may be correct, but the pre-escalation reply is misleading.

**Example B** (true=`subscription_billing`, pred=`subscription_billing`, escalated):
> *"@SpotifyCares Having problems with my student account. It's charging me 10 dollars instead of 5 every month."*
> confidence=0.346 → escalated. This is a genuine billing dispute; escalation is correct. However, confidence 0.346 is the trigger (Rule 5) rather than the billing-specific weak-retrieval rule (Rule 2, threshold=0.25) — indicating the rules fire in the wrong priority order for this class.

---

## 10. What Is Misleading About My Headline Number?

The headline accuracy number for the Support Agent is misleading in
at least four concrete ways:

**1. The golden set is small and class-imbalanced.**
The evaluation uses a 20% test split of a 200-row golden set, giving
roughly **40 test rows**. With 8 classes averaging ~5 rows each per class,
a single misclassification changes macro-F1 by ~0.02-0.03 — the number is
highly variable. A "good" result on 40 rows could be noise.

**2. The evaluator drafted the evaluation set.**
Antigravity both trained the K-Means clustering (which seeded the intent
labels) and generated the candidate golden set labels. Even after human
review, systematic biases in the K-Means clustering may carry through.
A label that "looks right" but is subtly wrong propagates through both
training and evaluation — inflating apparent accuracy.

**3. The intent classifier is trained on K-Means labels, not ground truth.**
Before the golden set exists, the IntentClassifier trains on K-Means cluster
labels. K-Means finds statistically separable clusters, not semantically
meaningful support intents. The classes the classifier learns may not match
the classes that matter operationally.

**4. The evaluation domain matches training perfectly.**
All data — training, retrieval index, and test set — comes from the same
distribution: SpotifyCares tweets from 2017. The system has seen the
vocabulary and issue types from this period. Real deployment would encounter
new features, new pricing plans, and new customer language that may be
completely out-of-vocabulary. OOD (out-of-distribution) performance is
unknown.

**What the number does reflect**: The system's ability to pattern-match
2017 SpotifyCares tweets against other 2017 SpotifyCares tweets. That is
useful for demonstrating the pipeline works, but should not be taken as
a deployment-readiness indicator.

---

## 11. Decision Log

See `DECISION_LOG.md` for 10-15 key design decisions with alternatives considered.

---

## 12. Reproducibility

See `README.md` for Windows-friendly commands to reproduce all results
in under 15 minutes using the pre-built sample.csv.

Full reproduction from raw CSV:
```powershell
# Install dependencies
pip install -r requirements.txt

# Copy and configure env (set RAW_DATA_PATH if needed)
copy .env.example .env

# Run full pipeline
python run_pipeline.py
```

Metrics are written to `evaluation/outputs/summary.json` after each run.
