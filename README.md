Customer Support Agent

AI agent pipeline for SpotifyCares support using the Twitter Customer Support dataset (TWCS).

## Quick Start (≤15 minutes, Windows)

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & Install

```powershell
git clone <repo-url>
cd Hiver_project
pip install -r requirements.txt
```

### 2. Configure paths (first-time only)

Copy the env template:
```powershell
copy .env.example .env
```

If you have the raw `twcs.csv`, set its path in `.env`:
```
RAW_DATA_PATH=C:\path\to\twcs.csv
```

If you're reproducing from the pre-built `data/processed/sample.csv` only (faster), skip this.

### 3. Run the pipeline

**From scratch (needs raw CSV):**
```powershell
python -X utf8 run_pipeline.py
```

**From pre-built sample (skip 60s CSV scan):**
```powershell
python -X utf8 run_pipeline.py --sample-only
```

⚠️ The pipeline will pause after generating `data/golden_set.csv` and ask you
to review the labels before evaluation. See "Manual Steps" below.

### 4. Get all metrics (standalone)

```powershell
python -X utf8 evaluation/evaluate.py
```

Output: `evaluation/outputs/comparison_table.csv`, `summary.json`, and report files.

---

## Measured Results (Final, post-correction golden set — 2026-09-15)

| System | Accuracy | Macro-F1 |
|--------|----------|----------|
| Baseline A (majority class) | 0.2000 | 0.0417 |
| Baseline B (TF-IDF + LinearSVC) | 0.7000 | 0.6422 |
| Support Agent | 0.7000 | 0.6422 |

> **Note**: The lower accuracy vs a pre-correction run (0.90) is intentional and
> more honest — it reflects the corrected golden set labels (32 intent corrections,
> 11 escalation corrections applied). See REPORT.md Section 7–8 for full discussion.

Escalation: Recall=1.00, Missed escalation rate=0%
Retrieval coverage: 100% (all 40 test queries found a similar example)
Reply quality (local judge): Overall=3.21/5
Human-vs-Judge agreement: Spearman ρ=0.3294, MAD=0.7172, within-1-point=70%

Full numbers in `evaluation/outputs/summary.json`.

---

## Project Structure

```
Hiver_project/
├── config.yaml                  # All defaults (no machine-specific paths)
├── .env.example                 # Copy to .env and set RAW_DATA_PATH
├── .gitignore                   # data/raw/ is gitignored
├── requirements.txt
├── run_pipeline.py              # End-to-end orchestrator
│
├── src/
│   ├── config_loader.py         # Reads config.yaml + .env
│   ├── agent.py                 # Core pipeline (retriever + intent + escalation + reply)
│   └── baselines.py             # Baseline A (majority) + Baseline B (TF-IDF+SVC)
│
├── scripts/
│   ├── sample_data.py           # Chunked CSV reader → sample.csv + brand_threads.csv
│   ├── build_index.py           # TF-IDF index builder
│   ├── derive_intents.py        # K-Means clustering → intent labels
│   └── generate_golden_set.py   # Drafts 200-row golden set candidates
│
├── evaluation/
│   ├── evaluate.py              # Full evaluation harness (one command)
│   ├── llm_judge.py             # LLM or local-fallback reply quality scorer
│   ├── judge_agreement_sheet.py # Generates human scoring sheet
│   └── compute_agreement.py     # Computes Spearman ρ after you fill in scores
│
├── data/
│   ├── raw/                     # ← gitignored; put twcs.csv here
│   ├── processed/               # sample.csv, brand_threads.csv, tfidf_index.pkl
│   └── golden_set.csv           # 200-row evaluation set (review required!)
│
├── REPORT.md                    # Full project report (all 12 sections)
└── DECISION_LOG.md              # 15 design decisions with alternatives
```

---

## Key Design Decisions

- **Brand**: SpotifyCares — 43,265 outbound tweets, 41,585 matched customer messages
- **No full CSV in memory**: Two-pass chunked read (100k rows/chunk)
- **Intent taxonomy**: 8 labels derived from K-Means clustering (not pre-guessed)
- **Escalation**: 5 explicit if/else rules (thresholds in config.yaml, not code)
- **LLM fallback**: ROUGE-1 + retrieval similarity if no `OPENAI_API_KEY`
- **Fixed seed**: `seed=42` everywhere; fully reproducible

---

## Manual Steps Required Before Submission

These two steps **must be done by you** — Antigravity can't do them:

### 1. Review the golden set (30-45 min)

```powershell
# Open in Excel or similar:
start data\golden_set.csv
```

For each row, check:
- Is the `intent` label correct? (e.g. is "account_access" the right label for this message?)
- Is `expected_action` correct? (`auto_handle` vs `escalate`)
- Add notes in the `notes` column where you corrected something

After reviewing, re-run evaluation:
```powershell
python -X utf8 evaluation/evaluate.py
```

### 2. Fill in judge-agreement scores (15 min)

```powershell
# Generate the sheet (already done if you ran the pipeline):
python -X utf8 evaluation/judge_agreement_sheet.py

# Open and fill in 'your_score' column (1-5):
start evaluation\outputs\judge_agreement_sheet.csv
```

Then compute correlation:
```powershell
python -X utf8 evaluation/compute_agreement.py
```

### 3. Read REPORT.md end-to-end

Make sure you can defend every number live, especially:
- Why Baseline B and Agent have the same accuracy (Section 7-8)
- The "misleading headline number" discussion (Section 10)
- The escalation precision vs recall tradeoff

---

## Adding an LLM Key (optional)

Add to `.env`:
```
OPENAI_API_KEY=sk-...
```

This enables:
- GPT-4o-mini reply generation in `src/agent.py`
- GPT-4o-mini judge scoring in `evaluation/llm_judge.py`

No code changes needed — the system detects the key automatically.

---

## Windows Notes

- Always use `python -X utf8` prefix (or set `PYTHONUTF8=1`) to avoid cp1252 Unicode errors
- Or add `$env:PYTHONUTF8=1` to your PowerShell profile

## Reproducing from Scratch

```powershell
pip install -r requirements.txt
copy .env.example .env
# Edit .env: set RAW_DATA_PATH=C:\path\to\twcs.csv
python -X utf8 scripts/sample_data.py
python -X utf8 scripts/build_index.py
python -X utf8 scripts/derive_intents.py
python -X utf8 scripts/generate_golden_set.py
# ← REVIEW data/golden_set.csv HERE (30-45 min)
python -X utf8 evaluation/evaluate.py
python -X utf8 evaluation/llm_judge.py
python -X utf8 evaluation/judge_agreement_sheet.py
# ← FILL IN your_score in evaluation/outputs/judge_agreement_sheet.csv (15 min)
python -X utf8 evaluation/compute_agreement.py
```
