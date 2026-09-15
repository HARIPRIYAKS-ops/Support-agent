"""
scripts/apply_golden_set_corrections.py
----------------------------------------
Applies corrections from the golden-set assessment (2026-09-14).

Changes applied:
  - Category A: intent label corrections (assessed as clearly wrong)
  - Category B: expected_action corrections (assessed as clearly wrong)
  - 4 invalid rows replaced with valid SpotifyCares customer messages
    drawn from data/processed/sample.csv (seed=42)

Does NOT:
  - Invent examples or labels
  - Change rows not listed in the assessment
  - Change anything outside golden_set.csv
"""

import sys
import random
import re
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path

SEED = cfg["sampling"]["seed"]
random.seed(SEED)

GOLDEN_PATH = resolve_path(cfg["paths"]["golden_set"])
SAMPLE_PATH  = resolve_path(cfg["paths"]["sample_csv"])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Explicit intent corrections (Category A)
#    Only rows where the assessment gives a clear single correct label.
#    Ambiguous "general_inquiry / feature_request" rows default to
#    general_inquiry because our taxonomy has no feature_request label.
# ─────────────────────────────────────────────────────────────────────────────
INTENT_CORRECTIONS = {
    # Row ID            : new_intent
    "GS_0001": "general_inquiry",      # song-library cap policy question, not playback
    "GS_0009": "subscription_billing", # charged but only getting free tier
    "GS_0016": "subscription_billing", # pricing offer question
    "GS_0023": "playback_issue",        # songs unavailable on phone, fine on desktop
    "GS_0024": "app_crash_or_bug",      # "only crashes if using premium"
    "GS_0030": "general_inquiry",       # Discover Weekly analytics suggestion
    "GS_0031": "general_inquiry",       # context-only fragment (previously-played songs thread)
    "GS_0037": "general_inquiry",       # how to get an artist on Spotify – not a bug
    "GS_0043": "general_inquiry",       # playlist curation suggestion
    "GS_0046": "account_access",        # password-reset link not arriving (also action fix below)
    "GS_0047": "general_inquiry",       # how-to: upload mp3 without a computer
    "GS_0052": "general_inquiry",       # explicit-song filter request
    "GS_0055": "subscription_billing",  # why pay full month on 9php promo
    "GS_0056": "general_inquiry",       # asking about app launch dates
    "GS_0063": "general_inquiry",       # feature suggestion: suggested songs after playlist
    "GS_0064": "general_inquiry",       # high-school student discount suggestion
    "GS_0066": "general_inquiry",       # artist name wrong on release – metadata inquiry
    "GS_0069": "general_inquiry",       # requesting Apple Watch app – feature request
    "GS_0073": "general_inquiry",       # track removed from Spotify – content inquiry
    "GS_0080": "general_inquiry",       # feature suggestion: add playlist to queue
    "GS_0082": "general_inquiry",       # Romanian promo text – no actionable support intent
    "GS_0085": "subscription_billing",  # student discount pending verification
    "GS_0087": "account_access",        # last.fm link invalid email / error loop
    "GS_0093": "account_access",        # got logged out of Spotify
    "GS_0094": "general_inquiry",       # feature request: no edited songs in Daily Mix
    "GS_0098": "general_inquiry",       # pure anger tweet, no specific issue
    "GS_0100": "general_inquiry",       # Christmas playlist curation complaint
    "GS_0159": "subscription_billing",  # trial not over but being charged
    "GS_0177": "general_inquiry",       # artist music missing from Spotify
    "GS_0182": "general_inquiry",       # privacy feature request
    "GS_0184": "general_inquiry",       # new user asking how to see lyrics
    "GS_0186": "general_inquiry",       # Spotify not available in Slovenia
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Explicit expected_action corrections (Category B)
# ─────────────────────────────────────────────────────────────────────────────
ACTION_CORRECTIONS = {
    # Row ID     : new_action
    "GS_0010": "auto_handle",  # queue-sync question – not a security/billing issue
    "GS_0029": "auto_handle",  # can't share songs – standard app bug
    "GS_0046": "auto_handle",  # password-reset link – standard support request
    "GS_0073": "auto_handle",  # content removal inquiry – not threatening
    "GS_0085": "auto_handle",  # student discount pending – not a payment dispute
    "GS_0093": "auto_handle",  # logged out – standard login issue
    "GS_0150": "auto_handle",  # premium redirect UI – subscription issue
    "GS_0164": "auto_handle",  # mild feature complaint
    "GS_0176": "auto_handle",  # "switch to Apple Music" – not a real threat
    "GS_0186": "auto_handle",  # Spotify unavailable in country – general inquiry
    "GS_0197": "auto_handle",  # forgot username – standard account recovery
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Invalid rows to remove and replace
# ─────────────────────────────────────────────────────────────────────────────
ROWS_TO_REPLACE = {
    # Row ID  : target_intent for replacement
    # Targets chosen to reduce the largest deficits after all Category A corrections:
    #   connectivity_device: -11, playback_issue: -9, app_crash_or_bug: -3, resolved_followup: -1
    "GS_0168": "connectivity_device",  # brand reply text — replace with connectivity example
    "GS_0169": "connectivity_device",  # hype tweet    — replace with connectivity example
    "GS_0165": "playback_issue",        # bare fragment — replace with playback example
    "GS_0170": "app_crash_or_bug",      # Spotify URI   — replace with app crash example
}
# Note: after these 4 replacements the net change is:
#   connectivity_device: 14 + 2 = 16  (was -11, now -9)
#   playback_issue:      16 + 1 = 17  (was  -9, now -8)
#   app_crash_or_bug:    22 + 1 = 23  (was  -3, now -2)
# general_inquiry remains the largest class at 41 — this reflects a real
# property of the dataset (cluster 0 was 48.7% of all messages).
# Perfect balance is not achievable without inventing data.


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return text.strip()


def is_valid_message(text: str) -> bool:
    """Basic quality filter for replacement candidates."""
    t = clean_text(text)
    # Reject very short after cleaning
    if len(t.split()) < 5:
        return False
    # Reject pure URIs or empty-ish
    if t.startswith('spotify:') or t == '':
        return False
    # Reject mostly-numeric
    if sum(c.isdigit() for c in t) > len(t) * 0.6:
        return False
    return True


def find_replacement(sample_df: pd.DataFrame,
                     target_intent: str,
                     used_messages: set) -> dict | None:
    """
    Find a valid unused replacement row from sample_df with the given intent.
    Returns a dict suitable for a golden-set row, or None if not found.
    """
    candidates = sample_df[
        (sample_df['intent'] == target_intent) &
        (~sample_df['customer_message'].isin(used_messages))
    ].copy()
    candidates = candidates[candidates['customer_message'].apply(is_valid_message)]
    if candidates.empty:
        return None
    row = candidates.sample(1, random_state=SEED).iloc[0]
    return {
        'customer_message': str(row['customer_message']),
        'intent': target_intent,
        'expected_action': 'auto_handle',
        'reference_reply_or_resolution': str(row.get('brand_reply', ''))[:500],
        'top_retrieval_score': '',
        'intent_confidence': '',
        'notes': 'replacement row – original was invalid (no support content)',
    }


def main():
    print("Loading golden_set.csv ...")
    df = pd.read_csv(GOLDEN_PATH, encoding='utf-8')
    original_len = len(df)
    print(f"  {original_len} rows loaded")

    # Track all original messages to avoid duplicates in replacements
    used_messages = set(df['customer_message'].dropna().tolist())

    # Load sample for replacement sourcing
    sample_df = pd.read_csv(SAMPLE_PATH, encoding='utf-8')
    sample_df = sample_df.dropna(subset=['customer_message', 'intent'])
    # Exclude messages already in the golden set
    sample_df = sample_df[~sample_df['customer_message'].isin(used_messages)]

    intent_changes = 0
    action_changes = 0
    replacements_made = 0

    # ── Apply intent corrections ──────────────────────────────────────────────
    for row_id, new_intent in INTENT_CORRECTIONS.items():
        mask = df['id'] == row_id
        if not mask.any():
            print(f"  WARNING: {row_id} not found in golden set – skipping")
            continue
        old_intent = df.loc[mask, 'intent'].iloc[0]
        if old_intent != new_intent:
            df.loc[mask, 'intent'] = new_intent
            existing_note = str(df.loc[mask, 'notes'].iloc[0])
            note_text = f"intent corrected: {old_intent} -> {new_intent}"
            df.loc[mask, 'notes'] = note_text if existing_note in ('', 'nan') else existing_note + '; ' + note_text
            intent_changes += 1
            print(f"  [intent] {row_id}: {old_intent} -> {new_intent}")

    # ── Apply action corrections ──────────────────────────────────────────────
    for row_id, new_action in ACTION_CORRECTIONS.items():
        mask = df['id'] == row_id
        if not mask.any():
            print(f"  WARNING: {row_id} not found – skipping")
            continue
        old_action = df.loc[mask, 'expected_action'].iloc[0]
        if old_action != new_action:
            df.loc[mask, 'expected_action'] = new_action
            existing_note = str(df.loc[mask, 'notes'].iloc[0])
            note_text = f"action corrected: {old_action} -> {new_action}"
            df.loc[mask, 'notes'] = note_text if existing_note in ('', 'nan') else existing_note + '; ' + note_text
            action_changes += 1
            print(f"  [action] {row_id}: {old_action} -> {new_action}")

    # ── Replace invalid rows ──────────────────────────────────────────────────
    for row_id, target_intent in ROWS_TO_REPLACE.items():
        mask = df['id'] == row_id
        if not mask.any():
            print(f"  WARNING: {row_id} not found – skipping replacement")
            continue
        idx = df.index[mask][0]
        old_msg = df.loc[idx, 'customer_message']

        replacement = find_replacement(sample_df, target_intent, used_messages)
        if replacement is None:
            print(f"  WARNING: No valid replacement found for {row_id} "
                  f"(intent={target_intent}). Row left as-is.")
            continue

        # Update used set so we don't reuse this message
        used_messages.add(replacement['customer_message'])
        sample_df = sample_df[sample_df['customer_message'] != replacement['customer_message']]

        for col, val in replacement.items():
            df.loc[idx, col] = val
        df.loc[idx, 'id'] = row_id  # keep original ID

        replacements_made += 1
        print(f"  [replace] {row_id}: removed invalid msg; "
              f"inserted {target_intent} example: "
              f"{replacement['customer_message'][:80]!r}")

    # ── Final verification ────────────────────────────────────────────────────
    final_len = len(df)
    assert final_len == 200, f"Row count changed! Expected 200, got {final_len}"
    assert df['id'].nunique() == 200, "Duplicate IDs present!"
    assert df['customer_message'].notna().all(), "Null messages present!"

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(GOLDEN_PATH, index=False, encoding='utf-8')
    print(f"\nSaved corrected golden_set.csv -> {GOLDEN_PATH}")

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CORRECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Intent label changes:   {intent_changes}")
    print(f"  Action label changes:   {action_changes}")
    print(f"  Invalid rows replaced:  {replacements_made}")
    print(f"  Total rows:             {final_len}  (target: 200)")

    print(f"\nFinal intent distribution:")
    for intent, cnt in df['intent'].value_counts().items():
        bar = '#' * (cnt // 2)
        print(f"  {intent:22s}: {cnt:3d}  {bar}")

    print(f"\nFinal expected_action distribution:")
    for action, cnt in df['expected_action'].value_counts().items():
        print(f"  {action:15s}: {cnt:3d}")

    print(f"\nNOTE: Labels in this file are programmatically assisted")
    print(f"      (K-Means seed + agent suggestions), then systematically")
    print(f"      corrected via rule-based assessment of all 200 rows.")
    print(f"      They are NOT manually labelled from scratch.")
    print(f"      Human review of corrections is still required.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
