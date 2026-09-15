"""
scripts/sample_data.py
---------------------
Reads the raw TWCS CSV in chunks (never loads full file into memory),
filters to the configured brand (SpotifyCares), and writes two outputs:

  data/processed/sample.csv       – up to `sampling.sample_size` inbound customer
                                    messages that SpotifyCares replied to
  data/processed/brand_threads.csv – matched (customer_message, brand_reply) pairs
                                    for retrieval grounding (all pairs, not just sample)

Run:
    python scripts/sample_data.py
    python scripts/sample_data.py --raw-path /path/to/twcs.csv  # override
"""

import sys
import os
import argparse
import random
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── allow running as script from repo root ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import cfg, resolve_path, get_raw_data_path

SEED = cfg["sampling"]["seed"]
random.seed(SEED)

BRAND = cfg["brand"]["name"]
SAMPLE_SIZE = cfg["sampling"]["sample_size"]
PROCESSED_DIR = resolve_path(cfg["paths"]["processed_dir"])


def parse_args():
    p = argparse.ArgumentParser(description="Build processed sample from raw TWCS CSV")
    p.add_argument("--raw-path", default=None,
                   help="Override path to twcs.csv (default: config.yaml / RAW_DATA_PATH env var)")
    p.add_argument("--chunksize", type=int, default=100_000)
    return p.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_path) if args.raw_path else get_raw_data_path()

    if not raw_path.exists():
        print(f"ERROR: Raw data not found at {raw_path}", file=sys.stderr)
        print("  Set RAW_DATA_PATH in .env, or use --raw-path flag.", file=sys.stderr)
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {raw_path} in chunks of {args.chunksize:,}...")
    print(f"Brand: {BRAND}  |  Target sample size: {SAMPLE_SIZE:,}")

    # ── Pass 1: collect brand outbound in_response_to IDs ─────────────────────
    brand_responded_to: set = set()
    total_rows = 0

    reader = pd.read_csv(raw_path, chunksize=args.chunksize, low_memory=False)
    for chunk in tqdm(reader, desc="Pass 1 – finding brand responses"):
        total_rows += len(chunk)
        out = chunk[(chunk["author_id"] == BRAND) & (chunk["inbound"] == False)]
        ids = out["in_response_to_tweet_id"].dropna().astype(int).tolist()
        brand_responded_to.update(ids)

    print(f"  Total rows scanned: {total_rows:,}")
    print(f"  Unique customer tweets {BRAND} replied to: {len(brand_responded_to):,}")

    # ── Pass 2: collect matching inbound messages + brand replies ──────────────
    # We need: inbound customer msg (tweet_id in brand_responded_to) 
    #       +  outbound brand reply (in_response_to_tweet_id in brand_responded_to)
    inbound_msgs: Dict[int, dict] = {}
    brand_replies: Dict[int, str] = {}  # key = in_response_to_tweet_id, value = reply text

    reader = pd.read_csv(raw_path, chunksize=args.chunksize, low_memory=False)
    for chunk in tqdm(reader, desc="Pass 2 – collecting messages"):
        # Inbound customer messages we care about
        inb = chunk[
            (chunk["inbound"] == True) &
            (chunk["tweet_id"].isin(brand_responded_to))
        ]
        for _, row in inb.iterrows():
            tid = int(row["tweet_id"])
            if tid not in inbound_msgs:
                inbound_msgs[tid] = {
                    "tweet_id": tid,
                    "customer_message": str(row["text"]) if pd.notna(row["text"]) else "",
                    "created_at": row["created_at"],
                }

        # Brand outbound replies
        out = chunk[(chunk["author_id"] == BRAND) & (chunk["inbound"] == False)]
        for _, row in out.iterrows():
            resp_to = row["in_response_to_tweet_id"]
            if pd.notna(resp_to):
                rtid = int(resp_to)
                if rtid not in brand_replies:
                    brand_replies[rtid] = str(row["text"]) if pd.notna(row["text"]) else ""

    print(f"  Matched inbound messages: {len(inbound_msgs):,}")
    print(f"  Brand reply texts collected: {len(brand_replies):,}")

    # ── Build threads (customer + brand reply) ─────────────────────────────────
    threads: List[dict] = []
    for tid, msg in inbound_msgs.items():
        reply = brand_replies.get(tid, "")
        if msg["customer_message"].strip() and reply.strip():
            threads.append({
                "tweet_id": tid,
                "customer_message": msg["customer_message"],
                "brand_reply": reply,
                "created_at": msg["created_at"],
            })

    print(f"  Complete threads (msg + reply): {len(threads):,}")

    # ── Write brand_threads.csv (all pairs) ───────────────────────────────────
    threads_path = PROCESSED_DIR / "brand_threads.csv"
    threads_df = pd.DataFrame(threads)
    threads_df.to_csv(threads_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(threads_df):,} threads -> {threads_path}")

    # -- Build sample.csv (stratified random, fixed seed) ---------------------
    rng = random.Random(SEED)
    sample_threads = rng.sample(threads, min(SAMPLE_SIZE, len(threads)))
    sample_df = pd.DataFrame(sample_threads)

    sample_path = PROCESSED_DIR / "sample.csv"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    print(f"Wrote {len(sample_df):,} rows  -> {sample_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
