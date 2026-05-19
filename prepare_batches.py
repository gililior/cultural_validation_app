"""One-time pre-processing for the cultural-specificity human-validation app.

What it does
------------
1. Loads the four cultural-detection ranker CSVs (top-2000 each) and takes
   the deduplicated union of their ``item_id`` columns.
2. Fetches the English MCQ for every union item from the HuggingFace
   dataset ``li-lab/MMLU-ProX`` (config ``en``, split ``test``).
3. Writes the joined pool to ``cultural_eval_pool.csv`` with columns:
       item_id, question_text, option_a ... option_j,
       correct_answer_letter, subject_category
4. Samples uniformly at random (SEED=42): a shared OVERLAP_SIZE-item block
   that every annotator labels (so one human-human kappa can be computed)
   plus NUM_BATCHES disjoint blocks of UNIQUE_PER_BATCH items for breadth.
   Each annotator's batch = overlap + unique = BATCH_SIZE items. Writes the
   batches (with an ``is_overlap`` flag) to the ``batches`` worksheet of the
   Google Sheet via gspread.

Usage
-----
    # build the pool CSV + upload the batches worksheet:
    python prepare_batches.py

    # only build the CSV, skip the Google Sheets upload (inspect first):
    python prepare_batches.py --csv-only

Requirements
------------
    pip install datasets gspread google-auth pandas

Credentials
-----------
Reads ``.streamlit/secrets.toml`` (the same file the Streamlit app uses):
it must contain a ``[gcp_service_account]`` block and a top-level
``GSPREAD_SHEET_ID`` key. See README.md.
"""
from __future__ import annotations

import argparse
import random
import sys
import tomllib
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
NUM_BATCHES = 20            # supports up to 20 annotators
BATCH_SIZE = 50             # total items each annotator labels
OVERLAP_SIZE = 30           # shared items every annotator labels (human-human kappa)
UNIQUE_PER_BATCH = BATCH_SIZE - OVERLAP_SIZE   # 20 unique items per annotator
SEED = 42

NUM_OPTIONS = 10
DATASET_NAME = "li-lab/MMLU-ProX"
DATASET_CONFIG = "en"
DATASET_SPLIT = "test"

# Resolve paths relative to this script so it works from any CWD.
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SUBSETS_DIR = REPO_ROOT / "results" / "cultural" / "subsets"
POOL_CSV = APP_DIR / "cultural_eval_pool.csv"
SECRETS_PATH = APP_DIR / ".streamlit" / "secrets.toml"

# The four rankers used in the paper (§5.3). Per the user's decision the
# "airatio" ranker is the *unfiltered* irt_ratio.csv.
RANKER_FILES = [
    "irt_ratio.csv",            # airatio  a^(2) / a^(1)
    "acc_max_minus_mean.csv",   # one_lang_stands_out
    "acc_gap_en.csv",           # english_anchored
    "random_items.csv",         # random
]

OPTION_LETTERS = [chr(ord("a") + i) for i in range(NUM_OPTIONS)]   # a..j

POOL_COLUMNS = (
    ["item_id", "question_text"]
    + [f"option_{ch}" for ch in OPTION_LETTERS]
    + ["correct_answer_letter", "subject_category"]
)


# --------------------------------------------------------------------------
# Step 1: union of the four rankers
# --------------------------------------------------------------------------
def load_union() -> list[str]:
    """Return the sorted deduplicated union of item_ids across the rankers."""
    union: set[str] = set()
    for fname in RANKER_FILES:
        path = SUBSETS_DIR / fname
        if not path.exists():
            sys.exit(f"ERROR: ranker file not found: {path}")
        df = pd.read_csv(path)
        if "item_id" not in df.columns:
            sys.exit(f"ERROR: {path} has no 'item_id' column")
        ids = set(df["item_id"].astype(str))
        print(f"  {fname:28s} {len(ids):5d} items")
        union |= ids
    # sorted() makes the downstream random sample deterministic.
    return sorted(union)


def parse_item_id(item_id: str) -> tuple[str, int]:
    """'law:1591' -> ('law', 1591)."""
    cat, qid = item_id.split(":", 1)
    return cat, int(qid)


# --------------------------------------------------------------------------
# Step 2: join with the English MMLU-ProX source
# --------------------------------------------------------------------------
def build_en_lookup(needed_keys: set[tuple[str, int]]) -> dict:
    """Return {(category, question_id): hf_row} for the English split.

    MMLU-ProX stores the per-item id under either ``question_id`` or
    ``question_id_src`` depending on dataset version; pick whichever field
    matches more of the item_ids we need (mirrors cultural_judge_run.py).
    """
    from datasets import load_dataset

    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG}/{DATASET_SPLIT}) ...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    candidates = [
        c for c in ("question_id", "question_id_src") if c in ds.column_names
    ]
    if not candidates:
        sys.exit(
            f"ERROR: neither 'question_id' nor 'question_id_src' present; "
            f"columns are {ds.column_names}"
        )

    best_field, best_hits = candidates[0], -1
    for field in candidates:
        present = {(r["category"], int(r[field])) for r in ds}
        hits = len(needed_keys & present)
        print(f"  field '{field}' matches {hits}/{len(needed_keys)} keys")
        if hits > best_hits:
            best_field, best_hits = field, hits
    if best_hits == 0:
        sys.exit(f"ERROR: no item_ids matched via {candidates}")

    return {(r["category"], int(r[best_field])): r for r in ds}


def build_pool(union_ids: list[str]) -> pd.DataFrame:
    """Join union item_ids with their English MCQ; return the pool DataFrame."""
    needed_keys = {parse_item_id(iid) for iid in union_ids}
    en_lookup = build_en_lookup(needed_keys)

    rows = []
    missing = 0
    for iid in union_ids:
        cat, qid = parse_item_id(iid)
        src = en_lookup.get((cat, qid))
        if src is None:
            missing += 1
            continue
        options = [src[f"option_{i}"] for i in range(NUM_OPTIONS)]
        row = {
            "item_id": iid,
            "question_text": src["question"],
            "correct_answer_letter": str(src["answer"]).strip().upper(),
            "subject_category": cat,
        }
        for ch, opt in zip(OPTION_LETTERS, options):
            row[f"option_{ch}"] = opt
        rows.append(row)

    if missing:
        print(f"WARNING: {missing} union item(s) not found in the source.")
    df = pd.DataFrame(rows, columns=POOL_COLUMNS)
    print(f"Built pool with {len(df)} items.")
    return df


# --------------------------------------------------------------------------
# Step 3: sample disjoint batches
# --------------------------------------------------------------------------
def make_batches(pool: pd.DataFrame) -> pd.DataFrame:
    """Sample the shared overlap block + the disjoint unique blocks.

    Every annotator's batch is OVERLAP_SIZE shared items (identical across
    all batches) plus UNIQUE_PER_BATCH items unique to that batch, for a
    total of BATCH_SIZE items. Within each batch the overlap and unique
    items are shuffled together (per-batch seed) so the shared items land
    at different, non-clustered positions for each annotator.

    Returns a DataFrame with columns:
        batch_id, item_id, position_in_batch, is_overlap
    batch_id is 1-indexed (1..NUM_BATCHES); position_in_batch is 1-indexed
    (1..BATCH_SIZE); is_overlap marks the shared block.
    """
    n_needed = OVERLAP_SIZE + NUM_BATCHES * UNIQUE_PER_BATCH
    available = sorted(pool["item_id"].astype(str))
    if len(available) < n_needed:
        sys.exit(
            f"ERROR: pool has {len(available)} items but {OVERLAP_SIZE} + "
            f"{NUM_BATCHES} x {UNIQUE_PER_BATCH} = {n_needed} are required."
        )

    rng = random.Random(SEED)
    sample = rng.sample(available, n_needed)   # uniform, no replacement
    overlap = sample[:OVERLAP_SIZE]
    rest = sample[OVERLAP_SIZE:]

    records = []
    for b in range(NUM_BATCHES):
        unique = rest[b * UNIQUE_PER_BATCH:(b + 1) * UNIQUE_PER_BATCH]
        items = (
            [(iid, True) for iid in overlap]
            + [(iid, False) for iid in unique]
        )
        random.Random(SEED + b + 1).shuffle(items)   # mix shared into batch
        for pos, (item_id, is_overlap) in enumerate(items, start=1):
            records.append({
                "batch_id": b + 1,
                "item_id": item_id,
                "position_in_batch": pos,
                "is_overlap": is_overlap,
            })
    df = pd.DataFrame(
        records,
        columns=["batch_id", "item_id", "position_in_batch", "is_overlap"],
    )
    print(
        f"Created {NUM_BATCHES} batches of {BATCH_SIZE} items "
        f"({OVERLAP_SIZE} shared + {UNIQUE_PER_BATCH} unique each); "
        f"{n_needed} distinct items used."
    )
    return df


# --------------------------------------------------------------------------
# Step 4: upload the batches worksheet
# --------------------------------------------------------------------------
def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        sys.exit(
            f"ERROR: {SECRETS_PATH} not found. Create it from "
            f"secrets.toml.example (see README.md)."
        )
    with SECRETS_PATH.open("rb") as fh:
        return tomllib.load(fh)


def upload_batches(batches: pd.DataFrame) -> None:
    """Create/replace the 'batches' worksheet with header + rows."""
    import gspread
    from google.oauth2.service_account import Credentials

    secrets = load_secrets()
    if "gcp_service_account" not in secrets or "GSPREAD_SHEET_ID" not in secrets:
        sys.exit(
            "ERROR: secrets.toml must contain a [gcp_service_account] block "
            "and a GSPREAD_SHEET_ID key."
        )

    creds = Credentials.from_service_account_info(
        secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    sh = client.open_by_key(secrets["GSPREAD_SHEET_ID"])

    try:
        ws = sh.worksheet("batches")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="batches", rows=len(batches) + 10, cols=4)

    header = ["batch_id", "item_id", "position_in_batch", "is_overlap"]
    values = [header] + batches.astype(object).values.tolist()
    ws.update(values=values, range_name="A1")
    print(f"Uploaded {len(batches)} rows to the 'batches' worksheet.")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv-only", action="store_true",
        help="Build cultural_eval_pool.csv only; skip the Google Sheets upload.",
    )
    args = ap.parse_args()

    print("Step 1/4: union of the four rankers")
    union_ids = load_union()
    print(f"  -> {len(union_ids)} unique items in the union\n")

    print("Step 2/4: join with English MMLU-ProX source")
    pool = build_pool(union_ids)
    pool.to_csv(POOL_CSV, index=False)
    print(f"  -> wrote {POOL_CSV}\n")

    print("Step 3/4: sample disjoint batches")
    batches = make_batches(pool)
    print()

    if args.csv_only:
        print("Step 4/4: SKIPPED (--csv-only). "
              "Re-run without --csv-only to upload the batches worksheet.")
        return 0

    print("Step 4/4: upload the 'batches' worksheet")
    upload_batches(batches)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
