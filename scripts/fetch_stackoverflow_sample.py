"""Pull a date-ranged sample of StackOverflow questions into a JSON file
shaped for the clustering app: a list of objects with a title (semantic
field), tags (lexical/list field), and the owning user id (join field).

The dataset is stored as ~59 parquet shards, each covering a rough date
range (with some overlap between neighboring shards). This script reads
each shard's parquet footer statistics to skip shards that can't overlap
the requested date range, then uses PyArrow's predicate pushdown to read
only matching row groups from the shards that remain -- avoiding a full
sequential scan of the ~58M-row dataset.

Source: mikex86/stackoverflow-posts on the Hugging Face Hub, itself a
mirror of the Stack Exchange Data Dump. See data/README.md for license
and attribution details.

Usage:
    python scripts/fetch_stackoverflow_sample.py \
        --start-date 2020-01-01 --end-date 2020-02-01 --limit 500
"""
import argparse
import json
import os
from datetime import datetime

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

QUESTION_POST_TYPE_ID = 1
DATASET_REPO = "mikex86/stackoverflow-posts"
COLUMNS = ["Title", "Tags", "OwnerUserId", "CreationDate", "ContentLicense", "PostTypeId"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Exclusive end date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=500, help="Max number of questions to pull")
    parser.add_argument("--output", default="data/stackoverflow_sample.json")
    parser.add_argument("--dataset", default=DATASET_REPO)
    return parser.parse_args()


def list_shard_paths(dataset):
    files = HfApi().list_repo_files(dataset, repo_type="dataset")
    shard_files = sorted(f for f in files if f.endswith(".parquet"))
    return [f"datasets/{dataset}/{f}" for f in shard_files]


def shard_overlaps_range(fs, path, start_iso, end_iso):
    metadata = pq.ParquetFile(path, filesystem=fs).metadata
    col_idx = metadata.schema.names.index("CreationDate")

    shard_min, shard_max = None, None
    for rg_i in range(metadata.num_row_groups):
        stats = metadata.row_group(rg_i).column(col_idx).statistics
        if stats is None or not stats.has_min_max:
            return True  # no stats to prove otherwise, so don't risk skipping it
        shard_min = stats.min if shard_min is None else min(shard_min, stats.min)
        shard_max = stats.max if shard_max is None else max(shard_max, stats.max)

    return shard_min is None or (shard_min < end_iso and shard_max >= start_iso)


def main():
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start_iso, end_iso = start.isoformat(), end.isoformat()

    fs = HfFileSystem()
    shard_paths = list_shard_paths(args.dataset)
    print(f"{len(shard_paths)} shards total")

    records = []
    for path in shard_paths:
        if not shard_overlaps_range(fs, path, start_iso, end_iso):
            continue

        table = pq.read_table(
            path,
            filesystem=fs,
            filters=[
                ("PostTypeId", "==", QUESTION_POST_TYPE_ID),
                ("CreationDate", ">=", start_iso),
                ("CreationDate", "<", end_iso),
            ],
            columns=COLUMNS,
        )
        print(f"{path.rsplit('/', 1)[-1]}: {table.num_rows} candidate rows")

        for row in table.to_pylist():
            if not row["Title"] or not row["Tags"] or row["OwnerUserId"] is None:
                continue

            records.append({
                "title": row["Title"],
                "tags": row["Tags"],
                "user_id": row["OwnerUserId"],
                "creation_date": row["CreationDate"],
                "content_license": row["ContentLicense"],
            })

            if len(records) >= args.limit:
                break

        if len(records) >= args.limit:
            break

    if not records:
        print("No matching rows found for the given date range.")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
