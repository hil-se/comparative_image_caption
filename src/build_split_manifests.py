"""Build auditable manifests for the official VICR and Flickr-Expert splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from embeddings_serialize import deserialize


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SCRIPT_DIR.parent
DEFAULT_EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings"
DEFAULT_OUTPUT_DIR = REPOSITORY_DIR / "manifests"
DATASETS = {
    "vicr": ("VICR", "VICR"),
    "flickr_expert": ("FlickrExpert", "Flickr8k"),
}
SPLITS = ("train", "val", "test")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_id(
    dataset: str,
    split: str,
    split_index: int,
    image: str,
    caption: str,
    ratings: list[float],
) -> str:
    payload = json.dumps(
        {
            "dataset": dataset,
            "split": split,
            "split_index": split_index,
            "image": image,
            "caption": caption,
            "ratings": [float(value) for value in ratings],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def build_manifest(
    dataset_key: str, embeddings_dir: Path, output_dir: Path
) -> dict[str, object]:
    dataset, source_prefix = DATASETS[dataset_key]
    rows: list[dict[str, object]] = []
    image_sets: dict[str, set[str]] = {}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    split_counts: dict[str, int] = {}

    for split in SPLITS:
        source_path = embeddings_dir / f"{source_prefix}-{split}-vilbert.emb"
        with source_path.open("rb") as stream:
            records = deserialize(stream)

        image_sets[split] = {record.image for record in records}
        pair_sets[split] = {(record.image, record.caption) for record in records}
        split_counts[split] = len(records)
        for split_index, record in enumerate(records):
            ratings = [float(value) for value in record.ratings]
            mean_rating = float(np.mean(ratings))
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "split_index": split_index,
                    "record_id": _record_id(
                        dataset,
                        split,
                        split_index,
                        record.image,
                        record.caption,
                        ratings,
                    ),
                    "image_id": _sha256_text(record.image),
                    "image_name": Path(record.image).name,
                    "rating_count": len(ratings),
                    "mean_rating": f"{mean_rating:.12g}",
                    "rounded_rating": int(np.round(mean_rating)),
                }
            )

    overlaps: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = {
            "images": len(image_sets[left].intersection(image_sets[right])),
            "image_caption_pairs": len(
                pair_sets[left].intersection(pair_sets[right])
            ),
        }

    duplicate_pair_counts = {
        split: split_counts[split] - len(pair_sets[split]) for split in SPLITS
    }
    if any(value["images"] for value in overlaps.values()):
        raise ValueError(f"{dataset} has image leakage across official splits")
    if any(value["image_caption_pairs"] for value in overlaps.values()):
        raise ValueError(
            f"{dataset} has image-caption leakage across official splits"
        )
    if any(duplicate_pair_counts.values()):
        raise ValueError(
            f"{dataset} has duplicate image-caption rows: {duplicate_pair_counts}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{dataset_key}_official.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": dataset,
        "source_prefix": source_prefix,
        "split_counts": split_counts,
        "unique_images": {
            split: len(image_sets[split]) for split in SPLITS
        },
        "duplicate_image_caption_pairs": duplicate_pair_counts,
        "cross_split_overlaps": overlaps,
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    summary_path = output_dir / f"{dataset_key}_official.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "all"), default="all"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_keys = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset_key in dataset_keys:
        summary = build_manifest(
            dataset_key,
            args.embeddings_dir.resolve(),
            args.output_dir.resolve(),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
