"""Paired ablation of the legacy and official same-image protocols.

The legacy protocol mirrors ``compare_same_image.py``: it combines all VICR
partitions, scales on all examples, creates both orientations of every pair,
and randomly splits those oriented pairs. The official protocol is imported
from ``compare_same_image_protocol.py`` and keeps the provided image-disjoint
train/validation/test partitions.

Both models use the same architecture, optimizer, training schedule, and run
seeds. This makes the evaluation protocol the principal experimental change.
Use ``--audit-only`` to report pair counts and exact reversed-pair leakage
without importing TensorFlow or training a model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compare_same_image_protocol import (
    DEFAULT_EMBEDDINGS_DIR,
    PairwiseExamples,
    SplitExamples,
    build_model,
    make_dataset,
    pair_accuracy,
    prepare_protocol_data,
    validate_protocol,
)
from embeddings_serialize import deserialize


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = (
    REPOSITORY_DIR / "results" / "compare_same_image_protocol_ablation.csv"
)


@dataclass(frozen=True)
class LegacyProtocolData:
    """Prepared data matching the released random-pair split."""

    train_pairs: PairwiseExamples
    test_pairs: PairwiseExamples
    test_examples: SplitExamples
    test_without_reversed_training_pair: PairwiseExamples
    canonical_pair_count: int
    reversed_pair_leakage_fraction: float


def _split_indices(
    item_count: int, test_fraction: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Match sklearn's shuffled train_test_split index selection."""

    test_count = int(np.ceil(item_count * test_fraction))
    permutation = np.random.RandomState(random_state).permutation(item_count)
    return permutation[test_count:], permutation[:test_count]


def _select_pairs(
    pairs: PairwiseExamples, indices: np.ndarray
) -> PairwiseExamples:
    return PairwiseExamples(
        first=pairs.first[indices],
        second=pairs.second[indices],
        labels=pairs.labels[indices],
    )


def _load_legacy_examples(embeddings_dir: Path) -> SplitExamples:
    records = []
    for split in ("train", "val", "test"):
        with (embeddings_dir / f"VICR-{split}-vilbert.emb").open("rb") as stream:
            records.extend(deserialize(stream))

    features = np.asarray(
        [
            np.concatenate([record.image_embedding, record.caption_embedding])
            for record in records
        ],
        dtype=np.float32,
    )
    ratings = np.asarray(
        [np.mean(record.ratings) for record in records], dtype=np.float32
    )
    ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
    images = np.asarray([record.image for record in records])

    # StandardScaler uses population variance, equivalent to NumPy's default.
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    features = ((features - feature_mean) / feature_scale).astype(np.float32)
    return SplitExamples(features=features, ratings=ratings, images=images)


def _generate_legacy_pairs(
    examples: SplitExamples,
) -> tuple[PairwiseExamples, np.ndarray, int]:
    image_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(examples.images):
        image_to_indices[str(image)].append(index)

    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    labels: list[int] = []
    canonical_ids: list[int] = []
    canonical_pair_count = 0

    for indices in image_to_indices.values():
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                left_rating = examples.ratings[left_index]
                right_rating = examples.ratings[right_index]
                if (
                    left_rating == right_rating
                    or abs(left_rating - right_rating) < 0.05
                ):
                    continue

                label = 1 if left_rating > right_rating else -1
                first.extend(
                    [examples.features[left_index], examples.features[right_index]]
                )
                second.extend(
                    [examples.features[right_index], examples.features[left_index]]
                )
                labels.extend([label, -label])
                canonical_ids.extend([canonical_pair_count, canonical_pair_count])
                canonical_pair_count += 1

    feature_count = examples.features.shape[1]
    pairs = PairwiseExamples(
        first=np.asarray(first, dtype=np.float32).reshape(-1, feature_count),
        second=np.asarray(second, dtype=np.float32).reshape(-1, feature_count),
        labels=np.asarray(labels, dtype=np.int8),
    )
    return pairs, np.asarray(canonical_ids, dtype=np.int64), canonical_pair_count


def prepare_legacy_data(
    embeddings_dir: Path, split_seed: int
) -> LegacyProtocolData:
    """Recreate the released pair split and measure its exact reversal leakage."""

    examples = _load_legacy_examples(embeddings_dir)
    pairs, canonical_ids, canonical_pair_count = _generate_legacy_pairs(examples)
    train_indices, test_indices = _split_indices(len(pairs), 0.2, split_seed)
    example_train_indices, example_test_indices = _split_indices(
        len(examples.ratings), 0.2, split_seed
    )
    del example_train_indices

    train_canonical_ids = set(canonical_ids[train_indices].tolist())
    reversed_pair_mask = np.asarray(
        [
            canonical_id in train_canonical_ids
            for canonical_id in canonical_ids[test_indices]
        ],
        dtype=bool,
    )
    no_reversal_indices = test_indices[~reversed_pair_mask]

    return LegacyProtocolData(
        train_pairs=_select_pairs(pairs, train_indices),
        test_pairs=_select_pairs(pairs, test_indices),
        test_examples=SplitExamples(
            features=examples.features[example_test_indices],
            ratings=examples.ratings[example_test_indices],
            images=examples.images[example_test_indices],
        ),
        test_without_reversed_training_pair=_select_pairs(
            pairs, no_reversal_indices
        ),
        canonical_pair_count=canonical_pair_count,
        reversed_pair_leakage_fraction=float(reversed_pair_mask.mean()),
    )


def audit_summary(legacy: LegacyProtocolData, official) -> None:
    official_counts = {
        split: len(pairs) for split, pairs in official.pairs.items()
    }
    print(
        "Legacy protocol: "
        f"{legacy.canonical_pair_count} unordered pairs, "
        f"{len(legacy.train_pairs)} oriented train pairs, "
        f"{len(legacy.test_pairs)} oriented test pairs"
    )
    print(
        "Legacy exact reversed-pair leakage: "
        f"{legacy.reversed_pair_leakage_fraction:.2%} of test pairs"
    )
    print(
        "Legacy test pairs without an exact reversed training pair: "
        f"{len(legacy.test_without_reversed_training_pair)}"
    )
    print(f"Official image-disjoint protocol: {official_counts}")


def _correlations(model, examples: SplitExamples) -> dict[str, float]:
    from scipy.stats import kendalltau, pearsonr, spearmanr

    predictions = model.encoder.predict(
        examples.features, batch_size=256, verbose=0
    ).ravel()
    return {
        "pearson": float(pearsonr(examples.ratings, predictions).statistic),
        "spearman": float(spearmanr(examples.ratings, predictions).statistic),
        "kendall_c": float(
            kendalltau(examples.ratings, predictions, variant="c").statistic
        ),
    }


def run_experiments(
    legacy: LegacyProtocolData,
    official,
    runs: int,
    epochs: int,
    batch_size: int,
    seed: int,
    margin: float,
    verbose: int,
):
    import pandas as pd
    import tensorflow as tf

    results: list[dict[str, float | int | str]] = []
    protocols = (
        (
            "legacy_random_pair_split",
            legacy.train_pairs,
            None,
            legacy.test_pairs,
            legacy.test_examples,
        ),
        (
            "official_image_split",
            official.pairs["train"],
            official.pairs["val"],
            official.pairs["test"],
            official.examples["test"],
        ),
    )

    for run_index in range(runs):
        run_seed = seed + run_index
        for (
            protocol_name,
            train_pairs,
            validation_pairs,
            test_pairs,
            test_examples,
        ) in protocols:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(run_seed)
            model = build_model(
                input_dim=train_pairs.first.shape[1], margin=margin
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
            )

            train_dataset = make_dataset(
                train_pairs, batch_size, run_seed, shuffle=True
            )
            validation_dataset = (
                make_dataset(
                    validation_pairs, batch_size, run_seed, shuffle=False
                )
                if validation_pairs is not None
                else None
            )
            model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                verbose=verbose,
            )

            result: dict[str, float | int | str] = {
                "run": run_index + 1,
                "seed": run_seed,
                "protocol": protocol_name,
                "train_pairs": len(train_pairs),
                "val_pairs": (
                    len(validation_pairs)
                    if validation_pairs is not None
                    else 0
                ),
                "test_pairs": len(test_pairs),
                "test_accuracy": pair_accuracy(model, test_pairs),
                "reversed_pair_leakage_fraction": (
                    legacy.reversed_pair_leakage_fraction
                    if protocol_name == "legacy_random_pair_split"
                    else 0.0
                ),
                "test_accuracy_without_exact_reversal": (
                    pair_accuracy(
                        model, legacy.test_without_reversed_training_pair
                    )
                    if protocol_name == "legacy_random_pair_split"
                    else np.nan
                ),
            }
            if validation_pairs is not None:
                result["val_accuracy"] = pair_accuracy(
                    model, validation_pairs
                )
            else:
                result["val_accuracy"] = np.nan
            result.update(_correlations(model, test_examples))
            results.append(result)
            print(
                f"Run {run_index + 1}/{runs}, {protocol_name}: "
                f"test_accuracy={result['test_accuracy']:.4f}"
            )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed random-pair split seed used by the released script.",
    )
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_dir = args.embeddings_dir.resolve()
    legacy = prepare_legacy_data(embeddings_dir, args.split_seed)
    official = prepare_protocol_data(embeddings_dir)
    validate_protocol(official)
    audit_summary(legacy, official)
    if args.audit_only:
        return

    results = run_experiments(
        legacy=legacy,
        official=official,
        runs=args.runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        margin=args.margin,
        verbose=args.verbose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print("\nProtocol summary:")
    print(
        results.groupby("protocol")[
            [
                "test_accuracy",
                "test_accuracy_without_exact_reversal",
                "pearson",
                "spearman",
                "kendall_c",
            ]
        ].agg(["mean", "std"])
    )
    print(f"\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
