"""Leakage-safe evaluation for same-image caption preferences.

This script implements the RQ2 protocol described in the paper while leaving
``compare_same_image.py`` unchanged for provenance. In particular, it:

* preserves the provided VICR train/validation/test partitions;
* fits feature scaling on the training partition only;
* creates one unordered caption pair at a time within each partition; and
* rounds mean ratings before removing ties, matching the paper's reported
  pair counts (5,066 train, 1,073 validation, and 1,701 test).

Run ``python compare_same_image_protocol.py --audit-only`` to validate the
data protocol without importing TensorFlow or training a model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embeddings_serialize import deserialize


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SCRIPT_DIR.parent
DEFAULT_EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings"
DEFAULT_OUTPUT = REPOSITORY_DIR / "results" / "compare_same_image_protocol.csv"
EXPECTED_PAIR_COUNTS = {"train": 5_066, "val": 1_073, "test": 1_701}


@dataclass(frozen=True)
class SplitExamples:
    """Individual image-caption examples from one official VICR split."""

    features: np.ndarray
    ratings: np.ndarray
    images: np.ndarray


@dataclass(frozen=True)
class PairwiseExamples:
    """One orientation of every non-tied same-image caption pair."""

    first: np.ndarray
    second: np.ndarray
    labels: np.ndarray

    def as_model_inputs(self) -> dict[str, np.ndarray]:
        return {"A": self.first, "B": self.second, "Label": self.labels}

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class ProtocolData:
    """Prepared individual and pairwise data for all official splits."""

    examples: dict[str, SplitExamples]
    pairs: dict[str, PairwiseExamples]


def load_split(path: Path) -> SplitExamples:
    """Load one serialized VICR split and construct multimodal features."""

    with path.open("rb") as input_file:
        records = deserialize(input_file)

    features = np.asarray(
        [
            np.concatenate([record.image_embedding, record.caption_embedding])
            for record in records
        ],
        dtype=np.float32,
    )
    # Rounding is required to recover the pair counts stated in the paper.
    ratings = np.asarray(
        [np.round(np.mean(record.ratings)) for record in records],
        dtype=np.float32,
    )
    images = np.asarray([record.image for record in records])
    return SplitExamples(features=features, ratings=ratings, images=images)


def generate_same_image_pairs(examples: SplitExamples) -> PairwiseExamples:
    """Create one unordered pair for every non-tied same-image caption pair."""

    image_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(examples.images):
        image_to_indices[str(image)].append(index)

    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    labels: list[int] = []

    for indices in image_to_indices.values():
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                left_rating = examples.ratings[left_index]
                right_rating = examples.ratings[right_index]
                if left_rating == right_rating:
                    continue

                first.append(examples.features[left_index])
                second.append(examples.features[right_index])
                labels.append(1 if left_rating > right_rating else -1)

    feature_count = examples.features.shape[1]
    return PairwiseExamples(
        first=np.asarray(first, dtype=np.float32).reshape(-1, feature_count),
        second=np.asarray(second, dtype=np.float32).reshape(-1, feature_count),
        labels=np.asarray(labels, dtype=np.int8),
    )


def prepare_protocol_data(embeddings_dir: Path) -> ProtocolData:
    """Load, scale, and pair the official train/validation/test partitions."""

    raw_examples = {
        split: load_split(embeddings_dir / f"VICR-{split}-vilbert.emb")
        for split in ("train", "val", "test")
    }

    training_features = raw_examples["train"].features
    training_mean = training_features.mean(axis=0)
    training_scale = training_features.std(axis=0)
    training_scale[training_scale == 0] = 1.0

    scaled_examples = {
        split: SplitExamples(
            features=(
                (examples.features - training_mean) / training_scale
            ).astype(np.float32),
            ratings=examples.ratings,
            images=examples.images,
        )
        for split, examples in raw_examples.items()
    }
    pairs = {
        split: generate_same_image_pairs(examples)
        for split, examples in scaled_examples.items()
    }
    return ProtocolData(examples=scaled_examples, pairs=pairs)


def validate_protocol(data: ProtocolData) -> None:
    """Reject partition overlap or pair-count drift before model training."""

    image_sets = {
        split: set(examples.images.tolist())
        for split, examples in data.examples.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = image_sets[left].intersection(image_sets[right])
        if overlap:
            raise ValueError(
                f"{left}/{right} image leakage detected: {len(overlap)} shared images"
            )

    actual_counts = {split: len(pairs) for split, pairs in data.pairs.items()}
    if actual_counts != EXPECTED_PAIR_COUNTS:
        raise ValueError(
            "Same-image pair counts do not match the paper: "
            f"expected {EXPECTED_PAIR_COUNTS}, found {actual_counts}"
        )

    for split, pairs in data.pairs.items():
        if not np.all(np.isin(pairs.labels, (-1, 1))):
            raise ValueError(f"{split} contains invalid pair labels")


def build_model(input_dim: int, margin: float):
    """Create the original encoder and hinge-loss model lazily."""

    import tensorflow as tf

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation=None),
        ]
    )

    class PairwiseModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.margin = margin
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        @property
        def metrics(self):
            return [self.loss_tracker]

        def call(self, features, training=False):
            first = self.encoder(features["A"], training=training)
            second = self.encoder(features["B"], training=training)
            return tf.squeeze(first - second, axis=-1)

        def hinge_loss(self, predictions, labels):
            labels = tf.cast(labels, tf.float32)
            return tf.reduce_mean(
                tf.maximum(0.0, self.margin - labels * predictions)
            )

        def train_step(self, features):
            with tf.GradientTape() as tape:
                predictions = self(features, training=True)
                loss = self.hinge_loss(predictions, features["Label"])
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, features):
            predictions = self(features, training=False)
            loss = self.hinge_loss(predictions, features["Label"])
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def predict_pairs(self, first, second):
            return self.encoder(first, training=False) - self.encoder(
                second, training=False
            )

    return PairwiseModel()


def pair_accuracy(model, pairs: PairwiseExamples) -> float:
    predictions = model.predict_pairs(pairs.first, pairs.second).numpy().ravel()
    predicted_labels = np.where(predictions > 0, 1, -1)
    return float(np.mean(predicted_labels == pairs.labels))


def make_dataset(pairs: PairwiseExamples, batch_size: int, seed: int, shuffle: bool):
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(pairs.as_model_inputs())
    if shuffle:
        dataset = dataset.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def run_experiments(
    data: ProtocolData,
    runs: int,
    epochs: int,
    batch_size: int,
    seed: int,
    margin: float,
    verbose: int,
) -> pd.DataFrame:
    """Train repeated same-image models and evaluate only on the test split."""

    import pandas as pd
    import tensorflow as tf
    from scipy.stats import kendalltau, pearsonr, spearmanr

    results: list[dict[str, float | int]] = []
    input_dim = data.examples["train"].features.shape[1]

    for run_index in range(runs):
        run_seed = seed + run_index
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(run_seed)

        model = build_model(input_dim=input_dim, margin=margin)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        train_dataset = make_dataset(
            data.pairs["train"], batch_size, run_seed, shuffle=True
        )
        validation_dataset = make_dataset(
            data.pairs["val"], batch_size, run_seed, shuffle=False
        )
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            verbose=verbose,
        )

        validation_accuracy = pair_accuracy(model, data.pairs["val"])
        test_accuracy = pair_accuracy(model, data.pairs["test"])

        test_examples = data.examples["test"]
        predicted_scores = model.encoder.predict(
            test_examples.features, batch_size=batch_size, verbose=0
        ).ravel()
        pearson = pearsonr(test_examples.ratings, predicted_scores).statistic
        spearman = spearmanr(test_examples.ratings, predicted_scores).statistic
        kendall_c = kendalltau(
            test_examples.ratings, predicted_scores, variant="c"
        ).statistic

        result = {
            "run": run_index + 1,
            "seed": run_seed,
            "train_pairs": len(data.pairs["train"]),
            "val_pairs": len(data.pairs["val"]),
            "test_pairs": len(data.pairs["test"]),
            "val_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "pearson": float(pearson),
            "spearman": float(spearman),
            "kendall_c": float(kendall_c),
        }
        results.append(result)
        print(
            f"Run {run_index + 1}/{runs}: "
            f"val_accuracy={validation_accuracy:.4f}, "
            f"test_accuracy={test_accuracy:.4f}"
        )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory containing the three serialized VICR split files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV for repeated-run metrics.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate split isolation and pair counts without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = prepare_protocol_data(args.embeddings_dir.resolve())
    validate_protocol(data)

    counts = {split: len(pairs) for split, pairs in data.pairs.items()}
    print(f"Protocol audit passed: {counts}")
    if args.audit_only:
        return

    results = run_experiments(
        data=data,
        runs=args.runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        margin=args.margin,
        verbose=args.verbose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print("\nSummary:")
    print(results.describe().loc[["mean", "std"]])
    print(f"\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
