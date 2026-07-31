"""Run controlled image-caption regression, hinge, and Bradley-Terry models.

Unlike the released entrypoints, this runner preserves the official dataset
partitions, fits feature scaling on training data only, uses matched seeds and
pair samples across comparative objectives, and reports both correlation and
pairwise-accuracy metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from embeddings_serialize import deserialize


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SCRIPT_DIR.parent
DEFAULT_EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings"
DATASETS = {
    "vicr": ("VICR", "VICR"),
    "flickr_expert": ("FlickrExpert", "Flickr8k"),
}
SPLITS = ("train", "val", "test")
OBJECTIVES = ("regression", "hinge", "bradley_terry")


@dataclass(frozen=True)
class SplitData:
    features: np.ndarray
    ratings: np.ndarray
    images: np.ndarray

    def __len__(self) -> int:
        return len(self.ratings)


@dataclass(frozen=True)
class PairData:
    first: np.ndarray
    second: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def load_splits(
    dataset_key: str,
    embeddings_dir: Path,
    embedding_tag: str,
    rating_mode: str,
) -> tuple[str, dict[str, SplitData], dict[str, str]]:
    dataset, source_prefix = DATASETS[dataset_key]
    raw: dict[str, SplitData] = {}
    source_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = embeddings_dir / f"{source_prefix}-{split}-{embedding_tag}.emb"
        with path.open("rb") as stream:
            records = deserialize(stream)
        features = np.asarray(
            [
                np.concatenate([record.image_embedding, record.caption_embedding])
                for record in records
            ],
            dtype=np.float32,
        )
        mean_ratings = np.asarray(
            [np.mean(record.ratings) for record in records], dtype=np.float32
        )
        ratings = (
            np.round(mean_ratings).astype(np.float32)
            if rating_mode == "rounded_mean"
            else mean_ratings
        )
        raw[split] = SplitData(
            features=features,
            ratings=ratings,
            images=np.asarray([record.image for record in records]),
        )
        source_hashes[split] = _sha256_file(path)

    feature_dim = raw["train"].features.shape[1]
    if any(split.features.shape[1] != feature_dim for split in raw.values()):
        raise ValueError("Embedding dimensions differ across official splits")
    image_sets = {name: set(split.images.tolist()) for name, split in raw.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = image_sets[left].intersection(image_sets[right])
        if overlap:
            raise ValueError(
                f"{dataset} {left}/{right} image leakage: {len(overlap)}"
            )

    train_mean = raw["train"].features.mean(axis=0)
    train_scale = raw["train"].features.std(axis=0)
    train_scale[train_scale == 0] = 1.0
    scaled = {
        name: SplitData(
            features=((split.features - train_mean) / train_scale).astype(
                np.float32
            ),
            ratings=split.ratings,
            images=split.images,
        )
        for name, split in raw.items()
    }
    return dataset, scaled, source_hashes


def _empty_pairs(feature_dim: int) -> PairData:
    return PairData(
        first=np.empty((0, feature_dim), dtype=np.float32),
        second=np.empty((0, feature_dim), dtype=np.float32),
        labels=np.empty((0,), dtype=np.int8),
    )


def generate_general_pairs(
    split: SplitData, pair_count: int, seed: int
) -> PairData:
    """Sample unique unordered, non-tied pairs with deterministic orientation."""

    if pair_count < 1:
        return _empty_pairs(split.features.shape[1])
    rng = np.random.default_rng(seed)
    item_count = len(split)
    selected: list[tuple[int, int]] = []
    seen: set[int] = set()
    attempts = 0
    max_attempts = max(pair_count * 100, 10_000)

    while len(selected) < pair_count and attempts < max_attempts:
        batch_size = min(max((pair_count - len(selected)) * 3, 1024), 100_000)
        left = rng.integers(0, item_count, size=batch_size)
        right = rng.integers(0, item_count, size=batch_size)
        attempts += batch_size
        for first_index, second_index in zip(left.tolist(), right.tolist()):
            if first_index == second_index:
                continue
            low, high = sorted((first_index, second_index))
            key = low * item_count + high
            if (
                key in seen
                or split.ratings[low] == split.ratings[high]
                or split.images[low] == split.images[high]
            ):
                continue
            seen.add(key)
            if rng.random() < 0.5:
                selected.append((low, high))
            else:
                selected.append((high, low))
            if len(selected) == pair_count:
                break

    if len(selected) != pair_count:
        raise ValueError(
            f"Could only sample {len(selected)} of {pair_count} requested pairs"
        )
    first_indices = np.fromiter((pair[0] for pair in selected), dtype=np.int64)
    second_indices = np.fromiter((pair[1] for pair in selected), dtype=np.int64)
    labels = np.where(
        split.ratings[first_indices] > split.ratings[second_indices], 1, -1
    ).astype(np.int8)
    return PairData(
        first=split.features[first_indices],
        second=split.features[second_indices],
        labels=labels,
    )


def generate_same_image_pairs(split: SplitData) -> PairData:
    image_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(split.images):
        image_to_indices[str(image)].append(index)

    first_indices: list[int] = []
    second_indices: list[int] = []
    labels: list[int] = []
    for indices in image_to_indices.values():
        for position, left_index in enumerate(indices):
            for right_index in indices[position + 1 :]:
                left_rating = split.ratings[left_index]
                right_rating = split.ratings[right_index]
                if left_rating == right_rating:
                    continue
                first_indices.append(left_index)
                second_indices.append(right_index)
                labels.append(1 if left_rating > right_rating else -1)

    if not labels:
        return _empty_pairs(split.features.shape[1])
    return PairData(
        first=split.features[np.asarray(first_indices)],
        second=split.features[np.asarray(second_indices)],
        labels=np.asarray(labels, dtype=np.int8),
    )


def build_encoder(input_dim: int):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation=None),
        ]
    )


def build_regression_model(input_dim: int, learning_rate: float):
    import tensorflow as tf

    model = build_encoder(input_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def build_pairwise_model(
    input_dim: int, objective: str, learning_rate: float, margin: float
):
    import tensorflow as tf

    encoder = build_encoder(input_dim)

    class PairwiseModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        @property
        def metrics(self):
            return [self.loss_tracker]

        def call(self, features, training=False):
            first = self.encoder(features["A"], training=training)
            second = self.encoder(features["B"], training=training)
            return tf.squeeze(first - second, axis=-1)

        def rank_loss(self, logits, labels):
            labels = tf.cast(labels, tf.float32)
            if objective == "hinge":
                return tf.reduce_mean(
                    tf.maximum(0.0, margin - labels * logits)
                )
            return tf.reduce_mean(tf.nn.softplus(-labels * logits))

        def train_step(self, data):
            features = data[0] if isinstance(data, tuple) else data
            with tf.GradientTape() as tape:
                logits = self(features, training=True)
                loss = self.rank_loss(logits, features["Label"])
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):
            features = data[0] if isinstance(data, tuple) else data
            logits = self(features, training=False)
            loss = self.rank_loss(logits, features["Label"])
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

    model = PairwiseModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def pair_dataset(
    pairs: PairData, batch_size: int, seed: int, shuffle: bool
):
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(
        {"A": pairs.first, "B": pairs.second, "Label": pairs.labels}
    )
    if shuffle:
        dataset = dataset.shuffle(
            len(pairs), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def callbacks(patience: int):
    import tensorflow as tf

    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=max(2, patience // 2),
            factor=0.3,
            min_lr=1e-6,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
    ]


def metrics_from_scores(ratings: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from scipy.stats import kendalltau, pearsonr, spearmanr

    errors = ratings - scores
    return {
        "mse": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "pearson": float(pearsonr(ratings, scores).statistic),
        "spearman": float(spearmanr(ratings, scores).statistic),
        "kendall_c": float(kendalltau(ratings, scores, variant="c").statistic),
    }


def accuracy_from_scores(model, pairs: PairData, batch_size: int) -> float:
    if len(pairs) == 0:
        return float("nan")
    first_scores = model.predict(pairs.first, batch_size=batch_size, verbose=0).ravel()
    second_scores = model.predict(pairs.second, batch_size=batch_size, verbose=0).ravel()
    predictions = np.where(first_scores - second_scores > 0, 1, -1)
    return float(np.mean(predictions == pairs.labels))


def _history_epochs(history) -> int:
    return len(history.history.get("loss", []))


def run_regression(
    splits: dict[str, SplitData],
    evaluation_pairs: dict[str, PairData],
    seeds: Iterable[int],
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    verbose: int,
) -> list[dict[str, object]]:
    import tensorflow as tf

    rows: list[dict[str, object]] = []
    for run_index, seed in enumerate(seeds, start=1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_regression_model(
            splits["train"].features.shape[1], learning_rate
        )
        history = model.fit(
            splits["train"].features,
            splits["train"].ratings,
            validation_data=(splits["val"].features, splits["val"].ratings),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks(patience),
            verbose=verbose,
            shuffle=True,
        )
        scores = model.predict(
            splits["test"].features, batch_size=batch_size, verbose=0
        ).ravel()
        row: dict[str, object] = {
            "objective": "regression",
            "pair_condition": "individual_ratings",
            "N": np.nan,
            "run": run_index,
            "seed": seed,
            "train_examples": len(splits["train"]),
            "train_pairs": 0,
            "val_pairs": 0,
            "epochs_trained": _history_epochs(history),
            "general_pair_accuracy": accuracy_from_scores(
                model, evaluation_pairs["general"], batch_size
            ),
            "same_image_accuracy": accuracy_from_scores(
                model, evaluation_pairs["same_image"], batch_size
            ),
        }
        row.update(metrics_from_scores(splits["test"].ratings, scores))
        rows.append(row)
        print(
            f"regression run {run_index}: pearson={row['pearson']:.4f}, "
            f"same_image_accuracy={row['same_image_accuracy']:.4f}",
            flush=True,
        )
    return rows


def run_pairwise(
    splits: dict[str, SplitData],
    evaluation_pairs: dict[str, PairData],
    objectives: Iterable[str],
    n_values: Iterable[int],
    seeds: Iterable[int],
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    margin: float,
    verbose: int,
    include_same_image_models: bool,
) -> list[dict[str, object]]:
    import tensorflow as tf

    rows: list[dict[str, object]] = []
    input_dim = splits["train"].features.shape[1]
    pairwise_objectives = [name for name in objectives if name != "regression"]
    for n_value in n_values:
        for run_index, seed in enumerate(seeds, start=1):
            train_pairs = generate_general_pairs(
                splits["train"], n_value * len(splits["train"]), seed
            )
            val_pairs = generate_general_pairs(
                splits["val"], n_value * len(splits["val"]), seed + 1_000_000
            )
            for objective in pairwise_objectives:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = build_pairwise_model(
                    input_dim, objective, learning_rate, margin
                )
                history = model.fit(
                    pair_dataset(train_pairs, batch_size, seed, shuffle=True),
                    validation_data=pair_dataset(
                        val_pairs, batch_size, seed, shuffle=False
                    ),
                    epochs=epochs,
                    callbacks=callbacks(patience),
                    verbose=verbose,
                )
                scores = model.encoder.predict(
                    splits["test"].features, batch_size=batch_size, verbose=0
                ).ravel()
                row: dict[str, object] = {
                    "objective": objective,
                    "pair_condition": "different_image",
                    "N": n_value,
                    "run": run_index,
                    "seed": seed,
                    "train_examples": len(splits["train"]),
                    "train_pairs": len(train_pairs),
                    "val_pairs": len(val_pairs),
                    "epochs_trained": _history_epochs(history),
                    "general_pair_accuracy": accuracy_from_scores(
                        model.encoder, evaluation_pairs["general"], batch_size
                    ),
                    "same_image_accuracy": accuracy_from_scores(
                        model.encoder,
                        evaluation_pairs["same_image"],
                        batch_size,
                    ),
                }
                row.update(metrics_from_scores(splits["test"].ratings, scores))
                rows.append(row)
                print(
                    f"{objective} N={n_value} run {run_index}: "
                    f"pearson={row['pearson']:.4f}, "
                    f"same_image_accuracy={row['same_image_accuracy']:.4f}",
                    flush=True,
                )

    if not include_same_image_models:
        return rows

    same_image_train = generate_same_image_pairs(splits["train"])
    same_image_val = generate_same_image_pairs(splits["val"])
    if len(same_image_train) == 0 or len(same_image_val) == 0:
        print("Skipping same-image models: no train/validation pairs", flush=True)
        return rows
    for run_index, seed in enumerate(seeds, start=1):
        for objective in pairwise_objectives:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_pairwise_model(input_dim, objective, learning_rate, margin)
            history = model.fit(
                pair_dataset(
                    same_image_train, batch_size, seed, shuffle=True
                ),
                validation_data=pair_dataset(
                    same_image_val, batch_size, seed, shuffle=False
                ),
                epochs=epochs,
                callbacks=callbacks(patience),
                verbose=verbose,
            )
            scores = model.encoder.predict(
                splits["test"].features, batch_size=batch_size, verbose=0
            ).ravel()
            row = {
                "objective": objective,
                "pair_condition": "same_image",
                "N": np.nan,
                "run": run_index,
                "seed": seed,
                "train_examples": len(splits["train"]),
                "train_pairs": len(same_image_train),
                "val_pairs": len(same_image_val),
                "epochs_trained": _history_epochs(history),
                "general_pair_accuracy": accuracy_from_scores(
                    model.encoder, evaluation_pairs["general"], batch_size
                ),
                "same_image_accuracy": accuracy_from_scores(
                    model.encoder, evaluation_pairs["same_image"], batch_size
                ),
            }
            row.update(metrics_from_scores(splits["test"].ratings, scores))
            rows.append(row)
            print(
                f"{objective} same-image run {run_index}: "
                f"same_image_accuracy={row['same_image_accuracy']:.4f}",
                flush=True,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--embedding-tag", required=True)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rating-mode", choices=("rounded_mean", "mean"), default="rounded_mean")
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--n-values", type=_parse_int_list, default=(1, 2, 3, 4, 5))
    parser.add_argument("--seeds", type=_parse_int_list, default=tuple(range(10)))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--skip-same-image-models", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd
    import tensorflow as tf

    objectives = tuple(
        item.strip() for item in args.objectives.split(",") if item.strip()
    )
    unknown_objectives = set(objectives) - set(OBJECTIVES)
    if unknown_objectives:
        raise ValueError(f"Unknown objectives: {sorted(unknown_objectives)}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (pass --overwrite): {args.output}")
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    dataset, splits, source_hashes = load_splits(
        args.dataset,
        args.embeddings_dir.resolve(),
        args.embedding_tag,
        args.rating_mode,
    )
    general_test_pairs = generate_general_pairs(
        splits["test"], min(5 * len(splits["test"]), 20_000), 2_000_000
    )
    same_image_test_pairs = generate_same_image_pairs(splits["test"])
    evaluation_pairs = {
        "general": general_test_pairs,
        "same_image": same_image_test_pairs,
    }
    print(
        f"{dataset}/{args.representation}: "
        f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
        f"examples, {len(general_test_pairs)} general test pairs, "
        f"{len(same_image_test_pairs)} same-image test pairs",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    if "regression" in objectives:
        rows.extend(
            run_regression(
                splits,
                evaluation_pairs,
                args.seeds,
                args.epochs,
                args.batch_size,
                args.patience,
                args.learning_rate,
                args.verbose,
            )
        )
    rows.extend(
        run_pairwise(
            splits,
            evaluation_pairs,
            objectives,
            args.n_values,
            args.seeds,
            args.epochs,
            args.batch_size,
            args.patience,
            args.learning_rate,
            args.margin,
            args.verbose,
            not args.skip_same_image_models,
        )
    )
    for row in rows:
        row.update(
            {
                "dataset": dataset,
                "representation": args.representation,
                "embedding_tag": args.embedding_tag,
                "rating_mode": args.rating_mode,
                "test_examples": len(splits["test"]),
                "general_test_pairs": len(general_test_pairs),
                "same_image_test_pairs": len(same_image_test_pairs),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    leading_columns = [
        "dataset",
        "representation",
        "embedding_tag",
        "objective",
        "pair_condition",
        "N",
        "run",
        "seed",
    ]
    results = results[leading_columns + [
        column for column in results.columns if column not in leading_columns
    ]]
    results.to_csv(args.output, index=False)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = None
    metadata = {
        "dataset": dataset,
        "representation": args.representation,
        "embedding_tag": args.embedding_tag,
        "rating_mode": args.rating_mode,
        "objectives": objectives,
        "n_values": args.n_values,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "include_same_image_models": not args.skip_same_image_models,
        "source_sha256": source_hashes,
        "result_sha256": _sha256_file(args.output),
        "repository_commit": commit,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(results)} result rows to {args.output.resolve()}")
    print(
        results.groupby(["objective", "pair_condition", "N"], dropna=False)[
            ["pearson", "spearman", "kendall_c", "same_image_accuracy"]
        ].agg(["mean", "std"])
    )


if __name__ == "__main__":
    main()
