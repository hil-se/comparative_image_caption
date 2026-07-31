"""Generate CLIP image and text embeddings for VICR or Flickr-Expert.

The source ``*.emb`` files provide image locators, captions, ratings, and the
official train/validation/test assignments. This script replaces only the
ViLBERT features and preserves every other field and row order.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np

from embeddings_serialize import Image_Caption_Embedding, deserialize, serialize


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "embeddings"
DATASETS = {
    "vicr": ("VICR", "VICR"),
    "flickr_expert": ("FlickrExpert", "Flickr8k"),
}
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug_model_name(model_name: str) -> str:
    return model_name.lower().replace("/", "-").replace("_", "-")


def load_source_records(
    dataset_key: str, source_dir: Path, max_records_per_split: int | None
) -> tuple[str, str, dict[str, list[Image_Caption_Embedding]], dict[str, str]]:
    dataset, source_prefix = DATASETS[dataset_key]
    records_by_split: dict[str, list[Image_Caption_Embedding]] = {}
    source_hashes: dict[str, str] = {}
    for split in SPLITS:
        source_path = source_dir / f"{source_prefix}-{split}-vilbert.emb"
        with source_path.open("rb") as stream:
            records = deserialize(stream)
        if max_records_per_split is not None:
            records = records[:max_records_per_split]
        records_by_split[split] = records
        source_hashes[split] = sha256_file(source_path)
    return dataset, source_prefix, records_by_split, source_hashes


def _valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _cache_path(url: str, cache_dir: Path) -> Path:
    extension = Path(urlparse(url).path).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        extension = ".img"
    return cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}{extension}"


def _download_one(url: str, destination: Path, retries: int) -> Path:
    import requests
    from PIL import Image

    if _valid_image(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                timeout=(15, 60),
                headers={"User-Agent": "comparative-image-caption-research/1.0"},
            )
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                image.verify()
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(response.content)
            os.replace(temporary, destination)
            return destination
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download {url}: {last_error}")


def _flickr_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Flickr8k image root not found: {root}")
    result: dict[str, Path] = {}
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        for path in root.rglob(extension):
            result.setdefault(path.name, path)
    if not result:
        raise FileNotFoundError(f"No images found beneath {root}")
    return result


def resolve_images(
    dataset_key: str,
    image_urls: Iterable[str],
    image_cache: Path,
    flickr8k_root: Path | None,
    workers: int,
    retries: int,
) -> dict[str, Path]:
    unique_urls = list(dict.fromkeys(image_urls))
    if dataset_key == "flickr_expert":
        if flickr8k_root is None:
            raise ValueError(
                "--flickr8k-root is required because validatedicr.org is no "
                "longer a reliable image host"
            )
        index = _flickr_index(flickr8k_root)
        missing = [url for url in unique_urls if Path(urlparse(url).path).name not in index]
        if missing:
            preview = ", ".join(Path(urlparse(url).path).name for url in missing[:5])
            raise FileNotFoundError(
                f"Flickr8k root is missing {len(missing)} required images: {preview}"
            )
        return {
            url: index[Path(urlparse(url).path).name] for url in unique_urls
        }

    destinations = {url: _cache_path(url, image_cache) for url in unique_urls}
    pending = {
        url: path for url, path in destinations.items() if not _valid_image(path)
    }
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_download_one, url, path, retries): url
                for url, path in pending.items()
            }
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 250 == 0 or completed == len(futures):
                    print(f"Downloaded {completed}/{len(futures)} images", flush=True)
    return destinations


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _as_tensor(features):
    if hasattr(features, "image_embeds"):
        return features.image_embeds
    if hasattr(features, "text_embeds"):
        return features.text_embeds
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def encode_texts(model, processor, captions: list[str], batch_size: int, device):
    import torch

    encoded: dict[str, np.ndarray] = {}
    for batch_index, batch in enumerate(_chunks(captions, batch_size), start=1):
        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = _as_tensor(model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features, dim=-1)
        for caption, feature in zip(batch, features.cpu().float().numpy()):
            encoded[caption] = feature
        if batch_index % 25 == 0:
            print(f"Encoded {len(encoded)}/{len(captions)} captions", flush=True)
    return encoded


def encode_images(
    model,
    processor,
    image_urls: list[str],
    image_paths: dict[str, Path],
    batch_size: int,
    device,
):
    import torch
    from PIL import Image

    encoded: dict[str, np.ndarray] = {}
    for batch_index, batch in enumerate(_chunks(image_urls, batch_size), start=1):
        images = []
        for url in batch:
            with Image.open(image_paths[url]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = _as_tensor(model.get_image_features(pixel_values=pixel_values))
            features = torch.nn.functional.normalize(features, dim=-1)
        for url, feature in zip(batch, features.cpu().float().numpy()):
            encoded[url] = feature
        if batch_index % 25 == 0:
            print(f"Encoded {len(encoded)}/{len(image_urls)} images", flush=True)
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--flickr8k-root", type=Path)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--download-retries", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-records-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import transformers
    from transformers import CLIPModel, CLIPProcessor

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dataset, source_prefix, records_by_split, source_hashes = load_source_records(
        args.dataset, args.source_dir.resolve(), args.max_records_per_split
    )
    all_records = [record for split in SPLITS for record in records_by_split[split]]
    image_urls = list(dict.fromkeys(record.image for record in all_records))
    captions = list(dict.fromkeys(record.caption for record in all_records))
    print(
        f"{dataset}: {len(all_records)} records, {len(image_urls)} images, "
        f"{len(captions)} captions on {device}",
        flush=True,
    )

    image_paths = resolve_images(
        args.dataset,
        image_urls,
        args.image_cache.resolve(),
        args.flickr8k_root.resolve() if args.flickr8k_root else None,
        args.download_workers,
        args.download_retries,
    )
    processor = CLIPProcessor.from_pretrained(
        args.model_name, revision=args.revision
    )
    model = CLIPModel.from_pretrained(args.model_name, revision=args.revision)
    model.eval().to(device)
    text_features = encode_texts(
        model, processor, captions, args.batch_size, device
    )
    image_features = encode_images(
        model, processor, image_urls, image_paths, args.batch_size, device
    )

    projection_dim = int(model.config.projection_dim)
    if any(feature.shape != (projection_dim,) for feature in text_features.values()):
        raise ValueError("Unexpected CLIP text feature shape")
    if any(feature.shape != (projection_dim,) for feature in image_features.values()):
        raise ValueError("Unexpected CLIP image feature shape")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slug_model_name(args.model_name)
    output_hashes: dict[str, str] = {}
    for split in SPLITS:
        output_path = args.output_dir / f"{source_prefix}-{split}-{model_slug}.emb"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists (pass --overwrite): {output_path}"
            )
        output_records = [
            Image_Caption_Embedding(
                image=record.image,
                caption=record.caption,
                image_embedding=image_features[record.image],
                caption_embedding=text_features[record.caption],
                ratings=list(record.ratings),
            )
            for record in records_by_split[split]
        ]
        with output_path.open("wb") as stream:
            serialize(output_records, stream)
        output_hashes[split] = sha256_file(output_path)
        print(f"Saved {len(output_records)} records to {output_path}", flush=True)

    metadata = {
        "dataset": dataset,
        "source_prefix": source_prefix,
        "model_name": args.model_name,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "projection_dim": projection_dim,
        "l2_normalized": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version,
        "record_counts": {
            split: len(records_by_split[split]) for split in SPLITS
        },
        "unique_images": len(image_urls),
        "unique_captions": len(captions),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
    }
    metadata_path = args.output_dir / f"{source_prefix}-{model_slug}.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
