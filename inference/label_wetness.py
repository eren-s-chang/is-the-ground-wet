import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from inference.app import (
    EMBEDDING_MODEL_ID,
    SEGFORMER_MODEL_ID,
    embed_crop,
    load_model,
    crop_surface_region,
    segment_surface,
)


def main():
    parser = argparse.ArgumentParser(description="Label a camera crop for wetness centroid training")
    parser.add_argument("--image-url", help="Camera image URL")
    parser.add_argument("--image-path", help="Local image path")
    parser.add_argument("--label", required=True, choices=["dry", "wet", "standing_water"], help="Wetness label")
    parser.add_argument("--output-dir", default="data/wetness-crops", help="Directory to store crops and JSONL")
    parser.add_argument("--note", default="", help="Optional note")
    args = parser.parse_args()

    if not args.image_url and not args.image_path:
        raise SystemExit("Provide --image-url or --image-path")

    image = load_image(args.image_url, args.image_path)
    processors, models, device = load_model()

    surface_mask, _ = segment_surface(image, processors["segformer_processor"], models["segformer_model"], device)
    crop, crop_box = crop_surface_region(image, surface_mask)
    embedding = embed_crop(crop, processors["embedding_processor"], models["embedding_model"], device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    name = build_crop_name(args.label, args.image_url, args.image_path, timestamp)
    crop_path = output_dir / f"{name}.png"
    crop.save(crop_path, format="PNG", optimize=True)

    entry = {
        "id": name,
        "label": args.label,
        "source": args.image_url or str(Path(args.image_path).resolve()),
        "cropPath": str(crop_path),
        "cropBox": {
            "x": crop_box[0],
            "y": crop_box[1],
            "width": crop_box[2] - crop_box[0],
            "height": crop_box[3] - crop_box[1],
        },
        "embedding": embedding.tolist(),
        "model": {
            "segformer": SEGFORMER_MODEL_ID,
            "embedding": EMBEDDING_MODEL_ID,
        },
        "note": args.note,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
    }

    jsonl_path = output_dir / "labels.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")

    print(f"saved {crop_path}")
    print(f"appended label to {jsonl_path}")


def load_image(image_url: str | None, image_path: str | None) -> Image.Image:
    if image_url:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")

    path = Path(image_path)
    if not path.exists():
        raise SystemExit(f"Image path not found: {path}")
    return Image.open(path).convert("RGB")


def build_crop_name(label: str, image_url: str | None, image_path: str | None, timestamp: int) -> str:
    if image_url:
        parsed = urlparse(image_url)
        base = Path(parsed.path).stem or "camera"
    else:
        base = Path(image_path).stem
    base = base.replace(" ", "-")
    return f"{label}-{base}-{timestamp}"


if __name__ == "__main__":
    main()
