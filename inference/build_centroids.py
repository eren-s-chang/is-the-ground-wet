import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build wetness centroids from labels.jsonl")
    parser.add_argument("--labels", default="data/wetness-crops/labels.jsonl", help="Path to labels.jsonl")
    parser.add_argument("--output", default="data/wetness-centroids.json", help="Output centroid JSON")
    args = parser.parse_args()

    label_path = Path(args.labels)
    if not label_path.exists():
        raise SystemExit(f"Label file not found: {label_path}")

    vectors = defaultdict(list)
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            label = str(payload.get("label", "")).strip().lower()
            embedding = payload.get("embedding")
            if not label or embedding is None:
                continue
            vectors[label].append(np.asarray(embedding, dtype=np.float32))

    centroids = {}
    counts = {}
    for label, items in vectors.items():
        if not items:
            continue
        stacked = np.stack(items, axis=0)
        centroid = np.mean(stacked, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[label] = centroid.tolist()
        counts[label] = len(items)

    output = {
        "centroids": centroids,
        "counts": counts,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
