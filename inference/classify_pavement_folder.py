import argparse
from pathlib import Path

from inference.pavement_classifier import PavementWetDryClassifier


def main():
    parser = argparse.ArgumentParser(description="Run pavement wet/dry classification on a folder of images")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument("--model-path", default="data/models/pavement-wetdry-efficientnet-b0.pt")
    parser.add_argument("--fallback-mode", default="heuristic", choices=["heuristic", "error"])
    parser.add_argument("--debug-image-dir", default="", help="Optional directory for ROI crops")
    args = parser.parse_args()

    classifier = PavementWetDryClassifier(
        model_path=args.model_path,
        fallback_mode=args.fallback_mode,
        debug_image_dir=args.debug_image_dir,
    )
    paths = sorted(path for path in Path(args.folder).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise SystemExit(f"No images found in {args.folder}")

    for path in paths:
        result = classifier.predict(path)
        print(f"{path.name}\t{result.label}\tconfidence={result.confidence:.3f}\twet={result.wet_probability:.3f}\tmode={result.debug['mode']}")


if __name__ == "__main__":
    main()
