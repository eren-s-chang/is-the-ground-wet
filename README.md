# Is The Ground Wet

Estimate whether a specific place's ground is still wet using local weather, recent precipitation, drying conditions, surface assumptions, and optional visual evidence.

## Quick Start

```bash
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The API server runs on `http://localhost:8787`.

For Dockerized camera inference, start the CPU fallback vision service in another terminal:

```bash
npm run dev:inference
```

For native macOS MPS inference with the fast pavement wet/dry classifier, use a clean Python 3.12 environment and run the local service:

```bash
conda create -n ground-vision python=3.12 -y
conda activate ground-vision
python -m pip install -r inference/requirements-vision-mps.txt
npm run dev:inference:local
```

The API defaults to `http://localhost:9000/inference/wetness` for local vision unless `DISABLE_LOCAL_VISION=1` is set. Docker Compose injects the internal `http://inference:9000/inference/wetness` endpoint for containerized runs. Apple MPS is only available when inference runs natively on macOS, not inside Docker.

## How It Works

The backend geocodes the input with OpenStreetMap Nominatim, falls back to Open-Meteo geocoding, fetches recent/current Open-Meteo weather and elevation, and runs a heuristic wetness model. Camera support accepts authorized image/frame URLs plus camera latitude, longitude, elevation, and observation time. Camera images are sent to a separate vision service when available and weighted by detection confidence, distance to target, elevation delta, and freshness. The spatial output is a precomputed hotspot surface, not a water-flow simulation.

The preferred local visual model combines SegFormer surface segmentation with a pavement wet/dry classifier. SegFormer supplies the surface mask overlay and determines the crop sent to the classifier; the classifier resizes that surface crop to 224x224 and runs a reusable EfficientNet-B0/MobileNetV3/ResNet18 checkpoint. The fixed bottom-center ROI is used only as a fallback when segmentation is disabled or unavailable. If no checkpoint is present, it uses an explicit prototype heuristic fallback or raises a clear error depending on `PAVEMENT_FALLBACK_MODE`. Place fine-tuned weights at `data/models/pavement-wetdry-efficientnet-b0.pt` or point `PAVEMENT_MODEL_PATH` to another checkpoint.

Run folder inference and optionally save ROI crops:

```bash
npm run classify:pavement -- data/sample-images --debug-image-dir data/debug-rois
```

Fine-tune on an RSCD/RoadSaW-style dataset:

```bash
npm run train:pavement -- --data-dir data --output data/models/pavement-wetdry-efficientnet-b0.pt
```

Expected dataset format:

```txt
data/train/dry
data/train/wet
data/val/dry
data/val/wet
```

### Quick labeling flow (nearest-centroid)

Label a handful of crops (dry, wet, standing_water) from cameras:

```bash
conda activate ground-vision
python inference/label_wetness.py --image-url "https://images.weatherstem.com/skycamera/alachua/uf/bhgs/snapshot.jpg" --label dry
python inference/label_wetness.py --image-url "https://images.weatherstem.com/skycamera/alachua/uf/bhgs/snapshot.jpg" --label wet
```

Build centroids:

```bash
python inference/build_centroids.py --labels data/wetness-crops/labels.jsonl --output data/wetness-centroids.json
```

Restart the inference service so it loads the centroid file (it reads `WETNESS_CENTROIDS_PATH`).

## API

- `GET /api/health`: service health.
- `POST /api/assess`: accepts `location`, `area`, and optional `cameraSignals`; returns probability, classification, precipitation series, model drivers, area assumptions, and caveats.
- `POST /api/assess/uf-sw-rec`: fixed University of Florida Southwest Recreation Center assessment with configured WeatherSTEM, Ventusky, and WeatherBug/FDOT public feeds.
- `POST /api/camera/analyze`: analyzes one authorized camera/image source and returns normalized visual evidence.

## Production Configuration

- `PORT`: API port, default `8787`.
- `VISION_INFERENCE_URL`: optional vision inference endpoint, for example `http://inference:9000/inference/wetness`.
- `DISABLE_LOCAL_VISION`: set to `1` to prevent the API from trying `http://localhost:9000/inference/wetness` during local development.
- `CORS_ORIGIN`: optional allowed browser origin in production.
- `RATE_LIMIT_WINDOW_MS`: API rate-limit window, default `60000`.
- `RATE_LIMIT_MAX`: max API requests per window, default `80`.
- `MAX_CAMERA_IMAGE_BYTES`: max camera image size, default `6000000`.
- `CAMERA_FETCH_TIMEOUT_MS`: camera validation timeout, default `4500`.
- `PUBLIC_CAMERA_ALLOWLIST_JSON`: optional JSON array of explicit public municipal/weather camera feeds with `id`, `label`, `region`, `latitude`, `longitude`, and `imageUrl` or `url`.
- `PAVEMENT_MODEL_PATH`: wet/dry checkpoint path, default `data/models/pavement-wetdry-efficientnet-b0.pt`.
- `PAVEMENT_ARCHITECTURE`: `efficientnet_b0`, `mobilenet_v3_large`, or `resnet18`, default `efficientnet_b0`.
- `PAVEMENT_CLASSIFIER_DEVICE`: optional `mps`, `cuda`, or `cpu`; auto-detects by default.
- `PAVEMENT_ROI_TYPE`: `bottom_crop` or `trapezoid`, default `bottom_crop`.
- `PAVEMENT_ROI`: normalized ROI coordinates `x0,y0,x1,y1`, default `0.2,0.45,0.8,0.98`.
- `PAVEMENT_WET_THRESHOLD`: wet probability threshold, default `0.55`.
- `PAVEMENT_FALLBACK_MODE`: `heuristic` or `error`, default `heuristic`.
- `PAVEMENT_DEBUG_IMAGE_DIR`: optional directory to save ROI crops during inference.
- `PAVEMENT_SEGMENTATION`: `segformer` to return semantic surface masks and classify the SegFormer surface crop, or `none` to classify only the fixed ROI. Default `segformer`.
- `PAVEMENT_SHEEN_WEIGHT`: blend weight for the classical (non-ML) heuristic when classifier weights exist, default `0.35`.
- `PAVEMENT_ROI_SHEEN_WEIGHT`: stronger classical-heuristic blend for fixed ROI fallback crops, default `0.65`.
- `SEGFORMER_MODEL_ID`: surface segmentation model, default `nvidia/segformer-b0-finetuned-ade-512-512`.
- `EMBEDDING_MODEL_ID`: wet/dry embedding model, default `google/siglip-large-patch16-256` (unused in VLM mode).
- `VLM_MODEL_ID`: Qwen2-VL model for crop classification, default `Qwen/Qwen2-VL-2B-Instruct`.
- `VLM_MAX_NEW_TOKENS`: max tokens for VLM JSON response, default `192`.
- `VLM_INPUT_MODE`: `full` sends the full camera image to Qwen2-VL while still showing the SegFormer crop; `crop` sends only the surface crop. Default `full`.
- `VLM_MAX_IMAGE_SIZE`: max camera image dimension before VLM processing, default `960`.
- `WETNESS_CENTROIDS_PATH`: optional nearest-centroid classifier JSON path, default `data/wetness-centroids.json`.
- `SURFACE_LABEL_TERMS`: comma-separated SegFormer labels to merge into the surface mask.
- `WETNESS_ZERO_SHOT_TEMPERATURE`: temperature for SigLIP prompt aggregation, default `1.2`.
- `WETNESS_ZERO_SHOT_FLOOR`: minimum class probability floor for SigLIP aggregation, default `0.08`.
- `SAM3_CHECKPOINT`: optional local SAM3 checkpoint path for native inference. Leave unset to download from `SAM3_REPO_ID`.
- `SAM3_REPO_ID`: Hugging Face repo for native SAM3 checkpoint download, default `1038lab/sam3`.
- `SAM3_CHECKPOINT_FILENAME`: checkpoint filename in `SAM3_REPO_ID`, default `sam3.pt`.
- `SAM3_BPE_PATH`: optional local SAM3 tokenizer vocab path. Leave unset to use the package asset.
- `SAM3_CONFIDENCE_THRESHOLD`: SAM3 prompt mask threshold for native inference, default `0.25`.

## Security Notes

Camera URLs are treated as sensitive user-provided inputs. The server blocks local/private/link-local hosts, resolves DNS before validation, enforces timeouts and image size checks, and avoids requiring camera credentials in the app. Automatic camera discovery only uses public metadata providers and explicit allowlisted public feeds; it does not scan networks. Do not use unauthorized camera feeds.

## Docker

```bash
docker compose up --build
```

The web app runs on `http://localhost:8787`; the optional Docker inference service runs on `http://localhost:9000`.

## Scripts

- `npm run dev`: run the Express API and Vite frontend together.
- `npm run dev:inference`: run the Dockerized CPU fallback inference service on port `9000`.
- `npm run dev:inference:local`: run the native SegFormer/SigLIP inference service on port `9000`, preferring MPS on macOS.
- `npm run dev:all`: run API, frontend, and inference service together.
- `npm run build`: type-check and build the frontend.
- `npm run test`: run backend model tests.
- `npm run verify`: run tests and build.

## Docs

- `docs/SPEC.md`: MVP technical spec.
- `docs/PHASES.md`: implementation phases and acceptance criteria.
