import os
import ipaddress
import socket
import base64
import io
import inspect
import json
from pathlib import Path
from functools import lru_cache

import requests
import torch
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image


MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "sam1").lower()
MODEL_ID = os.getenv("MODEL_ID", "vit_b")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "./sam_vit_b_01ec64.pth")
PAVEMENT_MODEL_PATH = os.getenv("PAVEMENT_MODEL_PATH", "data/models/pavement-wetdry-efficientnet-b0.pt")
PAVEMENT_ARCHITECTURE = os.getenv("PAVEMENT_ARCHITECTURE", "efficientnet_b0")
PAVEMENT_ROI_TYPE = os.getenv("PAVEMENT_ROI_TYPE", "bottom_crop")
PAVEMENT_ROI = tuple(float(value.strip()) for value in os.getenv("PAVEMENT_ROI", "0.2,0.45,0.8,0.98").split(","))
PAVEMENT_WET_THRESHOLD = float(os.getenv("PAVEMENT_WET_THRESHOLD", "0.55"))
PAVEMENT_FALLBACK_MODE = os.getenv("PAVEMENT_FALLBACK_MODE", "heuristic")
PAVEMENT_DEBUG_IMAGE_DIR = os.getenv("PAVEMENT_DEBUG_IMAGE_DIR", "")
PAVEMENT_SEGMENTATION = os.getenv("PAVEMENT_SEGMENTATION", "segformer").lower()
PAVEMENT_SHEEN_WEIGHT = float(os.getenv("PAVEMENT_SHEEN_WEIGHT", "0.35"))
PAVEMENT_ROI_SHEEN_WEIGHT = float(os.getenv("PAVEMENT_ROI_SHEEN_WEIGHT", "0.65"))
SEGFORMER_MODEL_ID = os.getenv("SEGFORMER_MODEL_ID", "nvidia/segformer-b0-finetuned-ade-512-512")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "google/siglip-large-patch16-256")
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
VLM_MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "192"))
VLM_MAX_IMAGE_SIZE = int(os.getenv("VLM_MAX_IMAGE_SIZE", "960"))
VLM_INPUT_MODE = os.getenv("VLM_INPUT_MODE", "full").lower()
WETNESS_CENTROIDS_PATH = os.getenv("WETNESS_CENTROIDS_PATH", "data/wetness-centroids.json")
SURFACE_LABEL_TERMS = [term.strip().lower() for term in os.getenv("SURFACE_LABEL_TERMS", "road,sidewalk,pavement,path,floor,earth,ground,terrain,field,grass,runway").split(",") if term.strip()]
WETNESS_ZERO_SHOT_PROMPTS = {
    "dry": [
        "a road surface with a matte dry texture and no reflections",
        "dry asphalt with even color and visible aggregate texture",
        "dry pavement that looks rough, dusty, and non reflective",
        "a ground surface with no dark wet patches and no pooling",
        "dry asphalt with sunlight glare but no darkened wet sheen",
        "dry pavement with bright highlights from sun but no reflective film",
    ],
    "wet": [
        "a road surface with glossy wet reflections and specular highlights",
        "darkened asphalt with continuous reflective sheen from moisture",
        "wet pavement reflecting sky and lights with visible glare",
        "a slick road surface with a reflective film and dark patches",
        "wet asphalt with reflective streaks and softened texture",
    ],
    "standing_water": [
        "standing water on pavement with mirror like reflections",
        "puddles on asphalt reflecting the sky and surroundings",
        "pooled water on a road surface with bright specular glare",
    ],
}
WETNESS_CLASS_DESCRIPTIONS = {
    "dry": "matte non-reflective dry surface",
    "wet": "glossy reflective wet surface",
    "standing_water": "mirror-like pooled water",
}
WETNESS_ZERO_SHOT_TEMPERATURE = float(os.getenv("WETNESS_ZERO_SHOT_TEMPERATURE", "1.2"))
WETNESS_ZERO_SHOT_FLOOR = float(os.getenv("WETNESS_ZERO_SHOT_FLOOR", "0.08"))
SAM3_CHECKPOINT = os.getenv("SAM3_CHECKPOINT", "")
SAM3_REPO_ID = os.getenv("SAM3_REPO_ID", "1038lab/sam3")
SAM3_CHECKPOINT_FILENAME = os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.pt")
SAM3_BPE_PATH = os.getenv("SAM3_BPE_PATH", "")
SAM3_PROMPTS = [prompt.strip() for prompt in os.getenv("SAM3_PROMPTS", "ground,asphalt,pavement,wet ground,dry ground,wet asphalt,dry asphalt,wet pavement,dry pavement,puddle,standing water").split(",") if prompt.strip()]
SAM3_CONFIDENCE_THRESHOLD = float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.25"))
SAM_POINTS_PER_SIDE = int(os.getenv("SAM_POINTS_PER_SIDE", "8"))
SAM_PRED_IOU_THRESH = float(os.getenv("SAM_PRED_IOU_THRESH", "0.9"))
SAM_STABILITY_SCORE_THRESH = float(os.getenv("SAM_STABILITY_SCORE_THRESH", "0.92"))
SAM_MIN_MASK_REGION_AREA = int(os.getenv("SAM_MIN_MASK_REGION_AREA", "400"))
SAM_MAX_IMAGE_SIZE = int(os.getenv("SAM_MAX_IMAGE_SIZE", "384"))

app = FastAPI(title="Ground Wetness Vision Inference")


class InferenceRequest(BaseModel):
    imageUrl: HttpUrl
    model: str | None = None
    prompts: list[str] | None = None
    roiPolygon: list[dict[str, float]] | None = None


@app.get("/health")
def health():
    if "vlm" in MODEL_PROVIDER:
        device = select_device()
        return {"ok": True, "model": selected_model_id(), "provider": MODEL_PROVIDER, "device": device, "status": "initializing"}
    _, _, device = load_model()
    return {"ok": True, "model": selected_model_id(), "provider": MODEL_PROVIDER, "device": device}


@app.post("/inference/wetness")
def infer_wetness(payload: InferenceRequest):
    processor, model, device = load_model()
    block_private_url(str(payload.imageUrl))

    try:
        response = requests.get(str(payload.imageUrl), stream=True, timeout=8)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not load image: {exc}") from exc

    resized = resize_image(image, VLM_MAX_IMAGE_SIZE if MODEL_PROVIDER == "segformer-vlm" else SAM_MAX_IMAGE_SIZE)
    if MODEL_PROVIDER == "sam3":
        segments, masks, wet, dry = extract_sam3_segments(resized, processor, model, payload.prompts or SAM3_PROMPTS, device)
        extra = {}
    elif MODEL_PROVIDER in {"pavement", "pavement-wetdry"}:
        segments, masks, wet, dry, extra = extract_pavement_wetness(resized, processor, model, device, payload.roiPolygon)
    elif MODEL_PROVIDER in {"segformer", "segformer-siglip", "segformer-vlm"}:
        segments, masks, wet, dry, extra = extract_segformer_wetness(resized, processor, model, device)
    else:
        segments, masks, wet, dry = extract_sam1_segments(resized, model)
        extra = {}
    wetness_score = None if wet + dry == 0 else wet / max(0.01, wet + dry)
    visual_confidence = min(1.0, wet + dry)

    return {
        "model": selected_model_id(),
        "detections": segments,
        "segments": segments,
        "masks": masks,
        "wetnessScore": wetness_score,
        "visualConfidence": visual_confidence,
        **extra,
    }


@lru_cache(maxsize=1)
def load_model():
    device = select_device()

    if MODEL_PROVIDER in {"pavement", "pavement-wetdry"}:
        from inference.pavement_classifier import PavementWetDryClassifier

        classifier = PavementWetDryClassifier(
            model_path=PAVEMENT_MODEL_PATH,
            architecture=PAVEMENT_ARCHITECTURE,
            device=os.getenv("PAVEMENT_CLASSIFIER_DEVICE", device),
            roi_type=PAVEMENT_ROI_TYPE,
            roi=PAVEMENT_ROI,
            wet_threshold=PAVEMENT_WET_THRESHOLD,
            fallback_mode=PAVEMENT_FALLBACK_MODE,
            debug_image_dir=PAVEMENT_DEBUG_IMAGE_DIR,
            sheen_weight=PAVEMENT_SHEEN_WEIGHT,
            roi_sheen_weight=PAVEMENT_ROI_SHEEN_WEIGHT,
        )
        segformer_processor = None
        segformer_model = None
        if PAVEMENT_SEGMENTATION == "segformer":
            try:
                from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
            except ImportError as exc:
                missing = getattr(exc, "name", None) or str(exc)
                raise RuntimeError(f"Segmentation dependency '{missing}' is missing. Install the native vision requirements.") from exc
            segformer_processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
            segformer_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID).to(str(classifier.device))
            segformer_model.eval()
        return {"segformer_processor": segformer_processor}, {"pavement_classifier": classifier, "segformer_model": segformer_model}, str(classifier.device)

    if MODEL_PROVIDER in {"segformer", "segformer-siglip", "segformer-vlm"}:
        try:
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(f"Vision dependency '{missing}' is missing. Install the native vision requirements.") from exc

        segformer_processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
        segformer_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID).to(device)
        segformer_model.eval()

        embedding_processor = None
        embedding_model = None
        if MODEL_PROVIDER == "segformer-siglip":
            embedding_processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)
            embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_ID).to(device)
            embedding_model.eval()

        vlm_processor = None
        vlm_model = None
        if MODEL_PROVIDER == "segformer-vlm":
            from transformers import AutoModelForVision2Seq
            vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
            dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
            vlm_model = AutoModelForVision2Seq.from_pretrained(VLM_MODEL_ID, torch_dtype=dtype).to(device)
            vlm_model.eval()

        return {
            "segformer_processor": segformer_processor,
            "embedding_processor": embedding_processor,
            "vlm_processor": vlm_processor,
        }, {
            "segformer_model": segformer_model,
            "embedding_model": embedding_model,
            "vlm_model": vlm_model,
        }, device

    if MODEL_PROVIDER == "sam3":
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(f"SAM3 dependency '{missing}' is missing. Run `pip install -r inference/requirements-sam3-mps.txt`.") from exc

        try:
            model = build_sam3_model(build_sam3_image_model, device)
        except Exception as exc:
            message = str(exc)
            if "Cannot access gated repo" in message or "GatedRepoError" in type(exc).__name__:
                raise RuntimeError(
                    "SAM3 checkpoint access is gated. Visit https://huggingface.co/facebook/sam3, "
                    "request access for the logged-in Hugging Face account, or set SAM3_CHECKPOINT "
                    "to a local sam3.pt checkpoint."
                ) from exc
            raise
        if hasattr(model, "to"):
            model.to(device)
        return Sam3Processor(model, device=device, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD), model, device

    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError as exc:
        raise RuntimeError("SAM1 dependencies are not installed. Run `pip install -r inference/requirements.txt`.") from exc

    sam = sam_model_registry[MODEL_ID](checkpoint=MODEL_CHECKPOINT)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        points_per_batch=32,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
        crop_n_layers=0,
    )
    return None, mask_generator, device


def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def selected_model_id():
    if MODEL_PROVIDER in {"pavement", "pavement-wetdry"}:
        return f"pavement-wetdry-{PAVEMENT_ARCHITECTURE}"
    if MODEL_PROVIDER == "segformer-vlm":
        return f"{SEGFORMER_MODEL_ID}+{VLM_MODEL_ID}"
    if MODEL_PROVIDER == "segformer-siglip":
        return f"{SEGFORMER_MODEL_ID}+{EMBEDDING_MODEL_ID}"
    if MODEL_PROVIDER == "segformer":
        return SEGFORMER_MODEL_ID
    return MODEL_ID


def build_sam3_model(build_sam3_image_model, device: str):
    signature = inspect.signature(build_sam3_image_model)
    kwargs = {}

    if "model_id" in signature.parameters:
        kwargs["model_id"] = MODEL_ID
    checkpoint_path = resolve_sam3_checkpoint_path()
    if "checkpoint_path" in signature.parameters and checkpoint_path:
        kwargs["checkpoint_path"] = checkpoint_path
    if "device" in signature.parameters:
        kwargs["device"] = device
    if "bpe_path" in signature.parameters:
        kwargs["bpe_path"] = resolve_sam3_bpe_path()

    return build_sam3_image_model(**kwargs)


def resolve_sam3_bpe_path():
    if SAM3_BPE_PATH:
        return SAM3_BPE_PATH

    try:
        import sam3
    except ImportError:
        return None

    package_path = Path(sam3.__file__).resolve().parent
    candidates = [
        package_path / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        package_path.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def resolve_sam3_checkpoint_path():
    if SAM3_CHECKPOINT:
        return SAM3_CHECKPOINT

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download SAM3 checkpoints. Run `python inference/install_sam3_mps.py`."
        ) from exc

    return hf_hub_download(repo_id=SAM3_REPO_ID, filename=SAM3_CHECKPOINT_FILENAME)


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def extract_sam1_segments(image: Image.Image, mask_generator):
    rgb = np.array(image)
    masks = mask_generator.generate(rgb)
    height, width = rgb.shape[:2]
    total_pixels = max(1, height * width)

    segments = []
    normalized_masks = []
    wet = 0.0
    dry = 0.0

    for mask in masks:
        binary = mask.get("segmentation")
        if binary is None:
            continue
        coverage = float(binary.sum()) / total_pixels
        if coverage < 0.002:
            continue
        label, wet_score, dry_score = classify_mask(rgb, binary)
        mask_png = encode_binary_mask(binary)
        segments.append({"className": label, "confidence": round(coverage, 4), "maskPng": mask_png})
        normalized_masks.append({"className": label, "coverage": round(coverage, 4), "maskPng": mask_png})
        wet += wet_score * coverage
        dry += dry_score * coverage

    segments.sort(key=lambda item: item["confidence"], reverse=True)
    segments = segments[:12]
    normalized_masks = [mask for mask in normalized_masks if mask["className"] in {segment["className"] for segment in segments}]
    return segments, normalized_masks, wet, dry


def extract_sam3_segments(image: Image.Image, processor, model, prompts: list[str], device: str):
    width, height = image.size
    total_pixels = max(1, width * height)
    segments = []
    normalized_masks = []
    wet = 0.0
    dry = 0.0

    with torch.inference_mode():
        state = processor.set_image(image)
        for prompt in prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get("masks")
            scores = output.get("scores")

            if masks is None:
                continue

            masks_array = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
            scores_array = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores if scores is not None else [])
            masks_array = normalize_mask_array(masks_array)

            for index, binary in enumerate(masks_array):
                coverage = float(binary.sum()) / total_pixels
                if coverage < 0.002:
                    continue
                label, wet_score, dry_score = label_prompt(prompt, float(scores_array[index]) if index < len(scores_array) else 1.0)
                mask_png = encode_binary_mask(binary)
                confidence = round(min(1.0, coverage), 4)
                segments.append({"className": label, "confidence": confidence, "maskPng": mask_png})
                normalized_masks.append({"className": label, "coverage": confidence, "maskPng": mask_png})
                wet += wet_score * coverage
                dry += dry_score * coverage

    segments.sort(key=lambda item: item["confidence"], reverse=True)
    segments = segments[:12]
    normalized_masks = normalized_masks[:12]
    return segments, normalized_masks, wet, dry


def extract_pavement_wetness(image: Image.Image, processors, models, device: str, roi_polygon: list[dict[str, float]] | None = None):
    from inference.pavement_classifier import crop_roi, encode_roi_png

    classifier = models["pavement_classifier"]
    surface_mask = None
    surface_confidence = 0.0
    crop_source = "fixed_roi"
    if processors and processors.get("segformer_processor") is not None and models.get("segformer_model") is not None:
        surface_mask, surface_confidence = segment_surface(image, processors["segformer_processor"], models["segformer_model"], device, fallback=False)

    if surface_mask is None:
        if roi_polygon:
            surface_mask = make_polygon_mask(image.size, roi_polygon)
            roi_image, roi_box = crop_polygon_region(image, surface_mask)
            roi_debug = normalized_crop_box(roi_box, image.size)
            result = classifier.predict_crop(roi_image, {"polygon": roi_polygon, **roi_debug})
            crop_source = "manual_polygon_roi"
        else:
            result = classifier.predict(image)
            roi_image, _ = crop_roi(image, classifier.roi_type, classifier.roi)
            roi_debug = result.debug.get("roi", {})
            surface_mask = make_roi_mask(image.size, roi_debug)
    else:
        roi_image, crop_box = crop_surface_region(image, surface_mask)
        roi_debug = normalized_crop_box(crop_box, image.size)
        result = classifier.predict_crop(roi_image, roi_debug)
        crop_source = "segformer_surface_crop"

    wetness_score = result.wet_probability
    visual_confidence = result.confidence
    wet = wetness_score * visual_confidence
    dry = (1.0 - wetness_score) * visual_confidence
    class_name = f"pavement_{result.label}"
    mask_png = encode_binary_mask(surface_mask)
    coverage = float(surface_mask.sum()) / max(1, surface_mask.size)
    detection = {"className": class_name, "confidence": round(visual_confidence, 4), "maskPng": mask_png}

    return [detection], [{"className": "surface", "coverage": round(coverage, 4), "confidence": round(surface_confidence or visual_confidence, 4), "maskPng": mask_png}], wet, dry, {
        "crop": {
            **roi_debug,
            "imagePng": encode_roi_png(roi_image),
        },
        "classifier": {
            "nearest": result.label,
            "nearestDescription": f"{result.label} pavement ROI",
            "confidence": round(visual_confidence, 4),
            "wetnessScore": round(wetness_score, 4),
            "margin": 0.0,
            "scores": result.debug.get("probabilities", {}),
            "method": result.debug.get("mode", "pavement-wetdry"),
            "inputMode": crop_source,
            "reason": "Classified from the SegFormer surface crop." if crop_source == "segformer_surface_crop" else "Classified from a fixed bottom-center pavement ROI fallback.",
            "debug": result.debug,
        },
        "classifierInputMode": crop_source,
        "segmentationModel": SEGFORMER_MODEL_ID if surface_confidence else "fixed-roi-mask",
    }


def make_roi_mask(size, roi):
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    x0 = int(width * float(roi.get("x", 0)))
    y0 = int(height * float(roi.get("y", 0)))
    x1 = int(width * float(roi.get("x", 0) + roi.get("width", 0)))
    y1 = int(height * float(roi.get("y", 0) + roi.get("height", 0)))
    mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return mask


def make_polygon_mask(size, polygon):
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    points = []
    for point in polygon:
        x = int(width * float(point.get("x", 0)))
        y = int(height * float(point.get("y", 0)))
        points.append([max(0, min(width - 1, x)), max(0, min(height - 1, y))])
    if len(points) < 3:
        return mask.astype(bool)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
    return mask.astype(bool)


def crop_polygon_region(image: Image.Image, mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        width, height = image.size
        return image.crop((0, 0, width, height)), (0, 0, width, height)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def normalized_crop_box(box, image_size):
    width, height = image_size
    x0, y0, x1, y1 = box
    return {
        "x": round(x0 / max(1, width), 4),
        "y": round(y0 / max(1, height), 4),
        "width": round((x1 - x0) / max(1, width), 4),
        "height": round((y1 - y0) / max(1, height), 4),
    }


def extract_segformer_wetness(image: Image.Image, processors, models, device: str):
    rgb = np.array(image)
    surface_mask, surface_confidence = segment_surface(image, processors["segformer_processor"], models["segformer_model"], device)
    crop, crop_box = crop_surface_region(image, surface_mask)
    mask_png = encode_binary_mask(surface_mask)
    coverage = float(surface_mask.sum()) / max(1, surface_mask.size)

    segments = []
    masks = []
    if coverage >= 0.002 and mask_png:
        segments.append({"className": "surface", "confidence": round(surface_confidence, 4), "maskPng": mask_png})
        masks.append({"className": "surface", "coverage": round(coverage, 4), "confidence": round(surface_confidence, 4), "maskPng": mask_png})

    classifier = None
    classifier_input = image if VLM_INPUT_MODE == "full" else crop
    classifier_input_mode = "full_image" if VLM_INPUT_MODE == "full" else "surface_crop"

    if models.get("vlm_model") is not None:
        classifier = classify_image_with_vlm(classifier_input, processors["vlm_processor"], models["vlm_model"], device, classifier_input_mode)
    elif models.get("embedding_model") is not None:
        classifier = classify_crop_with_embedding(crop, processors["embedding_processor"], models["embedding_model"], device)

    if classifier:
        wetness_score = float(classifier["wetnessScore"])
        visual_confidence = float(classifier["confidence"])
        wet = wetness_score * visual_confidence
        dry = (1.0 - wetness_score) * visual_confidence
    else:
        label, wet_score, dry_score = classify_mask(rgb, surface_mask)
        wet = wet_score * max(coverage, 0.01)
        dry = dry_score * max(coverage, 0.01)
        classifier = {
            "nearest": label,
            "confidence": round(max(wet_score, dry_score), 4),
            "wetnessScore": None if wet_score + dry_score == 0 else round(wet_score / max(0.01, wet_score + dry_score), 4),
            "scores": {},
            "method": "surface-hsv-heuristic",
        }

    return segments, masks, wet, dry, {
        "crop": encode_crop(crop, crop_box, image.size),
        "classifier": classifier,
        "classifierInputMode": classifier_input_mode,
    }


def segment_surface(image: Image.Image, processor, model, device: str, fallback: bool = True):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits, dim=1)[0]
        labels = probabilities.argmax(dim=0)

    surface_ids = get_surface_label_ids(model.config.id2label)
    if not surface_ids:
        if not fallback:
            return None, 0.0
        return fallback_surface_mask(image.size), 0.0

    label_ids = torch.tensor(surface_ids, device=labels.device)
    mask_tensor = torch.isin(labels, label_ids)
    confidence_tensor = probabilities[label_ids].sum(dim=0)
    mask = mask_tensor.detach().cpu().numpy().astype(bool)
    confidence = float(confidence_tensor[mask_tensor].mean().detach().cpu()) if bool(mask_tensor.any()) else 0.0

    if mask.sum() < image.width * image.height * 0.01:
        if not fallback:
            return None, 0.0
        return fallback_surface_mask(image.size), 0.0
    return mask, confidence


def get_surface_label_ids(id2label):
    surface_ids = []
    for raw_id, label in id2label.items():
        normalized = str(label).lower().replace("_", " ")
        if any(term in normalized for term in SURFACE_LABEL_TERMS):
            surface_ids.append(int(raw_id))
    return surface_ids


def fallback_surface_mask(size):
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    y0 = int(height * 0.45)
    y1 = int(height * 0.95)
    x0 = int(width * 0.2)
    x1 = int(width * 0.8)
    mask[y0:y1, x0:x1] = True
    return mask


def crop_surface_region(image: Image.Image, mask: np.ndarray):
    width, height = image.size
    lower_mask = mask.copy()
    lower_mask[: int(height * 0.35), :] = False
    lower_mask[:, : int(width * 0.08)] = False
    lower_mask[:, int(width * 0.92) :] = False

    if lower_mask.sum() < width * height * 0.01:
        box = fixed_crop_box(width, height)
        return image.crop(box), box

    components_count, labels, stats, _ = cv2.connectedComponentsWithStats(lower_mask.astype(np.uint8), 8)
    if components_count <= 1:
        box = fixed_crop_box(width, height)
        return image.crop(box), box

    best_label = max(range(1, components_count), key=lambda index: stats[index, cv2.CC_STAT_AREA])
    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    pad_x = int(width * 0.08)
    pad_y = int(height * 0.08)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)

    if (x1 - x0) * (y1 - y0) < width * height * 0.08:
        box = fixed_crop_box(width, height)
        return image.crop(box), box

    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def fixed_crop_box(width: int, height: int):
    return (int(width * 0.2), int(height * 0.45), int(width * 0.8), int(height * 0.95))


def classify_crop_with_embedding(crop: Image.Image, processor, model, device: str):
    centroid_result = classify_with_centroids(crop, processor, model, device)
    if centroid_result:
        return centroid_result
    return classify_with_zero_shot(crop, processor, model, device)


def classify_image_with_vlm(image: Image.Image, processor, model, device: str, input_mode: str):
    prompt = (
        "Classify the visible road, pavement, sidewalk, or ground surface in this image into the closest condition. "
        "Use the whole image for context, but base the condition only on the visible ground or road surface. "
        "Use dry when the surface looks matte, textured, evenly colored, or only has ordinary sun glare. "
        "Use mostly_dry when a small minority of the surface may be damp. "
        "Use mixed when dry and damp regions are both clearly visible. "
        "Use wet when most of the surface has a darkened moisture film or broad reflective sheen. "
        "Use standing_water when puddles or pooled water are visible. "
        "Choose the nearest class even if imperfect. Do not use unknown. "
        "Return only strict JSON with keys: condition, wetnessScore, confidence, reason. "
        "condition must be one of: dry, mostly_dry, mixed, wet, standing_water. "
        "The reason must describe visible evidence in this image, not repeat these instructions."
    )

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("qwen-vl-utils is required for Qwen2-VL inference. Run `pip install -r inference/requirements-vision-mps.txt`.") from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS)
    generated = generated[:, inputs["input_ids"].shape[1] :]
    output_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    parsed = parse_vlm_json(output_text)
    if not parsed:
        return None

    wetness_score = parsed.get("wetnessScore")
    confidence = parsed.get("confidence")
    reason = clean_vlm_reason(parsed.get("reason", ""))
    wetness_score = clamp_score(wetness_score)
    condition = coerce_vlm_condition(parsed.get("condition"), wetness_score)
    wetness_score = align_wetness_score_to_condition(condition, wetness_score)

    return {
        "nearest": condition,
        "nearestDescription": condition.replace("_", " "),
        "confidence": round(float(clamp_score(confidence)), 4),
        "wetnessScore": round(float(wetness_score), 4),
        "margin": 0.0,
        "scores": {},
        "method": "qwen2vl-full-image" if input_mode == "full_image" else "qwen2vl-crop",
        "inputMode": input_mode,
        "reason": str(reason)[:240],
    }


def classify_with_zero_shot(crop: Image.Image, processor, model, device: str):
    prompts = []
    prompt_labels = []
    for label, label_prompts in WETNESS_ZERO_SHOT_PROMPTS.items():
        for prompt in label_prompts:
            prompts.append(prompt)
            prompt_labels.append(label)

    inputs = processor(text=prompts, images=crop, padding="max_length", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0].detach().cpu().numpy().astype(np.float32)

    grouped_logits = {label: [] for label in WETNESS_ZERO_SHOT_PROMPTS}
    for label, logit in zip(prompt_labels, logits):
        grouped_logits[label].append(float(logit))

    grouped_scores = {label: max(values) if values else -999.0 for label, values in grouped_logits.items()}
    grouped_values = np.asarray(list(grouped_scores.values()), dtype=np.float32) / max(1e-6, WETNESS_ZERO_SHOT_TEMPERATURE)
    grouped_probabilities = softmax(grouped_values)
    scores = {label: float(grouped_probabilities[index]) for index, label in enumerate(grouped_scores.keys())}
    scores = apply_score_floor(scores, WETNESS_ZERO_SHOT_FLOOR)

    result = classifier_from_scores(scores, "siglip-reflectance-zero-shot")
    if result:
        result["promptGroups"] = {
            label: {
                "description": WETNESS_CLASS_DESCRIPTIONS.get(label, label),
                "prompts": label_prompts,
            }
            for label, label_prompts in WETNESS_ZERO_SHOT_PROMPTS.items()
        }
    return result


def classify_with_centroids(crop: Image.Image, processor, model, device: str):
    path = Path(WETNESS_CENTROIDS_PATH)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text())
        centroids = payload.get("centroids", payload)
    except Exception:
        return None

    embedding = embed_crop(crop, processor, model, device)
    scores = {}
    for label, values in centroids.items():
        centroid = np.asarray(values, dtype=np.float32)
        if centroid.ndim != 1 or centroid.size != embedding.size:
            continue
        scores[normalize_label(label)] = cosine_similarity(embedding, centroid)

    if not scores:
        return None
    labels = list(scores.keys())
    similarities = np.asarray([scores[label] for label in labels], dtype=np.float32)
    probabilities = softmax(similarities * 8.0)
    normalized_scores = {label: float(probabilities[index]) for index, label in enumerate(labels)}
    return classifier_from_scores(normalized_scores, "nearest-centroid")


def embed_crop(crop: Image.Image, processor, model, device: str):
    inputs = processor(images=crop, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            features = model.get_image_features(**inputs)
        else:
            outputs = model.vision_model(**inputs)
            features = outputs.pooler_output
    embedding = features[0].detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / max(norm, 1e-6)


def cosine_similarity(left: np.ndarray, right: np.ndarray):
    left_norm = left / max(float(np.linalg.norm(left)), 1e-6)
    right_norm = right / max(float(np.linalg.norm(right)), 1e-6)
    return float(np.dot(left_norm, right_norm))


def softmax(values: np.ndarray):
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / max(float(exp_values.sum()), 1e-6)


def apply_score_floor(scores: dict[str, float], floor: float):
    if not scores or floor <= 0:
        return scores
    floor = max(0.0, min(0.2, float(floor)))
    scaled = {label: max(floor, float(value)) for label, value in scores.items()}
    total = sum(scaled.values())
    if total <= 0:
        return scores
    return {label: value / total for label, value in scaled.items()}


def parse_vlm_json(text: str):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def clamp_score(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def coerce_vlm_condition(condition, wetness_score: float):
    normalized = normalize_label(condition or "")
    allowed = {"dry", "mostly_dry", "mixed", "wet", "standing_water"}
    if normalized in allowed:
        return normalized
    if wetness_score >= 0.82:
        return "standing_water"
    if wetness_score >= 0.62:
        return "wet"
    if wetness_score >= 0.42:
        return "mixed"
    if wetness_score >= 0.22:
        return "mostly_dry"
    return "dry"


def align_wetness_score_to_condition(condition: str, wetness_score: float):
    ranges = {
        "dry": (0.0, 0.2),
        "mostly_dry": (0.2, 0.42),
        "mixed": (0.42, 0.62),
        "wet": (0.62, 0.82),
        "standing_water": (0.82, 1.0),
    }
    low, high = ranges.get(condition, (0.0, 1.0))
    return max(low, min(high, wetness_score))


def clean_vlm_reason(reason):
    text = str(reason or "").strip()
    repeated_instruction_markers = [
        "do not use unknown",
        "return only strict json",
        "condition must be one of",
        "sun glare alone",
        "call wet only",
    ]
    if not text or any(marker in text.lower() for marker in repeated_instruction_markers):
        return "Classified from visible road-surface texture, reflectivity, darkening, and pooled-water cues."
    return text[:240]


def classifier_from_scores(scores: dict[str, float], method: str):
    if not scores:
        return None
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    nearest, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = max(0.0, float(top_score - second_score))
    wet_signal = float(scores.get("wet", 0.0))
    water_signal = float(scores.get("standing_water", 0.0))
    dry_signal = float(scores.get("dry", 0.0))
    wetness_score = max(wet_signal * 0.78, water_signal, 1.0 - dry_signal if nearest.startswith("dry") else wet_signal)
    if nearest.startswith("dry") and dry_signal >= wet_signal and dry_signal >= water_signal:
        wetness_score = min(0.28, max(0.0, 1.0 - dry_signal))
    confidence = min(1.0, 0.45 + margin * 2.2)
    return {
        "nearest": nearest,
        "nearestDescription": WETNESS_CLASS_DESCRIPTIONS.get(nearest, nearest.replace("_", " ")),
        "confidence": round(confidence, 4),
        "wetnessScore": round(float(np.clip(wetness_score, 0.0, 1.0)), 4),
        "margin": round(margin, 4),
        "scores": {label: round(float(value), 4) for label, value in scores.items()},
        "method": method,
    }


def encode_crop(crop: Image.Image, box, image_size):
    width, height = image_size
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG", optimize=True)
    x0, y0, x1, y1 = box
    return {
        "x": round(x0 / max(1, width), 4),
        "y": round(y0 / max(1, height), 4),
        "width": round((x1 - x0) / max(1, width), 4),
        "height": round((y1 - y0) / max(1, height), 4),
        "imagePng": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def normalize_mask_array(masks: np.ndarray):
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim == 4:
        masks = np.squeeze(masks, axis=1) if masks.shape[1] == 1 else masks[:, 0, :, :]
    return masks > 0


def label_prompt(prompt: str, score: float):
    normalized = normalize_label(prompt)
    score = max(0.0, min(1.0, score))
    if any(token in normalized for token in ["wet", "puddle", "water", "flood"]):
        return normalized, score, 0.0
    if any(token in normalized for token in ["dry", "pavement", "road", "ground"]):
        return normalized, 0.0, score
    return normalized, 0.0, 0.0


def classify_mask(rgb: np.ndarray, binary: np.ndarray):
    masked = rgb[binary]
    if masked.size == 0:
        return "uncertain", 0.0, 0.0
    hsv = cv2.cvtColor(masked.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0].astype(np.float32)
    sat = hsv[:, 1].astype(np.float32)
    val = hsv[:, 2].astype(np.float32)
    brightness = np.mean(val) / 255.0
    saturation = np.mean(sat) / 255.0
    contrast = np.std(val) / 255.0
    blue_bias = np.mean((hue > 90) & (hue < 140))

    wet_score = 0.0
    dry_score = 0.0

    if brightness > 0.55 and saturation < 0.35:
        wet_score += 0.6
    if blue_bias > 0.1:
        wet_score += 0.3
    if contrast < 0.08:
        wet_score += 0.2

    if saturation > 0.4 and brightness > 0.45:
        dry_score += 0.4
    if contrast > 0.1:
        dry_score += 0.3
    if brightness < 0.4:
        dry_score += 0.2

    if wet_score >= dry_score and wet_score > 0.2:
        return "wet", wet_score, 0.0
    if dry_score > wet_score and dry_score > 0.2:
        return "dry", 0.0, dry_score
    return "uncertain", 0.0, 0.0


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.BILINEAR)


def encode_binary_mask(mask: np.ndarray) -> str:
    if mask is None or mask.sum() == 0:
        return ""
    image = Image.fromarray((mask * 200).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def block_private_url(image_url: str):
    hostname = requests.utils.urlparse(image_url).hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Image URL host is missing")

    if hostname.lower() in {"localhost"} or hostname.lower().endswith(".localhost"):
        raise HTTPException(status_code=400, detail="Private camera hosts are blocked")

    try:
        addresses = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Image URL host could not be resolved") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise HTTPException(status_code=400, detail="Private camera hosts are blocked")
