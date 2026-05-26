import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


@dataclass
class PavementWetDryResult:
    label: str
    confidence: float
    wet_probability: float
    debug: dict[str, Any]


class PavementWetDryClassifier:
    def __init__(
        self,
        model_path: str = "data/models/pavement-wetdry-efficientnet-b0.pt",
        architecture: str = "efficientnet_b0",
        device: str | None = None,
        roi_type: str = "bottom_crop",
        roi: tuple[float, float, float, float] = (0.2, 0.45, 0.8, 0.98),
        wet_threshold: float = 0.55,
        fallback_mode: str = "heuristic",
        debug_image_dir: str = "",
        sheen_weight: float | None = None,
        roi_sheen_weight: float | None = None,
    ):
        self.model_path = Path(model_path)
        self.architecture = architecture
        self.device = torch.device(device or select_device())
        self.roi_type = roi_type
        self.roi = roi
        self.wet_threshold = wet_threshold
        self.fallback_mode = fallback_mode
        self.debug_image_dir = Path(debug_image_dir) if debug_image_dir else None
        self.sheen_weight = clamp01(sheen_weight if sheen_weight is not None else float(os.getenv("PAVEMENT_SHEEN_WEIGHT", "0.35")))
        self.roi_sheen_weight = clamp01(roi_sheen_weight if roi_sheen_weight is not None else float(os.getenv("PAVEMENT_ROI_SHEEN_WEIGHT", "0.65")))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._load_model() if self.model_path.exists() else None

        if self.model is None and self.fallback_mode == "error":
            raise FileNotFoundError(
                f"Pavement wet/dry weights not found at {self.model_path}. "
                "Place a fine-tuned checkpoint there or set PAVEMENT_FALLBACK_MODE=heuristic."
            )

    def predict(self, image_input: str | Path | Image.Image | np.ndarray | torch.Tensor) -> PavementWetDryResult:
        image = to_pil_image(image_input)
        roi_image, roi_box = crop_roi(image, self.roi_type, self.roi)
        debug = {
            "roiType": self.roi_type,
            "roi": normalized_box(roi_box, image.size),
            "modelPath": str(self.model_path),
            "architecture": self.architecture,
            "mode": "pretrained" if self.model is not None else f"fallback-{self.fallback_mode}",
        }

        if self.debug_image_dir:
            self.debug_image_dir.mkdir(parents=True, exist_ok=True)
            debug_path = self.debug_image_dir / f"roi-{len(list(self.debug_image_dir.glob('roi-*.png'))) + 1}.png"
            roi_image.save(debug_path)
            debug["debugImagePath"] = str(debug_path)

        if self.model is not None:
            model_probability = self._predict_model(roi_image)
            wet_probability, manual_debug = manual_classical_wet_probability(roi_image)
            wet_probability = blend_probability(model_probability, wet_probability, self.roi_sheen_weight)
            debug["probabilities"] = {"dry": round(1.0 - wet_probability, 4), "wet": round(wet_probability, 4)}
            debug["modelWetProbability"] = round(model_probability, 4)
            debug.update(manual_debug)
        elif self.fallback_mode == "heuristic":
            wet_probability, heuristic_debug = manual_classical_wet_probability(roi_image)
            debug.update(heuristic_debug)
        else:
            raise RuntimeError(
                f"Unsupported pavement classifier fallback mode '{self.fallback_mode}'. "
                "Use 'heuristic' or provide model weights."
            )

        label = "wet" if wet_probability >= self.wet_threshold else "dry"
        confidence = wet_probability if label == "wet" else 1.0 - wet_probability
        return PavementWetDryResult(label=label, confidence=float(confidence), wet_probability=float(wet_probability), debug=debug)

    def predict_crop(self, roi_image: Image.Image, roi_debug: dict[str, Any] | None = None) -> PavementWetDryResult:
        roi_image = roi_image.convert("RGB").resize((224, 224), Image.BILINEAR)
        debug = {
            "roiType": "segformer_surface_crop",
            "roi": roi_debug or {},
            "modelPath": str(self.model_path),
            "architecture": self.architecture,
            "mode": "pretrained" if self.model is not None else f"fallback-{self.fallback_mode}",
        }

        if self.debug_image_dir:
            self.debug_image_dir.mkdir(parents=True, exist_ok=True)
            debug_path = self.debug_image_dir / f"segformer-roi-{len(list(self.debug_image_dir.glob('segformer-roi-*.png'))) + 1}.png"
            roi_image.save(debug_path)
            debug["debugImagePath"] = str(debug_path)

        if self.model is not None:
            model_probability = self._predict_model(roi_image)
            sheen_probability, manual_debug = manual_classical_wet_probability(roi_image)
            wet_probability = blend_probability(model_probability, sheen_probability, self.sheen_weight)
            debug["probabilities"] = {"dry": round(1.0 - wet_probability, 4), "wet": round(wet_probability, 4)}
            debug["modelWetProbability"] = round(model_probability, 4)
            debug.update(manual_debug)
        elif self.fallback_mode == "heuristic":
            wet_probability, heuristic_debug = manual_classical_wet_probability(roi_image)
            debug.update(heuristic_debug)
        else:
            raise RuntimeError(
                f"Unsupported pavement classifier fallback mode '{self.fallback_mode}'. "
                "Use 'heuristic' or provide model weights."
            )

        label = "wet" if wet_probability >= self.wet_threshold else "dry"
        confidence = wet_probability if label == "wet" else 1.0 - wet_probability
        return PavementWetDryResult(label=label, confidence=float(confidence), wet_probability=float(wet_probability), debug=debug)

    def _predict_model(self, roi_image: Image.Image) -> float:
        tensor = self.transform(roi_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            if logits.ndim == 2 and logits.shape[1] == 2:
                probability = torch.softmax(logits, dim=1)[0, 1]
            else:
                probability = torch.sigmoid(logits.reshape(-1)[0])
        return float(probability.detach().cpu())

    def _load_model(self):
        model = build_model(self.architecture)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model


def build_model(architecture: str):
    if architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
        return model
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return model
    if architecture != "efficientnet_b0":
        raise ValueError(f"Unsupported pavement classifier architecture: {architecture}")
    model = models.efficientnet_b0(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    return model


def crop_roi(image: Image.Image, roi_type: str = "bottom_crop", roi: tuple[float, float, float, float] = (0.2, 0.45, 0.8, 0.98)):
    width, height = image.size
    if roi_type == "trapezoid":
        # The classifier consumes a rectangular crop; trapezoid is represented by a tight lower-road box.
        roi = (0.15, 0.42, 0.85, 0.98)
    x0 = int(width * clamp01(roi[0]))
    y0 = int(height * clamp01(roi[1]))
    x1 = int(width * clamp01(roi[2]))
    y1 = int(height * clamp01(roi[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid pavement ROI coordinates; expected x1>x0 and y1>y0")
    return image.crop((x0, y0, x1, y1)).resize((224, 224), Image.BILINEAR), (x0, y0, x1, y1)


def manual_classical_wet_probability(roi_image: Image.Image):
    rgb = np.asarray(roi_image.convert("RGB"))
    height, width = rgb.shape[:2]
    mask = trapezoid_mask(width, height)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    entropy_map = local_entropy_map(l_channel, bins=16, window=9)
    entropy_mean = masked_mean(entropy_map, mask)
    specular_ratio = specular_highlight_ratio(l_channel, a_channel, b_channel, mask)
    high_freq_energy = fft_high_frequency_energy(l_channel, mask)
    l_mean = masked_mean(l_channel / 255.0, mask)

    smoothness = max(0.0, min(1.0, 1.0 - entropy_mean))
    low_freq_energy = max(0.0, min(1.0, 1.0 - high_freq_energy))
    darkness = max(0.0, min(1.0, 1.0 - l_mean))

    wet_probability = (
        specular_ratio * 0.4
        + smoothness * 0.25
        + low_freq_energy * 0.2
        + darkness * 0.15
    )
    if specular_ratio < 0.01 and entropy_mean > 0.65:
        wet_probability *= 0.7
    wet_probability = max(0.0, min(1.0, wet_probability))
    return wet_probability, {
        "manualHeuristic": {
            "roiMask": "trapezoid",
            "entropyMean": round(entropy_mean, 4),
            "specularRatio": round(specular_ratio, 4),
            "highFreqEnergy": round(high_freq_energy, 4),
            "lMean": round(l_mean, 4),
            "smoothness": round(smoothness, 4),
            "lowFreqEnergy": round(low_freq_energy, 4),
            "darkness": round(darkness, 4),
            "wetProbability": round(wet_probability, 4),
        }
    }


def trapezoid_mask(width: int, height: int):
    top_y = int(height * 0.08)
    bottom_y = int(height * 0.98)
    top_left = (int(width * 0.28), top_y)
    top_right = (int(width * 0.72), top_y)
    bottom_left = (int(width * 0.04), bottom_y)
    bottom_right = (int(width * 0.96), bottom_y)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)], 1)
    return mask.astype(bool)


def local_entropy_map(l_channel: np.ndarray, bins: int = 16, window: int = 9):
    l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
    quantized = (l_uint8.astype(np.int32) * bins) // 256
    kernel = (window, window)
    entropy = np.zeros_like(l_channel, dtype=np.float32)
    for value in range(bins):
        mask = (quantized == value).astype(np.float32)
        prob = cv2.blur(mask, kernel)
        entropy -= prob * np.log2(np.maximum(prob, 1e-6))
    entropy = entropy / max(1.0, np.log2(bins))
    return np.clip(entropy, 0.0, 1.0)


def specular_highlight_ratio(l_channel: np.ndarray, a_channel: np.ndarray, b_channel: np.ndarray, mask: np.ndarray):
    chroma = np.sqrt((a_channel - 128.0) ** 2 + (b_channel - 128.0) ** 2)
    highlight = (l_channel > 210.0) & (chroma < 12.0)
    return masked_mean(highlight.astype(np.float32), mask)


def fft_high_frequency_energy(l_channel: np.ndarray, mask: np.ndarray):
    l_scaled = l_channel.copy()
    l_scaled[~mask] = 0.0
    fft = np.fft.fftshift(np.fft.fft2(l_scaled))
    magnitude = np.abs(fft)
    h, w = l_channel.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * 0.12)
    low_freq_mask = np.zeros((h, w), dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    low_freq_mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius] = True
    total_energy = float(np.sum(magnitude)) + 1e-6
    high_energy = float(np.sum(magnitude[~low_freq_mask]))
    return max(0.0, min(1.0, high_energy / total_energy))


def masked_mean(values: np.ndarray, mask: np.ndarray):
    masked_values = values[mask]
    if masked_values.size == 0:
        return 0.0
    return float(np.mean(masked_values))


def blend_probability(model_probability: float, sheen_probability: float, sheen_weight: float):
    sheen_weight = clamp01(sheen_weight)
    return max(0.0, min(1.0, (1.0 - sheen_weight) * float(model_probability) + sheen_weight * float(sheen_probability)))


def encode_roi_png(roi_image: Image.Image):
    buffer = io.BytesIO()
    roi_image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def to_pil_image(image_input: str | Path | Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, torch.Tensor):
        array = image_input.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.moveaxis(array, 0, -1)
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(image_input, np.ndarray):
        array = image_input
        if array.ndim == 3 and array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def normalized_box(box, image_size):
    width, height = image_size
    x0, y0, x1, y1 = box
    return {
        "x": round(x0 / max(1, width), 4),
        "y": round(y0 / max(1, height), 4),
        "width": round((x1 - x0) / max(1, width), 4),
        "height": round((y1 - y0) / max(1, height), 4),
    }


def select_device():
    env_device = os.getenv("PAVEMENT_CLASSIFIER_DEVICE", "").lower()
    if env_device:
        return env_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clamp01(value: float):
    return max(0.0, min(1.0, float(value)))
