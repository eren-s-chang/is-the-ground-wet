export const visualWetnessModel = {
  id: 'pavement-wetdry-efficientnet-b0',
  task: 'segformer-surface-mask-fixed-roi-pavement-classification',
  prompts: [],
  source: 'local checkpoint: data/models/pavement-wetdry-efficientnet-b0.pt',
};

export async function analyzeImageWetness({ imageUrl, roiPolygon, fetchJson = defaultFetchJson }) {
  const endpoint = process.env.VISION_INFERENCE_URL || (process.env.DISABLE_LOCAL_VISION === '1' ? '' : 'http://localhost:9000/inference/wetness');

  if (!endpoint || !imageUrl) {
    return {
      available: false,
      model: visualWetnessModel,
      detections: [],
      wetnessScore: null,
      visualConfidence: 0,
      caveats: ['Vision inference is configured for production but VISION_INFERENCE_URL is not set.'],
    };
  }

  const data = await fetchJson(new URL(endpoint), 'Vision inference', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ imageUrl, model: visualWetnessModel.id, roiPolygon }),
  });

  const detections = normalizeDetections(data.detections);
  const masks = normalizeMasks(data.masks || data.segments);
  const wetnessScore = scoreDetections(masks.length ? masks : detections, data.wetnessScore);
  const visualConfidence = Math.max(0, Math.min(1, Number(data.visualConfidence ?? maxConfidence(detections)) || 0));

  return {
    available: true,
    model: data.model || visualWetnessModel,
    detections,
    masks,
    wetnessScore,
    visualConfidence,
    crop: normalizeCrop(data.crop),
    classifier: normalizeClassifier(data.classifier),
    classifierInputMode: data.classifierInputMode ? String(data.classifierInputMode) : '',
    caveats: Array.isArray(data.caveats) ? data.caveats.map(String) : [],
  };
}

export function conditionFromVision(wetnessScore) {
  if (!Number.isFinite(wetnessScore)) return 'unknown';
  if (wetnessScore >= 0.82) return 'standing_water';
  if (wetnessScore >= 0.62) return 'wet';
  if (wetnessScore >= 0.42) return 'mixed';
  if (wetnessScore >= 0.22) return 'mostly_dry';
  return 'dry';
}

export function normalizeDetections(detections = []) {
  if (!Array.isArray(detections)) {
    return [];
  }

  return detections.slice(0, 20).map((detection) => {
    const normalized = {
      className: String(detection.className || detection.label || 'unknown').toLowerCase().replace(/\s+/g, '_'),
      confidence: Math.max(0, Math.min(1, Number(detection.confidence ?? detection.score) || 0)),
      boundingBox: detection.boundingBox || detection.box || null,
    };

    if (detection.maskPng) {
      normalized.maskPng = String(detection.maskPng).slice(0, 200000);
    }

    return normalized;
  });
}

export function normalizeMasks(masks = []) {
  if (!Array.isArray(masks)) {
    return [];
  }

  return masks.slice(0, 12).map((mask) => ({
    className: String(mask.className || mask.label || 'unknown').toLowerCase().replace(/\s+/g, '_'),
    coverage: Math.max(0, Math.min(1, Number(mask.coverage ?? mask.confidence) || 0)),
    confidence: Math.max(0, Math.min(1, Number(mask.coverage ?? mask.confidence) || 0)),
    maskPng: mask.maskPng ? String(mask.maskPng).slice(0, 200000) : undefined,
  }));
}

export function normalizePolygon(points) {
  if (!Array.isArray(points)) {
    return null;
  }

  const normalized = points
    .map((point) => ({
      x: clamp01(Number(point?.x)),
      y: clamp01(Number(point?.y)),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  return normalized.length >= 3 ? normalized.slice(0, 20) : null;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function scoreDetections(detections, providerScore) {
  const explicitScore = Number(providerScore);
  if (Number.isFinite(explicitScore)) {
    return Math.max(0, Math.min(1, explicitScore));
  }

  let wetEvidence = 0;
  let dryEvidence = 0;

  for (const detection of detections) {
    const confidence = Number(detection.coverage ?? detection.confidence) || 0;
    if (['wet_ground', 'standing_water', 'puddle', 'mud', 'water', 'flooded', 'wet'].includes(detection.className)) {
      wetEvidence += confidence;
    }

    if (['dry_ground', 'dry_pavement', 'dry_surface', 'road', 'sidewalk', 'ground', 'grass', 'dirt', 'soil', 'sand', 'asphalt', 'concrete', 'path', 'dry'].includes(detection.className)) {
      dryEvidence += confidence;
    }
  }

  if (wetEvidence === 0 && dryEvidence === 0) {
    return null;
  }

  return Math.max(0, Math.min(1, wetEvidence / Math.max(0.01, wetEvidence + dryEvidence)));
}

function maxConfidence(detections) {
  return detections.reduce((max, detection) => Math.max(max, detection.confidence), 0);
}

function normalizeCrop(crop) {
  if (!crop || typeof crop !== 'object') {
    return null;
  }

  return {
    x: Math.max(0, Math.min(1, Number(crop.x) || 0)),
    y: Math.max(0, Math.min(1, Number(crop.y) || 0)),
    width: Math.max(0, Math.min(1, Number(crop.width) || 0)),
    height: Math.max(0, Math.min(1, Number(crop.height) || 0)),
    imagePng: crop.imagePng ? String(crop.imagePng).slice(0, 300000) : undefined,
  };
}

function normalizeClassifier(classifier) {
  if (!classifier || typeof classifier !== 'object') {
    return null;
  }

  const scores = classifier.scores && typeof classifier.scores === 'object'
    ? Object.fromEntries(Object.entries(classifier.scores).map(([key, value]) => [String(key), Math.max(0, Math.min(1, Number(value) || 0))]))
    : {};

  return {
    nearest: classifier.nearest ? String(classifier.nearest) : 'unknown',
    nearestDescription: classifier.nearestDescription ? String(classifier.nearestDescription) : '',
    confidence: Math.max(0, Math.min(1, Number(classifier.confidence) || 0)),
    wetnessScore: Number.isFinite(Number(classifier.wetnessScore)) ? Math.max(0, Math.min(1, Number(classifier.wetnessScore))) : null,
    margin: Math.max(0, Math.min(1, Number(classifier.margin) || 0)),
    method: classifier.method ? String(classifier.method) : 'unknown',
    inputMode: classifier.inputMode ? String(classifier.inputMode) : '',
    reason: classifier.reason ? String(classifier.reason).slice(0, 240) : '',
    scores,
    promptGroups: classifier.promptGroups && typeof classifier.promptGroups === 'object' ? classifier.promptGroups : {},
  };
}

async function defaultFetchJson(url, serviceName, options) {
  const response = await fetch(url, options);
  const data = await response.json();

  if (!response.ok || data.error) {
    throw new Error(`${serviceName} error: ${data.reason || data.error || response.statusText}`);
  }

  return data;
}
