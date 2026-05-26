import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { assessWetness, normalizeArea, normalizeCameraSignals } from './wetnessModel.js';
import { fetchElevationGrid, fetchElevationMeters } from './elevationService.js';
import {
  cameraGeometryWeight,
  elevationDeltaMeters,
  freshnessWeight,
  haversineDistanceMeters,
  observationFreshnessMinutes,
} from './geometryService.js';
import { fetchCameraImageMetadata, validateCameraImageUrl } from './cameraImageService.js';
import { analyzeImageWetness, conditionFromVision, normalizePolygon, visualWetnessModel } from './visionInferenceClient.js';
import { buildAutomaticEvidenceInputs, discoverPublicSources } from './publicSourceService.js';

const app = express();
const port = Number(process.env.PORT || 8787);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distPath = path.resolve(__dirname, '../dist');
const nominatimUrl = 'https://nominatim.openstreetmap.org/search';
const geocodingUrl = 'https://geocoding-api.open-meteo.com/v1/search';
const forecastUrl = 'https://api.open-meteo.com/v1/forecast';
const currentVariables = [
  'temperature_2m',
  'relative_humidity_2m',
  'apparent_temperature',
  'is_day',
  'precipitation',
  'rain',
  'showers',
  'snowfall',
  'weather_code',
  'cloud_cover',
  'pressure_msl',
  'wind_speed_10m',
  'wind_direction_10m',
  'wind_gusts_10m',
];
const primaryHourlyVariables = [
  'temperature_2m',
  'relative_humidity_2m',
  'dew_point_2m',
  'precipitation',
  'rain',
  'showers',
  'snowfall',
  'snow_depth',
  'weather_code',
  'pressure_msl',
  'cloud_cover',
  'visibility',
  'evapotranspiration',
  'et0_fao_evapotranspiration',
  'vapour_pressure_deficit',
  'wind_speed_10m',
  'wind_gusts_10m',
  'soil_temperature_0cm',
  'soil_moisture_0_to_1cm',
  'shortwave_radiation',
  'direct_radiation',
  'diffuse_radiation',
  'uv_index',
];
const fallbackHourlyVariables = [
  'temperature_2m',
  'relative_humidity_2m',
  'precipitation',
  'rain',
  'showers',
  'snowfall',
  'weather_code',
  'cloud_cover',
  'wind_speed_10m',
  'wind_gusts_10m',
  'shortwave_radiation',
];
const rateLimitWindowMs = Number(process.env.RATE_LIMIT_WINDOW_MS || 60_000);
const rateLimitMax = Number(process.env.RATE_LIMIT_MAX || 80);
const rateLimitBuckets = new Map();
const cameraRoiOverrides = new Map();
const weatherBugGainesvilleTrafficUrl = 'https://www.weatherbug.com/traffic-cam/gainesville-fl-32603';
const ufSouthwestRecPlace = {
  query: 'Southwest Recreation Center, University of Florida',
  id: 'uf-sw-rec-fixed',
  provider: 'Fixed campus route',
  name: 'Southwest Recreation Center',
  displayName: 'Southwest Recreation Center, University of Florida, Gainesville, FL',
  admin1: 'Florida',
  admin2: 'Alachua County',
  country: 'United States',
  countryCode: 'US',
  latitude: 29.638383,
  longitude: -82.367133,
  elevationMeters: null,
  timezone: 'auto',
};
const ufSouthwestRecCameras = [
  {
    id: 'uf-weatherstem-ben-hill-griffin',
    label: 'UF WeatherSTEM live camera near Ben Hill Griffin Stadium',
    imageUrl: 'https://images.weatherstem.com/skycamera/alachua/uf/bhgs/snapshot.jpg',
    condition: 'unknown',
    confidence: 0.45,
    latitude: 29.6499,
    longitude: -82.3486,
    source: 'weatherstem',
    caveats: ['Public WeatherSTEM live feed supplied explicitly for this UF route.'],
  },
  {
    id: 'ventusky-uf-century-tower',
    label: 'Ventusky UF Century Tower webcam',
    imageUrl: 'https://webcams.ventusky.com/data/81/961662181/latest.jpg',
    condition: 'unknown',
    confidence: 0.38,
    latitude: 29.65,
    longitude: -82.35,
    source: 'ventusky',
    caveats: ['Public Ventusky page supplied explicitly for this UF route; camera is campus-adjacent, not at SWRC.'],
  },
  makeWeatherBugTrafficCamera({
    id: '433468',
    key: 'adbe452d15721ba1bbd5e9806514ad70d489ea97a98b49f686334fab2ffcf558',
    label: 'WeatherBug/FDOT I-75 @ MM 384.7',
    latitude: 29.628903,
    longitude: -82.393921,
    confidence: 0.3,
  }),
  makeWeatherBugTrafficCamera({
    id: '458984',
    key: '568577228538e546f527a2f290f313e4c671e12b2d62a73b8179b0d6971e559d',
    label: 'WeatherBug/FDOT I-75 @ MM 384.3',
    latitude: 29.623869,
    longitude: -82.390401,
    confidence: 0.28,
  }),
  makeWeatherBugTrafficCamera({
    id: '433469',
    key: '1ff8a29778cfdb489b77685a4da52604a4967ff9a4c96900dbfb2df4333fa46d',
    label: 'WeatherBug/FDOT I-75 @ MM 385.7/SW 20th Ave',
    latitude: 29.640094,
    longitude: -82.402643,
    confidence: 0.3,
  }),
  makeWeatherBugTrafficCamera({
    id: '433467',
    key: '3c78b0eb2dcd7e3ecb82273c1d9cdc764ff96bb176a9a965743548da6b086d14',
    label: 'WeatherBug/FDOT I-75 @ FL-24/Archer Rd',
    latitude: 29.617029,
    longitude: -82.385884,
    confidence: 0.27,
  }),
  makeWeatherBugTrafficCamera({
    id: '433470',
    key: '8daedb3309a4074fe708068ac34183c774d4da0f7352bfb4ad91ae6c9399b4ca',
    label: 'WeatherBug/FDOT I-75 @ MM 386.4',
    latitude: 29.649421,
    longitude: -82.409888,
    confidence: 0.27,
  }),
  makeWeatherBugTrafficCamera({
    id: '462878',
    key: 'dbc7676e36ec58fb24a7625678ec125ae4af5055dd9b50b1edc9c70921acafe9',
    label: 'WeatherBug/FDOT I-75 @ MM 386.9',
    latitude: 29.655649,
    longitude: -82.414841,
    confidence: 0.24,
  }),
];

app.use(express.json({ limit: '1mb' }));
app.use(applySecurityHeaders);
app.use(applyCors);
app.use(applyRateLimit);

app.get('/api/health', (_request, response) => {
  response.json({ ok: true, service: 'is-the-ground-wet', visualModel: visualWetnessModel.id });
});

app.get('/api/camera/image', async (request, response) => {
  try {
    const validation = await validateCameraImageUrl(String(request.query.url || ''));
    if (!validation.ok) {
      response.status(400).json({ error: validation.reason });
      return;
    }

    const upstream = await fetch(validation.normalizedUrl, { headers: { 'user-agent': 'is-the-ground-wet/0.1 image-preview' } });
    const contentType = upstream.headers.get('content-type') || '';

    if (!upstream.ok || !contentType.startsWith('image/')) {
      response.status(upstream.status || 502).json({ error: 'Camera image could not be loaded.' });
      return;
    }

    const bytes = Buffer.from(await upstream.arrayBuffer());
    response.setHeader('content-type', contentType);
    response.setHeader('cache-control', 'public, max-age=60');
    response.send(bytes);
  } catch (error) {
    response.status(502).json({ error: error.message || 'Camera image could not be loaded.' });
  }
});

app.post('/api/camera/analyze', async (request, response) => {
  try {
    const camera = normalizeCameraInputs([request.body?.camera || request.body])[0];
    const analysis = await analyzeCamera(camera, null);
    response.json(analysis);
  } catch (error) {
    const status = error.status || 500;
    response.status(status).json({ error: error.message || 'Unable to analyze camera image.', status });
  }
});

app.post('/api/camera/roi', (request, response) => {
  const cameraId = String(request.body?.cameraId || '').trim();
  const roiPolygon = normalizePolygon(request.body?.roiPolygon);

  if (!cameraId || !roiPolygon) {
    response.status(400).json({ error: 'Invalid cameraId or roiPolygon.' });
    return;
  }

  cameraRoiOverrides.set(cameraId, roiPolygon);
  response.json({ ok: true, cameraId, roiPolygon });
});

app.post('/api/assess', async (request, response) => {
  try {
    const assessmentRequest = parseAssessmentRequest(request.body);
    const place = await geocodeLocation(assessmentRequest.location);
    response.json(await runAssessment({ assessmentRequest, place }));
  } catch (error) {
    const status = error.status || 500;
    response.status(status).json({
      error: error.message || 'Unable to assess ground wetness.',
      status,
    });
  }
});

app.post('/api/assess/uf-sw-rec', async (_request, response) => {
  try {
    const cameras = await resolveWeatherBugTrafficCameras(ufSouthwestRecCameras);
    const assessmentRequest = {
      location: ufSouthwestRecPlace.displayName,
      area: normalizeArea({ radiusMeters: 1800, surface: 'grass', shade: 'full_sun', drainage: 'average', slope: 'flat', traffic: 'foot' }),
      cameras: normalizeCameraInputs(cameras),
    };

    response.json(await runAssessment({ assessmentRequest, place: { ...ufSouthwestRecPlace } }));
  } catch (error) {
    const status = error.status || 500;
    response.status(status).json({
      error: error.message || 'Unable to assess ground wetness.',
      status,
    });
  }
});

app.use(express.static(distPath));

app.get(/.*/, (_request, response) => {
  response.sendFile(path.join(distPath, 'index.html'));
});

app.listen(port, () => {
  console.log(`is-the-ground-wet listening on http://localhost:${port}`);
});

function parseAssessmentRequest(body) {
  const location = String(body?.location || '').trim();

  if (location.length < 2) {
    throw new HttpError(400, 'Enter a location with at least 2 characters.');
  }

  if (location.length > 200) {
    throw new HttpError(400, 'Location is too long.');
  }

  return {
    location,
    area: normalizeArea(body?.target || body?.area || deriveAreaFromAddress(location)),
    cameras: normalizeCameraInputs(body?.cameras || body?.cameraSignals),
  };
}

function makeWeatherBugTrafficCamera({ id, key, label, latitude, longitude, confidence }) {
  return {
    id: `weatherbug-fdot-${id}`,
    trafficCameraId: id,
    label,
    imageUrl: `https://cmn-trffc.pulse.weatherbug.net/media/trffc/v2/img/large?system=weatherbug-web&id=${id}&key=${key}&rate=300000`,
    condition: 'unknown',
    confidence,
    latitude,
    longitude,
    source: 'weatherbug',
    caveats: ['Public WeatherBug traffic camera page supplied explicitly for this UF route; road camera is downweighted for SWRC ground inference.'],
  };
}

async function resolveWeatherBugTrafficCameras(cameras) {
  const weatherBugCameras = cameras.filter((camera) => camera.source === 'weatherbug' && camera.trafficCameraId);

  if (weatherBugCameras.length === 0) {
    return cameras;
  }

  let pageHtml;

  try {
    pageHtml = await fetchText(new URL(weatherBugGainesvilleTrafficUrl), 'WeatherBug traffic cameras', {
      headers: {
        'user-agent': 'is-the-ground-wet/0.1 camera-refresh',
      },
    });
  } catch (error) {
    return cameras.map((camera) => camera.source === 'weatherbug'
      ? { ...camera, caveats: [...(camera.caveats || []), `WeatherBug dynamic refresh failed: ${error.message}`] }
      : camera);
  }

  const dynamicUrls = extractWeatherBugTrafficImageUrls(pageHtml);

  if (dynamicUrls.size === 0) {
    return cameras.map((camera) => camera.source === 'weatherbug'
      ? { ...camera, caveats: [...(camera.caveats || []), 'WeatherBug dynamic refresh found no traffic camera image URLs.'] }
      : camera);
  }

  return cameras.map((camera) => {
    if (camera.source !== 'weatherbug') {
      return camera;
    }

    const dynamicUrl = dynamicUrls.get(String(camera.trafficCameraId));

    if (!dynamicUrl) {
      return {
        ...camera,
        caveats: [...(camera.caveats || []), 'WeatherBug dynamic refresh did not include this camera, so the configured fallback URL was used.'],
      };
    }

    return {
      ...camera,
      imageUrl: dynamicUrl,
      url: dynamicUrl,
      caveats: [...(camera.caveats || []), 'WeatherBug image URL refreshed dynamically from the Gainesville traffic camera page.'],
    };
  });
}

function extractWeatherBugTrafficImageUrls(html) {
  const urlsById = new Map();
  const encodedMatches = String(html).matchAll(/_next\/image\?url=([^"'&\s<>]+)/g);
  const rawMatches = String(html).matchAll(/https:\/\/cmn-trffc\.pulse\.weatherbug\.net\/media\/trffc\/v2\/img\/large\?[^"'\s<>\\]+/g);

  for (const match of encodedMatches) {
    tryAddWeatherBugUrl(urlsById, decodeURIComponent(match[1]));
  }

  for (const match of rawMatches) {
    tryAddWeatherBugUrl(urlsById, match[0].replace(/&amp;/g, '&'));
  }

  return urlsById;
}

function tryAddWeatherBugUrl(urlsById, candidate) {
  let url;

  try {
    url = new URL(String(candidate).replace(/&amp;/g, '&'));
  } catch {
    return;
  }

  if (url.hostname !== 'cmn-trffc.pulse.weatherbug.net') {
    return;
  }

  const id = url.searchParams.get('id');

  if (!id || urlsById.has(id)) {
    return;
  }

  urlsById.set(id, url.toString());
}

async function runAssessment({ assessmentRequest, place }) {
  place.elevationMeters ??= await fetchElevationMeters(place, fetchJson);
  const terrainGrid = await fetchElevationGrid(place, fetchJson, { radiusMeters: assessmentRequest.area.radiusMeters });
  const publicSourceDiscovery = await discoverPublicSources(place, fetchJson);
  const automaticEvidence = buildAutomaticEvidenceInputs(publicSourceDiscovery.sources);
  const cameraSignals = await analyzeCameras([...automaticEvidence, ...assessmentRequest.cameras], place);
  const forecast = await fetchForecast(place);
  const result = assessWetness({ request: { ...assessmentRequest, cameraSignals, terrainGrid }, place, forecast });
  const usedSourceIds = new Set(automaticEvidence.map((source) => source.id.replace(/^auto-/, '')));

  return {
    ...result,
    publicSources: publicSourceDiscovery.sources.map((source) => ({
      ...source,
      usedAsEvidence: usedSourceIds.has(source.id),
    })),
    caveats: [...result.caveats, ...publicSourceDiscovery.caveats],
  };
}

function deriveAreaFromAddress(location) {
  const normalized = location.toLowerCase();

  if (normalized.includes('park') || normalized.includes('field') || normalized.includes('garden')) {
    return { radiusMeters: 24, surface: 'grass', shade: 'partial', drainage: 'average', slope: 'gentle', traffic: 'foot' };
  }

  if (normalized.includes('parking') || normalized.includes('lot') || normalized.includes('driveway')) {
    return { radiusMeters: 18, surface: 'asphalt', shade: 'full_sun', drainage: 'average', slope: 'flat', traffic: 'vehicle' };
  }

  if (normalized.includes('trail') || normalized.includes('path')) {
    return { radiusMeters: 16, surface: 'dirt', shade: 'mostly_shaded', drainage: 'average', slope: 'gentle', traffic: 'foot' };
  }

  return { radiusMeters: 12, surface: 'concrete', shade: 'partial', drainage: 'average', slope: 'gentle', traffic: 'foot' };
}

function normalizeCameraInputs(cameras = []) {
  return normalizeCameraSignals(cameras).map((camera) => ({
    ...camera,
    userConfidence: camera.confidence,
    imageUrl: camera.imageUrl || camera.url,
    roiPolygon: camera.roiPolygon || null,
  }));
}

async function analyzeCameras(cameras, targetPlace) {
  const analyses = [];

  for (const camera of cameras) {
    analyses.push(await analyzeCamera(camera, targetPlace));
  }

  return analyses;
}

async function analyzeCamera(camera, targetPlace) {
  const caveats = [...(camera.caveats || [])];
  const cameraWithElevation = { ...camera };

  if (cameraWithElevation.id && cameraRoiOverrides.has(cameraWithElevation.id)) {
    cameraWithElevation.roiPolygon = cameraRoiOverrides.get(cameraWithElevation.id);
  }

  if (!Number.isFinite(cameraWithElevation.elevationMeters) && Number.isFinite(cameraWithElevation.latitude) && Number.isFinite(cameraWithElevation.longitude)) {
    cameraWithElevation.elevationMeters = await fetchElevationMeters(cameraWithElevation, fetchJson);
  }

  const distanceToTargetMeters = targetPlace ? haversineDistanceMeters(cameraWithElevation, targetPlace) : null;
  const deltaMeters = targetPlace ? elevationDeltaMeters(cameraWithElevation.elevationMeters, targetPlace.elevationMeters) : null;
  const freshMinutes = observationFreshnessMinutes(cameraWithElevation.observedAt);
  let vision = null;

  if (cameraWithElevation.imageUrl) {
    const imageMetadata = await fetchCameraImageMetadata(cameraWithElevation.imageUrl);

    if (!imageMetadata.ok) {
      caveats.push(imageMetadata.reason);
    } else {
      try {
        vision = await analyzeImageWetness({ imageUrl: imageMetadata.normalizedUrl, roiPolygon: cameraWithElevation.roiPolygon, fetchJson });
        caveats.push(...vision.caveats);
      } catch (error) {
        caveats.push(`Vision inference failed: ${error.message}`);
      }
    }
  }

  const visionCondition = conditionFromVision(vision?.wetnessScore);
  const condition = visionCondition !== 'unknown' ? visionCondition : cameraWithElevation.condition;
  const visualConfidence = vision?.visualConfidence ?? 0;
  const confidence = vision?.wetnessScore == null
    ? cameraWithElevation.confidence
    : Math.max(cameraWithElevation.confidence * 0.5, visualConfidence);

  return {
    ...cameraWithElevation,
    condition,
    confidence,
    source: vision?.available ? 'vision' : cameraWithElevation.source,
    wetnessScore: vision?.wetnessScore ?? cameraWithElevation.wetnessScore,
    visualConfidence,
    detections: vision?.detections || cameraWithElevation.detections || [],
    masks: vision?.masks || cameraWithElevation.masks || [],
    crop: vision?.crop || cameraWithElevation.crop || null,
    classifier: vision?.classifier || cameraWithElevation.classifier || null,
    classifierInputMode: vision?.classifierInputMode || cameraWithElevation.classifierInputMode || '',
    model: vision?.model || null,
    distanceToTargetMeters,
    elevationDeltaMeters: deltaMeters,
    freshnessMinutes: freshMinutes,
    geometryWeight: cameraGeometryWeight({ distanceToTargetMeters, elevationDeltaMeters: deltaMeters }),
    freshnessWeight: freshnessWeight(freshMinutes),
    caveats,
  };
}

async function geocodeLocation(location) {
  let primaryGeocoderError = null;
  let nominatimMatch = null;

  try {
    nominatimMatch = await geocodeWithNominatim(location);
  } catch (error) {
    primaryGeocoderError = error;
  }

  if (nominatimMatch) {
    return nominatimMatch;
  }

  const openMeteoMatch = await geocodeWithOpenMeteo(location);

  if (openMeteoMatch) {
    return openMeteoMatch;
  }

  if (primaryGeocoderError) {
    throw new HttpError(502, `Address geocoding failed and fallback found no match: ${primaryGeocoderError.message}`);
  }

  throw new HttpError(400, `No location match found for "${location}".`);
}

async function geocodeWithNominatim(location) {
  const url = withSearchParams(nominatimUrl, {
    q: location,
    format: 'jsonv2',
    addressdetails: '1',
    limit: '1',
  });

  const data = await fetchJson(url, 'Address geocoding', {
    headers: {
      'user-agent': 'is-the-ground-wet/0.1 local-development',
    },
  });
  const match = Array.isArray(data) ? data[0] : null;

  if (!match) {
    return null;
  }

  const latitude = Number(match.lat);
  const longitude = Number(match.lon);

  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return null;
  }

  const address = match.address || {};
  const displayParts = String(match.display_name || '').split(',').map((part) => part.trim());

  return {
    query: location,
    id: match.place_id,
    provider: 'OpenStreetMap Nominatim',
    name: match.name || displayParts[0] || location,
    displayName: match.display_name || location,
    admin1: address.state || address.region || '',
    admin2: address.county || address.city || address.town || '',
    country: address.country || '',
    countryCode: address.country_code ? String(address.country_code).toUpperCase() : '',
    latitude,
    longitude,
    elevationMeters: null,
    timezone: 'auto',
  };
}

async function geocodeWithOpenMeteo(location) {
  const url = withSearchParams(geocodingUrl, {
    name: location,
    count: '1',
    language: 'en',
    format: 'json',
  });
  const data = await fetchJson(url, 'Geocoding');
  const match = data.results?.[0];

  if (!match) {
    return null;
  }

  return {
    query: location,
    id: match.id,
    provider: 'Open-Meteo Geocoding',
    name: match.name,
    displayName: [match.name, match.admin1, match.country].filter(Boolean).join(', '),
    admin1: match.admin1 || '',
    admin2: match.admin2 || '',
    country: match.country || '',
    countryCode: match.country_code || '',
    latitude: match.latitude,
    longitude: match.longitude,
    elevationMeters: match.elevation,
    timezone: match.timezone || 'auto',
  };
}

async function fetchForecast(place) {
  try {
    return await fetchForecastWithVariables(place, primaryHourlyVariables);
  } catch (error) {
    if (error.status !== 502) {
      throw error;
    }

    const fallback = await fetchForecastWithVariables(place, fallbackHourlyVariables);
    fallback.fallbackReason = error.message;
    return fallback;
  }
}

async function fetchForecastWithVariables(place, hourlyVariables) {
  const url = withSearchParams(forecastUrl, {
    latitude: String(place.latitude),
    longitude: String(place.longitude),
    timezone: 'auto',
    past_days: '3',
    forecast_days: '1',
    current: currentVariables.join(','),
    hourly: hourlyVariables.join(','),
  });

  return fetchJson(url, 'Weather');
}

async function fetchJson(url, serviceName, options = {}) {
  let response;

  try {
    response = await fetch(url, {
      ...options,
      headers: { accept: 'application/json', ...(options.headers || {}) },
    });
  } catch (error) {
    throw new HttpError(502, `${serviceName} request failed: ${error.message}`);
  }

  let data;

  try {
    data = await response.json();
  } catch {
    throw new HttpError(502, `${serviceName} returned a non-JSON response.`);
  }

  if (!response.ok || data.error) {
    throw new HttpError(502, `${serviceName} error: ${data.reason || response.statusText}`);
  }

  return data;
}

async function fetchText(url, serviceName, options = {}) {
  let response;

  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new HttpError(502, `${serviceName} request failed: ${error.message}`);
  }

  if (!response.ok) {
    throw new HttpError(502, `${serviceName} error: ${response.statusText}`);
  }

  try {
    return await response.text();
  } catch (error) {
    throw new HttpError(502, `${serviceName} returned unreadable text: ${error.message}`);
  }
}

function applySecurityHeaders(_request, response, next) {
  response.setHeader('x-content-type-options', 'nosniff');
  response.setHeader('referrer-policy', 'strict-origin-when-cross-origin');
  next();
}

function applyCors(request, response, next) {
  const allowedOrigin = process.env.CORS_ORIGIN;

  if (allowedOrigin && request.headers.origin === allowedOrigin) {
    response.setHeader('access-control-allow-origin', allowedOrigin);
    response.setHeader('vary', 'origin');
  }

  response.setHeader('access-control-allow-methods', 'GET,POST,OPTIONS');
  response.setHeader('access-control-allow-headers', 'content-type');

  if (request.method === 'OPTIONS') {
    response.sendStatus(204);
    return;
  }

  next();
}

function applyRateLimit(request, response, next) {
  if (!request.path.startsWith('/api/')) {
    next();
    return;
  }

  const now = Date.now();
  const key = request.ip || request.socket.remoteAddress || 'unknown';
  const bucket = rateLimitBuckets.get(key) || { resetAt: now + rateLimitWindowMs, count: 0 };

  if (bucket.resetAt <= now) {
    bucket.resetAt = now + rateLimitWindowMs;
    bucket.count = 0;
  }

  bucket.count += 1;
  rateLimitBuckets.set(key, bucket);

  if (bucket.count > rateLimitMax) {
    response.status(429).json({ error: 'Too many API requests. Please wait and try again.', status: 429 });
    return;
  }

  next();
}

function withSearchParams(baseUrl, params) {
  const url = new URL(baseUrl);

  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });

  return url;
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}
