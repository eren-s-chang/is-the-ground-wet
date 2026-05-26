const overpassUrl = 'https://overpass-api.de/api/interpreter';
const wikimediaUrl = 'https://commons.wikimedia.org/w/api.php';
const defaultAllowlistedFeeds = [
  {
    id: 'nyc-dot-times-square-earthcam',
    region: 'New York City',
    label: 'Times Square public webcam landing page',
    latitude: 40.758,
    longitude: -73.9855,
    url: 'https://www.earthcam.com/usa/newyork/timessquare/',
    directImage: false,
    notes: 'Allowlisted public webcam page near Times Square; no direct still image endpoint is assumed.',
  },
];

export async function discoverPublicSources(place, fetchJson, options = {}) {
  if (!Number.isFinite(place?.latitude) || !Number.isFinite(place?.longitude)) {
    return { sources: [], caveats: ['Public source discovery skipped because the target location was unavailable.'] };
  }

  const providers = [
    createProvider('OpenStreetMap', () => discoverOpenStreetMapSources(place, fetchJson)),
    createProvider('Wikimedia Commons', () => discoverWikimediaSources(place, fetchJson)),
    createProvider('Allowlisted municipal/weather feeds', () => discoverAllowlistedFeedSources(place, options.allowlistedFeeds)),
  ];
  const settled = await Promise.allSettled(providers.map((provider) => provider.run()));
  const sources = [];
  const caveats = [];

  for (const result of settled) {
    if (result.status === 'fulfilled') {
      sources.push(...result.value.sources);
      caveats.push(...result.value.caveats);
    } else {
      caveats.push(result.reason?.message || 'A public source search failed.');
    }
  }

  return {
    sources: dedupeSources(sources).slice(0, 12).map(addEvidenceEligibility),
    caveats,
  };
}

export function buildAutomaticEvidenceInputs(sources, now = new Date()) {
  return sources
    .filter((source) => source.usableAsEvidence)
    .slice(0, 3)
    .map((source, index) => ({
      id: `auto-${source.id}`,
      label: `Auto source ${index + 1}: ${source.label}`,
      imageUrl: source.url,
      condition: 'unknown',
      confidence: 0.45,
      latitude: source.latitude,
      longitude: source.longitude,
      elevationMeters: null,
      observedAt: source.kind === 'public_geotagged_image' ? '' : now.toISOString(),
      source: 'public_discovery',
      caveats: [source.evidenceWarning],
    }));
}

async function discoverOpenStreetMapSources(place, fetchJson) {
  const query = `[out:json][timeout:8];(
    node(around:1200,${place.latitude},${place.longitude})["man_made"="surveillance"];
    node(around:1200,${place.latitude},${place.longitude})["surveillance:type"="camera"];
    node(around:1200,${place.latitude},${place.longitude})["camera:type"];
    node(around:1200,${place.latitude},${place.longitude})["webcam"];
    node(around:1200,${place.latitude},${place.longitude})["tourism"="viewpoint"]["website"];
  );out center tags 20;`;
  const url = new URL(overpassUrl);
  url.searchParams.set('data', query);
  const data = await fetchJson(url, 'OpenStreetMap public source search');
  const elements = Array.isArray(data.elements) ? data.elements : [];

  return {
    sources: elements.slice(0, 20).map((element) => {
      const tags = element.tags || {};
      const latitude = Number(element.lat ?? element.center?.lat);
      const longitude = Number(element.lon ?? element.center?.lon);

      return {
        id: `osm-${element.type}-${element.id}`,
        provider: 'OpenStreetMap',
        kind: tags.webcam ? 'public_webcam_metadata' : 'public_camera_metadata',
        label: tags.name || tags.operator || tags.surveillance || 'Mapped public camera/source',
        latitude: Number.isFinite(latitude) ? latitude : null,
        longitude: Number.isFinite(longitude) ? longitude : null,
        distanceToTargetMeters: estimateDistanceMeters(place, { latitude, longitude }),
        url: tags.website || tags.url || tags['contact:website'] || '',
        notes: compact([
          tags.surveillance ? `surveillance=${tags.surveillance}` : '',
          tags['camera:type'] ? `camera:type=${tags['camera:type']}` : '',
          tags.webcam ? `webcam=${tags.webcam}` : '',
        ]).join(', '),
      };
    }),
    caveats: ['OpenStreetMap results are public metadata only and may not include accessible image frames.'],
  };
}

async function discoverWikimediaSources(place, fetchJson) {
  const url = new URL(wikimediaUrl);
  url.searchParams.set('action', 'query');
  url.searchParams.set('format', 'json');
  url.searchParams.set('origin', '*');
  url.searchParams.set('generator', 'geosearch');
  url.searchParams.set('ggscoord', `${place.latitude}|${place.longitude}`);
  url.searchParams.set('ggsradius', '1200');
  url.searchParams.set('ggslimit', '8');
  url.searchParams.set('prop', 'imageinfo|coordinates');
  url.searchParams.set('iiprop', 'url|mime');
  const data = await fetchJson(url, 'Wikimedia public image search');
  const pages = Object.values(data.query?.pages || {});

  return {
    sources: pages.map((page) => {
      const coordinates = page.coordinates?.[0] || {};
      const imageInfo = page.imageinfo?.[0] || {};

      return {
        id: `commons-${page.pageid}`,
        provider: 'Wikimedia Commons',
        kind: 'public_geotagged_image',
        label: String(page.title || 'Nearby public image').replace(/^File:/, ''),
        latitude: Number.isFinite(coordinates.lat) ? coordinates.lat : null,
        longitude: Number.isFinite(coordinates.lon) ? coordinates.lon : null,
        distanceToTargetMeters: estimateDistanceMeters(place, { latitude: coordinates.lat, longitude: coordinates.lon }),
        url: imageInfo.url || '',
        notes: imageInfo.mime ? `mime=${imageInfo.mime}` : 'public geotagged image',
      };
    }),
    caveats: ['Wikimedia results are nearby public images, not necessarily live cameras or current ground conditions.'],
  };
}

function discoverAllowlistedFeedSources(place, allowlistedFeeds = loadAllowlistedFeeds()) {
  const feeds = Array.isArray(allowlistedFeeds) ? allowlistedFeeds : [];
  const sources = feeds
    .map((feed) => normalizeAllowlistedFeed(feed, place))
    .filter((source) => source && source.distanceToTargetMeters != null && source.distanceToTargetMeters <= 25000)
    .sort((a, b) => a.distanceToTargetMeters - b.distanceToTargetMeters)
    .slice(0, 8);

  return {
    sources,
    caveats: [
      'Allowlisted municipal/weather feeds are explicitly configured public sources; they are not discovered by scanning networks.',
    ],
  };
}

function normalizeAllowlistedFeed(feed, place) {
  const latitude = Number(feed?.latitude);
  const longitude = Number(feed?.longitude);
  const url = String(feed?.imageUrl || feed?.url || '');

  if (!feed?.id || !Number.isFinite(latitude) || !Number.isFinite(longitude) || !url) {
    return null;
  }

  return {
    id: `allowlist-${feed.id}`,
    provider: 'Configured Public Feed',
    kind: feed.directImage || feed.imageUrl ? 'allowlisted_public_image' : 'allowlisted_public_camera_page',
    label: String(feed.label || feed.id),
    latitude,
    longitude,
    distanceToTargetMeters: estimateDistanceMeters(place, { latitude, longitude }),
    url,
    notes: compact([feed.region ? `region=${feed.region}` : '', feed.notes || 'explicit public feed allowlist']).join(', '),
  };
}

function loadAllowlistedFeeds() {
  if (!process.env.PUBLIC_CAMERA_ALLOWLIST_JSON) {
    return defaultAllowlistedFeeds;
  }

  try {
    const feeds = JSON.parse(process.env.PUBLIC_CAMERA_ALLOWLIST_JSON);
    return Array.isArray(feeds) ? feeds : defaultAllowlistedFeeds;
  } catch {
    return defaultAllowlistedFeeds;
  }
}

function createProvider(name, run) {
  return { name, run };
}

function dedupeSources(sources) {
  const seen = new Set();
  return sources.filter((source) => {
    const key = source.url || `${source.provider}:${source.label}:${source.latitude}:${source.longitude}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function addEvidenceEligibility(source) {
  const url = String(source.url || '');
  const directImage = /\.(avif|gif|jpe?g|png|webp)(\?|#|$)/i.test(url);
  const usableAsEvidence = Boolean(url) && (
    source.kind === 'public_geotagged_image' ||
    source.kind === 'allowlisted_public_image' ||
    directImage
  );

  return {
    ...source,
    usableAsEvidence,
    usedAsEvidence: false,
    evidenceWarning: usableAsEvidence
      ? 'Automatically discovered public source; may be stale, indirect, or not representative of the target ground.'
      : 'Metadata only; no direct public image URL was available for automatic visual evidence.',
  };
}

function estimateDistanceMeters(a, b) {
  if (!Number.isFinite(b?.latitude) || !Number.isFinite(b?.longitude)) {
    return null;
  }

  const radius = 6371000;
  const dLat = toRadians(b.latitude - a.latitude);
  const dLon = toRadians(b.longitude - a.longitude);
  const lat1 = toRadians(a.latitude);
  const lat2 = toRadians(b.latitude);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return Math.round(radius * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h)));
}

function toRadians(value) {
  return (value * Math.PI) / 180;
}

function compact(values) {
  return values.filter(Boolean);
}
