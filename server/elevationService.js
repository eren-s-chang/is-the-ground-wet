const elevationUrl = 'https://api.open-meteo.com/v1/elevation';

export async function fetchElevationMeters({ latitude, longitude }, fetchJson) {
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return null;
  }

  const url = new URL(elevationUrl);
  url.searchParams.set('latitude', String(latitude));
  url.searchParams.set('longitude', String(longitude));

  try {
    const data = await fetchJson(url, 'Elevation');
    const elevation = Array.isArray(data.elevation) ? Number(data.elevation[0]) : Number(data.elevation);
    return Number.isFinite(elevation) ? elevation : null;
  } catch {
    return null;
  }
}

export async function fetchElevationGrid({ latitude, longitude }, fetchJson, options = {}) {
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return null;
  }

  const size = normalizeOddSize(options.size || 5);
  const radiusMeters = clamp(20, 300, Number(options.radiusMeters) || 90);
  const midpoint = Math.floor(size / 2);
  const points = [];
  const cosLatitude = Math.max(0.01, Math.cos((latitude * Math.PI) / 180));

  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      const x = (column - midpoint) / midpoint;
      const y = (row - midpoint) / midpoint;
      const eastMeters = x * radiusMeters;
      const northMeters = -y * radiusMeters;
      points.push({
        x,
        y,
        latitude: latitude + northMeters / 111_320,
        longitude: longitude + eastMeters / (111_320 * cosLatitude),
      });
    }
  }

  const url = new URL(elevationUrl);
  url.searchParams.set('latitude', points.map((point) => point.latitude.toFixed(6)).join(','));
  url.searchParams.set('longitude', points.map((point) => point.longitude.toFixed(6)).join(','));

  try {
    const data = await fetchJson(url, 'Elevation grid');
    const elevations = Array.isArray(data.elevation) ? data.elevation : [data.elevation];
    const cells = points.map((point, index) => ({
      ...point,
      x: round(point.x, 2),
      y: round(point.y, 2),
      latitude: round(point.latitude, 6),
      longitude: round(point.longitude, 6),
      elevationMeters: Number.isFinite(Number(elevations[index])) ? Number(elevations[index]) : null,
    }));

    if (cells.filter((cell) => Number.isFinite(cell.elevationMeters)).length < 9) {
      return null;
    }

    return {
      model: 'open-meteo-elevation-grid-v1',
      size,
      radiusMeters,
      cells,
    };
  } catch {
    return null;
  }
}

function normalizeOddSize(size) {
  const normalized = Math.round(Number(size));
  if (!Number.isFinite(normalized)) {
    return 5;
  }

  return clamp(3, 7, normalized % 2 === 0 ? normalized + 1 : normalized);
}

function clamp(min, max, value) {
  return Math.min(max, Math.max(min, value));
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
