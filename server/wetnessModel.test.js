import test from 'node:test';
import assert from 'node:assert/strict';
import { assessWetness, clamp01, normalizeArea, normalizeCameraSignals } from './wetnessModel.js';

test('clamp01 bounds values to a probability range', () => {
  assert.equal(clamp01(-1), 0);
  assert.equal(clamp01(0.42), 0.42);
  assert.equal(clamp01(2), 1);
  assert.equal(clamp01(Number.NaN), 0);
});

test('normalizers keep site and camera inputs bounded', () => {
  assert.deepEqual(normalizeArea({ radiusMeters: 999, surface: 'clay', shade: 'deep_shade' }), {
    radiusMeters: 500,
    surface: 'clay',
    shade: 'deep_shade',
    drainage: 'average',
    slope: 'gentle',
    traffic: 'none',
  });

  const [camera] = normalizeCameraSignals([{ label: 'Gate', condition: 'wet', confidence: 2 }]);
  assert.equal(camera.id, 'camera-1');
  assert.equal(camera.label, 'Gate');
  assert.equal(camera.condition, 'wet');
  assert.equal(camera.confidence, 1);
  assert.equal(camera.geometryWeight, 1);
});

test('camera geometry and freshness weights reduce visual evidence impact', () => {
  const place = { name: 'Test Place', latitude: 1, longitude: 2, elevationMeters: 50, timezone: 'UTC' };
  const baseRequest = { area: { surface: 'concrete', shade: 'partial', drainage: 'average' } };
  const forecast = makeForecast('neutral');
  const nearFresh = assessWetness({
    request: { ...baseRequest, cameraSignals: [{ condition: 'wet', confidence: 0.9, geometryWeight: 0.95, freshnessWeight: 0.9 }] },
    place,
    forecast,
    generatedAt: new Date('2026-05-24T12:00:00Z'),
  });
  const farStale = assessWetness({
    request: { ...baseRequest, cameraSignals: [{ condition: 'wet', confidence: 0.9, geometryWeight: 0.15, freshnessWeight: 0.2 }] },
    place,
    forecast,
    generatedAt: new Date('2026-05-24T12:00:00Z'),
  });

  assert.ok(nearFresh.probabilityWet > farStale.probabilityWet);
  assert.equal(nearFresh.geometry.cameraCount, 1);
  assert.equal(nearFresh.wetnessField.cells.length, 225);
  assert.ok(nearFresh.wetnessField.cells.every((cell) => cell.probability >= 0 && cell.probability <= 1));
});

test('recent wet weather scores higher than dry sunny weather', () => {
  const place = { name: 'Test Place', latitude: 1, longitude: 2, timezone: 'UTC' };
  const request = { area: { surface: 'grass', shade: 'partial', drainage: 'average' } };
  const dry = assessWetness({ request, place, forecast: makeForecast('dry'), generatedAt: new Date('2026-05-24T12:00:00Z') });
  const wet = assessWetness({ request, place, forecast: makeForecast('wet'), generatedAt: new Date('2026-05-24T12:00:00Z') });

  assert.ok(dry.probabilityWet >= 0 && dry.probabilityWet <= 1);
  assert.ok(wet.probabilityWet >= 0 && wet.probabilityWet <= 1);
  assert.ok(wet.probabilityWet > dry.probabilityWet + 0.25);
});

test('terrain grid drives hotspot surface when available', () => {
  const place = { name: 'Test Place', latitude: 1, longitude: 2, timezone: 'UTC' };
  const terrainGrid = makeTerrainGrid();
  const result = assessWetness({
    request: { area: { surface: 'grass', shade: 'partial', drainage: 'average' }, terrainGrid },
    place,
    forecast: makeForecast('neutral'),
    generatedAt: new Date('2026-05-24T12:00:00Z'),
  });
  const lowSpot = result.wetnessField.cells.find((cell) => cell.x === 1 && cell.y === 0);
  const highSpot = result.wetnessField.cells.find((cell) => cell.x === -1 && cell.y === 0);

  assert.equal(result.wetnessField.model, 'terrain-hotspot-surface-v1');
  assert.equal(result.wetnessField.cells.length, 225);
  assert.ok(lowSpot.probability > highSpot.probability);
});

test('camera evidence transparently changes the probability', () => {
  const place = { name: 'Test Place', latitude: 1, longitude: 2, timezone: 'UTC' };
  const baseRequest = { area: { surface: 'concrete', shade: 'partial', drainage: 'average' } };
  const forecast = makeForecast('neutral');
  const base = assessWetness({ request: baseRequest, place, forecast, generatedAt: new Date('2026-05-24T12:00:00Z') });
  const visualWet = assessWetness({
    request: { ...baseRequest, cameraSignals: [{ condition: 'standing_water', confidence: 0.9 }] },
    place,
    forecast,
    generatedAt: new Date('2026-05-24T12:00:00Z'),
  });
  const visualDry = assessWetness({
    request: { ...baseRequest, cameraSignals: [{ condition: 'dry', confidence: 0.9 }] },
    place,
    forecast,
    generatedAt: new Date('2026-05-24T12:00:00Z'),
  });

  assert.ok(visualWet.probabilityWet > base.probabilityWet);
  assert.ok(visualDry.probabilityWet < base.probabilityWet);
});

function makeForecast(kind) {
  const currentTime = '2026-05-24T12:00';
  const hourly = {
    time: [],
    precipitation: [],
    rain: [],
    showers: [],
    snowfall: [],
    temperature_2m: [],
    relative_humidity_2m: [],
    cloud_cover: [],
    wind_speed_10m: [],
    shortwave_radiation: [],
    uv_index: [],
    evapotranspiration: [],
    et0_fao_evapotranspiration: [],
    vapour_pressure_deficit: [],
    soil_moisture_0_to_1cm: [],
  };

  for (let hour = -72; hour <= 0; hour += 1) {
    const time = addHours(currentTime, hour);
    const daylight = hour % 24 >= -6 && hour % 24 <= 6;
    const wetRecent = kind === 'wet' && hour >= -3;
    const neutralRecent = kind === 'neutral' && hour >= -18 && hour <= -16;
    const precipitation = wetRecent ? 2.2 : neutralRecent ? 0.3 : 0;
    const dry = kind === 'dry';

    hourly.time.push(time);
    hourly.precipitation.push(precipitation);
    hourly.rain.push(precipitation);
    hourly.showers.push(0);
    hourly.snowfall.push(0);
    hourly.temperature_2m.push(dry ? 28 : wetRecent ? 11 : 17);
    hourly.relative_humidity_2m.push(dry ? 36 : wetRecent ? 94 : 66);
    hourly.cloud_cover.push(dry ? 15 : wetRecent ? 96 : 55);
    hourly.wind_speed_10m.push(dry ? 18 : wetRecent ? 4 : 8);
    hourly.shortwave_radiation.push(daylight ? (dry ? 620 : wetRecent ? 30 : 220) : 0);
    hourly.uv_index.push(daylight ? (dry ? 7 : wetRecent ? 0.2 : 2) : 0);
    hourly.evapotranspiration.push(dry ? 0.18 : wetRecent ? 0.01 : 0.04);
    hourly.et0_fao_evapotranspiration.push(dry ? 0.2 : wetRecent ? 0.01 : 0.05);
    hourly.vapour_pressure_deficit.push(dry ? 1.8 : wetRecent ? 0.18 : 0.7);
    hourly.soil_moisture_0_to_1cm.push(dry ? 0.08 : wetRecent ? 0.44 : 0.22);
  }

  const currentIndex = hourly.time.length - 1;

  return {
    current: {
      time: currentTime,
      temperature_2m: hourly.temperature_2m[currentIndex],
      relative_humidity_2m: hourly.relative_humidity_2m[currentIndex],
      precipitation: hourly.precipitation[currentIndex],
      rain: hourly.rain[currentIndex],
      showers: 0,
      snowfall: 0,
      cloud_cover: hourly.cloud_cover[currentIndex],
      wind_speed_10m: hourly.wind_speed_10m[currentIndex],
      wind_gusts_10m: hourly.wind_speed_10m[currentIndex] + 6,
      weather_code: hourly.precipitation[currentIndex] > 0 ? 61 : 1,
    },
    current_units: {
      temperature_2m: 'C',
      precipitation: 'mm',
      wind_speed_10m: 'km/h',
    },
    hourly,
  };
}

function makeTerrainGrid() {
  const cells = [];

  for (let row = 0; row < 5; row += 1) {
    for (let column = 0; column < 5; column += 1) {
      const x = (column - 2) / 2;
      const y = (row - 2) / 2;
      cells.push({ x, y, elevationMeters: 10 - (x + 1) * 5 + y });
    }
  }

  return {
    model: 'test-terrain',
    size: 5,
    radiusMeters: 50,
    cells,
  };
}

function addHours(start, hours) {
  const date = new Date(`${start}:00Z`);
  date.setUTCHours(date.getUTCHours() + hours);
  return date.toISOString().slice(0, 16);
}
