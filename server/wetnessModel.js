const SURFACE_PROFILES = {
  asphalt: {
    label: 'Asphalt',
    retention: 0.72,
    heatGain: 1.2,
    description: 'dark hardscape sheds water but can keep low spots wet',
  },
  concrete: {
    label: 'Concrete',
    retention: 0.82,
    heatGain: 1.08,
    description: 'hardscape with moderate surface retention',
  },
  grass: {
    label: 'Grass',
    retention: 1.18,
    heatGain: 0.86,
    description: 'vegetation holds moisture near the surface',
  },
  dirt: {
    label: 'Bare dirt',
    retention: 1.36,
    heatGain: 0.82,
    description: 'exposed soil can remain damp after rain',
  },
  clay: {
    label: 'Clay soil',
    retention: 1.72,
    heatGain: 0.72,
    description: 'clay drains slowly and holds water',
  },
  gravel: {
    label: 'Gravel',
    retention: 0.58,
    heatGain: 1.0,
    description: 'coarse aggregate drains quickly',
  },
  mulch: {
    label: 'Mulch',
    retention: 1.52,
    heatGain: 0.7,
    description: 'organic cover absorbs and shades water',
  },
  wood: {
    label: 'Wood decking',
    retention: 1.08,
    heatGain: 0.9,
    description: 'porous boards can stay damp in shade',
  },
};

const SHADE_PROFILES = {
  full_sun: { label: 'Full sun', drying: 1.22, logit: -0.26 },
  partial: { label: 'Partial shade', drying: 1.0, logit: 0 },
  mostly_shaded: { label: 'Mostly shaded', drying: 0.74, logit: 0.32 },
  deep_shade: { label: 'Deep shade', drying: 0.54, logit: 0.56 },
};

const DRAINAGE_PROFILES = {
  poor: { label: 'Poor drainage', logit: 0.58 },
  average: { label: 'Average drainage', logit: 0 },
  good: { label: 'Good drainage', logit: -0.36 },
};

const SLOPE_PROFILES = {
  flat: { label: 'Flat', logit: 0.18 },
  gentle: { label: 'Gentle slope', logit: -0.06 },
  steep: { label: 'Steep slope', logit: -0.3 },
};

const TRAFFIC_PROFILES = {
  none: { label: 'No traffic', logit: 0 },
  foot: { label: 'Foot traffic', logit: 0.07 },
  vehicle: { label: 'Vehicle traffic', logit: 0.16 },
};

const CAMERA_CONDITIONS = {
  standing_water: { label: 'Standing water visible', logit: 1.9 },
  wet: { label: 'Wet surface visible', logit: 1.15 },
  mixed: { label: 'Mixed wet/dry surface', logit: 0.35 },
  mostly_dry: { label: 'Mostly dry surface', logit: -0.68 },
  dry: { label: 'Dry surface visible', logit: -1.08 },
  unknown: { label: 'Unknown visual condition', logit: 0 },
};

export function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.min(1, Math.max(0, value));
}

export function assessWetness({ request, place, forecast, generatedAt = new Date() }) {
  const area = normalizeArea(request?.area);
  const cameraSignals = normalizeCameraSignals(request?.cameraSignals);
  const terrainGrid = normalizeTerrainGrid(request?.terrainGrid);
  const current = normalizeCurrent(forecast?.current, forecast?.current_units);
  const hourlyRecords = normalizeHourly(forecast?.hourly, current.time, generatedAt);
  const recentRecords = hourlyRecords.filter((record) => record.hoursAgo >= 0 && record.hoursAgo <= 72);
  const recent12 = recentRecords.filter((record) => record.hoursAgo <= 12);
  const recent24 = recentRecords.filter((record) => record.hoursAgo <= 24);
  const recent6 = recentRecords.filter((record) => record.hoursAgo <= 6);
  const surface = SURFACE_PROFILES[area.surface] || SURFACE_PROFILES.grass;
  const shade = SHADE_PROFILES[area.shade] || SHADE_PROFILES.partial;
  const drainage = DRAINAGE_PROFILES[area.drainage] || DRAINAGE_PROFILES.average;
  const slope = SLOPE_PROFILES[area.slope] || SLOPE_PROFILES.gentle;
  const traffic = TRAFFIC_PROFILES[area.traffic] || TRAFFIC_PROFILES.none;
  const weatherNow = current.values;

  const totals = {
    last1h: sumPrecip(recentRecords, 1),
    last3h: sumPrecip(recentRecords, 3),
    last6h: sumPrecip(recentRecords, 6),
    last12h: sumPrecip(recentRecords, 12),
    last24h: sumPrecip(recentRecords, 24),
    last48h: sumPrecip(recentRecords, 48),
    last72h: sumPrecip(recentRecords, 72),
  };

  const dryingMultiplier = getDryingMultiplier({ weatherNow, recent6, recent12, recent24, shade, surface });
  const halfLifeHours = clamp(2.8, 22, 8 * surface.retention / dryingMultiplier);
  const weightedMoistureMm = recentRecords.reduce((total, record) => {
    const ageWeight = Math.pow(0.5, record.hoursAgo / halfLifeHours);
    return total + record.precipitationMm * ageWeight;
  }, 0);

  const drivers = [];
  let logit = -0.82;

  const weightedPrecipImpact = clamp(0, 2.4, Math.log1p(weightedMoistureMm * surface.retention) * 0.78);
  logit += pushDriver(drivers, {
    label: 'Recency-weighted precipitation',
    value: `${formatNumber(weightedMoistureMm, 1)} mm effective`,
    impact: weightedPrecipImpact,
    description: 'recent rain and snow count more than older precipitation',
  });

  if (totals.last1h > 0.05) {
    logit += pushDriver(drivers, {
      label: 'Precipitation in the last hour',
      value: `${formatNumber(totals.last1h, 1)} mm`,
      impact: clamp(0.2, 1.4, 0.55 + totals.last1h * 0.24),
      description: 'active or very recent precipitation strongly implies wet ground',
    });
  }

  if ((weatherNow.precipitationMm ?? 0) > 0.02) {
    logit += pushDriver(drivers, {
      label: 'Precipitation right now',
      value: `${formatNumber(weatherNow.precipitationMm, 1)} mm`,
      impact: 1.18,
      description: 'current precipitation overrides many drying signals',
    });
  }

  const humidityImpact = humidityLogit(weatherNow.relativeHumidityPercent, average(recent6, 'relativeHumidityPercent'));
  logit += pushDriver(drivers, {
    label: 'Humidity',
    value: formatPercent(weatherNow.relativeHumidityPercent ?? average(recent6, 'relativeHumidityPercent')),
    impact: humidityImpact,
    description: 'humid air slows evaporation',
  });

  const temperatureImpact = temperatureLogit(weatherNow.temperatureC ?? average(recent6, 'temperatureC'));
  logit += pushDriver(drivers, {
    label: 'Temperature',
    value: formatDegrees(weatherNow.temperatureC),
    impact: temperatureImpact,
    description: 'warm surfaces dry faster; cold surfaces stay wet longer',
  });

  const windImpact = windLogit(weatherNow.windSpeedKph ?? average(recent6, 'windSpeedKph'));
  logit += pushDriver(drivers, {
    label: 'Wind',
    value: formatSpeed(weatherNow.windSpeedKph),
    impact: windImpact,
    description: 'wind increases evaporation from exposed ground',
  });

  const solarImpact = solarLogit({
    shortwaveMean: average(recent6, 'shortwaveRadiationWm2'),
    uvMean: average(recent6, 'uvIndex'),
    cloudCover: weatherNow.cloudCoverPercent ?? average(recent6, 'cloudCoverPercent'),
    shade,
    surface,
  });
  logit += pushDriver(drivers, {
    label: 'Sun, UV, and cloud cover',
    value: `${formatNumber(average(recent6, 'shortwaveRadiationWm2'), 0)} W/m2 avg`,
    impact: solarImpact,
    description: 'sunlight, UV, and low cloud cover accelerate drying',
  });

  const evapotranspiration24h = sumField(recent24, 'evapotranspirationMm');
  const eto24h = sumField(recent24, 'et0EvapotranspirationMm');
  const evapImpact = evapotranspirationLogit(Math.max(evapotranspiration24h, eto24h));
  logit += pushDriver(drivers, {
    label: 'Evapotranspiration',
    value: `${formatNumber(Math.max(evapotranspiration24h, eto24h), 2)} mm/24h`,
    impact: evapImpact,
    description: 'reference evapotranspiration estimates atmospheric drying demand',
  });

  const vpdImpact = vaporPressureDeficitLogit(average(recent6, 'vapourPressureDeficitKpa'));
  logit += pushDriver(drivers, {
    label: 'Vapor pressure deficit',
    value: `${formatNumber(average(recent6, 'vapourPressureDeficitKpa'), 2)} kPa`,
    impact: vpdImpact,
    description: 'higher deficit means the air can pull more water off surfaces',
  });

  const soilImpact = soilMoistureLogit(average(recent12, 'soilMoistureM3m3'));
  logit += pushDriver(drivers, {
    label: 'Near-surface soil moisture',
    value: `${formatNumber(average(recent12, 'soilMoistureM3m3'), 2)} m3/m3`,
    impact: soilImpact,
    description: 'wet near-surface soil can keep unpaved areas damp',
  });

  const siteImpact = siteLogit({ area, surface, drainage, slope, traffic });
  logit += pushDriver(drivers, {
    label: 'Surface and site assumptions',
    value: `${surface.label}, ${shade.label.toLowerCase()}, ${drainage.label.toLowerCase()}`,
    impact: siteImpact + shade.logit,
    description: surface.description,
  });

  const cameraImpacts = cameraSignals.map((signal) => {
    const profile = CAMERA_CONDITIONS[signal.condition] || CAMERA_CONDITIONS.unknown;
    const evidenceWeight = numberOrDefault(signal.geometryWeight, 1) * numberOrDefault(signal.freshnessWeight, 1);
    const visualWeight = signal.source === 'vision' ? Math.max(0.25, numberOrDefault(signal.visualConfidence, 0.5)) : 1;
    const impact = profile.logit * signal.confidence * evidenceWeight * visualWeight;
    logit += impact;
    return {
      ...signal,
      conditionLabel: profile.label,
      impact: round(impact, 3),
    };
  });

  if (cameraImpacts.length > 0) {
    pushDriver(drivers, {
      label: 'Camera or visual evidence',
      value: `${cameraImpacts.length} observation${cameraImpacts.length === 1 ? '' : 's'}`,
      impact: cameraImpacts.reduce((total, signal) => total + signal.impact, 0),
      description: 'visual evidence is weighted by confidence, distance, elevation delta, and observation age',
    });
  }

  const probabilityWet = clamp01(logistic(logit));
  const classification = classify(probabilityWet);
  const caveats = buildCaveats({ recentRecords, cameraSignals, forecast });
  const sortedDrivers = drivers
    .filter((driver) => Math.abs(driver.impact) >= 0.01)
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

  return {
    place,
    generatedAt: generatedAt.toISOString(),
    probabilityWet,
    probabilityPercent: Math.round(probabilityWet * 100),
    classification,
    explanation: buildExplanation({ probabilityWet, classification, drivers: sortedDrivers, totals }),
    current,
    precipitation: {
      totals,
      effectiveMoistureMm: round(weightedMoistureMm, 2),
      dryingHalfLifeHours: round(halfLifeHours, 1),
      series: recentRecords
        .slice()
        .sort((a, b) => a.hoursAgo - b.hoursAgo)
        .map((record) => ({
          time: record.time,
          hoursAgo: round(record.hoursAgo, 1),
          precipitationMm: round(record.precipitationMm, 2),
          temperatureC: nullableRound(record.temperatureC, 1),
          humidityPercent: nullableRound(record.relativeHumidityPercent, 0),
        })),
    },
    drivers: sortedDrivers,
    area: {
      ...area,
      surfaceLabel: surface.label,
      shadeLabel: shade.label,
      drainageLabel: drainage.label,
      slopeLabel: slope.label,
      trafficLabel: traffic.label,
      estimatedSquareMeters: round(Math.PI * area.radiusMeters * area.radiusMeters, 1),
    },
    cameraSignals: cameraImpacts,
    cameras: cameraImpacts,
    geometry: summarizeGeometry({ place, cameraSignals: cameraImpacts }),
    wetnessField: buildWetnessField({ probabilityWet, area, weatherNow, totals, terrainGrid }),
    confidence: round(estimateConfidence({ recentRecords, cameraSignals: cameraImpacts, forecast }), 2),
    caveats,
    model: {
      name: 'heuristic-weather-visual-v1',
      logit: round(logit, 3),
      dryingMultiplier: round(dryingMultiplier, 2),
    },
  };
}

export function normalizeArea(area = {}) {
  return {
    radiusMeters: clamp(1, 500, numberOrDefault(area.radiusMeters, 12)),
    surface: normalizeChoice(area.surface, Object.keys(SURFACE_PROFILES), 'grass'),
    shade: normalizeChoice(area.shade, Object.keys(SHADE_PROFILES), 'partial'),
    drainage: normalizeChoice(area.drainage, Object.keys(DRAINAGE_PROFILES), 'average'),
    slope: normalizeChoice(area.slope, Object.keys(SLOPE_PROFILES), 'gentle'),
    traffic: normalizeChoice(area.traffic, Object.keys(TRAFFIC_PROFILES), 'none'),
  };
}

export function normalizeCameraSignals(cameraSignals = []) {
  if (!Array.isArray(cameraSignals)) {
    return [];
  }

  return cameraSignals.slice(0, 8).map((signal, index) => {
    const latitude = maybeNumber(signal?.latitude);
    const longitude = maybeNumber(signal?.longitude);
    const elevationMeters = maybeNumber(signal?.elevationMeters);
    const distanceToTargetMeters = maybeNumber(signal?.distanceToTargetMeters);
    const elevationDeltaMeters = maybeNumber(signal?.elevationDeltaMeters);

    return {
      id: String(signal?.id || `camera-${index + 1}`),
      label: String(signal?.label || `Camera ${index + 1}`).slice(0, 80),
      url: signal?.url || signal?.imageUrl ? String(signal.url || signal.imageUrl).slice(0, 500) : '',
      imageUrl: signal?.imageUrl || signal?.url ? String(signal.imageUrl || signal.url).slice(0, 500) : '',
      condition: normalizeChoice(signal?.condition, Object.keys(CAMERA_CONDITIONS), 'unknown'),
      confidence: clamp01(numberOrDefault(signal?.confidence ?? signal?.userConfidence, 0.6)),
      latitude,
      longitude,
      elevationMeters,
      observedAt: signal?.observedAt ? String(signal.observedAt) : '',
      source: signal?.source ? String(signal.source) : 'manual',
      wetnessScore: maybeNumber(signal?.wetnessScore),
      visualConfidence: clamp01(numberOrDefault(signal?.visualConfidence, signal?.source === 'vision' ? 0 : 1)),
      distanceToTargetMeters,
      elevationDeltaMeters,
      freshnessMinutes: maybeNumber(signal?.freshnessMinutes),
      geometryWeight: clamp01(numberOrDefault(signal?.geometryWeight, 1)),
      freshnessWeight: clamp01(numberOrDefault(signal?.freshnessWeight, 1)),
      detections: Array.isArray(signal?.detections) ? signal.detections.slice(0, 20) : [],
      masks: Array.isArray(signal?.masks) ? signal.masks.slice(0, 12) : [],
      crop: signal?.crop && typeof signal.crop === 'object' ? signal.crop : null,
      classifier: signal?.classifier && typeof signal.classifier === 'object' ? signal.classifier : null,
      classifierInputMode: signal?.classifierInputMode ? String(signal.classifierInputMode) : '',
      roiPolygon: normalizePolygon(signal?.roiPolygon),
      caveats: Array.isArray(signal?.caveats) ? signal.caveats.map(String) : [],
    };
  });
}

function normalizePolygon(points) {
  if (!Array.isArray(points)) {
    return null;
  }

  const normalized = points
    .map((point) => ({
      x: clamp01(numberOrDefault(point?.x, Number.NaN)),
      y: clamp01(numberOrDefault(point?.y, Number.NaN)),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  return normalized.length >= 3 ? normalized.slice(0, 20) : null;
}

function normalizeTerrainGrid(terrainGrid) {
  if (!terrainGrid || !Array.isArray(terrainGrid.cells)) {
    return null;
  }

  const size = Math.round(numberOrDefault(terrainGrid.size, Math.sqrt(terrainGrid.cells.length)));
  const radiusMeters = clamp(20, 300, numberOrDefault(terrainGrid.radiusMeters, 90));
  const cells = terrainGrid.cells
    .map((cell) => ({
      x: maybeNumber(cell?.x),
      y: maybeNumber(cell?.y),
      elevationMeters: maybeNumber(cell?.elevationMeters),
    }))
    .filter((cell) => cell.x != null && cell.y != null && cell.elevationMeters != null);

  if (size < 3 || cells.length < 9) {
    return null;
  }

  return {
    model: String(terrainGrid.model || 'elevation-grid'),
    size,
    radiusMeters,
    cells,
  };
}

function normalizeCurrent(current = {}, units = {}) {
  return {
    time: current.time || null,
    units,
    values: {
      temperatureC: maybeNumber(current.temperature_2m),
      apparentTemperatureC: maybeNumber(current.apparent_temperature),
      relativeHumidityPercent: maybeNumber(current.relative_humidity_2m),
      precipitationMm: maybeNumber(current.precipitation),
      rainMm: maybeNumber(current.rain),
      showersMm: maybeNumber(current.showers),
      snowfallCm: maybeNumber(current.snowfall),
      cloudCoverPercent: maybeNumber(current.cloud_cover),
      pressureHPa: maybeNumber(current.pressure_msl),
      windSpeedKph: maybeNumber(current.wind_speed_10m),
      windDirectionDeg: maybeNumber(current.wind_direction_10m),
      windGustsKph: maybeNumber(current.wind_gusts_10m),
      weatherCode: maybeNumber(current.weather_code),
      isDay: maybeNumber(current.is_day),
    },
  };
}

function normalizeHourly(hourly = {}, currentTime, generatedAt) {
  const times = Array.isArray(hourly.time) ? hourly.time : [];
  const referenceTime = currentTime || generatedAt.toISOString();
  const referenceMs = parseTime(referenceTime) ?? generatedAt.getTime();

  return times
    .map((time, index) => {
      const hoursAgo = ((referenceMs - (parseTime(time) ?? referenceMs)) / 36e5);
      const precipitationMm = Math.max(
        0,
        maybeNumber(hourly.precipitation?.[index]) ?? 0,
        (maybeNumber(hourly.rain?.[index]) ?? 0) +
          (maybeNumber(hourly.showers?.[index]) ?? 0) +
          (maybeNumber(hourly.snowfall?.[index]) ?? 0),
      );

      return {
        time,
        hoursAgo,
        precipitationMm,
        rainMm: maybeNumber(hourly.rain?.[index]),
        showersMm: maybeNumber(hourly.showers?.[index]),
        snowfallCm: maybeNumber(hourly.snowfall?.[index]),
        snowDepthM: maybeNumber(hourly.snow_depth?.[index]),
        temperatureC: maybeNumber(hourly.temperature_2m?.[index]),
        dewPointC: maybeNumber(hourly.dew_point_2m?.[index]),
        relativeHumidityPercent: maybeNumber(hourly.relative_humidity_2m?.[index]),
        weatherCode: maybeNumber(hourly.weather_code?.[index]),
        pressureHPa: maybeNumber(hourly.pressure_msl?.[index]),
        cloudCoverPercent: maybeNumber(hourly.cloud_cover?.[index]),
        visibilityM: maybeNumber(hourly.visibility?.[index]),
        evapotranspirationMm: maybeNumber(hourly.evapotranspiration?.[index]),
        et0EvapotranspirationMm: maybeNumber(hourly.et0_fao_evapotranspiration?.[index]),
        vapourPressureDeficitKpa: maybeNumber(hourly.vapour_pressure_deficit?.[index]),
        windSpeedKph: maybeNumber(hourly.wind_speed_10m?.[index]),
        windGustsKph: maybeNumber(hourly.wind_gusts_10m?.[index]),
        soilTemperatureC: maybeNumber(hourly.soil_temperature_0cm?.[index]),
        soilMoistureM3m3: maybeNumber(hourly.soil_moisture_0_to_1cm?.[index]),
        shortwaveRadiationWm2: maybeNumber(hourly.shortwave_radiation?.[index]),
        directRadiationWm2: maybeNumber(hourly.direct_radiation?.[index]),
        diffuseRadiationWm2: maybeNumber(hourly.diffuse_radiation?.[index]),
        uvIndex: maybeNumber(hourly.uv_index?.[index]),
      };
    })
    .filter((record) => Number.isFinite(record.hoursAgo));
}

function getDryingMultiplier({ weatherNow, recent6, recent12, recent24, shade, surface }) {
  const humidity = weatherNow.relativeHumidityPercent ?? average(recent6, 'relativeHumidityPercent') ?? 70;
  const temperature = weatherNow.temperatureC ?? average(recent6, 'temperatureC') ?? 15;
  const windSpeed = weatherNow.windSpeedKph ?? average(recent6, 'windSpeedKph') ?? 7;
  const radiation = average(recent6, 'shortwaveRadiationWm2') ?? 100;
  const evapotranspiration = Math.max(
    sumField(recent24, 'evapotranspirationMm'),
    sumField(recent24, 'et0EvapotranspirationMm'),
  );
  const vpd = average(recent12, 'vapourPressureDeficitKpa') ?? 0.55;

  const humidityFactor = clamp(0.42, 1.32, 1.25 - humidity / 130);
  const temperatureFactor = clamp(0.55, 1.45, 0.7 + temperature / 35);
  const windFactor = clamp(0.76, 1.36, 0.86 + windSpeed / 60);
  const radiationFactor = clamp(0.72, 1.42, 0.82 + radiation / 650);
  const evapFactor = clamp(0.84, 1.38, 0.94 + evapotranspiration / 7);
  const vpdFactor = clamp(0.82, 1.34, 0.92 + vpd / 4);

  return humidityFactor * temperatureFactor * windFactor * radiationFactor * evapFactor * vpdFactor * shade.drying * surface.heatGain;
}

function humidityLogit(currentHumidity, recentHumidity) {
  const humidity = currentHumidity ?? recentHumidity;
  if (humidity == null) return 0;
  if (humidity >= 92) return 0.52;
  if (humidity >= 82) return 0.32;
  if (humidity >= 68) return 0.12;
  if (humidity <= 38) return -0.34;
  if (humidity <= 52) return -0.18;
  return 0;
}

function temperatureLogit(temperature) {
  if (temperature == null) return 0;
  if (temperature <= 1) return 0.44;
  if (temperature <= 7) return 0.2;
  if (temperature >= 30) return -0.38;
  if (temperature >= 21) return -0.22;
  return 0;
}

function windLogit(windSpeed) {
  if (windSpeed == null) return 0;
  if (windSpeed <= 3) return 0.16;
  if (windSpeed >= 26) return -0.34;
  if (windSpeed >= 14) return -0.2;
  return 0;
}

function solarLogit({ shortwaveMean, uvMean, cloudCover, shade, surface }) {
  let impact = 0;

  if (shortwaveMean != null) {
    if (shortwaveMean >= 520) impact -= 0.46;
    else if (shortwaveMean >= 260) impact -= 0.28;
    else if (shortwaveMean <= 45) impact += 0.12;
  }

  if (uvMean != null) {
    if (uvMean >= 6) impact -= 0.16;
    else if (uvMean >= 3) impact -= 0.08;
  }

  if (cloudCover != null) {
    if (cloudCover >= 88) impact += 0.2;
    else if (cloudCover <= 25) impact -= 0.12;
  }

  return impact * shade.drying * surface.heatGain;
}

function evapotranspirationLogit(mm24h) {
  if (!Number.isFinite(mm24h) || mm24h <= 0) return 0;
  if (mm24h >= 4) return -0.54;
  if (mm24h >= 2) return -0.34;
  if (mm24h >= 0.8) return -0.16;
  return 0;
}

function vaporPressureDeficitLogit(vpd) {
  if (vpd == null) return 0;
  if (vpd >= 1.8) return -0.38;
  if (vpd >= 1.05) return -0.22;
  if (vpd <= 0.25) return 0.24;
  if (vpd <= 0.45) return 0.12;
  return 0;
}

function soilMoistureLogit(soilMoisture) {
  if (soilMoisture == null) return 0;
  if (soilMoisture >= 0.42) return 0.36;
  if (soilMoisture >= 0.3) return 0.22;
  if (soilMoisture <= 0.1) return -0.16;
  return 0;
}

function siteLogit({ area, surface, drainage, slope, traffic }) {
  const areaSquareMeters = Math.PI * area.radiusMeters * area.radiusMeters;
  const areaImpact = clamp(0, 0.28, Math.log10(Math.max(1, areaSquareMeters) / 80) * 0.09);
  const surfaceImpact = (surface.retention - 1) * 0.34;
  return areaImpact + surfaceImpact + drainage.logit + slope.logit + traffic.logit;
}

function pushDriver(drivers, driver) {
  const normalized = {
    ...driver,
    impact: round(driver.impact, 3),
    direction: driver.impact > 0.01 ? 'wetter' : driver.impact < -0.01 ? 'drier' : 'neutral',
  };

  drivers.push(normalized);
  return driver.impact;
}

function sumPrecip(records, hours) {
  return round(
    records
      .filter((record) => record.hoursAgo <= hours)
      .reduce((total, record) => total + record.precipitationMm, 0),
    2,
  );
}

function sumField(records, field) {
  return records.reduce((total, record) => total + (record[field] ?? 0), 0);
}

function average(records, field) {
  const values = records.map((record) => record[field]).filter((value) => Number.isFinite(value));
  if (values.length === 0) {
    return null;
  }

  return values.reduce((total, value) => total + value, 0) / values.length;
}

function normalizeChoice(value, allowed, fallback) {
  const normalized = String(value || '').trim().toLowerCase();
  return allowed.includes(normalized) ? normalized : fallback;
}

function numberOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function maybeNumber(value) {
  if (value === '') {
    return null;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function clamp(min, max, value) {
  return Math.min(max, Math.max(min, value));
}

function logistic(logit) {
  return 1 / (1 + Math.exp(-logit));
}

function classify(probabilityWet) {
  if (probabilityWet >= 0.82) return 'Very likely wet';
  if (probabilityWet >= 0.6) return 'Likely wet';
  if (probabilityWet >= 0.35) return 'Mixed/uncertain';
  return 'Likely dry';
}

function parseTime(value) {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildExplanation({ probabilityWet, classification, drivers, totals }) {
  const topDrivers = drivers.slice(0, 3).map((driver) => driver.label.toLowerCase());
  const topDriverText = topDrivers.length > 0 ? topDrivers.join(', ') : 'limited available data';
  return `${classification} at ${Math.round(probabilityWet * 100)}%. The model weighed ${topDriverText}. Recent precipitation totals are ${formatNumber(totals.last6h, 1)} mm in 6h and ${formatNumber(totals.last24h, 1)} mm in 24h.`;
}

function buildCaveats({ recentRecords, cameraSignals, forecast }) {
  const caveats = [
    'This is a heuristic estimate, not a calibrated safety guarantee.',
    'Weather-grid data may miss microclimates, sprinklers, roof runoff, shade pockets, or covered surfaces.',
  ];

  if (recentRecords.length < 24) {
    caveats.push('Less than 24 hours of hourly weather data was available for the model window.');
  }

  if (cameraSignals.length === 0) {
    caveats.push('No camera or visual evidence was provided, so the result relies on weather and site assumptions.');
  }

  if (cameraSignals.some((signal) => signal.source === 'vision' && signal.detections?.length === 0)) {
    caveats.push('At least one camera image had no visual wetness detections, so camera evidence was downweighted.');
  }

  const hourly = forecast?.hourly || {};
  const optionalFields = [
    ['uv_index', 'UV index'],
    ['soil_moisture_0_to_1cm', 'near-surface soil moisture'],
    ['evapotranspiration', 'evapotranspiration'],
    ['vapour_pressure_deficit', 'vapor pressure deficit'],
  ];
  const missing = optionalFields.filter(([field]) => !Array.isArray(hourly[field])).map(([, label]) => label);

  if (missing.length > 0) {
    caveats.push(`Some optional variables were unavailable: ${missing.join(', ')}.`);
  }

  return caveats;
}

function summarizeGeometry({ place, cameraSignals }) {
  const distances = cameraSignals.map((signal) => signal.distanceToTargetMeters).filter((value) => Number.isFinite(value));
  const elevationDeltas = cameraSignals.map((signal) => signal.elevationDeltaMeters).filter((value) => Number.isFinite(value));

  return {
    targetElevationMeters: Number.isFinite(place?.elevationMeters) ? place.elevationMeters : null,
    cameraCount: cameraSignals.length,
    nearestCameraDistanceMeters: distances.length > 0 ? Math.min(...distances) : null,
    maxElevationDeltaMeters: elevationDeltas.length > 0 ? Math.max(...elevationDeltas.map((value) => Math.abs(value))) : null,
  };
}

function buildWetnessField({ probabilityWet, area, weatherNow, totals, terrainGrid }) {
  const drainageMultiplier = { poor: 1.18, average: 1, good: 0.78 }[area.drainage] ?? 1;
  const terrain = buildTerrainSurface(terrainGrid);
  const recentRainBoost = clamp01((totals.last6h + totals.last24h * 0.25) / 8);
  const cells = [];
  const gridSize = 15;
  const midpoint = Math.floor(gridSize / 2);

  for (let row = 0; row < gridSize; row += 1) {
    for (let column = 0; column < gridSize; column += 1) {
      const x = (column - midpoint) / midpoint;
      const y = (row - midpoint) / midpoint;
      const terrainCell = nearestTerrainCell(terrain, x, y);
      const lowSpotBoost = terrainCell ? terrainCell.lowSpotScore * 0.18 : 0;
      const highSpotDrying = terrainCell ? terrainCell.highSpotScore * 0.1 : 0;
      const radialDistance = Math.min(1, Math.sqrt(x * x + y * y));
      const centerPooling = (1 - radialDistance) * 0.08 * drainageMultiplier;
      const edgeDrying = radialDistance * (area.drainage === 'good' ? 0.09 : 0.04);
      const focalBoost = Math.max(0, 1 - radialDistance / 0.55) * 0.06;
      const probability = clamp01(probabilityWet + lowSpotBoost + centerPooling + recentRainBoost * 0.1 - highSpotDrying - edgeDrying);

      cells.push({
        x: round(x, 2),
        y: round(y, 2),
        probability: round(clamp01(probability + focalBoost), 2),
        hotspot: classifyHotspot(probability),
        elevationMeters: nullableRound(terrainCell?.elevationMeters, 1),
      });
    }
  }

  return {
    model: terrain ? 'terrain-hotspot-surface-v1' : 'synthetic-hotspot-surface-v1',
    radiusMeters: area.radiusMeters,
    assumptions: {
      slope: area.slope,
      drainage: area.drainage,
      windDirectionDeg: weatherNow.windDirectionDeg ?? null,
      recentRainBoost: round(recentRainBoost, 2),
      terrainModel: terrain?.model || null,
    },
    cells,
  };
}

function buildTerrainSurface(terrainGrid) {
  if (!terrainGrid) {
    return null;
  }

  const elevations = terrainGrid.cells.map((cell) => cell.elevationMeters).filter((value) => Number.isFinite(value));

  if (elevations.length < 9) {
    return null;
  }

  const minElevation = Math.min(...elevations);
  const maxElevation = Math.max(...elevations);
  const range = Math.max(0.1, maxElevation - minElevation);

  return {
    model: terrainGrid.model,
    cells: terrainGrid.cells.map((cell) => {
      const normalized = (cell.elevationMeters - minElevation) / range;
      return {
        ...cell,
        lowSpotScore: clamp01(1 - normalized),
        highSpotScore: clamp01(normalized),
      };
    }),
  };
}

function nearestTerrainCell(terrain, x, y) {
  if (!terrain) {
    return null;
  }

  return terrain.cells.reduce((nearest, cell) => {
    const distance = Math.hypot(cell.x - x, cell.y - y);
    return !nearest || distance < nearest.distance ? { ...cell, distance } : nearest;
  }, null);
}

function classifyHotspot(probability) {
  if (probability >= 0.75) return 'high';
  if (probability >= 0.52) return 'medium';
  if (probability >= 0.32) return 'low';
  return 'dry';
}

function estimateConfidence({ recentRecords, cameraSignals, forecast }) {
  let confidence = 0.46;

  if (recentRecords.length >= 48) confidence += 0.14;
  else if (recentRecords.length >= 24) confidence += 0.08;

  if (cameraSignals.length > 0) {
    const bestCameraConfidence = Math.max(...cameraSignals.map((signal) => signal.confidence * signal.geometryWeight * signal.freshnessWeight));
    confidence += Math.min(0.24, bestCameraConfidence * 0.24);
  }

  const hourly = forecast?.hourly || {};
  if (Array.isArray(hourly.soil_moisture_0_to_1cm)) confidence += 0.06;
  if (Array.isArray(hourly.evapotranspiration)) confidence += 0.05;
  if (Array.isArray(hourly.uv_index)) confidence += 0.03;

  return clamp01(confidence);
}

function formatPercent(value) {
  return value == null ? 'unavailable' : `${Math.round(value)}%`;
}

function formatDegrees(value) {
  return value == null ? 'unavailable' : `${formatNumber(value, 1)} C`;
}

function formatSpeed(value) {
  return value == null ? 'unavailable' : `${formatNumber(value, 1)} km/h`;
}

function formatNumber(value, digits) {
  return value == null || !Number.isFinite(value) ? 'unavailable' : String(round(value, digits));
}

function nullableRound(value, digits) {
  return value == null || !Number.isFinite(value) ? null : round(value, digits);
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
