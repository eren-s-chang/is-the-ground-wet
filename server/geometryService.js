export function haversineDistanceMeters(a, b) {
  if (!isCoordinate(a) || !isCoordinate(b)) {
    return null;
  }

  const earthRadiusMeters = 6371000;
  const latitude1 = toRadians(a.latitude);
  const latitude2 = toRadians(b.latitude);
  const deltaLatitude = toRadians(b.latitude - a.latitude);
  const deltaLongitude = toRadians(b.longitude - a.longitude);
  const sinLatitude = Math.sin(deltaLatitude / 2);
  const sinLongitude = Math.sin(deltaLongitude / 2);
  const centralAngle =
    sinLatitude * sinLatitude + Math.cos(latitude1) * Math.cos(latitude2) * sinLongitude * sinLongitude;

  return Math.round(earthRadiusMeters * 2 * Math.atan2(Math.sqrt(centralAngle), Math.sqrt(1 - centralAngle)));
}

export function elevationDeltaMeters(cameraElevationMeters, targetElevationMeters) {
  if (!Number.isFinite(cameraElevationMeters) || !Number.isFinite(targetElevationMeters)) {
    return null;
  }

  return Math.round((cameraElevationMeters - targetElevationMeters) * 10) / 10;
}

export function cameraGeometryWeight({ distanceToTargetMeters, elevationDeltaMeters }) {
  const distanceWeight = distanceToTargetMeters == null ? 0.65 : Math.max(0.18, Math.exp(-distanceToTargetMeters / 450));
  const elevationWeight = elevationDeltaMeters == null ? 0.75 : Math.max(0.25, Math.exp(-Math.abs(elevationDeltaMeters) / 55));

  return Math.round(distanceWeight * elevationWeight * 1000) / 1000;
}

export function observationFreshnessMinutes(observedAt, now = new Date()) {
  if (!observedAt) {
    return null;
  }

  const observedMs = Date.parse(observedAt);

  if (!Number.isFinite(observedMs)) {
    return null;
  }

  return Math.max(0, Math.round((now.getTime() - observedMs) / 60000));
}

export function freshnessWeight(freshnessMinutes) {
  if (freshnessMinutes == null) {
    return 0.7;
  }

  return Math.round(Math.max(0.12, Math.exp(-freshnessMinutes / 180)) * 1000) / 1000;
}

function isCoordinate(value) {
  return (
    Number.isFinite(value?.latitude) &&
    Number.isFinite(value?.longitude) &&
    Math.abs(value.latitude) <= 90 &&
    Math.abs(value.longitude) <= 180
  );
}

function toRadians(value) {
  return (value * Math.PI) / 180;
}
