import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cameraGeometryWeight,
  elevationDeltaMeters,
  freshnessWeight,
  haversineDistanceMeters,
  observationFreshnessMinutes,
} from './geometryService.js';

test('haversineDistanceMeters calculates approximate distance', () => {
  const distance = haversineDistanceMeters(
    { latitude: 40.7128, longitude: -74.006 },
    { latitude: 40.7138, longitude: -74.006 },
  );

  assert.ok(distance > 100 && distance < 120);
});

test('cameraGeometryWeight downweights far and elevation-mismatched cameras', () => {
  const near = cameraGeometryWeight({ distanceToTargetMeters: 20, elevationDeltaMeters: 2 });
  const far = cameraGeometryWeight({ distanceToTargetMeters: 1600, elevationDeltaMeters: 120 });

  assert.ok(near > far);
});

test('freshness and elevation helpers are bounded', () => {
  assert.equal(elevationDeltaMeters(125.24, 100), 25.2);
  assert.equal(observationFreshnessMinutes('2026-05-24T11:00:00Z', new Date('2026-05-24T12:00:00Z')), 60);
  assert.ok(freshnessWeight(5) > freshnessWeight(500));
});
