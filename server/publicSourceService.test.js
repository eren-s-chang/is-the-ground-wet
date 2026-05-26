import test from 'node:test';
import assert from 'node:assert/strict';
import { buildAutomaticEvidenceInputs, discoverPublicSources } from './publicSourceService.js';

test('discoverPublicSources combines public metadata sources', async () => {
  const place = { latitude: 45, longitude: -122 };
  const result = await discoverPublicSources(place, async (url) => {
    if (String(url).includes('overpass')) {
      return {
        elements: [
          {
            type: 'node',
            id: 1,
            lat: 45.001,
            lon: -122.001,
            tags: { name: 'Public plaza camera', man_made: 'surveillance', website: 'https://example.com/cam' },
          },
        ],
      };
    }

    return {
      query: {
        pages: {
          10: {
            pageid: 10,
            title: 'File:Nearby ground.jpg',
            coordinates: [{ lat: 45.002, lon: -122.002 }],
            imageinfo: [{ url: 'https://example.com/image.jpg', mime: 'image/jpeg' }],
          },
        },
      },
    };
  });

  assert.equal(result.sources.length, 2);
  assert.equal(result.sources[0].provider, 'OpenStreetMap');
  assert.equal(result.sources[1].provider, 'Wikimedia Commons');
  assert.equal(result.sources[0].usableAsEvidence, false);
  assert.equal(result.sources[1].usableAsEvidence, true);
  assert.ok(result.caveats.length >= 2);
});

test('buildAutomaticEvidenceInputs promotes usable public image sources', () => {
  const evidence = buildAutomaticEvidenceInputs([
    {
      id: 'commons-1',
      label: 'Ground image',
      kind: 'public_geotagged_image',
      latitude: 45,
      longitude: -122,
      url: 'https://example.com/ground.jpg',
      usableAsEvidence: true,
      evidenceWarning: 'public image',
    },
    {
      id: 'osm-1',
      label: 'Metadata only',
      kind: 'public_camera_metadata',
      url: '',
      usableAsEvidence: false,
    },
  ], new Date('2026-05-24T12:00:00Z'));

  assert.equal(evidence.length, 1);
  assert.equal(evidence[0].source, 'public_discovery');
  assert.equal(evidence[0].imageUrl, 'https://example.com/ground.jpg');
});

test('discoverPublicSources includes nearby explicit allowlisted feeds', async () => {
  const place = { latitude: 45, longitude: -122 };
  const result = await discoverPublicSources(place, async (url) => {
    if (String(url).includes('overpass')) {
      return { elements: [] };
    }

    return { query: { pages: {} } };
  }, {
    allowlistedFeeds: [
      {
        id: 'city-feed-1',
        region: 'Test City',
        label: 'City traffic still image',
        latitude: 45.0004,
        longitude: -122.0004,
        imageUrl: 'https://traffic.example.gov/camera.jpg',
        directImage: true,
      },
      {
        id: 'far-feed',
        label: 'Far feed',
        latitude: 41,
        longitude: -70,
        imageUrl: 'https://traffic.example.gov/far.jpg',
        directImage: true,
      },
    ],
  });

  assert.equal(result.sources.length, 1);
  assert.equal(result.sources[0].provider, 'Configured Public Feed');
  assert.equal(result.sources[0].kind, 'allowlisted_public_image');
  assert.equal(result.sources[0].usableAsEvidence, true);
});
