import test from 'node:test';
import assert from 'node:assert/strict';
import { conditionFromVision, normalizeDetections, visualWetnessModel } from './visionInferenceClient.js';

test('selected pretrained model is documented for surface wetness classification', () => {
  assert.equal(visualWetnessModel.id, 'pavement-wetdry-efficientnet-b0');
  assert.equal(visualWetnessModel.task, 'segformer-surface-mask-fixed-roi-pavement-classification');
  assert.equal(visualWetnessModel.prompts.length, 0);
});

test('vision wetness score maps to model conditions', () => {
  assert.equal(conditionFromVision(0.9), 'standing_water');
  assert.equal(conditionFromVision(0.7), 'wet');
  assert.equal(conditionFromVision(0.5), 'mixed');
  assert.equal(conditionFromVision(0.1), 'dry');
});

test('detections are normalized and bounded', () => {
  assert.deepEqual(normalizeDetections([{ label: 'Wet Ground', score: 2 }]), [
    { className: 'wet_ground', confidence: 1, boundingBox: null },
  ]);
});
