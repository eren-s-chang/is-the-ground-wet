import test from 'node:test';
import assert from 'node:assert/strict';
import { isPrivateHost } from './cameraImageService.js';

test('isPrivateHost blocks local and private network addresses', () => {
  assert.equal(isPrivateHost('localhost'), true);
  assert.equal(isPrivateHost('127.0.0.1'), true);
  assert.equal(isPrivateHost('10.1.2.3'), true);
  assert.equal(isPrivateHost('172.16.1.2'), true);
  assert.equal(isPrivateHost('192.168.1.8'), true);
  assert.equal(isPrivateHost('example.com'), false);
});
