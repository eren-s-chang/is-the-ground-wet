import dns from 'node:dns/promises';
import net from 'node:net';

const defaultMaxImageBytes = Number(process.env.MAX_CAMERA_IMAGE_BYTES || 6_000_000);
const defaultFetchTimeoutMs = Number(process.env.CAMERA_FETCH_TIMEOUT_MS || 4500);

export async function validateCameraImageUrl(imageUrl) {
  if (!imageUrl) {
    return { ok: false, reason: 'No image URL was provided.' };
  }

  let parsed;

  try {
    parsed = new URL(imageUrl);
  } catch {
    return { ok: false, reason: 'Image URL is invalid.' };
  }

  if (!['https:', 'http:'].includes(parsed.protocol)) {
    return { ok: false, reason: 'Only HTTP and HTTPS image URLs are supported.' };
  }

  if (isPrivateHost(parsed.hostname)) {
    return { ok: false, reason: 'Private, local, and link-local camera hosts are blocked by default.' };
  }

  try {
    const addresses = await dns.lookup(parsed.hostname, { all: true });
    if (addresses.some((address) => isPrivateHost(address.address))) {
      return { ok: false, reason: 'Camera host resolves to a private or local network address.' };
    }
  } catch {
    return { ok: false, reason: 'Camera host could not be resolved.' };
  }

  return { ok: true, normalizedUrl: parsed.toString() };
}

export async function fetchCameraImageMetadata(imageUrl, options = {}) {
  const validation = await validateCameraImageUrl(imageUrl);

  if (!validation.ok) {
    return { ok: false, reason: validation.reason };
  }

  const maxImageBytes = options.maxImageBytes || defaultMaxImageBytes;
  const timeoutMs = options.timeoutMs || defaultFetchTimeoutMs;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(validation.normalizedUrl, { method: 'HEAD', signal: controller.signal });
    const contentLength = Number(response.headers.get('content-length'));
    const contentType = response.headers.get('content-type') || '';

    if (!response.ok) {
      return { ok: false, reason: `Camera image returned HTTP ${response.status}.` };
    }

    if (contentLength > maxImageBytes) {
      return { ok: false, reason: 'Camera image exceeds the configured size limit.' };
    }

    if (contentType && !contentType.startsWith('image/') && !contentType.includes('multipart/x-mixed-replace')) {
      return { ok: false, reason: 'Camera URL did not return an image content type.' };
    }

    return { ok: true, normalizedUrl: validation.normalizedUrl, contentLength, contentType };
  } catch (error) {
    return { ok: false, reason: error.name === 'AbortError' ? 'Camera image validation timed out.' : 'Camera image could not be validated.' };
  } finally {
    clearTimeout(timeout);
  }
}

export function isPrivateHost(hostname) {
  const normalized = String(hostname || '').toLowerCase().replace(/^\[|\]$/g, '');

  if (!normalized || normalized === 'localhost' || normalized.endsWith('.localhost')) {
    return true;
  }

  if (net.isIP(normalized) === 4) {
    const parts = normalized.split('.').map(Number);
    return (
      parts[0] === 10 ||
      parts[0] === 127 ||
      (parts[0] === 169 && parts[1] === 254) ||
      (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) ||
      (parts[0] === 192 && parts[1] === 168) ||
      parts[0] === 0
    );
  }

  if (net.isIP(normalized) === 6) {
    return normalized === '::1' || normalized.startsWith('fc') || normalized.startsWith('fd') || normalized.startsWith('fe80');
  }

  return false;
}
