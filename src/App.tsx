import { type FormEvent, type ReactNode, useEffect, useRef, useState } from 'react';
import { CircleMarker, MapContainer, TileLayer, useMap } from 'react-leaflet';
import type { LatLngExpression } from 'leaflet';
import L from 'leaflet';
import 'leaflet.heat/dist/leaflet-heat';

type CameraSignalState = {
  id: string;
  label: string;
  imageUrl: string;
  url?: string;
  condition: string;
  confidence: number;
  latitude: string;
  longitude: string;
  elevationMeters: string;
  observedAt: string;
};

type Driver = {
  label: string;
  value: string;
  impact: number;
  direction: 'wetter' | 'drier' | 'neutral';
  description: string;
};

type AssessmentResult = {
  place: {
    name: string;
    displayName?: string;
    latitude: number;
    longitude: number;
    elevationMeters?: number | null;
    provider?: string;
  };
  probabilityWet: number;
  probabilityPercent: number;
  classification: string;
  confidence: number;
  explanation: string;
  generatedAt: string;
  current: { time: string | null; values: Record<string, number | null> };
  precipitation: {
    totals: Record<string, number>;
    effectiveMoistureMm: number;
    dryingHalfLifeHours: number;
    series: Array<{ time: string; hoursAgo: number; precipitationMm: number; temperatureC: number | null; humidityPercent: number | null }>;
  };
  drivers: Driver[];
  area: {
    radiusMeters: number;
    surfaceLabel: string;
    shadeLabel: string;
    drainageLabel: string;
    slopeLabel: string;
    trafficLabel: string;
    estimatedSquareMeters: number;
  };
  cameraSignals: Array<CameraSignalState & {
    conditionLabel: string;
    impact: number;
    distanceToTargetMeters: number | null;
    elevationDeltaMeters: number | null;
    freshnessMinutes: number | null;
    geometryWeight: number;
    freshnessWeight: number;
    wetnessScore: number | null;
    visualConfidence: number;
    source: string;
    detections: Array<{ className: string; confidence: number; maskPng?: string }>;
    segments?: Array<{ className: string; confidence: number; maskPng?: string }>;
    masks?: Array<{ className: string; coverage: number; maskPng?: string }>;
    crop?: { x: number; y: number; width: number; height: number; imagePng?: string } | null;
    classifier?: { nearest: string; nearestDescription?: string; confidence: number; wetnessScore: number | null; margin?: number; method: string; inputMode?: string; reason?: string; scores: Record<string, number>; promptGroups?: Record<string, { description?: string; prompts?: string[] }> } | null;
    classifierInputMode?: string;
    roiPolygon?: Array<{ x: number; y: number }> | null;
    caveats: string[];
  }>;
  geometry: {
    targetElevationMeters: number | null;
    cameraCount: number;
    nearestCameraDistanceMeters: number | null;
    maxElevationDeltaMeters: number | null;
  };
  wetnessField: {
    model: string;
    radiusMeters: number;
    assumptions: { drainage: string; terrainModel: string | null; recentRainBoost: number };
    cells: Array<{ x: number; y: number; probability: number; hotspot?: 'dry' | 'low' | 'medium' | 'high'; elevationMeters?: number | null }>;
  };
  caveats: string[];
  model: { name: string; dryingMultiplier: number };
  publicSources: Array<{
    id: string;
    provider: string;
    kind: string;
    label: string;
    latitude: number | null;
    longitude: number | null;
    distanceToTargetMeters: number | null;
    url: string;
    notes: string;
    usableAsEvidence: boolean;
    usedAsEvidence: boolean;
    evidenceWarning: string;
  }>;
};

const conditionOptions = [
  ['standing_water', 'Standing water'],
  ['wet', 'Wet'],
  ['mixed', 'Mixed wet/dry'],
  ['mostly_dry', 'Mostly dry'],
  ['dry', 'Dry'],
  ['unknown', 'Unknown'],
];

const loadingSteps = [
  'Resolving location and terrain',
  'Fetching recent weather',
  'Loading public camera frames',
  'Segmenting visible road surface',
  'Running local VLM classifier',
  'Combining camera and weather evidence',
];

export default function App() {
  const [location, setLocation] = useState('');
  const [cameraSignals, setCameraSignals] = useState<CameraSignalState[]>([]);
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState('Assessing');
  const [loadingStep, setLoadingStep] = useState(0);

  useEffect(() => {
    if (!loading) return;
    setLoadingStep(0);
    const interval = window.setInterval(() => {
      setLoadingStep((current) => Math.min(loadingSteps.length - 1, current + 1));
    }, 3500);
    return () => window.clearInterval(interval);
  }, [loading]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await runAssessment('/api/assess', {
      location,
      cameras: cameraSignals.map((signal) => ({
        ...signal,
        latitude: optionalNumber(signal.latitude),
        longitude: optionalNumber(signal.longitude),
        elevationMeters: optionalNumber(signal.elevationMeters),
      })),
    }, 'Assessing target');
  }

  async function handleUfAssessment() {
    setLocation('Southwest Recreation Center, University of Florida');
    await runAssessment('/api/assess/uf-sw-rec', {}, 'Assessing UF route');
  }

  async function runAssessment(endpoint: string, body: Record<string, unknown>, label: string) {
    setLoading(true);
    setLoadingLabel(label);
    setLoadingStep(0);
    setError('');

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      const payload = await response.json();

      if (!response.ok) throw new Error(payload.error || 'Assessment failed.');
      setResult(payload);
    } catch (assessmentError) {
      setError(assessmentError instanceof Error ? assessmentError.message : 'Assessment failed.');
    } finally {
      setLoading(false);
    }
  }

  function addCameraSignal() {
    setCameraSignals((current) => [...current, {
      id: crypto.randomUUID(),
      label: `Camera ${current.length + 1}`,
      imageUrl: '',
      condition: 'wet',
      confidence: 0.7,
      latitude: '',
      longitude: '',
      elevationMeters: '',
      observedAt: '',
    }]);
  }

  function updateCameraSignal(id: string, patch: Partial<CameraSignalState>) {
    setCameraSignals((current) => current.map((signal) => (signal.id === id ? { ...signal, ...patch } : signal)));
  }

  function removeCameraSignal(id: string) {
    setCameraSignals((current) => current.filter((signal) => signal.id !== id));
  }

  function usePublicSource(source: AssessmentResult['publicSources'][number]) {
    if (!source.url || !source.usableAsEvidence) return;
    setCameraSignals((current) => current.some((signal) => signal.imageUrl === source.url) ? current : [...current, {
      id: crypto.randomUUID(),
      label: source.label,
      imageUrl: source.url,
      condition: 'unknown',
      confidence: 0.45,
      latitude: source.latitude == null ? '' : String(source.latitude),
      longitude: source.longitude == null ? '' : String(source.longitude),
      elevationMeters: '',
      observedAt: '',
    }]);
  }

  return (
    <main className="min-h-screen bg-[#08090d] text-[#f7f8ff] antialiased">
      <nav className="sticky top-0 z-30 border-b border-white/10 bg-[#08090d]/85 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-[1440px] items-center justify-between px-4 text-sm">
          <div className="flex items-center gap-3">
            <div className="grid size-7 place-items-center rounded-lg border border-white/10 bg-white/[0.04] text-[11px] font-semibold">GW</div>
            <span className="font-medium tracking-tight">Ground Wetness</span>
            <span className="hidden rounded-full border border-white/10 px-2 py-1 text-xs text-white/50 sm:inline">Weather + vision + terrain</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-white/50">
            <span>{result ? result.model.name : 'heuristic-weather-visual-v1'}</span>
            <span className="size-1.5 rounded-full bg-emerald-400" />
          </div>
        </div>
      </nav>

      <div className="mx-auto grid max-w-[1440px] gap-4 px-4 py-4 lg:grid-cols-[360px_1fr]">
        <aside className="space-y-4 lg:sticky lg:top-[72px] lg:self-start">
          <form onSubmit={handleSubmit} className="rounded-2xl border border-white/10 bg-[#111219] shadow-2xl shadow-black/20">
            <PanelHeader kicker="Assessment" title="Target" meta={cameraSignals.length ? `${cameraSignals.length} queued` : 'Auto evidence'} />
            <div className="space-y-3 p-4 pt-0">
              <label className="block space-y-2">
                <span className="text-xs font-medium text-white/55">Address or place</span>
                <input
                  required
                  value={location}
                  onChange={(event) => setLocation(event.target.value)}
                  placeholder="123 Main St, Portland, OR"
                  className="h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 text-sm outline-none transition hover:border-white/20 focus:border-indigo-400/70"
                />
              </label>
              <button disabled={loading} className="h-10 w-full rounded-lg bg-[#f4f4f5] text-sm font-medium text-black transition hover:bg-white disabled:cursor-wait disabled:opacity-60" type="submit">
                {loading ? loadingLabel : 'Assess ground wetness'}
              </button>
              <button disabled={loading} className="h-10 w-full rounded-lg border border-indigo-400/25 bg-indigo-400/10 text-sm font-medium text-indigo-100 transition hover:bg-indigo-400/15 disabled:cursor-wait disabled:opacity-60" type="button" onClick={handleUfAssessment}>
                Run UF Southwest Rec route
              </button>
              {loading ? <LoadingProgress activeStep={loadingStep} label={loadingLabel} /> : null}
              {error ? <p className="rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-xs leading-5 text-red-100">{error}</p> : null}
            </div>
          </form>

          <section className="rounded-2xl border border-white/10 bg-[#111219]">
            <PanelHeader kicker="Fallback" title="Manual evidence" meta="Advanced" />
            <div className="space-y-3 p-4 pt-0">
              <p className="text-xs leading-5 text-white/45">Use only authorized public images or feeds. Automatic public discovery and the UF route run first.</p>
              <button type="button" onClick={addCameraSignal} className="h-9 rounded-lg border border-white/10 px-3 text-xs font-medium text-white/75 transition hover:border-white/20 hover:bg-white/[0.04]">Add evidence</button>
              {cameraSignals.map((signal) => (
                <div key={signal.id} className="space-y-2 rounded-xl border border-white/10 bg-black/15 p-3">
                  <div className="flex gap-2">
                    <input className="h-9 flex-1 rounded-lg border border-white/10 bg-black/20 px-2 text-xs outline-none" value={signal.label} onChange={(event) => updateCameraSignal(signal.id, { label: event.target.value })} />
                    <button type="button" onClick={() => removeCameraSignal(signal.id)} className="rounded-lg border border-white/10 px-2 text-xs text-white/60 hover:bg-white/[0.04]">Remove</button>
                  </div>
                  <input className="h-9 w-full rounded-lg border border-white/10 bg-black/20 px-2 text-xs outline-none" value={signal.imageUrl} onChange={(event) => updateCameraSignal(signal.id, { imageUrl: event.target.value })} placeholder="Authorized HTTPS image/frame URL" />
                  <div className="grid grid-cols-2 gap-2">
                    <select className="h-9 rounded-lg border border-white/10 bg-black/20 px-2 text-xs" value={signal.condition} onChange={(event) => updateCameraSignal(signal.id, { condition: event.target.value })}>
                      {conditionOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                    </select>
                    <input className="h-9 rounded-lg border border-white/10 bg-black/20 px-2 text-xs" value={signal.latitude} onChange={(event) => updateCameraSignal(signal.id, { latitude: event.target.value })} placeholder="Lat" />
                    <input className="h-9 rounded-lg border border-white/10 bg-black/20 px-2 text-xs" value={signal.longitude} onChange={(event) => updateCameraSignal(signal.id, { longitude: event.target.value })} placeholder="Lon" />
                    <input className="h-9 rounded-lg border border-white/10 bg-black/20 px-2 text-xs" value={signal.elevationMeters} onChange={(event) => updateCameraSignal(signal.id, { elevationMeters: event.target.value })} placeholder="Elevation" />
                  </div>
                </div>
              ))}
            </div>
          </section>
        </aside>

        <section className="min-w-0">
          {result ? <Dashboard result={result} onUsePublicSource={usePublicSource} /> : <EmptyState />}
        </section>
      </div>
    </main>
  );
}

function Dashboard({ result, onUsePublicSource }: { result: AssessmentResult; onUsePublicSource: (source: AssessmentResult['publicSources'][number]) => void }) {
  const current = result.current.values;
  const visualCount = result.cameraSignals.filter((signal) => signal.source === 'vision').length;
  const [expandedSignal, setExpandedSignal] = useState<AssessmentResult['cameraSignals'][number] | null>(null);

  return (
    <div className="space-y-4">
      <section className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <div className="rounded-2xl border border-white/10 bg-[#111219] p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-2xl">
              <p className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-white/40">Resolved place</p>
              <h1 className="text-2xl font-semibold tracking-[-0.04em] text-white md:text-4xl">{result.place.displayName || result.place.name}</h1>
              <p className="mt-3 text-sm leading-6 text-white/50">{result.explanation}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4 text-right">
              <div className="text-5xl font-semibold tracking-[-0.08em]">{result.probabilityPercent}%</div>
              <div className="mt-1 text-sm text-white/55">{result.classification}</div>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 rounded-2xl border border-white/10 bg-[#111219] p-4">
          <Metric label="Confidence" value={`${Math.round(result.confidence * 100)}%`} />
          <Metric label="Visual model" value={visualCount ? `${visualCount} active` : 'Not active'} />
          <Metric label="Cameras" value={String(result.cameraSignals.length)} />
          <Metric label="Effective rain" value={`${result.precipitation.effectiveMoistureMm} mm`} />
        </div>
      </section>

      <HotspotMap result={result} />

      <section className="grid gap-4 xl:grid-cols-3">
        <Panel title="Current Signals" kicker="Weather" meta={result.current.time || 'live'}>
          <div className="grid grid-cols-2 gap-2">
            <Metric label="Temperature" value={formatMetric(current.temperatureC, ' C')} />
            <Metric label="Humidity" value={formatMetric(current.relativeHumidityPercent, '%')} />
            <Metric label="Wind" value={formatMetric(current.windSpeedKph, ' km/h')} />
            <Metric label="Cloud" value={formatMetric(current.cloudCoverPercent, '%')} />
            <Metric label="Now precip" value={formatMetric(current.precipitationMm, ' mm')} />
            <Metric label="Drying half-life" value={`${result.precipitation.dryingHalfLifeHours} h`} />
          </div>
        </Panel>
        <Panel title="Top Drivers" kicker="Explainability" meta={`${result.drivers.length} drivers`} className="xl:col-span-2">
          <div className="divide-y divide-white/10">
            {result.drivers.slice(0, 7).map((driver) => (
              <div key={driver.label} className="grid grid-cols-[1fr_auto] gap-4 py-3 transition hover:bg-white/[0.025]">
                <div>
                  <p className="text-sm font-medium text-white/85">{driver.label}</p>
                  <p className="mt-1 text-xs leading-5 text-white/45">{driver.description}</p>
                </div>
                <div className="text-right text-xs">
                  <p className="text-white/50">{driver.value}</p>
                  <p className={driver.impact > 0 ? 'mt-1 font-medium text-sky-300' : 'mt-1 font-medium text-emerald-300'}>{driver.impact > 0 ? '+' : ''}{driver.impact}</p>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1fr_420px]">
        <Panel title="Vision Evidence" kicker="Cameras" meta={`${result.cameraSignals.length} observations`}>
          <div className="space-y-2">
            {result.cameraSignals.map((signal) => <CameraRow key={signal.id} signal={signal} onExpand={() => setExpandedSignal(signal)} />)}
          </div>
        </Panel>
        <Panel title="Public Sources" kicker="Discovery" meta={`${result.publicSources.length} found`}>
          <div className="space-y-2">
            {result.publicSources.slice(0, 8).map((source) => (
              <div key={source.id} className="rounded-xl border border-white/10 p-3 text-xs transition hover:bg-white/[0.03]">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium text-white/80">{source.label}</p>
                    <p className="mt-1 text-white/40">{source.provider} · {formatMetric(source.distanceToTargetMeters, ' m')}</p>
                  </div>
                  {source.usableAsEvidence && !source.usedAsEvidence ? <button type="button" onClick={() => onUsePublicSource(source)} className="rounded-md border border-white/10 px-2 py-1 text-white/60 hover:bg-white/[0.04]">Use</button> : null}
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </section>

      <Panel title="Recent Precipitation" kicker="Timeline" meta={`${result.precipitation.effectiveMoistureMm} mm effective`}>
        <PrecipitationChart series={result.precipitation.series} />
      </Panel>

      <Panel title="Caveats" kicker="Risk" meta={`${result.caveats.length} notes`}>
        <div className="grid gap-2 md:grid-cols-2">
          {result.caveats.map((caveat) => <p key={caveat} className="rounded-xl border border-white/10 bg-black/10 p-3 text-xs leading-5 text-white/45">{caveat}</p>)}
        </div>
      </Panel>

      {expandedSignal ? <CameraLightbox signal={expandedSignal} onClose={() => setExpandedSignal(null)} /> : null}
    </div>
  );
}

function HotspotMap({ result }: { result: AssessmentResult }) {
  const zoom = chooseMapZoom(result.place.latitude);
  const center: LatLngExpression = [result.place.latitude, result.place.longitude];
  const metersPerUnit = result.wetnessField.radiusMeters;
  const heatPoints = result.wetnessField.cells.map((cell) => {
    const point = offsetLatLng(center, cell.x * metersPerUnit, cell.y * metersPerUnit);
    return [point[0], point[1], clampNumber(0.2, 0.9, cell.probability)] as [number, number, number];
  });

  return (
    <Panel title="Wetness Radar" kicker="Spatial model" meta={result.wetnessField.model}>
      <div className="relative h-[360px] w-full overflow-hidden rounded-xl border border-white/10">
        <MapContainer center={center} zoom={zoom} scrollWheelZoom className="h-full w-full">
          <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" className="leaflet-slate" />
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" className="leaflet-vignette" />
          <MapSync center={center} zoom={zoom} />
          <HeatLayerControl points={heatPoints} />
          <CircleMarker center={center} radius={4} pathOptions={{ color: '#f8fafc', weight: 2, fillColor: '#f8fafc', fillOpacity: 1 }} />
        </MapContainer>
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(56,189,248,0.16),rgba(14,116,144,0.12),rgba(8,9,13,0.85))]" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.08),transparent_55%)]" />
          <div className="absolute bottom-4 left-4 flex items-center gap-2 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[11px] text-white/70">Radar composite · updated {new Date(result.generatedAt).toLocaleTimeString()}</div>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-xs text-white/45">
        <span>Target radius {result.wetnessField.radiusMeters} m</span>
        <span>Terrain {result.wetnessField.assumptions.terrainModel || 'synthetic'}</span>
        <span>Hotspots show relative wet likelihood, not surveyed pooling.</span>
      </div>
    </Panel>
  );
}

function CameraRow({ signal, onExpand }: { signal: AssessmentResult['cameraSignals'][number]; onExpand: () => void }) {
  const active = signal.source === 'vision' && signal.wetnessScore != null;
  const detections = signal.segments?.length ? signal.segments : signal.detections;
  const masks = signal.masks?.length ? signal.masks : detections.map((item) => ({ className: item.className, coverage: item.confidence, maskPng: item.maskPng }));
  const topDetections = [...detections].sort((a, b) => b.confidence - a.confidence).slice(0, 4);
  const overlays = [...masks]
    .filter((item) => item.maskPng)
    .sort((a, b) => b.coverage - a.coverage)
    .slice(0, 3);
  const previewUrl = signal.imageUrl || signal.url;
  const proxiedPreviewUrl = previewUrl ? `/api/camera/image?url=${encodeURIComponent(previewUrl)}` : '';
  const classifier = signal.classifier;
  return (
    <div className="rounded-xl border border-white/10 p-3 text-xs transition hover:bg-white/[0.03]">
      {proxiedPreviewUrl ? (
        <div className="mb-3 overflow-hidden rounded-lg border border-white/10 bg-black/40">
          <button type="button" onClick={onExpand} className="group relative block w-full text-left" aria-label={`Expand ${signal.label} segmentation preview`}>
            <img
              src={proxiedPreviewUrl}
              alt={`${signal.label} capture`}
              loading="lazy"
              className="h-36 w-full object-cover"
            />
            {overlays.map((overlay, index) => (
              <img
                key={`${signal.id}-${overlay.className}`}
                src={overlay.maskPng}
                alt={`${overlay.className} mask`}
                className={`absolute inset-0 h-36 w-full object-cover ${index === 0 ? 'mix-blend-screen opacity-65' : 'mix-blend-screen opacity-35'}`}
              />
            ))}
            <span className="absolute bottom-2 right-2 rounded-full border border-white/15 bg-black/65 px-2 py-1 text-[11px] font-medium text-white/75 opacity-0 transition group-hover:opacity-100">Expand</span>
          </button>
        </div>
      ) : null}
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-medium text-white/80">{signal.label}</p>
          <p className="mt-1 text-white/40">{signal.source} · {signal.conditionLabel} · impact {signal.impact > 0 ? '+' : ''}{signal.impact.toFixed(3)}</p>
        </div>
        <span className={active ? 'rounded-full bg-emerald-400/15 px-2 py-1 text-emerald-200' : 'rounded-full bg-white/[0.04] px-2 py-1 text-white/45'}>{active ? 'vision' : 'metadata'}</span>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-2 text-white/45">
        <span>dist {formatMetric(signal.distanceToTargetMeters, ' m')}</span>
        <span>geo {signal.geometryWeight}</span>
        <span>score {signal.wetnessScore == null ? 'none' : Math.round(signal.wetnessScore * 100)}</span>
        <span>det {detections.length}</span>
      </div>
      {classifier ? (
        <div className="mt-2 rounded-lg border border-white/10 bg-black/15 p-2 text-[11px] text-white/45">
          <span className="text-white/65">{classifier.nearestDescription || classifier.nearest.replace(/_/g, ' ')}</span>
          <span> · {classifier.method}</span>
          {classifier.inputMode || signal.classifierInputMode ? <span> · {(classifier.inputMode || signal.classifierInputMode)?.replace(/_/g, ' ')}</span> : null}
          <span> · conf {Math.round(classifier.confidence * 100)}%</span>
        </div>
      ) : null}
      {topDetections.length ? (
        <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-white/45">
          {topDetections.map((detection) => (
            <span key={`${signal.id}-${detection.className}`} className="rounded-full border border-white/10 px-2 py-0.5">
              {detection.className.replace(/_/g, ' ')} {Math.round(detection.confidence * 100)}%
            </span>
          ))}
        </div>
      ) : null}
      {signal.caveats.length ? <p className="mt-2 truncate text-white/30">{signal.caveats[0]}</p> : null}
    </div>
  );
}

function CameraLightbox({ signal, onClose }: { signal: AssessmentResult['cameraSignals'][number]; onClose: () => void }) {
  const detections = signal.segments?.length ? signal.segments : signal.detections;
  const masks = signal.masks?.length ? signal.masks : detections.map((item) => ({ className: item.className, coverage: item.confidence, maskPng: item.maskPng }));
  const overlays = [...masks]
    .filter((item) => item.maskPng)
    .sort((a, b) => b.coverage - a.coverage)
    .slice(0, 6);
  const topDetections = [...detections].sort((a, b) => b.confidence - a.confidence).slice(0, 8);
  const previewUrl = signal.imageUrl || signal.url;
  const proxiedPreviewUrl = previewUrl ? `/api/camera/image?url=${encodeURIComponent(previewUrl)}` : '';
  const classifierScores = signal.classifier ? Object.entries(signal.classifier.scores).sort((a, b) => b[1] - a[1]).slice(0, 5) : [];
  const [drawingMode, setDrawingMode] = useState(false);
  const [polygonPoints, setPolygonPoints] = useState<Array<{ x: number; y: number }>>(signal.roiPolygon || []);
  const polygonClosed = polygonPoints.length >= 3 && !drawingMode;
  const previewRef = useRef<HTMLDivElement | null>(null);
  const isUfWeatherstem = signal.id === 'uf-weatherstem-ben-hill-griffin' || (signal.label || '').toLowerCase().includes('weatherstem');
  const polygonForSave = polygonPoints.map((point) => ({ x: Number(point.x.toFixed(4)), y: Number(point.y.toFixed(4)) }));

  function handleCanvasClick(event: React.MouseEvent<HTMLDivElement>) {
    if (!drawingMode || !previewRef.current) return;
    const rect = previewRef.current.getBoundingClientRect();
    const x = clampNumber(0, 1, (event.clientX - rect.left) / rect.width);
    const y = clampNumber(0, 1, (event.clientY - rect.top) / rect.height);
    setPolygonPoints((current) => [...current, { x, y }]);
  }

  function closePolygon() {
    if (polygonPoints.length < 3) return;
    setDrawingMode(false);
  }

  function resetPolygon() {
    setPolygonPoints([]);
    setDrawingMode(true);
  }

  async function savePolygon() {
    if (polygonPoints.length < 3 || !isUfWeatherstem) return;
    try {
      await fetch('/api/camera/roi', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ cameraId: signal.id, roiPolygon: polygonForSave }),
      });
    } catch {
      // swallow; UI still shows local preview
    }
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/85 p-3 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label={`${signal.label} segmentation detail`} onClick={onClose}>
      <div className="max-h-[92vh] w-full max-w-5xl overflow-hidden rounded-2xl border border-white/10 bg-[#111219] shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-start justify-between gap-4 border-b border-white/10 p-4">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-white/35">Segmentation Preview</p>
            <h2 className="mt-1 text-base font-semibold text-white/90">{signal.label}</h2>
            <p className="mt-1 text-xs text-white/45">{signal.source} · {signal.conditionLabel} · score {signal.wetnessScore == null ? 'none' : Math.round(signal.wetnessScore * 100)}</p>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg border border-white/10 px-3 py-2 text-xs font-medium text-white/65 hover:bg-white/[0.04]">Close</button>
        </div>
        <div className="grid gap-4 p-4 lg:grid-cols-[1fr_260px]">
          <div className="overflow-hidden rounded-xl border border-white/10 bg-black/50">
            <div className={`relative ${drawingMode ? 'cursor-crosshair' : ''}`} ref={previewRef} onClick={handleCanvasClick} role={drawingMode ? 'button' : undefined} aria-label={drawingMode ? 'Add ROI point' : undefined}>
              {proxiedPreviewUrl ? <img src={proxiedPreviewUrl} alt={`${signal.label} expanded capture`} className="max-h-[68vh] w-full object-contain" /> : null}
              {overlays.map((overlay, index) => (
                <img
                  key={`${signal.id}-expanded-${overlay.className}-${index}`}
                  src={overlay.maskPng}
                  alt={`${overlay.className} expanded mask`}
                  className={`absolute inset-0 h-full w-full object-contain ${index === 0 ? 'mix-blend-screen opacity-70' : 'mix-blend-screen opacity-35'}`}
                />
              ))}
              {polygonPoints.length ? (
                <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
                  <polygon
                    points={polygonPoints.map((point) => `${point.x},${point.y}`).join(' ')}
                    fill={polygonClosed ? 'rgba(56,189,248,0.12)' : 'rgba(56,189,248,0.04)'}
                    stroke="rgba(56,189,248,0.85)"
                    strokeWidth="0.006"
                  />
                </svg>
              ) : null}
            </div>
          </div>
          <div className="space-y-3">
            <Metric label="Impact" value={`${signal.impact > 0 ? '+' : ''}${signal.impact.toFixed(3)}`} />
            <Metric label="Detections" value={String(detections.length)} />
            {signal.crop?.imagePng ? (
              <div className="overflow-hidden rounded-xl border border-white/10 bg-black/20">
                <img src={signal.crop.imagePng} alt="Classifier pavement crop" className="h-32 w-full object-cover" />
                <div className="p-3 text-[11px] text-white/45">Classifier crop · x {Math.round(signal.crop.x * 100)}% y {Math.round(signal.crop.y * 100)}%</div>
              </div>
            ) : null}
            {signal.classifier ? (
              <div className="rounded-xl border border-white/10 bg-black/10 p-3">
                <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-white/35">Wet/Dry Classifier</p>
                <p className="mt-2 text-sm font-medium text-white/80">{signal.classifier.nearestDescription || signal.classifier.nearest.replace(/_/g, ' ')} · {Math.round(signal.classifier.confidence * 100)}%</p>
                <p className="mt-1 text-xs text-white/45">{signal.classifier.method}{signal.classifier.inputMode || signal.classifierInputMode ? ` · ${(signal.classifier.inputMode || signal.classifierInputMode)?.replace(/_/g, ' ')}` : ''}</p>
                {signal.classifier.reason ? <p className="mt-2 text-[11px] text-white/45">{signal.classifier.reason}</p> : null}
                <div className="mt-3 space-y-2">
                  {classifierScores.map(([label, score]) => (
                    <div key={`${signal.id}-classifier-${label}`}>
                      <div className="flex justify-between text-[11px] text-white/45"><span>{signal.classifier?.promptGroups?.[label]?.description || label.replace(/_/g, ' ')}</span><span>{Math.round(score * 100)}%</span></div>
                      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-cyan-300" style={{ width: `${Math.round(score * 100)}%` }} /></div>
                    </div>
                  ))}
                </div>
                {signal.classifier.promptGroups?.[signal.classifier.nearest]?.prompts?.length ? (
                  <p className="mt-3 text-[11px] leading-4 text-white/35">Matched against: {signal.classifier.promptGroups[signal.classifier.nearest].prompts?.[0]}</p>
                ) : null}
              </div>
            ) : null}
            {isUfWeatherstem ? (
              <div className="rounded-xl border border-white/10 bg-black/10 p-3 text-[11px] text-white/45">
                <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-white/35">UF WeatherSTEM ROI</p>
                <p className="mt-2">Click on the image to draw a polygon around the pavement. Minimum 3 points.</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <button type="button" onClick={() => setDrawingMode((current) => !current)} className="rounded-md border border-white/10 px-2 py-1 text-white/70 hover:bg-white/[0.04]">{drawingMode ? 'Stop drawing' : 'Draw polygon'}</button>
                  <button type="button" onClick={closePolygon} className="rounded-md border border-white/10 px-2 py-1 text-white/70 hover:bg-white/[0.04]">Close</button>
                  <button type="button" onClick={resetPolygon} className="rounded-md border border-white/10 px-2 py-1 text-white/70 hover:bg-white/[0.04]">Reset</button>
                  <button type="button" onClick={savePolygon} className="rounded-md border border-cyan-400/40 bg-cyan-400/10 px-2 py-1 text-cyan-100 hover:bg-cyan-400/20">Save ROI</button>
                </div>
                {polygonPoints.length ? <p className="mt-2 text-white/35">{polygonPoints.length} points captured.</p> : null}
              </div>
            ) : null}
            <div className="rounded-xl border border-white/10 bg-black/10 p-3">
              <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-white/35">Top Masks</p>
              <div className="mt-3 flex flex-wrap gap-2 text-xs text-white/55">
                {topDetections.length ? topDetections.map((detection) => (
                  <span key={`${signal.id}-expanded-label-${detection.className}`} className="rounded-full border border-white/10 px-2 py-1">
                    {detection.className.replace(/_/g, ' ')} {Math.round(detection.confidence * 100)}%
                  </span>
                )) : <span>No masks returned</span>}
              </div>
            </div>
            {signal.caveats.length ? <p className="rounded-xl border border-white/10 bg-black/10 p-3 text-xs leading-5 text-white/45">{signal.caveats[0]}</p> : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="grid min-h-[640px] place-items-center rounded-2xl border border-white/10 bg-[#111219] p-8 text-center">
      <div className="max-w-xl">
        <p className="mb-3 text-xs font-medium uppercase tracking-[0.18em] text-white/40">Awaiting assessment</p>
        <h1 className="text-4xl font-semibold tracking-[-0.05em]">Map likely wet spots before you step outside.</h1>
        <p className="mt-4 text-sm leading-6 text-white/45">Run a general target or the UF Southwest Rec route to combine weather, public cameras, terrain, and a precomputed hotspot surface.</p>
      </div>
    </div>
  );
}

function Panel({ kicker, title, meta, actions, className = '', children }: { kicker: string; title: string; meta?: string; actions?: ReactNode; className?: string; children: ReactNode }) {
  return <section className={`rounded-2xl border border-white/10 bg-[#111219] ${className}`}><PanelHeader kicker={kicker} title={title} meta={meta} actions={actions} /><div className="p-4 pt-0">{children}</div></section>;
}

function PanelHeader({ kicker, title, meta, actions }: { kicker: string; title: string; meta?: string; actions?: ReactNode }) {
  return <div className="flex items-start justify-between gap-3 p-4"><div><p className="text-[11px] font-medium uppercase tracking-[0.16em] text-white/35">{kicker}</p><h2 className="mt-1 text-base font-semibold tracking-[-0.02em] text-white/90">{title}</h2></div><div className="flex items-center gap-2">{meta ? <span className="rounded-full border border-white/10 px-2 py-1 text-xs text-white/45">{meta}</span> : null}{actions}</div></div>;
}

function LoadingProgress({ activeStep, label }: { activeStep: number; label: string }) {
  return (
    <div className="rounded-xl border border-cyan-300/15 bg-cyan-300/[0.04] p-3 text-xs">
      <div className="flex items-center justify-between gap-3">
        <span className="font-medium text-cyan-100">{label}</span>
        <span className="text-white/35">{activeStep + 1}/{loadingSteps.length}</span>
      </div>
      <div className="mt-3 space-y-2">
        {loadingSteps.map((step, index) => (
          <div key={step} className="flex items-center gap-2 text-white/45">
            <span className={`size-2 rounded-full ${index < activeStep ? 'bg-emerald-300' : index === activeStep ? 'animate-pulse bg-cyan-300' : 'bg-white/15'}`} />
            <span className={index === activeStep ? 'text-white/75' : ''}>{step}</span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-4 text-white/35">Local VLM inference can take a while on first run while model weights warm up.</p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-white/10 bg-black/10 p-3 transition hover:bg-white/[0.03]"><p className="text-[11px] font-medium uppercase tracking-[0.12em] text-white/35">{label}</p><p className="mt-2 text-sm font-medium text-white/85">{value}</p></div>;
}

function PrecipitationChart({ series }: { series: AssessmentResult['precipitation']['series'] }) {
  const chartSeries = [...series].sort((a, b) => b.hoursAgo - a.hoursAgo);
  const maxPrecipitation = Math.max(0.2, ...chartSeries.map((point) => point.precipitationMm));
  const width = 100;
  const height = 36;
  const barWidth = Math.max(0.45, width / Math.max(1, chartSeries.length) - 0.2);
  return <svg className="h-36 w-full overflow-visible" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"><line x1="0" x2="100" y1="35" y2="35" stroke="rgba(255,255,255,0.12)" strokeWidth="0.4" />{chartSeries.map((point, index) => { const barHeight = (point.precipitationMm / maxPrecipitation) * 30; const x = (index / Math.max(1, chartSeries.length)) * width; return <rect key={`${point.time}-${index}`} x={x} y={35 - barHeight} width={barWidth} height={barHeight} rx="0.25" fill="#8b5cf6" opacity="0.85" />; })}</svg>;
}

function MapSync({ center, zoom }: { center: LatLngExpression; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom, { animate: false });
  }, [map, center, zoom]);
  return null;
}

function HeatLayerControl({ points }: { points: Array<[number, number, number]> }) {
  const map = useMap();

  useEffect(() => {
    if (!points.length) return undefined;
    const layer = (window as typeof window & { L?: { heatLayer?: (points: Array<[number, number, number]>, options: Record<string, unknown>) => { addTo: (mapInstance: unknown) => void; remove: () => void } } }).L?.heatLayer?.(points, {
      radius: 40,
      blur: 28,
      minOpacity: 0.35,
      maxZoom: 18,
      gradient: {
        0.1: '#0ea5e9',
        0.35: '#38bdf8',
        0.55: '#a855f7',
        0.72: '#f59e0b',
        0.9: '#f97316',
      },
    });

    if (layer) {
      layer.addTo(map as unknown);
    }

    return () => {
      if (layer) map.removeLayer(layer as unknown as L.Layer);
    };
  }, [map, points]);

  return null;
}

function offsetLatLng(origin: LatLngExpression, metersEast: number, metersNorth: number) {
  const point = Array.isArray(origin) ? origin : [origin.lat, origin.lng];
  const earthRadius = 6378137;
  const deltaLat = metersNorth / earthRadius;
  const deltaLng = metersEast / (earthRadius * Math.cos((point[0] * Math.PI) / 180));
  return [point[0] + (deltaLat * 180) / Math.PI, point[1] + (deltaLng * 180) / Math.PI] as [number, number];
}

function chooseMapZoom(latitude: number) {
  return Math.abs(latitude) > 65 ? 15 : 16;
}

function hotspotColor(probability: number) {
  if (probability >= 0.75) return '#38bdf8';
  if (probability >= 0.52) return '#8b5cf6';
  if (probability >= 0.32) return '#f59e0b';
  return '#64748b';
}

function formatMetric(value: number | null | undefined, unit: string) {
  if (value == null || !Number.isFinite(value)) return 'unavailable';
  return `${Math.round(value * 10) / 10}${unit}`;
}

function optionalNumber(value: string) {
  if (value.trim() === '') return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function clampNumber(min: number, max: number, value: number) {
  return Math.min(max, Math.max(min, value));
}
