# Handoff

## Status Summary
- Built a UF ground-wetness app with dynamic camera evidence, explainable overlays, and a radar-style wetness heatmap across UF.
- Preferred inference path is `pavement-wetdry` (SegFormer surface mask + lightweight pavement classifier); fallback is a classical heuristic pipeline.
- Added a polygon ROI drawing tool in the camera lightbox and wired it to the UF WeatherSTEM camera. ROI is sent to inference when SegFormer fails and is stored in-memory on the API server.
- Updated SW Rec center to `29.638383, -82.367133` and expanded UF overlay radius to `1800m` with denser 15x15 grid.

## Key Decisions
- Camera feeds are noisy; for now use SegFormer + lightweight classifier. User wants to likely switch to a powerful external model API next.
- ROI should be user-drawn for the UF WeatherSTEM camera to avoid “wrong patch” crops; polygon ROI is supported in UI and inference.
- UF map overlay should cover campus (not just a tiny radius) with a radar-like heat surface and a focal emphasis around SW Rec.

## What Changed (High Level)
- Frontend map: switched to Leaflet heatmap via `leaflet.heat` with radar-style overlay and styled tiles.
- Backend wetness field: grid density increased to 15x15; radial focal boost added; UF assessment radius set to 1800m.
- UF SW Rec coordinates updated from W3W to `29.638383, -82.367133`.
- Inference: accepts optional `roiPolygon` and uses it when SegFormer surface mask is unavailable.
- UI: polygon ROI drawing in the camera lightbox; saved to server via `/api/camera/roi`.

## Known Limitations
- ROI storage is in-memory only (lost on server restart).
- Camera quality and coverage are weak; user expects to move to a powerful external model API.
- Heatmap depends on local heuristics and terrain grid; not a true radar feed.

## Files Touched (Most Relevant)
- `src/App.tsx` (radar map, heat layer, ROI drawing UI)
- `src/styles.css` (map styling)
- `server/index.js` (UF place/area, ROI endpoint, camera analysis passthrough)
- `server/wetnessModel.js` (wetness field grid density and focal boost, polygon normalization)
- `server/visionInferenceClient.js` (ROI passthrough + normalizePolygon)
- `inference/app.py` (roiPolygon handling for pavement provider)
- `inference/pavement_classifier.py` (classical heuristic pipeline)

## How to Run
- API + client: `npm run dev`
- Local inference: `npm run dev:inference:local`
- Docker inference: `npm run dev:inference`

## Tests
- `npm test`
- `npm run build`
- `python -m py_compile inference/app.py`

## Next Steps (Planned)
- Replace local camera inference with a powerful external model API (user request for tomorrow).
- Persist ROI polygons to disk (e.g., `data/roi-polygons.json`) so they survive restarts.
- Consider attaching ROI polygon to specific camera IDs and show ROI even after reloads.
