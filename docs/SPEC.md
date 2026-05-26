# Is The Ground Wet Technical Spec

## Product Goal
Estimate the probability that the ground at a specific place is still wet by combining local weather, recent precipitation history, drying conditions, surface/area characteristics, and optional camera/visual evidence.

## MVP Scope
- Address/place search using OpenStreetMap Nominatim, with Open-Meteo geocoding as a fallback.
- Current and recent local weather using Open-Meteo forecast data.
- Wetness probability model using precipitation recency, humidity, temperature, wind, solar/UV/radiation, evapotranspiration, vapor pressure deficit, soil moisture, cloud cover, surface type, shade, drainage, slope, traffic, and area radius.
- Optional user-supplied camera evidence entries with observation labels and confidence.
- Single-page UI that accepts a location and site assumptions, calls the backend, and displays probability, status, explanation, weather drivers, precipitation history, and area assumptions.
- No API credentials required for the default experience.

## Non-Goals
- Scraping or discovering private IP cameras.
- Computer vision classification of arbitrary camera feeds.
- Property-boundary or parcel-area calculation.
- Guaranteed safety/operational decision making.
- Long-term persistence or user accounts.

## Architecture
- React + Vite frontend served in development by Vite and in production by Express.
- Node/Express backend provides API endpoints and keeps weather/geocoding logic out of the browser.
- Backend calls OpenStreetMap Nominatim for address geocoding and Open-Meteo for weather data.
- Wetness model is a deterministic scoring function returning probability, drivers, assumptions, and caveats.
- Frontend renders the result and a compact precipitation time-series visualization.

## Data Model
- `AssessmentRequest`
  - `location`: free-form address/place string.
  - `area`: radius in meters plus surface, shade, drainage, slope, and traffic assumptions.
  - `cameraSignals`: optional visual observations with label, URL, observed condition, and confidence.
- `AssessmentResult`
  - `place`: name, country/admin region, latitude, longitude, timezone.
  - `probabilityWet`: 0-1 probability.
  - `classification`: `Likely dry`, `Mixed/uncertain`, `Likely wet`, or `Very likely wet`.
  - `current`: current weather variables and units.
  - `precipitation`: hourly recent series and rolling totals.
  - `drivers`: scored model factors with impact direction.
  - `area`: normalized site assumptions and estimated area in square meters.
  - `cameraSignals`: normalized evidence and model impact.
  - `explanation`: short human-readable summary.
  - `caveats`: uncertainty and missing-data notes.

## API Contracts
- `GET /api/health`
  - Returns `{ "ok": true, "service": "is-the-ground-wet" }`.
- `POST /api/assess`
  - Request body: `AssessmentRequest` JSON.
  - Success: `AssessmentResult` JSON.
  - `400`: missing/invalid location or no geocoding match.
  - `502`: upstream weather/geocoding failure.

## Frontend Structure
- `src/main.tsx`: React entry point.
- `src/App.tsx`: form state, API call, result rendering.
- `src/styles.css`: responsive visual design.
- Vite proxy sends `/api/*` requests to local Express during development.

## Backend Structure
- `server/index.js`: Express app, request validation, Open-Meteo calls, static asset serving.
- `server/wetnessModel.js`: deterministic wetness probability model.
- `server/wetnessModel.test.js`: focused model unit tests.

## Deployment Plan
- Build frontend with `npm run build`.
- Run `npm start` in a Node 20+ environment.
- Express serves `dist/` and the `/api/*` endpoints on `PORT` or `8787`.
- No default secrets are needed. Future camera-provider integrations should use server-side credentials only.

## Risks
- Open-Meteo coverage and variable availability vary by location.
- Public weather grids may not reflect a shaded driveway, courtyard, or microclimate.
- Camera evidence is user-provided in the MVP and can be stale or inaccurate.
- The model is heuristic until calibrated against real observations.
- Snow/ice, irrigation, sprinklers, runoff, and indoor/covered surfaces add uncertainty.

## MVP Acceptance Criteria
- A user can enter a location and site assumptions from the browser.
- The backend geocodes the location and fetches current/recent weather without credentials.
- The app returns and displays a wetness probability, classification, explanation, precipitation series, weather drivers, and area estimate.
- Optional camera evidence changes the model output in a transparent way.
- `npm run verify` passes.
