# Implementation Phases

## Phase 1: Project Scaffold and Spec
- Goal: Establish a runnable full-stack project and document the MVP contract.
- Tasks: Add package scripts, Vite/React entry points, Express server shell, spec, and phase docs.
- Likely files touched: `package.json`, `index.html`, `vite.config.ts`, `tsconfig.json`, `src/*`, `server/*`, `docs/*`, `README.md`.
- Verification command: `npm run verify`.
- Acceptance criteria: Dependencies install, app builds, server has a health endpoint, and docs describe the planned MVP.

## Phase 2: Weather and Geocoding API
- Goal: Resolve an input location and retrieve relevant weather time series.
- Tasks: Add address geocoding call, Open-Meteo forecast call, request validation, upstream error handling, and normalized weather payloads.
- Likely files touched: `server/index.js`, `server/wetnessModel.js`.
- Verification command: `npm run test` and `npm run build`.
- Acceptance criteria: `/api/assess` returns a structured result for a valid location and clear errors for invalid input.

## Phase 3: Wetness Model
- Goal: Produce an explainable probability using weather, site assumptions, and visual evidence.
- Tasks: Implement rolling precipitation totals, drying load, surface modifiers, camera signal impacts, classifications, caveats, and unit tests.
- Likely files touched: `server/wetnessModel.js`, `server/wetnessModel.test.js`.
- Verification command: `npm run test`.
- Acceptance criteria: Model output is bounded between 0 and 1, wet recent-rain scenarios score higher than dry scenarios, and camera evidence affects probability.

## Phase 4: Frontend Experience
- Goal: Provide a polished responsive interface for assessment input and output.
- Tasks: Build location/site form, optional camera evidence controls, API loading/error states, probability display, driver cards, and precipitation chart.
- Likely files touched: `src/App.tsx`, `src/styles.css`.
- Verification command: `npm run build`.
- Acceptance criteria: Browser UI can submit an assessment and render all major response sections on desktop and mobile.

## Phase 5: Final Verification and Review
- Goal: Validate the MVP and clean up critical issues.
- Tasks: Run full verification, inspect git diff, fix critical review findings only, and document remaining risks.
- Likely files touched: any changed MVP files.
- Verification command: `npm run verify`.
- Acceptance criteria: Full verification passes or the work stops with a clear blocker.
