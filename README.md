# Lung Cancer Detection — Frontend

A React and TypeScript web client for a research platform exploring lung disease analysis across CT scans, lung X-rays, and clinical data. The frontend communicates with the Flask API supplied by the full project; this repository contains the frontend client only and does not contain model weights, patient datasets, API keys, or the backend implementation.

## Capabilities

- CT and lung X-ray analysis screens backed by the model outputs returned by Flask.
- Clinical assessment using remembered details or an uploaded CSV, with cohort SHAP and LIME explanations when provided by the API.
- Per-analysis class probabilities, model outputs, image explanations, and downloadable reports.
- A lung-disease educational assistant connected to the backend chat API.
- A documentation view with saved notebook test metrics, original confusion-matrix plots, and explicit evaluation limitations.
- Responsive layouts, accessible labels, analysis loading indicators, and an interactive 3D lung model served by the backend.

All model scores and explanations describe experimental model behavior. They are not diagnoses, medical advice, proof of disease, or evidence of clinical readiness. The synthetic CSV is artificial demo data and is not suitable for patient care or clinical research.

## Requirements

- Node.js 18 or newer and npm.
- The full project’s Flask backend running locally at `http://127.0.0.1:5000` for analysis, login, chat, reports, model assets, and other API-backed features.

## Run locally

1. Start the Flask backend from the full project directory containing `app.py`:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

   Use the backend project’s documented environment setup if you have not created its virtual environment yet.

2. In another terminal, start the frontend:

   ```powershell
   cd frontend
   npm ci
   npm run dev -- --host 127.0.0.1
   ```

3. Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

The Vite development server proxies API and session requests to the Flask server on port 5000. Analysis requires a working backend and any required sign-in session. The hero’s 3D model is loaded from the backend’s `/static/models/realistic_human_lungs.glb` asset.

## Main API routes used by the client

- `GET /api/health` and `GET /api/models` for backend status and available models.
- `POST /api/predict/image` for CT or lung X-ray uploads.
- `POST /api/predict/clinical` for manually entered clinical inputs.
- `POST /api/predict/csv` for clinical cohort CSV analysis.
- `POST /api/chat` for the educational assistant.
- `POST /download_report` for report generation.
- `/login` and `/register` for backend-managed authentication.

The exact response values and available explanation images depend on the backend and its loaded checkpoints. The client does not manufacture missing model metrics or history.

## Build

```powershell
npm run build
npm run preview
```

The production frontend must be served alongside a configured backend for API-backed features to work.

## Secrets and privacy

Never put Gemini keys, Flask secrets, or other private credentials in frontend source or `VITE_*` variables. Configure secrets only on the backend using its private environment file, which must not be committed. This frontend sends requests to the local backend; chat responses and uploaded data are handled according to the backend implementation.

## Research transparency

The Documentation page shows values saved in the project notebooks, not freshly measured or independently validated results. The CT and lung X-ray notebooks include an `UNKNOWN` class made from generic Unsplash images; this does not establish reliable recognition of unfamiliar medical images. Image-level splits do not establish patient-level separation. Review the full project’s notebooks and dataset notes before interpreting the reported scores.