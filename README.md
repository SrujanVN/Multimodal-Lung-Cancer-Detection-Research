# Lung Cancer Detection Research Platform

> A research and education platform for exploring CT, lung X-ray, and clinical-data classification with machine-learning models and explainability tools.

**Research use only.** This application is an experimental prototype, not a medical device or a diagnostic service. Model outputs are not diagnoses or treatment advice. Do not use them to make decisions about a person's care; consult a qualified healthcare professional.

## Table of contents

- [Overview](#overview)
- [Demo video](#demo-video)
- [Capabilities](#capabilities)
- [Technology](#technology)
- [Repository structure](#repository-structure)
- [Run with Docker Compose](#run-with-docker-compose)
- [Run locally without Docker](#run-locally-without-docker)
- [Configuration](#configuration)
- [API overview](#api-overview)
- [Models, data, and evaluation](#models-data-and-evaluation)
- [Explainability](#explainability)
- [Educational chat assistant](#educational-chat-assistant)
- [Privacy and security](#privacy-and-security)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Responsible use and limitations](#responsible-use-and-limitations)
- [License and attribution](#license-and-attribution)

## Overview

This project combines a React web interface with a Flask API and trained model artifacts. Users can explore image and clinical-data analysis workflows, compare model outputs, review explainability visualizations, download reports, and ask the educational chat assistant general questions about lung cancer and the application.

The supported image labels and clinical prediction target come from the project's research datasets and notebooks. They should not be interpreted as a general-purpose lung-cancer diagnosis.

## Demo video

> GitHub README pages do not allow embedded YouTube players. Select the preview below to play the video on YouTube.

[![Watch the Lung Cancer Detection Research Platform demo](https://img.youtube.com/vi/gb8Gy-uV1K4/hqdefault.jpg)](https://youtu.be/gb8Gy-uV1K4)

[Open the demo on YouTube](https://youtu.be/gb8Gy-uV1K4)

## Capabilities

- **CT analysis:** individual model inference and ensemble workflows using the CT checkpoints included in `models/`.
- **Lung X-ray analysis:** individual model inference and ensemble workflows. The dataset includes multiple thoracic/lung finding classes and is not cancer-only.
- **Clinical analysis:** single-entry and CSV cohort prediction using the trained tabular MLP and scaler. CSV analysis can produce cohort-level SHAP and LIME summaries when the required packages are installed.
- **Image explainability:** Grad-CAM overlays and image LIME explanations for inspecting model behavior.
- **Reports and history:** analysis results can be reviewed in the interface and reports can be downloaded.
- **Educational assistant:** a safety-oriented chatbot for general lung-cancer education and application concepts. Gemini-backed generation is optional and configured on the backend.
- **Research materials:** model training/evaluation notebooks, selected saved confusion-matrix figures, and a synthetic clinical CSV example.

## Technology

| Layer | Main technologies |
| --- | --- |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Lucide icons |
| Backend | Python, Flask, PyTorch, torchvision, scikit-learn |
| Explainability and reports | SHAP, LIME, Matplotlib, ReportLab |
| Chat integration | Google GenAI SDK (optional; backend environment key) |
| Local deployment | Docker Compose and Nginx |

## Repository structure

```text
.
├── app.py                         # Flask API and server-rendered routes
├── chatbot/                       # Educational assistant, sources, Sheets integration
├── docs/                          # Project documentation artifacts
├── frontend/                      # React + TypeScript + Vite application
│   └── public/notebook-results/   # Selected saved evaluation figures
├── models/                        # Image and clinical model checkpoints
├── notebooks/                     # CT and lung X-ray research notebooks
├── lungcancer_csv_Notebook.ipynb  # Clinical-data notebook
├── scaler.pkl                     # Clinical feature scaler
├── static/                        # Static assets, uploads, and reports
├── templates/                     # Flask templates
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

Local environment files, databases, user uploads, generated reports, virtual environments, caches, and dependency folders should remain untracked.

## Run with Docker Compose

Docker Compose builds the Flask API and the production React frontend served through Nginx. In this configuration, image inference uses CPU PyTorch. The local database, uploads, and generated reports use persistent project folders.

### Requirements

- Docker Desktop with Docker Compose enabled.
- Sufficient disk space and memory for the Python dependencies and model checkpoints.

### Start the application

From the repository root:

```powershell
Copy-Item .env.example .env
notepad .env
docker compose up --build -d
```

Set a strong, private `SECRET_KEY` in `.env`. `GEMINI_API_KEY` is optional and enables Gemini-backed chat. Never commit `.env` or paste secrets into frontend settings.

Open [http://localhost:8080](http://localhost:8080). The default host port can be changed with `APP_PORT` in `.env`.

Useful commands:

```powershell
docker compose logs -f
docker compose ps
docker compose down
```

The first build downloads dependencies and may take several minutes. `docker compose down` stops containers while data in the persistent folders remains on the host.

## Run locally without Docker

### Requirements

- Python 3.10 or newer (the notebooks record Python 3.11).
- Node.js and npm.
- The model checkpoints and scaler referenced by the backend.

### Start the backend

From the repository root, create a virtual environment and install Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally, set a strong `SECRET_KEY`, and optionally configure `GEMINI_API_KEY`. Then start Flask:

```powershell
python app.py
```

The API normally listens on `http://127.0.0.1:5000`.

### Start the frontend

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, normally [http://127.0.0.1:5173](http://127.0.0.1:5173). The development server proxies `/api` and `/static` requests to Flask. To create a production bundle, run `npm run build` from `frontend/`.

## Configuration

Configuration belongs in the backend `.env` file unless noted. Use `.env.example` to see the supported variable names.

| Setting | Purpose |
| --- | --- |
| `SECRET_KEY` | Protects Flask sessions; use a long, random private value. |
| `GEMINI_API_KEY` | Optional backend-only key for Gemini chat responses. |
| `APP_PORT` | Optional host port for the Docker Compose web application. |
| Google Sheets webhook settings | Optional integration. Use a protected endpoint and provide a clear privacy notice. Never export passwords or authentication secrets. |

Do not put API keys in React/Vite variables, source files, commits, screenshots, or public documentation. If a key has been exposed, revoke it and create a replacement.

## API overview

The Flask application currently exposes these principal routes:

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Check API health and model readiness. |
| `GET /api/models` | List available model options. |
| `POST /api/predict/image` | Submit an image for model inference. |
| `POST /api/predict/clinical` | Submit a clinical-data entry. |
| `POST /api/predict/csv` | Submit a clinical CSV cohort. |
| `POST /api/chat` | Send a message to the educational assistant. |

Image and CSV prediction routes accept multipart form data. Refer to `app.py` and `frontend/src/services/api.ts` for the current request fields and response format.

## Models, data, and evaluation

The checkpoints in `models/` are the trained artifacts currently wired into the application. CT and X-ray workflows include ResNet50, DenseNet121, Inception-v3, and EfficientNet-B3 checkpoints, along with ensemble workflows. Clinical inference uses a trained MLP, `scaler.pkl`, and the 15-feature encoding/order defined in the backend and clinical notebook.

Use the included synthetic clinical CSV only to understand the expected format. Its rows are fabricated examples; they are not patient data or valid research observations.

Saved notebook metrics and confusion matrices are exploratory results from those notebooks, not independent clinical validation. The image workflows do not establish patient-level separation, and the X-ray workflow includes an `UNKNOWN` class assembled from generic web images. These choices can inflate or distort reported performance. Review data provenance, licensing, label quality, subject-level splitting, and external validation before making research claims.

## Explainability

- **Grad-CAM** creates a coarse image-region map associated with a selected convolutional model output.
- **Image LIME** perturbs image regions and estimates which superpixels influenced a local prediction.
- **Clinical LIME** presents local feature contributions for one submitted row.
- **Cohort SHAP and LIME** summarize feature contributions across CSV model outputs.

Explanations describe model behavior under their assumptions. They do not show causation, prove a model is correct, or establish that a highlighted area is cancer. SHAP/LIME visualizations may be unavailable if optional dependencies are missing.

## Educational chat assistant

The chatbot provides educational information about lung cancer and can explain concepts used by this application, such as model confidence and explainability methods. When configured, generation uses Gemini through the backend. The frontend must never receive the API key. The service includes selected National Cancer Institute references; generated explanations are not a substitute for checking current medical sources or speaking with a clinician.

The assistant must not diagnose a person, prescribe medication, select an individual's treatment, or predict personal survival. For severe or rapidly worsening symptoms, seek urgent medical care.

## Privacy and security

- Do not commit `.env`, API keys, passwords, databases, uploads, reports, or identifiable health information.
- Use only data you are authorized to process; prefer synthetic or properly de-identified data for demonstrations.
- Explain data collection and obtain appropriate consent before exporting information. Do not log or store passwords in a spreadsheet.
- Before public deployment, configure HTTPS, access controls, secure session cookies, request limits, storage protection, and retention/deletion policies.
- Check dataset, pretrained-weight, model, and 3D-asset licenses before redistribution.
- The local research setup has not been security-reviewed for production or regulated health-data use.

## Deployment

The frontend is configured for Vercel with `frontend/` as the project root. `frontend/vercel.ts` configures the Vite build, SPA fallback, and API/static proxy rewrites. Set `BACKEND_URL` in Vercel Preview and Production to the HTTPS origin of a separately hosted backend. It must be an origin such as `https://api.example.com`, with no path. The Vercel build requires this value to be HTTPS.

Deploy the backend separately, using a host that supports persistent writable storage for `instance/`, `static/uploads/`, and `static/reports/`. Set backend secrets such as `SECRET_KEY` and optional `GEMINI_API_KEY` in the hosting provider's secret manager. Do not set Gemini credentials in Vercel frontend variables. A public demo should use synthetic data only.

## Troubleshooting

- **Model not found:** start the backend from the repository root and confirm the expected checkpoint files exist under `models/`.
- **Chat says Gemini is not configured:** check `GEMINI_API_KEY` in the backend `.env` and restart Flask.
- **SHAP or LIME is unavailable:** install the project requirements in the active Python environment and restart the backend.
- **Frontend cannot reach Flask:** check that the backend is listening on port 5000 and that the Vite proxy in `frontend/vite.config.ts` is active.
- **Image explanations are slow:** Grad-CAM/LIME may need repeated inference passes; CPU-only inference can take longer.
- **Docker health check fails:** inspect `docker compose logs` and verify model files and environment configuration are present.

## Responsible use and limitations

This repository is a research prototype, not a medical device. Model outputs can be inaccurate, biased, poorly calibrated, or affected by data leakage and differences between training data and new inputs. Never use a prediction to diagnose or rule out disease, choose treatment, or delay care. A qualified healthcare professional should interpret clinical findings and imaging.

## License and attribution

No repository license file is currently included. Without a license, this README does not grant reuse or redistribution rights. Add a project license only after confirming ownership and permissions. Dataset citations, pretrained-model references, and third-party asset attributions should be added here once their provenance and licensing have been verified.
