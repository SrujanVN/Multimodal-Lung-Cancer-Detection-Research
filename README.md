# Lung Cancer Detection Research Platform

A research and education web application for exploring lung disease image and clinical-data classification. The project brings a React interface together with a Flask API, trained PyTorch models, explainability views, saved notebook results, and an educational chat assistant.

> **Research use only.** Predictions and explanations are experimental model outputs. They are not diagnoses, screening decisions, or treatment advice. Do not use this application to make decisions about a person's care. A qualified healthcare professional must interpret clinical findings.

## What is included

- **CT image analysis:** ResNet50, DenseNet121, Inception-v3, and EfficientNet-B3 checkpoints, with individual-model and ensemble workflows.
- **Lung X-ray analysis:** ResNet50, DenseNet121, Inception-v3, and EfficientNet-B3 checkpoints, with individual-model and ensemble workflows. The labels in this research dataset include thoracic/lung findings and are not a cancer-only diagnosis.
- **Clinical data analysis:** a trained tabular MLP and scaler for individual entries and CSV cohorts. Local LIME explanations and cohort SHAP and LIME summaries are generated when the corresponding packages are available.
- **Image explainability:** Grad-CAM and LIME overlays are provided to help inspect model behavior. Highlighted pixels do not establish disease.
- **Educational chat:** a lung-cancer information assistant with safety guidance and source links. Gemini is optional and configured on the backend using an environment variable.
- **Research transparency:** training/evaluation notebooks, selected saved confusion-matrix figures, and a clearly labelled synthetic clinical CSV example.
- **Application pages:** analysis, report download, sign-in/registration, project information, documentation, and a responsive React frontend.

## Repository layout

```text
.
├── app.py                         # Flask API and server-rendered routes
├── chatbot/                       # Educational assistant service, sources, Sheets script
├── docs/                          # Project documentation artifacts
├── frontend/                      # React + TypeScript + Vite interface
│   └── public/notebook-results/   # Selected notebook confusion-matrix figures
├── lungcancer_csv_Notebook.ipynb  # Clinical model research notebook
├── models/
│   ├── ct_models/                 # Four CT model checkpoints
│   ├── xray models/               # Four lung X-ray model checkpoints
│   └── csv_best_model.pth         # Clinical MLP checkpoint
├── notebooks/
│   ├── ct notebooks/
│   └── xray notebooks/
├── scaler.pkl                     # Clinical feature scaler
├── static/                        # Flask static assets and 3D lung model
├── templates/                     # Flask templates
├── requirements.txt
└── README.md
```

Local secrets, virtual environments, dependency folders, generated reports/uploads, user database files, and caches are intentionally excluded from version control.

## Technology

- Frontend: React 19, TypeScript, Vite, Tailwind CSS, Lucide icons, and Google `<model-viewer>`.
- Backend: Flask, PyTorch/torchvision, scikit-learn, SHAP, LIME, Matplotlib, ReportLab, and Google GenAI SDK.
- Authentication: Flask-backed user registration and login. Keep local database files private; do not commit them.

## Run locally

### Requirements

- Python 3.10 or newer (the notebooks record Python 3.11).
- Node.js and npm.
- Enough memory and disk space for PyTorch, the model checkpoints, and image inference. A CUDA-enabled PyTorch install is optional; CPU inference may be slower.

### Run the full app with Docker Compose

Docker Compose runs the Flask inference/API service and a production-built React app behind Nginx. The model checkpoints and clinical scaler are included in the backend image so it can run on a Docker host without model bind mounts. The SQLite database, uploads, and generated reports persist in local folders. Image inference uses CPU-only PyTorch in this setup.

Create your private environment file once, then set a strong `SECRET_KEY`. `GEMINI_API_KEY` is optional and enables Gemini-backed chat:

```powershell
Copy-Item .env.example .env
notepad .env
docker compose up --build -d
```

Open [http://localhost:8080](http://localhost:8080). Compose waits for the API health check, which verifies that all eight image checkpoints and the clinical model/scaler load. Follow startup and inference logs with `docker compose logs -f`; shut down the stack with `docker compose down`. The persistent `instance`, `static/uploads`, and `static/reports` folders remain on the host after shutdown. Set `APP_PORT` in `.env` to change the browser port.

The initial build downloads Python and Node dependencies and can take several minutes. Docker Desktop should have enough memory for PyTorch and the models. Keep `.env` private; it is ignored by Git and excluded from the Docker build context.

### Deploy the frontend on Vercel

The Vercel project should use `frontend/` as its Root Directory. `frontend/vercel.ts` configures the Vite build, SPA fallback, and same-origin proxy rewrites for the Flask API and returned image/report URLs. Set `BACKEND_URL` in the Vercel project's Preview and Production environments to the public HTTPS **origin** of the separately hosted backend (for example `https://api.example.com`, with no path). Vercel will fail the build if this URL is missing or is not HTTPS.

Build and deploy the Dockerized backend on a host that supports persistent container storage. Set `SECRET_KEY` and, optionally, `GEMINI_API_KEY` as backend environment secrets. Do not add Gemini credentials to Vercel frontend variables. Keep writable storage persistent for `instance/`, `static/uploads/`, and `static/reports/`; predictions or uploads may contain sensitive data. Do not use real patient scans or identifiable clinical records for a public demonstration. After the backend has a public HTTPS address, add it as `BACKEND_URL` in Vercel, connect the Git repository, choose the `frontend` root, and deploy.

### Backend

From the repository root, create and activate a virtual environment, install the dependencies, and create a private `.env` file from the example:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally. Keep all values private and never commit this file. `SECRET_KEY` should be a long random value. Set `GEMINI_API_KEY` only if you want Gemini-backed chat. The Google Sheets webhook fields are optional and should only be configured with a protected endpoint and an explicit privacy notice. Do not record passwords or other authentication secrets in a spreadsheet.

Start Flask:

```powershell
python app.py
```

The API defaults to `http://127.0.0.1:5000`. Check `GET /api/health` to confirm the server is responding. The API also exposes `GET /api/models`, `POST /api/predict/image`, `POST /api/predict/clinical`, `POST /api/predict/csv`, and `POST /api/chat`. The image/CSV endpoints accept multipart form data; see the frontend service in `frontend/src/services/api.ts` and `app.py` for the current request/response fields.

### Frontend

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite (normally `http://127.0.0.1:5173`). The Vite development server proxies `/api` and `/static` requests to the Flask server. For a production frontend bundle, run `npm run build` from `frontend/`; serve the generated `frontend/dist` with your chosen hosting setup and configure the API origin accordingly.

## Models and data

The checkpoints in `models/` are the trained artifacts currently wired into this project. CT and X-ray checkpoints are approximately 28–98 MB each. Clinical inference expects the 15 feature columns and encoding/scaling order implemented by the backend and described in the clinical notebook. Use the included synthetic CSV only to understand the required format; its rows are fabricated examples, not patient data or research observations.

Notebook outputs include high test-set scores and confusion matrices. These are reproduced from the saved notebooks, not an independent validation. In particular, the notebook workflows do not establish patient-level separation for image data, and the X-ray workflows include an `UNKNOWN` class built from generic web images. These design choices can inflate or distort reported metrics. Treat all scores as exploratory; they do not demonstrate clinical validity, generalization, or safety. Review dataset provenance, licensing, label quality, subject-level splitting, and external validation before making research claims.

## Explainability

- **Grad-CAM** produces a coarse spatial map of image regions that influenced a selected convolutional model output.
- **Image LIME** probes predictions using perturbed image regions and shows which superpixels locally affected the output.
- **Clinical LIME** describes local feature contributions for an individual row.
- **CSV SHAP and LIME** summarize how features contributed to the model outputs across a submitted cohort.

These methods describe model behavior under their assumptions; they do not show causal factors, confirm a lesion, or prove that a model is correct. SHAP/LIME may be unavailable when optional dependencies are missing.

## Chat assistant and sources

The chat service is educational and application-aware. It can explain general lung-cancer concepts, the application's models, and common explainability terms. When configured, it uses the Gemini API key stored in backend environment configuration; never place the key in frontend code or commit it. The service also includes selected National Cancer Institute references. Retrieved links and generated answers should still be checked against current authoritative medical information. The chatbot must not diagnose, prescribe, or replace a clinician.

## Privacy and security

- Never commit `.env`, API keys, passwords, database files, user uploads, generated reports, or personal medical information.
- Run this project only with data you are authorized to process. Use synthetic or appropriately de-identified data for demonstrations.
- Provide clear notice and consent before collecting or exporting any user information. Authentication logs should not contain passwords or unnecessary identifiers.
- Configure production secrets, HTTPS, access controls, secure session cookies, request limits, and retention/deletion policies before deployment. The local research setup is not a production security review.
- Check dataset and model licenses and attribution requirements before redistributing or deploying artifacts.

## Troubleshooting

- **Model not found:** run the backend from the repository root and check that every expected checkpoint under `models/` is present.
- **Chat says the model is not configured:** add a valid `GEMINI_API_KEY` to the backend `.env`, then restart Flask. Never add the key to Vite variables or browser code.
- **SHAP/LIME unavailable:** install the pinned backend requirements in the active environment and restart Flask.
- **Frontend cannot reach the API:** confirm Flask is listening on port 5000 and use the Vite development server configuration in `frontend/vite.config.ts`.
- **Slow image explanations:** image explainability can require many inference passes, particularly on CPU.

## Responsible use

This repository is a research prototype, not a medical device. Outputs may be inaccurate, biased, poorly calibrated, or affected by data leakage and dataset shift. Never use a prediction to diagnose, rule out disease, select treatment, or delay care. For symptoms or findings, consult a qualified healthcare professional; for severe or rapidly worsening symptoms, seek urgent medical care.

## License and acknowledgements

No license file is included at present. Unless a license is added, reuse and redistribution rights are not granted by this README. Check the licenses of the training data, pretrained weights, model checkpoints, 3D asset, and third-party packages before redistribution. Add dataset citations and source attributions here as their provenance is confirmed.
