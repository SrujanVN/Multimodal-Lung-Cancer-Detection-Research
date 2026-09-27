# Frontend developer notes

This directory contains the React 19, TypeScript, and Vite client. The complete project keeps its Flask server and trained model files in a separate backend project checkout; they are not included in this frontend-only repository.

## Development

Requirements: Node.js 18+ and npm.

```powershell
npm ci
npm run dev -- --host 127.0.0.1
```

The Vite development server is available at `http://127.0.0.1:5173` and proxies API, login, report, upload, and backend static-file requests to `http://127.0.0.1:5000`. Start the full project’s Flask app separately for integrated analysis features.

## Production build

```powershell
npm run build
npm run preview
```

## Frontend routes and capabilities

The application is a single-page client with Home, Analysis, About, More Info, Login, Register, and Documentation views. The Analysis view supports CT, lung X-ray, manual clinical input, and clinical CSV workflows. The frontend displays only prediction and explanation data returned by the backend.

The Documentation view presents saved notebook results and confusion matrices. Treat them as research outputs, not clinical validation. Synthetic CSV examples are demo-only. The 3D hero model is loaded from the backend static asset route and is not stored in this frontend directory.

## Configuration and secrets

Do not expose API keys in browser code or frontend environment variables. Configure credentials in the backend environment. API and session behavior is implemented in `src/services/api.ts` and the Flask service.