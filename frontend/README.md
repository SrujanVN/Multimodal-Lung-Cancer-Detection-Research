# React frontend

The React + TypeScript + Vite client is in `frontend/`. The existing Flask application and model files remain at the repository root.

## Local development

1. Start the backend from the project root: `python app.py` (Flask listens on port 5000).
2. In `frontend/`, run `npm install` then `npm run dev` (Vite defaults to port 5173).

The Vite dev server proxies the existing Flask auth, inference form, report, documentation, and explanation paths. The current image inference routes return rendered Jinja HTML rather than JSON; the client therefore submits image and clinical forms to those existing routes. No prediction/history/metrics APIs were invented.
