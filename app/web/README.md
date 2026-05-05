# Agastya Frontend (React + Vite)

Frontend for the Agastya contract-risk web UI.

## Features

- Contract upload workflow
- Risk dashboard visualizations
- Clause-level exploration and annotation views
- Integration with backend endpoints:
  - `POST /analyze`
  - `POST /classify-clause`

## Development

From this directory (`app/web`):

```bash
npm install
npm run dev
```

The app uses `VITE_API_URL` for backend routing.
If not provided, it defaults to `http://localhost:8000`.

Example:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Build

```bash
npm run build
npm run preview
```
