# Just looking for stocks
A website for stocks

## Overview
- Flask API + simple frontend (Chart.js) for stock info, history, and LSTM-based close-price predictions.
- Data via Yahoo Finance (yfinance and direct search/screener endpoints).
- Models are cached per ticker under `models/` (gitignored); trained on-demand with a lightweight LSTM.
- Recommended to run locally on your own machine: you avoid network egress limits, keep API keys out of hosted logs, and can leverage your GPU/Metal stack directly.

## Quickstart (CPU, recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m stock_app.app   # serves on http://127.0.0.1:5000 (set PORT to override)
```

## GPU acceleration (optional)
- Default install (`requirements.txt`) uses the CPU build.
- Apple Silicon (Metal): Requires Python 3.10–3.12 and macOS 12+. `requirements-metal.txt` pins the Metal build (tf-macos 2.16 + tensorflow-metal). Create a venv with a supported Python, then `pip install -r requirements-metal.txt`. Verify with:
  ```python
  python - <<'PY'
  import tensorflow as tf
  print(tf.config.list_physical_devices("GPU"))
  PY
  ```
- Windows/Linux (NVIDIA): install matching CUDA/cuDNN for TensorFlow 2.20, then `pip install tensorflow==2.20.0` (the package will use GPU if CUDA is present). Verify with the snippet above.

## Run with Metal venv
```bash
source .venv-metal/bin/activate  # created with python3.11 -m venv .venv-metal
python -m stock_app.app
```

## Usage (UI)
- Open `http://127.0.0.1:5000` (or your configured PORT).
- Search bar supports tickers/company names; quick-picks show most active symbols.
- Range tabs: intraday (1d/1w) and longer windows; interval dropdown lets you force granularity.
- Charts: zoom/pan (wheel/pinch; drag to pan; Shift+drag to box-zoom), crosshair shows date/price.
- Tables default to summary; “Show all” toggles full data.

## API (informal)
- `/api/info?ticker=TSM`
- `/api/history?ticker=TSM&range=1m&interval=auto` (range supports 1d,1w,1m,3m,6m,ytd,1y,2y,5y,10y,all; interval optional)
- `/api/predict?ticker=TSM`
- `/api/search?q=apple`
- `/api/symbols` (most active + fallback list)

## Development notes
- Models dir: created automatically (`models/`, gitignored). On first predict per ticker, an LSTM is trained and cached (tiny LRU to control memory).
- Search: uses Yahoo Finance search, autocomplete fallback, then local map.
- History: picks sensible defaults per range; interval override allowed (short intervals only for <6m).
- Frontend: plain HTML/JS; Chart.js + chartjs-plugin-zoom.
- Local-first: everything runs on localhost; no credentials are required. If you do host it, ensure TLS and apply rate limits to avoid being blocked by Yahoo for excessive requests.
- Fetch/cache: history and full-history calls are cached briefly in-memory to reduce Yahoo hits; caches auto-evict and expire.
- Training: prediction runs a short training loop (few epochs) with a small validation split and early stopping to stabilize forecasts.
- UI resilience: info/history/prediction load independently; one API hiccup won’t blank the page.
- Hosting (prod): run behind a WSGI server, e.g. `gunicorn "stock_app.app:app"` (Linux/macOS) or `waitress-serve --port=5000 stock_app.app:app` (Windows-friendly). Put nginx/HTTPS in front, set `PORT` if needed, and add simple rate limiting to avoid Yahoo throttling.
- Predictions are experimental and not investment advice; treat outputs as a guide, not ground truth.

## Contributing
- Prefer PRs with small, focused changes.
- Keep dependencies minimal; pin GPU/Metal deps in `requirements-metal.txt` only.
- Tests: none included; if adding, document how to run.
- Coding style: Python 3.11+, Flask; avoid breaking API params. Frontend in vanilla JS; keep UI concise.

## Code ownership
- No CODEOWNERS file yet; if you add one, include maintainers who review API, model, and UI changes separately. For now, please tag maintainers in PR descriptions.
