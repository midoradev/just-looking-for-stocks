# Just looking for stocks
A website + notebook for stocks

## Run (CPU default)
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
