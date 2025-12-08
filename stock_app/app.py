import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .stock_service import (
    find_ticker,
    get_history,
    get_stock_info,
    list_symbols,
    predict_prices,
    search_symbols,
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "stocks_dashboard"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
# Disable static caching so changes are picked up immediately after restart.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
CORS(app)

RANGE_MAP = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "ytd": "ytd",
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "all": 4000,
}


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/info")
def api_info():
    ticker_input = request.args.get("ticker", "").strip()
    if not ticker_input:
        return jsonify({"error": "ticker is required"}), 400
    try:
        ticker = find_ticker(ticker_input)
        info = get_stock_info(ticker)
        return jsonify(info)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.route("/api/history")
def api_history():
    ticker_input = request.args.get("ticker", "").strip()
    range_key = request.args.get("range", "1m")
    interval = request.args.get("interval", "").strip().lower() or None
    if not ticker_input:
        return jsonify({"error": "ticker is required"}), 400

    days_value = RANGE_MAP.get(range_key, RANGE_MAP["1m"])
    if days_value == "ytd":
        today = datetime.now().date()
        year_start = datetime(year=today.year, month=1, day=1).date()
        days = max((today - year_start).days + 1, 1)
    else:
        days = int(days_value)
    try:
        ticker = find_ticker(ticker_input)
        df, eff_interval = get_history(ticker, days, range_key=range_key, interval=interval)
        return jsonify({"symbol": ticker, "range": range_key, "interval": eff_interval, "prices": df.to_dict(orient="records")})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict")
def api_predict():
    ticker_input = request.args.get("ticker", "").strip()
    range_key = request.args.get("range", "1m")
    interval = request.args.get("interval", "").strip().lower() or None
    price_field = request.args.get("price_field", "close").strip().lower() or "close"
    if not ticker_input:
        return jsonify({"error": "ticker is required"}), 400

    days_value = RANGE_MAP.get(range_key, RANGE_MAP["1m"])
    if days_value == "ytd":
        today = datetime.now().date()
        year_start = datetime(year=today.year, month=1, day=1).date()
        days = max((today - year_start).days + 1, 1)
    else:
        days = int(days_value)
    try:
        ticker = find_ticker(ticker_input)
        result = predict_prices(
            ticker,
            days,
            range_key=range_key,
            interval=interval,
            price_field=price_field,
        )
        result["symbol"] = ticker
        result["range"] = range_key
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.route("/api/symbols")
def api_symbols():
    return jsonify(list_symbols())


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    try:
        return jsonify(search_symbols(query))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    # Bind to all interfaces so it works with localhost/127.0.0.1 and over LAN if needed.
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
