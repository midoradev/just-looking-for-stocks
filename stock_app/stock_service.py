from datetime import datetime, timedelta
from pathlib import Path
from difflib import get_close_matches
import re
import time
import math
from urllib.parse import urlencode
from collections import OrderedDict
import gc

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from keras import backend as K
from keras.layers import Dense, LSTM, Input
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Map common company names to tickers
STOCK_MAP = {
    "apple": "AAPL",
    "google": "GOOG",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "tsmc": "TSM",
    "meta": "META",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "asml": "ASML",
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR.parent / "models").resolve()
MODEL_DIR.mkdir(exist_ok=True)
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
YAHOO_AUTOC_URL = "https://autoc.finance.yahoo.com/autoc"
YAHOO_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.\-^=]{1,10}$")
MODEL_CACHE: "OrderedDict[tuple[str, int], tuple[Sequential, MinMaxScaler]]" = OrderedDict()
MAX_MODEL_CACHE = 2
MIN_PRED_POINTS = 22  # minimum closes required to attempt a forecast
HISTORY_CACHE: "OrderedDict[tuple[str, int, str, str], tuple[float, pd.DataFrame, str]]" = OrderedDict()
HISTORY_CACHE_TTL = 300  # seconds
HISTORY_CACHE_MAX = 12
FULL_HISTORY_CACHE: "OrderedDict[str, tuple[float, pd.DataFrame]]" = OrderedDict()
FULL_HISTORY_TTL = 3600  # seconds
FULL_HISTORY_MAX = 12


def find_ticker(user_input: str) -> str:
    user_input = user_input.strip()
    if not user_input:
        raise ValueError("Stock not found. Try a valid company name or ticker.")

    lower = user_input.lower()
    if user_input.upper() in STOCK_MAP.values():
        return user_input.upper()
    matches = get_close_matches(lower, STOCK_MAP.keys(), n=1, cutoff=0.5)
    if matches:
        return STOCK_MAP[matches[0]]

    # If the input already looks like a ticker, return it directly.
    if SYMBOL_PATTERN.match(user_input):
        return user_input.upper()

    # Fallback: search the wider market by company name.
    matches = search_symbols(user_input, limit=5)
    if matches:
        return matches[0]["symbol"]
    raise ValueError("Stock not found. Try a valid company name or ticker.")


def list_symbols() -> list[dict]:
    """Return a list of symbols for the UI picker (most actives with fallback)."""
    try:
        most_active = fetch_most_active(count=25)
        if most_active:
            return most_active
    except Exception:
        pass

    seen = set()
    items = []
    for name, sym in STOCK_MAP.items():
        if sym in seen:
            continue
        seen.add(sym)
        items.append({"symbol": sym, "name": name.title()})
    # keep consistent order
    return sorted(items, key=lambda x: x["symbol"])


def fetch_most_active(count: int = 25) -> list[dict]:
    """Fetch most active stocks from Yahoo Finance screener."""
    headers = {"User-Agent": "Mozilla/5.0 (stock-viewer)"}
    params = {
        "scrIds": "most_actives",
        "count": count,
        "lang": "en-US",
        "region": "US",
        "formatted": "false",
    }
    resp = requests.get(YAHOO_SCREENER_URL, headers=headers, params=params, timeout=6)
    resp.raise_for_status()
    data = resp.json() or {}
    quotes = data.get("finance", {}).get("result", [])
    if quotes:
        quotes = quotes[0].get("quotes", [])
    results = []
    for q in quotes:
        symbol = q.get("symbol")
        name = q.get("shortName") or q.get("longName") or symbol
        if symbol:
            results.append({"symbol": symbol.upper(), "name": name})
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for row in results:
        sym = row["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        unique.append(row)
    return unique[:count]


def _dedupe(rows: list[dict], limit: int) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        sym = row["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        unique.append(row)
        if len(unique) >= limit:
            break
    return unique


def _search_primary(query: str, limit: int, headers: dict) -> list[dict]:
    params = {"q": query, "quotesCount": limit, "newsCount": 0, "listsCount": 0}
    resp = requests.get(f"{YAHOO_SEARCH_URL}?{urlencode(params)}", headers=headers, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for item in data.get("quotes", []):
        quote_type = (item.get("quoteType") or "").lower()
        if quote_type and quote_type not in {"equity", "etf", "mutualfund"}:
            continue
        rows.append(
            {
                "symbol": item.get("symbol"),
                "name": item.get("longname") or item.get("shortname") or item.get("symbol"),
            }
        )
    return rows


def _search_autoc(query: str, headers: dict) -> list[dict]:
    params = {"query": query, "region": "1", "lang": "en"}
    resp = requests.get(f"{YAHOO_AUTOC_URL}?{urlencode(params)}", headers=headers, timeout=6)
    resp.raise_for_status()
    data = resp.json() or {}
    rows = []
    for item in data.get("ResultSet", {}).get("Result", []):
        quote_type = (item.get("typeDisp") or "").lower()
        if quote_type and quote_type not in {"equity", "etf", "fund", "mutualfund"}:
            continue
        rows.append({"symbol": item.get("symbol"), "name": item.get("name")})
    return rows


def search_symbols(query: str, limit: int = 15) -> list[dict]:
    """Search Yahoo Finance for symbols matching the query with a local and autoc fallback."""
    query = query.strip()
    if not query:
        return []

    headers = {"User-Agent": "Mozilla/5.0 (stock-viewer)"}
    collected: list[dict] = []

    try:
        collected.extend(_search_primary(query, limit, headers))
    except Exception:
        collected = []

    if not collected:
        try:
            collected.extend(_search_autoc(query, headers))
        except Exception:
            collected = []

    if not collected:
        if SYMBOL_PATTERN.match(query):
            collected.append({"symbol": query.upper(), "name": query.upper()})
        fuzzy = get_close_matches(query.lower(), STOCK_MAP.keys(), n=5, cutoff=0.4)
        for name in fuzzy:
            collected.append({"symbol": STOCK_MAP[name], "name": name.title()})

    return _dedupe(collected, limit)


def get_stock_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    return {
        "symbol": ticker,
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "marketCap": info.get("marketCap"),
        "currentPrice": info.get("currentPrice"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "website": info.get("website"),
    }


SHORT_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "1h", "1d"}
LONG_INTERVALS = {"1d", "5d", "1wk", "1mo", "3mo"}
YF_TIMEOUT = 6


def _choose_interval(range_key: str | None, requested: str | None) -> str:
    defaults = {
        "1d": "1m",
        "1w": "5m",
        "1m": "1d",
        "3m": "1d",
        "6m": "1d",
    }
    default = defaults.get(range_key or "", "1d")
    if not requested:
        return default
    if range_key in {"1d", "1w", "1m", "3m", "6m"}:
        return requested if requested in SHORT_INTERVALS else default
    return requested if requested in LONG_INTERVALS else default


def _choose_period(range_key: str | None, days: int, interval: str) -> dict:
    """Return kwargs for yf history: either period or start/end."""
    short_map = {
        "1d": "1d",
        "1w": "7d",
        "1m": "1mo",
        "3m": "3mo",
        "6m": "6mo",
    }
    period = short_map.get(range_key or "")
    if interval in SHORT_INTERVALS and period:
        return {"period": period}
    # fall back to start/end for long windows
    end = datetime.now()
    start = end - timedelta(days=days)
    return {"start": start, "end": end}


def get_history(
    ticker: str, days: int, range_key: str | None = None, interval: str | None = None
) -> tuple[pd.DataFrame, str]:
    """Return historical prices with flexible interval selection."""
    ticker_upper = ticker.upper()
    cache_key = (ticker_upper, days, range_key or "", interval or "")
    now = time.time()
    cached = HISTORY_CACHE.get(cache_key)
    if cached:
        ts, cached_df, cached_interval = cached
        if now - ts < HISTORY_CACHE_TTL:
            HISTORY_CACHE.move_to_end(cache_key)
            return cached_df.copy(), cached_interval
        HISTORY_CACHE.pop(cache_key, None)

    stock = yf.Ticker(ticker)
    eff_interval = _choose_interval(range_key, interval)
    kwargs = _choose_period(range_key, days, eff_interval)
    try:
        df = stock.history(interval=eff_interval, timeout=YF_TIMEOUT, **kwargs)
    except Exception:
        # Retry with default if the requested interval is unsupported for this range.
        eff_interval = _choose_interval(range_key, None)
        kwargs = _choose_period(range_key, days, eff_interval)
        df = stock.history(interval=eff_interval, timeout=YF_TIMEOUT, **kwargs)
    if df.empty:
        try:
            # Fall back to a longer daily window to avoid empty responses.
            df = _get_full_history(ticker).reset_index()
            eff_interval = "1d"
        except Exception as exc:
            raise ValueError("No data available for this ticker and range.") from exc
    df = df.reset_index()
    # Use datetime for intraday-like intervals, date for longer windows.
    if eff_interval in {"1m", "2m", "5m", "15m", "30m", "1h"} and "Datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["Datetime"]).dt.strftime("%Y-%m-%d %H:%M")
    else:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    final_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    HISTORY_CACHE[cache_key] = (now, final_df.copy(), eff_interval)
    HISTORY_CACHE.move_to_end(cache_key)
    while len(HISTORY_CACHE) > HISTORY_CACHE_MAX:
        HISTORY_CACHE.popitem(last=False)
    return final_df, eff_interval


def _get_full_history(ticker: str) -> pd.DataFrame:
    key = ticker.upper()
    now = time.time()
    cached = FULL_HISTORY_CACHE.get(key)
    if cached:
        ts, df_cached = cached
        if now - ts < FULL_HISTORY_TTL:
            FULL_HISTORY_CACHE.move_to_end(key)
            return df_cached.copy()
        FULL_HISTORY_CACHE.pop(key, None)

    stock = yf.Ticker(key)
    df = stock.history(start="2012-01-01", end=datetime.now(), timeout=YF_TIMEOUT)
    if df.empty:
        # Try Yahoo's full span if the bounded query returns nothing.
        df = stock.history(period="max", timeout=YF_TIMEOUT)
    if df.empty:
        raise ValueError("No historical data available for this ticker.")

    FULL_HISTORY_CACHE[key] = (now, df.copy())
    FULL_HISTORY_CACHE.move_to_end(key)
    while len(FULL_HISTORY_CACHE) > FULL_HISTORY_MAX:
        FULL_HISTORY_CACHE.popitem(last=False)
    return df


def _warmup_predict(model, lookback: int):
    """Prime the model predict function to avoid repeated retracing warnings."""
    try:
        dummy = np.zeros((1, lookback, 1), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception:
        # If warmup fails, skip without breaking the main flow.
        pass


def _cache_model(ticker: str, lookback: int, model: Sequential, scaler: MinMaxScaler):
    """Store the model with a tiny LRU cache to keep memory in check."""
    key = (ticker, lookback)
    MODEL_CACHE[key] = (model, scaler)
    MODEL_CACHE.move_to_end(key)
    while len(MODEL_CACHE) > MAX_MODEL_CACHE:
        _, (old_model, _) = MODEL_CACHE.popitem(last=False)
        try:
            old_model.reset_states()
        except Exception:
            pass
        del old_model
        gc.collect()


def load_or_train_model(ticker: str, dataset: np.ndarray, lookback: int):
    dataset = dataset.astype(np.float32)
    key = (ticker, lookback)
    if key in MODEL_CACHE:
        MODEL_CACHE.move_to_end(key)
        return MODEL_CACHE[key]

    # Prefer lookback-specific files; fall back to legacy names when lookback is 60.
    model_path = MODEL_DIR / f"{ticker}_lb{lookback}_model.keras"
    scaler_path = MODEL_DIR / f"{ticker}_lb{lookback}_scaler.pkl"
    legacy_model_path = MODEL_DIR / f"{ticker}_model.keras"
    legacy_scaler_path = MODEL_DIR / f"{ticker}_scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        _cache_model(ticker, lookback, model, scaler)
        _warmup_predict(model, lookback)
        return model, scaler
    if lookback == 60 and legacy_model_path.exists() and legacy_scaler_path.exists():
        model = load_model(legacy_model_path)
        scaler = joblib.load(legacy_scaler_path)
        _cache_model(ticker, lookback, model, scaler)
        _warmup_predict(model, lookback)
        return model, scaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset).astype(np.float32)
    training_data_len = _compute_train_len(len(dataset), lookback)
    train_data = scaled_data[0:training_data_len, :]

    x_train, y_train = [], []
    for i in range(lookback, len(train_data)):
        x_train.append(train_data[i - lookback : i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    val_split = 0.1 if len(x_train) > 20 else 0.0
    callbacks = []
    if val_split > 0:
        callbacks.append(EarlyStopping(patience=1, restore_best_weights=True, monitor="val_loss"))
    model.fit(
        x_train,
        y_train,
        batch_size=1,
        epochs=5,
        verbose=0,
        shuffle=False,
        validation_split=val_split,
        callbacks=callbacks,
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    _cache_model(ticker, lookback, model, scaler)
    _warmup_predict(model, lookback)
    return model, scaler


def _format_date_column(df: pd.DataFrame, interval: str) -> pd.Series:
    if "Date" in df.columns:
        col = df["Date"]
    elif "Datetime" in df.columns:
        col = df["Datetime"]
    else:
        col = df.index
    if interval in {"1m", "2m", "5m", "15m", "30m", "1h"}:
        return pd.to_datetime(col).dt.strftime("%Y-%m-%d %H:%M")
    return pd.to_datetime(col).dt.strftime("%Y-%m-%d")


def _next_timestamp_label(last_label: str, interval: str) -> str:
    """Return the next timestamp label in the same format as the input label."""
    try:
        ts = pd.to_datetime(last_label)
    except Exception:
        return last_label
    if pd.isna(ts):
        return last_label
    step_map = {
        "1m": pd.Timedelta(minutes=1),
        "2m": pd.Timedelta(minutes=2),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "5d": pd.Timedelta(days=5),
        "1wk": pd.Timedelta(weeks=1),
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
    }
    step = step_map.get(interval, pd.Timedelta(days=1))
    try:
        next_ts = ts + step
    except Exception:
        next_ts = ts
    has_time = ":" in str(last_label) or interval in {"1m", "2m", "5m", "15m", "30m", "1h"}
    fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
    return next_ts.strftime(fmt)


def _interval_step_delta(interval: str):
    step_map = {
        "1m": pd.Timedelta(minutes=1),
        "2m": pd.Timedelta(minutes=2),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "5d": pd.Timedelta(days=5),
        "1wk": pd.Timedelta(weeks=1),
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
    }
    return step_map.get(interval, pd.Timedelta(days=1))


def _compute_horizon_steps(last_label: str, interval: str) -> int:
    """Return how many future steps to forecast to reach session close for intraday."""
    try:
        ts = pd.to_datetime(last_label)
    except Exception:
        return 5
    if pd.isna(ts):
        return 5
    intraday = {"1m", "2m", "5m", "15m", "30m", "1h"}
    if interval not in intraday:
        return 5
    close_ts = ts.normalize() + pd.Timedelta(hours=16)
    if ts >= close_ts:
        return 5
    remaining_minutes = (close_ts - ts).total_seconds() / 60
    step = _interval_step_delta(interval)
    if isinstance(step, pd.Timedelta):
        step_minutes = step / pd.Timedelta(minutes=1)
    elif isinstance(step, pd.DateOffset):
        # DateOffset isn't expected for intraday, but fall back to its n value if present.
        step_minutes = (step.n or 1) * 24 * 60
    else:
        step_minutes = 60
    if not step_minutes or step_minutes <= 0:
        return 5
    steps = math.ceil(remaining_minutes / step_minutes)
    return max(3, min(int(steps), 600))  # cap to avoid runaway while covering a full session


def _multi_step_forecast(
    scaled_data: np.ndarray, model: Sequential, scaler: MinMaxScaler, lookback: int, steps: int
) -> list[float]:
    """Iteratively forecast multiple steps ahead from the last lookback window."""
    window = scaled_data[-lookback:, :].copy()
    forecasts: list[float] = []
    for _ in range(steps):
        pred_scaled = model.predict(np.reshape(window, (1, lookback, 1)).astype(np.float32), verbose=0)
        pred_val = float(scaler.inverse_transform(pred_scaled)[0, 0])
        forecasts.append(pred_val)
        # slide the window forward with the predicted scaled value
        window = np.vstack([window[1:], pred_scaled.reshape(1, 1)])
    return forecasts


def _choose_lookback(length: int) -> int:
    if length >= 100:
        return 60
    if length >= 80:
        return 50
    if length >= 60:
        return 40
    if length >= 40:
        return 30
    return 20


def _compute_train_len(length: int, lookback: int) -> int:
    base = int(np.ceil(length * 0.9))
    train_len = min(length - 1, max(base, lookback + 1))
    if train_len <= lookback and length > lookback + 1:
        train_len = length - 1
    return train_len


def _price_field_label(price_field: str) -> str:
    labels = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "current": "Current",
    }
    return labels.get(price_field.lower(), "Close")


def _select_price_series(df: pd.DataFrame, price_field: str) -> pd.Series:
    mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "current": "Close",  # Current price aligns to the latest close in history
    }
    col = mapping.get(price_field.lower(), "Close")
    if col not in df.columns:
        raise ValueError("Requested price field is not available in the dataset.")
    return df[col]


def predict_prices(
    ticker: str,
    days: int,
    range_key: str | None = None,
    interval: str | None = None,
    price_field: str = "close",
) -> dict:
    # Try requested window/interval first; fall back to full daily history only if nothing returns.
    try:
        df, eff_interval = get_history(ticker, days, range_key=range_key, interval=interval)
    except Exception:
        df, eff_interval = _get_full_history(ticker).reset_index(), "1d"

    dates = _format_date_column(df, eff_interval).reset_index(drop=True)
    price_series = _select_price_series(df, price_field).reset_index(drop=True)
    dataset = price_series.values.astype(np.float32).reshape(-1, 1)
    using_full_for_training = False
    if len(dataset) < MIN_PRED_POINTS:
        # Train on full history, but return predictions sliced to the requested window.
        df_full = _get_full_history(ticker).reset_index()
        dates_full = _format_date_column(df_full, "1d").reset_index(drop=True)
        price_series_full = _select_price_series(df_full, price_field).reset_index(drop=True)
        dataset_full = price_series_full.values.astype(np.float32).reshape(-1, 1)
        if len(dataset_full) < MIN_PRED_POINTS:
            raise ValueError("Not enough historical data to build a prediction window yet.")
        dates = dates_full
        price_series = price_series_full
        dataset = dataset_full
        eff_interval = "1d"
        using_full_for_training = True

    lookback = _choose_lookback(len(dataset))
    training_data_len = _compute_train_len(len(dataset), lookback)

    model, scaler = load_or_train_model(ticker, dataset, lookback)

    scaled_data = scaler.transform(dataset).astype(np.float32)

    # Predict across the whole window (minus the initial lookback) so charts stay aligned to the selected range.
    preds_full = []
    for i in range(lookback, len(dataset)):
        window = scaled_data[i - lookback : i, :]
        window = np.reshape(window, (1, lookback, 1)).astype(np.float32)
        pred = model.predict(window, verbose=0)
        preds_full.append(pred[0, 0])
    if not preds_full:
        raise ValueError("Not enough data to build a prediction window.")
    preds_full = np.array(preds_full).reshape(-1, 1)
    preds_full = scaler.inverse_transform(preds_full)[:, 0]

    # Bias-correct predictions so they stay anchored to the latest observed price.
    last_actual = float(price_series.iloc[-1])
    last_pred = float(preds_full[-1])
    bias = last_actual - last_pred
    preds_full = preds_full + bias

    horizon = _compute_horizon_steps(str(dates.iloc[-1]), eff_interval)
    raw_forecasts = _multi_step_forecast(scaled_data, model, scaler, lookback, max(1, horizon))
    forecast_values = [float(v + bias) for v in raw_forecasts]
    forecast_labels = []
    label_cursor = str(dates.iloc[-1])
    for _ in forecast_values:
        label_cursor = _next_timestamp_label(label_cursor, eff_interval)
        forecast_labels.append(label_cursor)

    predicted_next_value = float(forecast_values[0])
    next_date_label = forecast_labels[0] if forecast_labels else _next_timestamp_label(str(dates.iloc[-1]), eff_interval)
    predicted_close_value = float(forecast_values[-1])
    forecast_path = [
        {"Date": forecast_labels[i], "Prediction": float(forecast_values[i])}
        for i in range(len(forecast_values))
    ]

    # RMSE on validation portion only.
    val_start = max(training_data_len, lookback)
    y_true = price_series.iloc[val_start:].to_numpy()
    y_hat = preds_full[(val_start - lookback) :]
    rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2))) if len(y_true) and len(y_hat) else 0.0

    valid = pd.DataFrame(
        {
            "Date": dates.iloc[lookback:].reset_index(drop=True),
            "Actual": price_series.iloc[lookback:].reset_index(drop=True),
            "Prediction": preds_full,
        }
    )

    if using_full_for_training and len(df) > 0:
        # Slice predictions to requested window length so charts stay in sync with the selected range.
        tail_len = len(df)
        if tail_len > len(valid):
            tail_len = len(valid)
        valid = valid.tail(tail_len).reset_index(drop=True)

    payload = {
        "rmse": rmse,
        "predicted_next_price": predicted_next_value,
        "predicted_next_date": next_date_label,
        "predicted_close_price": predicted_close_value,
        "next_point": {"Date": next_date_label, "Prediction": predicted_next_value},
        "forecast_path": forecast_path,
        "validation": valid.to_dict(orient="records"),
        "interval": eff_interval,
        "price_field": price_field.lower(),
        "price_field_label": _price_field_label(price_field),
    }
    # Preserve backwards compatibility for consumers still looking for the close-specific key.
    if price_field.lower() == "close":
        payload["predicted_next_close"] = predicted_next_value
    return payload
