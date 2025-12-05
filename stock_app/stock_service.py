from datetime import datetime, timedelta
from pathlib import Path
from difflib import get_close_matches
import re
import time
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


def predict_prices(ticker: str, days: int, range_key: str | None = None, interval: str | None = None) -> dict:
    # Try requested window/interval first; fall back to full daily history if not enough data.
    try:
        df, eff_interval = get_history(ticker, days, range_key=range_key, interval=interval)
    except Exception:
        df, eff_interval = _get_full_history(ticker).reset_index(), "1d"
    else:
        if len(df) < MIN_PRED_POINTS:
            try:
                df_full = _get_full_history(ticker).reset_index()
                if len(df_full) > len(df):
                    df, eff_interval = df_full, "1d"
            except Exception:
                pass

    dates = _format_date_column(df, eff_interval).reset_index(drop=True)
    close_series = df["Close"].reset_index(drop=True)
    dataset = close_series.values.astype(np.float32).reshape(-1, 1)
    if len(dataset) < MIN_PRED_POINTS:
        raise ValueError("Not enough historical data to build a prediction window yet.")

    lookback = _choose_lookback(len(dataset))
    training_data_len = _compute_train_len(len(dataset), lookback)

    model, scaler = load_or_train_model(ticker, dataset, lookback)

    scaled_data = scaler.transform(dataset).astype(np.float32)
    test_data = scaled_data[training_data_len - lookback :, :]

    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(lookback, len(test_data)):
        x_test.append(test_data[i - lookback : i, 0])
    x_test = np.array(x_test, dtype=np.float32)
    if x_test.size == 0:
        raise ValueError("Not enough data to build a prediction window.")
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))

    valid = pd.DataFrame(
        {
            "Date": dates.iloc[training_data_len:].reset_index(drop=True),
            "Close": close_series.iloc[training_data_len:].reset_index(drop=True),
        }
    )
    valid["Prediction"] = predictions[:, 0]

    return {
        "rmse": rmse,
        "predicted_next_close": float(predictions[-1, 0]),
        "validation": valid[["Date", "Close", "Prediction"]].to_dict(orient="records"),
        "interval": eff_interval,
    }
