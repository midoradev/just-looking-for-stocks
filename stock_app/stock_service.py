from datetime import datetime, timedelta
from pathlib import Path
from difflib import get_close_matches
import re
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
MODEL_CACHE: "OrderedDict[str, tuple[Sequential, MinMaxScaler]]" = OrderedDict()
MAX_MODEL_CACHE = 2


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
    stock = yf.Ticker(ticker)
    eff_interval = _choose_interval(range_key, interval)
    kwargs = _choose_period(range_key, days, eff_interval)
    try:
        df = stock.history(interval=eff_interval, **kwargs)
    except Exception:
        # Retry with default if the requested interval is unsupported for this range.
        eff_interval = _choose_interval(range_key, None)
        kwargs = _choose_period(range_key, days, eff_interval)
        df = stock.history(interval=eff_interval, **kwargs)
    if df.empty:
        raise ValueError("No data available for this ticker and range.")
    df = df.reset_index()
    # Use datetime for intraday-like intervals, date for longer windows.
    if eff_interval in {"1m", "2m", "5m", "15m", "30m", "1h"} and "Datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["Datetime"]).dt.strftime("%Y-%m-%d %H:%M")
    else:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]], eff_interval


def _get_full_history(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start="2012-01-01", end=datetime.now())
    if df.empty:
        raise ValueError("No historical data available for this ticker.")
    return df


def _warmup_predict(model):
    """Prime the model predict function to avoid repeated retracing warnings."""
    try:
        dummy = np.zeros((1, 60, 1), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception:
        # If warmup fails, skip without breaking the main flow.
        pass


def _cache_model(ticker: str, model: Sequential, scaler: MinMaxScaler):
    """Store the model with a tiny LRU cache to keep memory in check."""
    MODEL_CACHE[ticker] = (model, scaler)
    MODEL_CACHE.move_to_end(ticker)
    while len(MODEL_CACHE) > MAX_MODEL_CACHE:
        _, (old_model, _) = MODEL_CACHE.popitem(last=False)
        try:
            old_model.reset_states()
        except Exception:
            pass
        del old_model
        gc.collect()


def load_or_train_model(ticker: str, dataset: np.ndarray):
    dataset = dataset.astype(np.float32)
    if ticker in MODEL_CACHE:
        MODEL_CACHE.move_to_end(ticker)
        return MODEL_CACHE[ticker]

    model_path = MODEL_DIR / f"{ticker}_model.keras"
    scaler_path = MODEL_DIR / f"{ticker}_scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        _cache_model(ticker, model, scaler)
        _warmup_predict(model)
        return model, scaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset).astype(np.float32)
    training_data_len = int(np.ceil(len(dataset) * 0.95))
    train_data = scaled_data[0:training_data_len, :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
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
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    _cache_model(ticker, model, scaler)
    _warmup_predict(model)
    return model, scaler


def predict_prices(ticker: str) -> dict:
    df = _get_full_history(ticker)
    data = df[["Close"]]
    dataset = data.values.astype(np.float32)
    if len(dataset) < 80:
        raise ValueError("Not enough data to build a prediction window.")

    model, scaler = load_or_train_model(ticker, dataset)

    scaled_data = scaler.transform(dataset).astype(np.float32)
    training_data_len = int(np.ceil(len(dataset) * 0.95))
    test_data = scaled_data[training_data_len - 60 :, :]

    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])
    x_test = np.array(x_test, dtype=np.float32)
    if x_test.size == 0:
        raise ValueError("Not enough data to build a prediction window.")
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))

    valid = data[training_data_len:].copy()
    valid["Prediction"] = predictions[:, 0]
    valid = valid.reset_index()
    valid["Date"] = valid["Date"].dt.strftime("%Y-%m-%d")

    return {
        "rmse": rmse,
        "predicted_next_close": float(predictions[-1, 0]),
        "validation": valid[["Date", "Close", "Prediction"]].to_dict(orient="records"),
    }
