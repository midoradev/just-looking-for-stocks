const state = {
  ticker: "AAPL",
  range: "1m",
  interval: "auto",
  priceField: "close",
  showFullHistory: false,
  showFullPred: false,
  historyData: [],
  validationData: [],
  forecastPath: [],
  nextPoint: null,
  theme: "light",
};
if (globalThis.ChartZoom) {
  Chart.register(globalThis.ChartZoom);
}
const crosshairPlugin = {
  id: "crosshair",
  afterDatasetsDraw(chart) {
    const enabled = chart.options?.plugins?.crosshair?.enabled;
    if (!enabled) return;
  const active = chart.getActiveElements?.();
  if (!active?.length) return;
  const { ctx, chartArea } = chart;
  const { left, right, top, bottom } = chartArea;
  const { element } = active[0];
  const x = element.x;
  const y = element.y;
    const { datasetIndex, index } = active[0];
    const ds = chart.data.datasets?.[datasetIndex];
    const val = ds?.data?.[index];
    const price = typeof val === "object" && val !== null ? val.y ?? val : val;
  const label = chart.data.labels?.[index];
  const styles = getComputedStyle(document.documentElement);
  const textColor = styles.getPropertyValue("--text").trim() || "#1b2333";
  const panelColor = styles.getPropertyValue("--panel").trim() || "#ffffff";
  const borderColor = styles.getPropertyValue("--border").trim() || "#d7deea";
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = borderColor || "#94a3b8";
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    if (label) {
      const labelText = String(label);
      const padding = 6;
      const height = 20;
      ctx.font = "12px Inter, system-ui";
      const width = ctx.measureText(labelText).width + padding * 2;
      const lx = Math.min(Math.max(x - width / 2, left + 2), right - width - 2);
      const ly = top + 4;
      ctx.fillStyle = panelColor || "#f8fafc";
      ctx.strokeStyle = borderColor || "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(lx, ly, width, height, 4);
      } else {
        ctx.rect(lx, ly, width, height);
      }
      ctx.fillStyle = panelColor || "#f8fafc";
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.fillText(labelText, lx + padding, ly + height - 6);
    }
    if (price !== undefined) {
      const txt = typeof price === "number" ? price.toLocaleString("en-US", { maximumFractionDigits: 2 }) : String(price);
      const padding = 6;
      const height = 20;
      ctx.font = "12px Inter, system-ui";
      const width = ctx.measureText(txt).width + padding * 2;
      const rx = right - width - 6;
      const ry = y - height / 2;
      ctx.fillStyle = panelColor || "#f8fafc";
      ctx.strokeStyle = borderColor || "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(rx, ry, width, height, 4);
      } else {
        ctx.rect(rx, ry, width, height);
      }
      ctx.fillStyle = panelColor || "#f8fafc";
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.fillText(txt, rx + padding, ry + height - 6);
    }
    ctx.restore();
  },
};
Chart.register(crosshairPlugin);
let quickPickList = [];
let priceChart, predChart;
let searchTimer = null;
let searchSeq = 0;
let activeLoadToken = 0;

const tickerInput = document.getElementById("tickerInput");
const rangeTabs = document.getElementById("rangeTabs");
const intervalSelect = document.getElementById("intervalSelect");
const priceFieldSelect = document.getElementById("priceFieldSelect");
const toggleHistoryRowsBtn = document.getElementById("toggleHistoryRows");
const togglePredRowsBtn = document.getElementById("togglePredRows");
const themeToggle = document.getElementById("themeToggle");
rangeTabs.addEventListener("click", (e) => {
  const btn = e.target.closest(".range-btn");
  if (!btn) return;
  for (const b of document.querySelectorAll(".range-btn")) {
    b.classList.remove("active");
  }
  btn.classList.add("active");
  state.range = btn.dataset.range;
  loadAll();
});

intervalSelect.addEventListener("change", () => {
  state.interval = intervalSelect.value;
  loadAll();
});

priceFieldSelect.addEventListener("change", () => {
  state.priceField = priceFieldSelect.value;
  loadAll();
});

toggleHistoryRowsBtn.addEventListener("click", () => {
  state.showFullHistory = !state.showFullHistory;
  renderHistoryTable(state.historyData);
});

togglePredRowsBtn.addEventListener("click", () => {
  state.showFullPred = !state.showFullPred;
  renderPredTable(state.validationData, state.forecastPath);
});

function applyTheme(theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
  themeToggle.textContent = theme === "dark" ? "Switch to light" : "Switch to dark";
}

themeToggle.addEventListener("click", () => {
  state.theme = state.theme === "dark" ? "light" : "dark";
  applyTheme(state.theme);
  // Rebuild charts/tables so chart styling matches the active theme.
  loadAll();
});
document.getElementById("searchForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const val = tickerInput.value.trim();
  if (!val) return;
  state.ticker = val.toUpperCase();
  loadAll();
});

tickerInput.addEventListener("input", (e) => {
  const val = e.target.value.trim();
  clearTimeout(searchTimer);
  if (val.length < 2) {
    searchSeq += 1;
    renderOptions(quickPickList);
    return;
  }
  searchTimer = setTimeout(() => {
    runSearch(val);
  }, 200);
});

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function formatUSD(num) {
  if (num === null || num === undefined || Number.isNaN(num)) return "—";
  return "$" + Number(num).toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function formatDateLabel(range, value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const pad = (n) => String(n).padStart(2, "0");
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const yearShort = String(date.getFullYear()).slice(-2);
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const hasTime = value.includes(":");
  const shortRanges = new Set(["1d", "1w", "1m", "3m", "6m", "ytd"]);
  if (shortRanges.has(range)) {
    return hasTime ? `${month}/${day} ${hours}:${minutes}` : `${month}/${day}`;
  }
  return `${yearShort}/${month}/${day}`;
}

function setCursor(el, cursor) {
  if (el) el.style.cursor = cursor;
}

function priceFieldLabel(value) {
  switch ((value || "").toLowerCase()) {
    case "open":
      return "Open";
    case "high":
      return "High";
    case "low":
      return "Low";
    case "current":
      return "Current";
    case "close":
    default:
      return "Close";
  }
}

function pickPrice(row, field) {
  const key = (field || "").toLowerCase();
  if (key === "open") return row.Open;
  if (key === "high") return row.High;
  if (key === "low") return row.Low;
  // "current" maps to the latest traded/close price in the history payload
  return row.Close;
}

function trendClass(current, previous) {
  if (current === null || current === undefined || previous === null || previous === undefined) return "";
  if (Number.isNaN(current) || Number.isNaN(previous)) return "";
  if (current > previous) return "price-up";
  if (current < previous) return "price-down";
  return "";
}

function applyPriceForecast(forecastPoints) {
  if (!priceChart || !forecastPoints?.length) return;
  const base = priceChart.data.datasets?.[0];
  if (!base) return;
  const originalLabels = [...priceChart.data.labels];
  const originalData = Array.isArray(base.data) ? [...base.data] : [];
  const futurePoints = forecastPoints
    .filter((p) => p?.Date && p?.Prediction !== null && p?.Prediction !== undefined)
    .map((p) => ({ label: formatDateLabel(state.range, p.Date), value: p.Prediction }));
  if (!futurePoints.length) return;

  const labels = [...originalLabels];
  for (const { label } of futurePoints) {
    if (!labels.includes(label)) {
      labels.push(label);
    }
  }
  if (labels.length < 2) return;

  const actualData = labels.map((_, idx) => (idx < originalData.length ? originalData[idx] : null));
  const forecastData = labels.map(() => null);
  const lastActualIdx = Math.max(0, originalData.length - 1);
  const lastActualValue = originalData[lastActualIdx];
  if (lastActualValue !== null && lastActualValue !== undefined) {
    forecastData[lastActualIdx] = lastActualValue;
  }

  for (const { label, value } of futurePoints) {
    const idx = labels.indexOf(label);
    if (idx >= 0) {
      forecastData[idx] = value;
    }
  }

  const finalLabel = futurePoints[futurePoints.length - 1]?.label;
  const pointRadius = labels.map((lbl) => (lbl === finalLabel ? 5 : 0));
  const pointBg = labels.map((lbl) => (lbl === finalLabel ? "#f97316" : "rgba(249,115,22,0.12)"));

  let forecastDs = priceChart.data.datasets.find((ds) => ds._isForecast);
  if (forecastDs) {
    forecastDs.data = forecastData;
    forecastDs.pointRadius = pointRadius;
    forecastDs.pointBackgroundColor = pointBg;
  } else {
    forecastDs = {
      label: "Forecast",
      data: forecastData,
      borderColor: "#f97316",
      backgroundColor: "rgba(249,115,22,0.12)",
      borderDash: [6, 4],
      pointRadius,
      pointBackgroundColor: pointBg,
      tension: 0.25,
      fill: false,
      _isForecast: true,
    };
    priceChart.data.datasets.push(forecastDs);
  }

  base.data = actualData;
  priceChart.data.labels = labels;
  priceChart.update();
}

function zoomOptions(canvas) {
  const styles = getComputedStyle(document.documentElement);
  const textColor = styles.getPropertyValue("--text").trim() || "#1b2333";
  const borderColor = styles.getPropertyValue("--border").trim() || "#d7deea";
  const gridColor = styles.getPropertyValue("--grid")?.trim() || borderColor;
  return {
    textColor,
    borderColor,
    pan: { enabled: true, mode: "xy", threshold: 0 },
    zoom: {
      wheel: { enabled: true, speed: 0.05, modifierKey: null },
      pinch: { enabled: true },
      drag: {
        enabled: true,
        modifierKey: "shift",
        mode: "xy",
        borderColor: "#0052cc",
        backgroundColor: "rgba(0,82,204,0.08)",
        threshold: 6,
      },
      mode: "xy",
    },
    limits: {
      x: { min: "original", max: "original" },
      y: { min: "original", max: "original" },
    },
    scales: {
      x: {
        ticks: { color: textColor, maxRotation: 0 },
        grid: { color: gridColor },
      },
      y: {
        ticks: { color: textColor },
        grid: { color: gridColor },
      },
    },
    legendColor: textColor,
  };
}

const panHandlers = new WeakMap();
function attachPanHandlers(chart, canvas) {
  if (!canvas || !chart || typeof chart.pan !== "function") return;
  const prev = panHandlers.get(canvas);
  if (prev) {
    canvas.removeEventListener("pointerdown", prev.down);
    canvas.removeEventListener("pointermove", prev.move);
    canvas.removeEventListener("pointerup", prev.up);
    canvas.removeEventListener("pointerleave", prev.up);
  }
  let isPanning = false;
  let lastX = 0;
  let lastY = 0;
  let pending = { x: 0, y: 0 };
  let raf = null;
  const flushPan = () => {
    raf = null;
    if (!isPanning) return;
    chart.pan(pending, undefined, "none");
    pending = { x: 0, y: 0 };
  };
  const down = (e) => {
    // Skip if using shift for box-zoom or not primary button.
    if (e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    isPanning = true;
    lastX = e.clientX;
    lastY = e.clientY;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch (error_) {
      console.error(error_);
    }
    setCursor(canvas, "grabbing");
  };
  const move = (e) => {
    if (!isPanning) return;
    e.preventDefault();
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    pending.x += dx;
    pending.y += dy;
    lastX = e.clientX;
    lastY = e.clientY;
    if (!raf) raf = requestAnimationFrame(flushPan);
  };
  const up = (e) => {
    if (!isPanning) return;
    isPanning = false;
    pending = { x: 0, y: 0 };
    setCursor(canvas, "grab");
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch (error_) {
      console.error(error_);
    }
  };
  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointerleave", up);
  panHandlers.set(canvas, { down, move, up });
}

function setError(msg) {
  const el = document.getElementById("error");
  if (msg) {
    el.style.display = "block";
    el.textContent = msg;
  } else {
    el.style.display = "none";
    el.textContent = "";
  }
}

function chartLoaderEl(key) {
  return document.querySelector(`[data-chart-loader="${key}"]`);
}

function showChartLoader(key, title, message) {
  const el = chartLoaderEl(key);
  if (!el) return;
  const titleEl = el.querySelector(".chart-loader__title");
  const msgEl = el.querySelector(".chart-loader__message");
  const spinner = el.querySelector(".chart-loader__spinner");
  if (titleEl && title) titleEl.textContent = title;
  if (msgEl && message) msgEl.textContent = message;
  if (spinner) spinner.style.display = "block";
  el.classList.remove("error");
  el.style.display = "flex";
}

function showChartError(key, message) {
  const el = chartLoaderEl(key);
  if (!el) return;
  const titleEl = el.querySelector(".chart-loader__title");
  const msgEl = el.querySelector(".chart-loader__message");
  if (titleEl) titleEl.textContent = "Unable to load chart";
  if (msgEl) msgEl.textContent = message || "Something went wrong.";
  el.classList.add("error");
  el.style.display = "flex";
}

function hideChartLoader(key) {
  const el = chartLoaderEl(key);
  if (!el) return;
  el.style.display = "none";
  el.classList.remove("error");
}

async function loadInfo(loadToken, ticker) {
  const info = await fetchJSON(`/api/info?ticker=${encodeURIComponent(ticker)}&t=${Date.now()}`);
  if (loadToken !== activeLoadToken) return;
  document.getElementById("infoName").textContent = info.name || "—";
  document.getElementById("infoSymbol").textContent = info.symbol ? info.symbol : "—";
  document.getElementById("infoSector").textContent = info.sector || "—";
  document.getElementById("infoIndustry").textContent = info.industry || "—";
  document.getElementById("infoPrice").textContent = formatUSD(info.currentPrice);
  document.getElementById("infoMarketCap").textContent = info.marketCap ? info.marketCap.toLocaleString("en-US") : "—";
  const range = (info.fiftyTwoWeekLow || "—") + " / " + (info.fiftyTwoWeekHigh || "—");
  document.getElementById("info52w").textContent = range;
  document.getElementById("infoWebsite").textContent = info.website || "—";
}

function renderHistoryTable(data) {
  const tbody = document.querySelector("#historyTable tbody");
  tbody.innerHTML = "";
  const rows = state.showFullHistory ? data : data.slice(-10);
  const startIdx = state.showFullHistory ? 0 : Math.max(0, data.length - rows.length);
  for (let i = 0; i < rows.length; i += 1) {
    const p = rows[i];
    const prev = data[startIdx + i - 1] || null;
    const cls = trendClass(p.Close, prev?.Close);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${p.Date}</td><td>${p.Open}</td><td>${p.High}</td><td>${p.Low}</td><td class="${cls}">${p.Close}</td><td>${p.Volume}</td>`;
    if (cls) tr.classList.add(cls);
    tbody.appendChild(tr);
  }
  const meta = document.getElementById("historyRowsMeta");
  const btn = document.getElementById("toggleHistoryRows");
  meta.textContent = state.showFullHistory ? `Showing all ${data.length}` : `Showing latest ${rows.length} of ${data.length}`;
  btn.textContent = state.showFullHistory ? "Show summary" : "Show all";
}

async function loadHistory(loadToken, ticker, range, interval, priceField) {
  showChartLoader("price", `Preparing ${priceFieldLabel(priceField)} chart…`, "Fetching price history");
  const query = new URLSearchParams({
    ticker,
    range,
    t: Date.now().toString(),
  });
  if (interval && interval !== "auto") {
    query.set("interval", interval);
  }
  try {
    const data = await fetchJSON(`/api/history?${query.toString()}`);
    if (loadToken !== activeLoadToken) return;
    const prices = data.prices || [];
    state.historyData = prices;
    renderHistoryTable(prices);
    let intervalLabel = "auto";
    if (data.interval) {
      intervalLabel = data.interval;
    } else if (interval && interval !== "auto") {
      intervalLabel = interval;
    }
    document.getElementById("priceMeta").textContent = `${data.symbol} • ${range.toUpperCase()} • ${intervalLabel} • ${prices.length} rows`;

    // Build price chart for the selected field
    const labels = prices.map((p) => formatDateLabel(range, p.Date));
    const values = prices.map((p) => pickPrice(p, priceField));
    const fieldLabel = priceFieldLabel(priceField);
    document.getElementById("priceChartLabel").textContent = `Price chart (${fieldLabel})`;
    if (priceChart) priceChart.destroy();
    const priceCanvas = document.getElementById("priceChart");
    const ctx = priceCanvas.getContext("2d");
    const zoomOpts = zoomOptions(priceCanvas);
    priceChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: `${fieldLabel} (${intervalLabel})`,
            data: values,
            borderColor: "#0052cc",
            backgroundColor: "rgba(0,82,204,0.1)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: zoomOpts.scales,
        plugins: {
          legend: { display: true, labels: { color: zoomOpts.legendColor } },
          zoom: { zoom: zoomOpts.zoom, pan: zoomOpts.pan, limits: zoomOpts.limits },
          crosshair: { enabled: true },
        },
      },
    });
    attachPanHandlers(priceChart, priceCanvas);
    const reset = document.getElementById("resetPriceZoom");
    reset.onclick = () => {
      if (priceChart) priceChart.resetZoom();
      setCursor(priceCanvas, "grab");
    };
    applyPriceForecast(state.forecastPath);
    hideChartLoader("price");
  } catch (err) {
    if (loadToken === activeLoadToken) {
      showChartError("price", err?.message || "Unable to load price chart.");
    }
    throw err;
  }
}

function renderPredTable(validation, forecastPath) {
  const tbody = document.querySelector("#predTable tbody");
  tbody.innerHTML = "";
  const baseRows = [...validation].map((row) => {
    const parsed = new Date(row.Date);
    return { ...row, _dateObj: parsed, _raw: row.Date, _isFuture: false, _isFinal: false };
  });
  const futureRows = (forecastPath || []).map((row, idx, arr) => {
    const parsed = new Date(row.Date);
    return {
      Date: row.Date,
      Prediction: row.Prediction,
      Actual: null,
      _dateObj: Number.isNaN(parsed?.getTime?.()) ? new Date(Date.now()) : parsed,
      _raw: row.Date,
      _isFuture: true,
      _isFinal: idx === arr.length - 1,
    };
  });
  const tableRows = [...futureRows, ...baseRows].sort((a, b) => b._dateObj - a._dateObj);
  const rows = state.showFullPred ? tableRows : tableRows.slice(0, 10);
  let prevActual = null;
  let prevPred = null;
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row._isFuture) tr.classList.add(row._isFinal ? "future-row-final" : "future-row");
    const actual = row.Actual === null || row.Actual === undefined ? "—" : formatUSD(row.Actual);
    const prediction = row.Prediction === null || row.Prediction === undefined ? "—" : formatUSD(row.Prediction);
    const actualCls = trendClass(row.Actual, prevActual);
    const predCls = trendClass(row.Prediction, prevPred);
    tr.innerHTML = `<td>${formatDateLabel(state.range, row._raw)}</td><td class="${actualCls}">${actual}</td><td class="${predCls}">${prediction}</td>`;
    tbody.appendChild(tr);
    prevActual = row.Actual !== null && row.Actual !== undefined ? row.Actual : prevActual;
    prevPred = row.Prediction !== null && row.Prediction !== undefined ? row.Prediction : prevPred;
  }
  const meta = document.getElementById("predRowsMeta");
  const btn = document.getElementById("togglePredRows");
  meta.textContent = state.showFullPred ? `Showing all ${tableRows.length}` : `Showing latest ${rows.length} of ${tableRows.length}`;
  btn.textContent = state.showFullPred ? "Show summary" : "Show all";
}

async function loadPrediction(loadToken, ticker, priceField) {
  showChartLoader(
    "pred",
    `Preparing ${priceFieldLabel(priceField)} prediction…`,
    "Training model and forecasting next price",
  );
  const query = new URLSearchParams({
    ticker,
    range: state.range,
    interval: state.interval,
    price_field: priceField,
    t: Date.now().toString(),
  });
  try {
    const data = await fetchJSON(`/api/predict?${query.toString()}`);
    if (loadToken !== activeLoadToken) return;
    const fieldLabel = data.price_field_label || priceFieldLabel(priceField);
    const forecastPath = Array.isArray(data.forecast_path) ? data.forecast_path : [];
    state.forecastPath = forecastPath;
    const nextPoint =
      forecastPath[0] ||
      (data.next_point?.Date ? data.next_point : null) ||
      (data.predicted_next_date
        ? { Date: data.predicted_next_date, Prediction: data.predicted_next_price ?? data.predicted_next_close }
        : null);
    state.nextPoint = nextPoint;
    document.getElementById("predictionLabel").textContent = `Next predicted ${fieldLabel.toLowerCase()}`;
    document.getElementById("predictionValue").textContent = formatUSD(data.predicted_next_price ?? data.predicted_next_close);
    const whenLabel = nextPoint?.Date ? formatDateLabel(state.range, nextPoint.Date) : "—";
    document.getElementById("predictionTime").textContent = nextPoint ? `${whenLabel} (${data.interval || state.interval})` : "—";
    const closeForecast =
      data.predicted_close_price ??
      (forecastPath.length ? forecastPath[forecastPath.length - 1].Prediction : null) ??
      null;
    document.getElementById("predictionCloseValue").textContent =
      closeForecast === null ? "—" : formatUSD(closeForecast);
    document.getElementById("rmseValue").textContent = formatUSD(data.rmse);
    document.getElementById("predActualHeader").textContent = `Actual ${fieldLabel}`;

    const validation = data.validation || [];
    state.validationData = validation;
    renderPredTable(validation, forecastPath);
    applyPriceForecast(forecastPath);

    // Build prediction chart
    const labels = validation.map((d) => formatDateLabel(state.range, d.Date));
    const actual = validation.map((d) => d.Actual);
    const preds = validation.map((d) => d.Prediction);
    if (forecastPath?.length) {
      for (const pt of forecastPath) {
        labels.push(formatDateLabel(state.range, pt.Date));
        actual.push(null);
        preds.push(pt.Prediction ?? null);
      }
    }
    const finalIdx = forecastPath?.length ? labels.length - 1 : -1;
    if (predChart) predChart.destroy();
    const predCanvas = document.getElementById("predChart");
    const ctx = predCanvas.getContext("2d");
    const zoomOpts = zoomOptions(predCanvas);
    const pointRadius = preds.map((_, idx) => (idx === finalIdx && forecastPath?.length ? 4 : 0));
    const pointBg = preds.map((_, idx) => (idx === finalIdx && forecastPath?.length ? "#e11d48" : "rgba(244,161,30,0.12)"));
    predChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: `Actual ${fieldLabel}`,
            data: actual,
            borderColor: "#008f5d",
            backgroundColor: "rgba(0,143,93,0.12)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
            fill: true,
          },
          {
            label: `Predicted ${fieldLabel}`,
            data: preds,
            borderColor: "#f4a11e",
            backgroundColor: "rgba(244,161,30,0.12)",
            borderWidth: 2,
            pointRadius,
            pointBackgroundColor: pointBg,
            tension: 0.25,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: zoomOpts.scales,
        plugins: {
          legend: { display: true, labels: { color: zoomOpts.legendColor } },
          zoom: { zoom: zoomOpts.zoom, pan: zoomOpts.pan, limits: zoomOpts.limits },
          crosshair: { enabled: true },
        },
      },
    });
    attachPanHandlers(predChart, predCanvas);
    const reset = document.getElementById("resetPredZoom");
    reset.onclick = () => {
      if (predChart) predChart.resetZoom();
      setCursor(predCanvas, "grab");
    };
    hideChartLoader("pred");
  } catch (err) {
    if (loadToken === activeLoadToken) {
      showChartError("pred", err?.message || "Unable to load prediction chart.");
    }
    throw err;
  }
}

function renderOptions(list) {
  const dl = document.getElementById("symbolList");
  dl.innerHTML = "";
  for (const { symbol, name } of list) {
    const opt = document.createElement("option");
    opt.value = symbol;
    opt.label = name;
    dl.appendChild(opt);
  }
}

async function loadQuickPicks() {
  try {
    const list = await fetchJSON(`/api/symbols?t=${Date.now()}`);
    quickPickList = list;
    renderOptions(list);
    const qp = document.getElementById("quickPicks");
    qp.innerHTML = "";
    for (const { symbol, name } of list) {
      const chip = document.createElement("div");
      chip.className = "chip";
      chip.innerHTML = `<span>${symbol}</span><span class="muted">${name}</span>`;
      chip.onclick = () => {
        state.ticker = symbol;
        tickerInput.value = symbol;
        loadAll();
      };
      qp.appendChild(chip);
    }
  } catch (err) {
    console.error(err);
    setError("Unable to load symbols right now.");
  }
}

async function runSearch(query) {
  const seq = ++searchSeq;
  try {
    const results = await fetchJSON(`/api/search?q=${encodeURIComponent(query)}&t=${Date.now()}`);
    if (seq !== searchSeq) return;
    renderOptions(results);
  } catch (err) {
    console.error(err);
    setError("Search unavailable right now.");
  }
}

async function loadAll() {
  state.ticker = (tickerInput.value.trim() || state.ticker || "").toUpperCase();
  tickerInput.value = state.ticker;
  state.interval = intervalSelect.value || "auto";
  state.priceField = priceFieldSelect.value || "close";
  activeLoadToken += 1;
  const loadToken = activeLoadToken;
  if (priceChart) {
    priceChart.destroy();
    priceChart = null;
  }
  if (predChart) {
    predChart.destroy();
    predChart = null;
  }
  showChartLoader("price", "Preparing price chart…", "Fetching latest candles");
  showChartLoader("pred", "Preparing prediction chart…", "Training model and forecasting next price");
  document.querySelector("#historyTable tbody").innerHTML = "";
  document.querySelector("#predTable tbody").innerHTML = "";
  document.getElementById("priceMeta").textContent = "Loading...";
  document.getElementById("predictionValue").textContent = "—";
  document.getElementById("predictionTime").textContent = "—";
  document.getElementById("predictionCloseValue").textContent = "—";
  document.getElementById("rmseValue").textContent = "—";
  state.historyData = [];
  state.validationData = [];
  state.forecastPath = [];
  state.nextPoint = null;
  setError("");
  const tasks = [
    { name: "Info", run: () => loadInfo(loadToken, state.ticker) },
    { name: "History", run: () => loadHistory(loadToken, state.ticker, state.range, state.interval, state.priceField) },
    { name: "Prediction", run: () => loadPrediction(loadToken, state.ticker, state.priceField) },
  ];
  const results = await Promise.allSettled(tasks.map((t) => t.run()));
  const errors = tasks
    .map((task, idx) => ({ task: task.name, result: results[idx] }))
    .filter((entry) => entry.result.status === "rejected")
    .map((entry) => `${entry.task}: ${entry.result.reason?.message || "Request failed"}`);
  if (errors.length) {
    setError(errors.join(" • "));
  }
}

async function bootstrap() {
  tickerInput.value = state.ticker;
  priceFieldSelect.value = state.priceField;
  applyTheme(state.theme);
  await loadQuickPicks();
  await loadAll();
}

await bootstrap();
