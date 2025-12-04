const state = {
  ticker: "AAPL",
  range: "1m",
  interval: "auto",
  showFullHistory: false,
  showFullPred: false,
  historyData: [],
  validationData: [],
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
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#94a3b8";
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
      ctx.fillStyle = "#0f172a";
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(lx, ly, width, height, 4);
      } else {
        ctx.rect(lx, ly, width, height);
      }
      ctx.fillStyle = "#f8fafc";
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#0f172a";
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
      ctx.fillStyle = "#0f172a";
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(rx, ry, width, height, 4);
      } else {
        ctx.rect(rx, ry, width, height);
      }
      ctx.fillStyle = "#f8fafc";
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#0f172a";
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
  loadHistory(activeLoadToken, state.ticker, state.range, state.interval);
});

intervalSelect.addEventListener("change", () => {
  state.interval = intervalSelect.value;
  loadHistory(activeLoadToken, state.ticker, state.range, state.interval);
});

toggleHistoryRowsBtn.addEventListener("click", () => {
  state.showFullHistory = !state.showFullHistory;
  renderHistoryTable(state.historyData);
});

togglePredRowsBtn.addEventListener("click", () => {
  state.showFullPred = !state.showFullPred;
  renderPredTable(state.validationData);
});

function applyTheme(theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
  themeToggle.textContent = theme === "dark" ? "Switch to light" : "Switch to dark";
}

themeToggle.addEventListener("click", () => {
  state.theme = state.theme === "dark" ? "light" : "dark";
  applyTheme(state.theme);
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

function setCursor(el, cursor) {
  if (el) el.style.cursor = cursor;
}

function zoomOptions(canvas) {
  return {
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
    if (isPanning) return;
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
    if (isPanning === false) return;
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
    if (isPanning === false) return;
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
  for (const p of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${p.Date}</td><td>${p.Open}</td><td>${p.High}</td><td>${p.Low}</td><td>${p.Close}</td><td>${p.Volume}</td>`;
    tbody.appendChild(tr);
  }
  const meta = document.getElementById("historyRowsMeta");
  const btn = document.getElementById("toggleHistoryRows");
  meta.textContent = state.showFullHistory ? `Showing all ${data.length}` : `Showing latest ${rows.length} of ${data.length}`;
  btn.textContent = state.showFullHistory ? "Show summary" : "Show all";
}

async function loadHistory(loadToken, ticker, range, interval) {
  const query = new URLSearchParams({
    ticker,
    range,
    t: Date.now().toString(),
  });
  if (interval && interval !== "auto") {
    query.set("interval", interval);
  }
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

  // Build price chart (close)
  const labels = prices.map((p) => p.Date);
  const closes = prices.map((p) => p.Close);
  if (priceChart) priceChart.destroy();
  const priceCanvas = document.getElementById("priceChart");
  const ctx = priceCanvas.getContext("2d");
  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `Close (${intervalLabel})`,
          data: closes,
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
      scales: { x: { ticks: { maxRotation: 0 } } },
      plugins: {
        legend: { display: true },
        zoom: zoomOptions(priceCanvas),
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
}

function renderPredTable(validation) {
  const tbody = document.querySelector("#predTable tbody");
  tbody.innerHTML = "";
  const tableRows = [...validation].sort((a, b) => new Date(b.Date) - new Date(a.Date));
  const rows = state.showFullPred ? tableRows : tableRows.slice(0, 10);
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.Date}</td><td>${formatUSD(row.Close)}</td><td>${formatUSD(row.Prediction)}</td>`;
    tbody.appendChild(tr);
  }
  const meta = document.getElementById("predRowsMeta");
  const btn = document.getElementById("togglePredRows");
  meta.textContent = state.showFullPred ? `Showing all ${tableRows.length}` : `Showing latest ${rows.length} of ${tableRows.length}`;
  btn.textContent = state.showFullPred ? "Show summary" : "Show all";
}

async function loadPrediction(loadToken, ticker) {
  const data = await fetchJSON(`/api/predict?ticker=${encodeURIComponent(ticker)}&t=${Date.now()}`);
  if (loadToken !== activeLoadToken) return;
  document.getElementById("predictionValue").textContent = formatUSD(data.predicted_next_close);
  document.getElementById("rmseValue").textContent = formatUSD(data.rmse);

  const validation = data.validation || [];
  state.validationData = validation;
  renderPredTable(validation);

  // Build prediction chart
  const labels = validation.map((d) => d.Date);
  const actual = validation.map((d) => d.Close);
  const preds = validation.map((d) => d.Prediction);
  if (predChart) predChart.destroy();
  const predCanvas = document.getElementById("predChart");
  const ctx = predCanvas.getContext("2d");
  predChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual",
          data: actual,
          borderColor: "#008f5d",
          backgroundColor: "rgba(0,143,93,0.12)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
        },
        {
          label: "Predicted",
          data: preds,
          borderColor: "#f4a11e",
          backgroundColor: "rgba(244,161,30,0.12)",
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
      scales: { x: { ticks: { maxRotation: 0 } } },
      plugins: {
        legend: { display: true },
        zoom: zoomOptions(predCanvas),
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
  document.querySelector("#historyTable tbody").innerHTML = "";
  document.querySelector("#predTable tbody").innerHTML = "";
  document.getElementById("priceMeta").textContent = "Loading...";
  document.getElementById("predictionValue").textContent = "—";
  document.getElementById("rmseValue").textContent = "—";
  state.historyData = [];
  state.validationData = [];
  setError("");
  try {
    await Promise.all([
      loadInfo(loadToken, state.ticker),
      loadHistory(loadToken, state.ticker, state.range, state.interval),
      loadPrediction(loadToken, state.ticker),
    ]);
  } catch (err) {
    console.error(err);
    setError(err.message || "Unknown error");
  }
}

async function bootstrap() {
  tickerInput.value = state.ticker;
  applyTheme(state.theme);
  await loadQuickPicks();
  await loadAll();
}

await bootstrap();
