const board = document.querySelector("#board");
const boardCtx = board.getContext("2d");
const latencyCanvas = document.querySelector("#latency-canvas");
const latencyCtx = latencyCanvas.getContext("2d");
const latencyHistory = [];
let latestState = null;
let socket = null;
let reconnectTimer = null;

const $ = (selector) => document.querySelector(selector);
const pad = (value, size = 2) => String(value ?? 0).padStart(size, "0");
const percent = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;

function fitCanvas(canvas, ctx) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}

function drawBoard(timestamp = 0) {
  const rect = fitCanvas(board, boardCtx);
  boardCtx.clearRect(0, 0, rect.width, rect.height);
  if (!latestState?.game) return;
  const game = latestState.game;
  const cellW = rect.width / game.width;
  const cellH = rect.height / game.height;

  boardCtx.fillStyle = "#030303";
  boardCtx.fillRect(0, 0, rect.width, rect.height);
  boardCtx.strokeStyle = "rgba(255,255,255,0.055)";
  boardCtx.lineWidth = 1;
  for (let x = 1; x < game.width; x++) {
    boardCtx.beginPath(); boardCtx.moveTo(x * cellW, 0); boardCtx.lineTo(x * cellW, rect.height); boardCtx.stroke();
  }
  for (let y = 1; y < game.height; y++) {
    boardCtx.beginPath(); boardCtx.moveTo(0, y * cellH); boardCtx.lineTo(rect.width, y * cellH); boardCtx.stroke();
  }

  const pulse = 0.65 + Math.sin(timestamp / 180) * 0.18;
  const [foodX, foodY] = game.food;
  const fx = (foodX + 0.5) * cellW;
  const fy = (foodY + 0.5) * cellH;
  boardCtx.save();
  boardCtx.shadowColor = "#ffffff"; boardCtx.shadowBlur = 16 * pulse;
  boardCtx.fillStyle = "#030303";
  boardCtx.strokeStyle = "#ffffff"; boardCtx.lineWidth = 2;
  boardCtx.beginPath(); boardCtx.arc(fx, fy, Math.min(cellW, cellH) * 0.22, 0, Math.PI * 2); boardCtx.fill(); boardCtx.stroke();
  boardCtx.strokeStyle = `rgba(255,255,255,${pulse})`; boardCtx.lineWidth = 1;
  boardCtx.beginPath(); boardCtx.arc(fx, fy, Math.min(cellW, cellH) * (0.31 + pulse * 0.08), 0, Math.PI * 2); boardCtx.stroke();
  boardCtx.restore();

  [...game.snake].reverse().forEach(([x, y], reverseIndex) => {
    const index = game.snake.length - 1 - reverseIndex;
    const inset = Math.max(1.5, Math.min(cellW, cellH) * 0.08);
    const alpha = 0.48 + (1 - index / Math.max(1, game.snake.length)) * 0.48;
    boardCtx.fillStyle = index === 0 ? "#ffffff" : `rgba(235,235,231,${alpha})`;
    if (index === 0) { boardCtx.shadowColor = "#ffffff"; boardCtx.shadowBlur = 12; }
    boardCtx.fillRect(x * cellW + inset, y * cellH + inset, cellW - inset * 2, cellH - inset * 2);
    boardCtx.shadowBlur = 0;
  });
}

function drawLatency() {
  const rect = fitCanvas(latencyCanvas, latencyCtx);
  latencyCtx.clearRect(0, 0, rect.width, rect.height);
  latencyCtx.strokeStyle = "rgba(255,255,255,0.12)";
  latencyCtx.beginPath(); latencyCtx.moveTo(0, rect.height - 0.5); latencyCtx.lineTo(rect.width, rect.height - 0.5); latencyCtx.stroke();
  if (latencyHistory.length < 2) return;
  const max = Math.max(50, ...latencyHistory);
  latencyCtx.beginPath();
  latencyHistory.forEach((value, index) => {
    const x = index / 59 * rect.width;
    const y = rect.height - (value / max) * (rect.height - 5) - 2;
    index ? latencyCtx.lineTo(x, y) : latencyCtx.moveTo(x, y);
  });
  latencyCtx.strokeStyle = "#ffffff";
  latencyCtx.lineWidth = 1.5;
  latencyCtx.shadowColor = "#ffffff"; latencyCtx.shadowBlur = 6;
  latencyCtx.stroke(); latencyCtx.shadowBlur = 0;
}

function updateOverlay(state) {
  const overlay = $("#board-overlay");
  const title = $("#overlay-title");
  const copy = $("#overlay-copy");
  if (state.model_status === "loading") {
    overlay.classList.add("visible"); title.textContent = "LOADING LOCAL MODEL"; copy.textContent = "Preparing CUDA decision engine…";
  } else if (state.model_status === "error") {
    overlay.classList.add("visible"); title.textContent = "MODEL ERROR"; copy.textContent = state.error || "Unknown inference error";
  } else if (!state.game.alive) {
    overlay.classList.add("visible"); title.textContent = state.game.won ? "BOARD COMPLETE" : "RUN TERMINATED";
    copy.textContent = state.game.death_reason || "Start a new round to continue";
  } else {
    overlay.classList.remove("visible");
  }
}

function updateUI(state) {
  latestState = state;
  const game = state.game;
  const decision = state.decision;
  $("#backend-name").textContent = state.backend.toUpperCase();
  $("#rail-backend").textContent = state.backend.toUpperCase();
  $("#model-name").textContent = state.model_label;
  $("#device").textContent = state.device;
  $("#seed").textContent = pad(game.seed, 3);
  $("#steps").textContent = pad(game.steps, 4);
  $("#score").textContent = pad(game.score);
  $("#length").textContent = pad(game.length);
  $("#interventions").textContent = pad(state.interventions);
  $("#thinking-tag").textContent = state.thinking ? "INFERENCE" : state.paused ? "PAUSED" : "READY";
  $("#thinking-tag").classList.toggle("active", state.thinking);
  $("#pause-button").innerHTML = state.paused ? "<span>▶</span> RESUME" : "<span>Ⅱ</span> PAUSE";
  $("#speed-slider").value = state.target_fps;
  $("#speed-value").textContent = Math.round(state.target_fps);
  $("#shield-toggle").checked = state.shield_enabled;

  ["UP", "DOWN", "LEFT", "RIGHT"].forEach((direction) => {
    const row = document.querySelector(`[data-direction="${direction}"]`);
    const probability = decision?.probabilities?.[direction];
    row.querySelector(".prob-fill").style.width = `${(probability || 0) * 100}%`;
    row.querySelector(".prob-value").textContent = probability == null ? "—" : percent(probability);
    const safe = decision?.safe_actions?.includes(direction);
    const tag = row.querySelector(".safe-tag");
    tag.textContent = decision ? (safe ? "SAFE" : "BLOCK") : "WAIT";
    tag.classList.toggle("safe", Boolean(safe));
    row.classList.toggle("executed", decision?.executed === direction);
  });

  $("#raw-action").textContent = decision?.raw_action || "—";
  $("#executed-action").textContent = decision?.executed || "—";
  $("#shield-flag").textContent = decision?.shielded ? "SHIELD CUT" : "DIRECT";
  $("#shield-flag").classList.toggle("cut", Boolean(decision?.shielded));
  $("#risk").textContent = percent(decision?.dead_end_risk);
  $("#reachable").textContent = percent(decision?.food_reachable);
  $("#latency").textContent = decision ? `${decision.latency_ms.toFixed(1)} MS` : "—";
  $("#rate").textContent = decision ? `${decision.model_rate.toFixed(1)} /S` : "—";
  if (decision?.latency_ms != null) {
    latencyHistory.push(decision.latency_ms);
    if (latencyHistory.length > 60) latencyHistory.shift();
  }
  updateOverlay(state);
  drawLatency();
}

function sendControl(action, value = null) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "control", action, value }));
}

function connect() {
  clearTimeout(reconnectTimer);
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.addEventListener("open", () => {
    $("#live-chip").className = "live-chip live";
    $("#live-label").textContent = "LIVE LOCAL";
    $("#connection-note").textContent = "SOCKET / STABLE";
  });
  socket.addEventListener("message", (event) => updateUI(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    $("#live-chip").className = "live-chip error";
    $("#live-label").textContent = "RECONNECTING";
    $("#connection-note").textContent = "SOCKET / RETRY";
    reconnectTimer = setTimeout(connect, 1000);
  });
}

$("#pause-button").addEventListener("click", () => sendControl("pause"));
$("#new-button").addEventListener("click", () => sendControl("new_game"));
$("#speed-slider").addEventListener("change", (event) => sendControl("speed", Number(event.target.value)));
$("#speed-slider").addEventListener("input", (event) => $("#speed-value").textContent = event.target.value);
$("#shield-toggle").addEventListener("change", (event) => sendControl("shield", event.target.checked));
window.addEventListener("keydown", (event) => {
  if (event.code === "Space") { event.preventDefault(); sendControl("pause"); }
  if (event.key.toLowerCase() === "r") sendControl("new_game");
  if (event.key === "+" || event.key === "=") sendControl("speed", Math.min(30, Number($("#speed-slider").value) + 2));
  if (event.key === "-") sendControl("speed", Math.max(1, Number($("#speed-slider").value) - 2));
});
window.addEventListener("resize", () => { drawBoard(performance.now()); drawLatency(); });

function animate(timestamp) {
  drawBoard(timestamp);
  requestAnimationFrame(animate);
}

connect();
requestAnimationFrame(animate);
