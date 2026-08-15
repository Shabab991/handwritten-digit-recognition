/* script.js — Digit Recognition (enhanced)
   ==========================================================================
   Fixes vs. the original script.js:
   1. REAL BUG: `canvas.addEventListener("ontouchstart", ...)` used the
      HTML-attribute-style event name ("ontouchstart") instead of the
      addEventListener event name ("touchstart"). This means touch/mobile
      drawing never actually worked in the original — the listener was
      registered for an event that doesn't exist. Fixed here.
   2. `tf.loadModel()` is deprecated/removed in modern TensorFlow.js —
      replaced with `tf.loadLayersModel()`.
   3. No error handling if the model fails to load (e.g. wrong path,
      network issue) — the original would just silently never predict
      anything with no explanation. Added a visible status message and
      try/catch.
   4. Only showed the single top prediction — added a full probability
      distribution across all 10 digits (the softmax output), which is
      the actually interesting part of a classifier's output for a
      data-analyst-facing portfolio piece: it shows you understand the
      model isn't just returning an answer, it's returning a distribution.
   5. Added prediction history so multiple attempts are visible at once.
   ========================================================================== */

let model;
const modelStatusEl = document.getElementById("model-status");

async function loadModel() {
    try {
        model = await tf.loadLayersModel("model/model.json");
        modelStatusEl.textContent = "Model loaded — ready to predict";
        modelStatusEl.classList.add("ready");
    } catch (err) {
        console.error("Model failed to load:", err);
        modelStatusEl.textContent = "Model failed to load — check that model/model.json is being served correctly (see README).";
        modelStatusEl.classList.add("error");
    }
}
loadModel();

// ---------------------------------------------------------------------
// CANVAS DRAWING
// ---------------------------------------------------------------------
const mainCanvas = document.getElementById("main-canvas");
const mainCtx = mainCanvas.getContext("2d");
const smallCanvas = document.getElementById("small-canvas"); // 28x28 buffer fed to the model
const smallCtx = smallCanvas.getContext("2d");

let isDrawing = false;

function resetCanvas() {
    mainCtx.fillStyle = "black";
    mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
    mainCtx.lineWidth = 18;
    mainCtx.lineCap = "round";
    mainCtx.strokeStyle = "white";
}
resetCanvas();

function getPos(evt) {
    const rect = mainCanvas.getBoundingClientRect();
    const scaleX = mainCanvas.width / rect.width;
    const scaleY = mainCanvas.height / rect.height;
    const point = evt.touches ? evt.touches[0] : evt;
    return {
        x: (point.clientX - rect.left) * scaleX,
        y: (point.clientY - rect.top) * scaleY,
    };
}

function startDraw(evt) {
    evt.preventDefault();
    isDrawing = true;
    const { x, y } = getPos(evt);
    mainCtx.beginPath();
    mainCtx.moveTo(x, y);
}

function draw(evt) {
    if (!isDrawing) return;
    evt.preventDefault();
    const { x, y } = getPos(evt);
    mainCtx.lineTo(x, y);
    mainCtx.stroke();
}

function endDraw() {
    if (!isDrawing) return;
    isDrawing = false;
    predict();
}

// Mouse events
mainCanvas.addEventListener("mousedown", startDraw);
mainCanvas.addEventListener("mousemove", draw);
mainCanvas.addEventListener("mouseup", endDraw);
mainCanvas.addEventListener("mouseleave", endDraw);

// Touch events — fixed event names ("touchstart", not "ontouchstart")
mainCanvas.addEventListener("touchstart", startDraw, { passive: false });
mainCanvas.addEventListener("touchmove", draw, { passive: false });
mainCanvas.addEventListener("touchend", endDraw);

document.getElementById("erase").addEventListener("click", () => {
    resetCanvas();
    document.getElementById("prediction").textContent = "—";
    document.getElementById("confidence").textContent = "Draw a digit to see a prediction";
    document.getElementById("distribution").innerHTML = "";
});

// ---------------------------------------------------------------------
// PREDICTION
// ---------------------------------------------------------------------
const history = []; // { digit, confidence }
const MAX_HISTORY = 6;

async function predict() {
    if (!model) return; // model still loading or failed to load

    // Downscale the 300x300 drawing to the 28x28 the model expects
    smallCtx.drawImage(mainCanvas, 0, 0, smallCanvas.width, smallCanvas.height);

    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(smallCanvas, 1)
            .toFloat()
            .div(255.0)
            .reshape([1, 28, 28, 1]);
    });

    const output = model.predict(tensor);
    const probabilities = await output.data(); // Float32Array of 10 values, softmax output
    tensor.dispose();
    output.dispose();

    renderPrediction(probabilities);
}

function renderPrediction(probabilities) {
    const probs = Array.from(probabilities);
    const predictedDigit = probs.indexOf(Math.max(...probs));
    const confidence = probs[predictedDigit];

    document.getElementById("prediction").textContent = predictedDigit;
    document.getElementById("confidence").innerHTML =
        `Confidence: <strong>${(confidence * 100).toFixed(1)}%</strong>`;

    renderDistribution(probs, predictedDigit);
    addToHistory(predictedDigit, confidence);
}

function renderDistribution(probs, topDigit) {
    const container = document.getElementById("distribution");
    container.innerHTML = "";
    probs.forEach((p, digit) => {
        const row = document.createElement("div");
        row.className = "dist-row";
        row.innerHTML = `
            <span class="dist-digit">${digit}</span>
            <span class="dist-track"><span class="dist-fill ${digit === topDigit ? "top" : ""}" style="width:${(p * 100).toFixed(1)}%"></span></span>
            <span class="dist-pct">${(p * 100).toFixed(1)}%</span>
        `;
        container.appendChild(row);
    });
}

function addToHistory(digit, confidence) {
    history.unshift({ digit, confidence });
    if (history.length > MAX_HISTORY) history.pop();

    const historyEl = document.getElementById("history");
    historyEl.innerHTML = "";
    history.forEach((entry) => {
        const item = document.createElement("div");
        item.className = "history-item";
        item.innerHTML = `
            <span class="history-digit">${entry.digit}</span>
            <span class="history-conf">${(entry.confidence * 100).toFixed(1)}% confidence</span>
        `;
        historyEl.appendChild(item);
    });
}
