/* ═══════════════════════════════════════════════════════════════════════
   IMAGE-COMPARE.JS — KI-gestützte Bilderkennung & Ergebnis-Vergleich
   100% kostenlos · Tesseract.js + TensorFlow.js · Läuft im Browser
   ═══════════════════════════════════════════════════════════════════════
   Architektur:
     1. BRAIN-IMPORT     — Konfiguration aus image-compare-brain.js
     2. TENSORFLOW.JS    — Monitor-Erkennung & Auto-Crop
     3. PREPROCESSING    — Modulare Bild-Pipeline (Web Worker)
     4. OCR-ENGINE       — Tesseract.js Lazy-Loading & Multi-Pass
     5. SCORE-PARSING    — Regelbasierte Punktzahl-Extraktion
     6. UI-RENDERING     — Overlay, Upload, Fortschritt, Vergleich
     7. PUBLIC API       — init(), open(), createGameOverButton()
   ═══════════════════════════════════════════════════════════════════════ */

window.ImageCompare = (function () {
  'use strict';

  /* ═══ 1. BRAIN-IMPORT ═══════════════════════════════════════════════
     Alle Konstanten kommen aus image-compare-brain.js
     (muss VOR diesem Script geladen werden)
     ═══════════════════════════════════════════════════════════════════ */
  const Brain = window.ImageCompareBrain;
  if (!Brain) {
    console.error('[ImageCompare] FEHLER: image-compare-brain.js muss VOR image-compare.js geladen werden!');
  }
  const SCORE_CONFIG = Brain.SCORE_CONFIG;
  const OCR_PASSES = Brain.OCR_PASSES;
  const cleanOCRText = Brain.cleanOCRText;

  /* ─── PRIVATE STATE ──────────────────────── */
  let _isProcessing = false;

  /* ═══ 2. TENSORFLOW.JS — Monitor-Erkennung ═════════════════════════
     Erkennt ob ein Foto einen Ergebnis-Monitor zeigt und liefert
     optional eine Bounding Box für Auto-Crop.
     ═══════════════════════════════════════════════════════════════════ */

  let _tfModel = null;
  let _tfLoadFailed = false;

  /**
   * Lädt TensorFlow.js on-demand per CDN.
   * Wird nur aufgerufen wenn ein Modell konfiguriert ist.
   */
  function ensureTensorFlow() {
    return new Promise((resolve, reject) => {
      if (typeof tf !== 'undefined') { resolve(); return; }
      if (document.querySelector('script[data-ic-tfjs]')) {
        const check = setInterval(() => {
          if (typeof tf !== 'undefined') { clearInterval(check); resolve(); }
        }, 200);
        setTimeout(() => { clearInterval(check); reject(new Error('TF.js-Timeout')); }, 30000);
        return;
      }
      const sc = document.createElement('script');
      sc.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js';
      sc.dataset.icTfjs = '1';
      sc.onload = () => {
        const check = setInterval(() => {
          if (typeof tf !== 'undefined') { clearInterval(check); resolve(); }
        }, 100);
        setTimeout(() => { clearInterval(check); reject(new Error('TF.js-Timeout')); }, 15000);
      };
      sc.onerror = () => reject(new Error('TensorFlow.js konnte nicht geladen werden.'));
      document.head.appendChild(sc);
    });
  }

  /**
   * Lädt das benutzerdefinierte TF-Modell (einmalig, dann gecacht).
   * ══════════════════════════════════════════════════════════════════
   * ██  Der Modell-Pfad wird in image-compare-brain.js definiert:  ██
   * ██  → Brain.MODEL_PATH (Standard: './model/model.json')       ██
   * ══════════════════════════════════════════════════════════════════
   */
  async function loadTFModel() {
    if (_tfModel) return _tfModel;
    if (_tfLoadFailed) return null;
    try {
      await ensureTensorFlow();
      // ── Speicher-Optimierung: WebGL-Backend bevorzugen ──
      await tf.setBackend('webgl');
      await tf.ready();
      _tfModel = await tf.loadLayersModel(Brain.MODEL_PATH);
      console.info('[ImageCompare] TF-Modell geladen:', Brain.MODEL_PATH);
      return _tfModel;
    } catch (err) {
      _tfLoadFailed = true;
      console.info('[ImageCompare] Monitor-Modell nicht gefunden, überspringe TF-Analyse.', err.message);
      return null;
    }
  }

  /**
   * Prüft ob das Foto einen Ergebnis-Monitor zeigt.
   * Nutzt tf.tidy() für automatisches Tensor-Cleanup → kein RAM-Leak!
   *
   * ═══════════════════════════════════════════════════════════════════
   * Dein Teachable Machine Modell hat 2 Klassen:
   *   Index 0 = "Monitor"  (Ergebnis-Anzeige erkannt)
   *   Index 1 = "Nichts"   (kein Monitor)
   * Ausgabe ist Softmax: [monitor_prob, nichts_prob]
   * ═══════════════════════════════════════════════════════════════════
   *
   * @param {HTMLImageElement} imgEl — das hochgeladene Bild
   * @returns {Promise<{isMonitor: boolean, confidence: number, boundingBox: object|null}>}
   */
  async function detectMonitor(imgEl) {
    const model = await loadTFModel();
    if (!model) {
      // Kein Modell → Fallback: gehe davon aus, es IST ein Monitor
      return { isMonitor: true, confidence: 0, boundingBox: null };
    }

    const inputSize = Brain.MODEL_INPUT_SIZE; // 224 (aus metadata.json)

    // ── Speicher-optimiert: tf.tidy() räumt ALLE Zwischen-Tensoren auf ──
    const prediction = tf.tidy(() => {
      // Bild → Tensor (3 Kanäle, uint8)
      const imgTensor = tf.browser.fromPixels(imgEl);
      // Resize auf Modell-Eingabegröße (224×224)
      const resized = tf.image.resizeBilinear(imgTensor, [inputSize, inputSize]);
      // ── WICHTIG: Teachable Machine / MobileNet Normalisierung ──
      // Nicht ÷255, sondern ÷127.5 − 1  →  Bereich wird [-1, +1]
      const normalized = resized.div(127.5).sub(1.0);
      // Batch-Dimension: [224,224,3] → [1,224,224,3]
      const batched = normalized.expandDims(0);
      // Inferenz
      return model.predict(batched);
    });

    // Ergebnis auslesen — Softmax [Monitor_prob, Nichts_prob]
    let result;
    try {
      const data = await prediction.data();
      // data[0] = Wahrsch. "Monitor", data[1] = Wahrsch. "Nichts"
      const monitorConf = data[0];
      result = {
        isMonitor: monitorConf >= Brain.MONITOR_CONFIDENCE_THRESHOLD,
        confidence: monitorConf,
        // Teachable Machine liefert keine Bounding Box —
        // das Modell klassifiziert nur "Monitor" vs. "Nichts"
        boundingBox: null
      };
      console.info(`[ImageCompare] TF-Ergebnis: Monitor=${(monitorConf * 100).toFixed(1)}%, Nichts=${(data[1] * 100).toFixed(1)}%`);
    } catch (e) {
      console.warn('[ImageCompare] TF-Inferenz fehlgeschlagen:', e);
      result = { isMonitor: true, confidence: 0, boundingBox: null };
    } finally {
      // ── Speicher: Output-Tensor manuell disposen ──
      if (Array.isArray(prediction)) prediction.forEach(t => t.dispose());
      else prediction.dispose();
    }

    return result;
  }

  /* ─── PRIVATE STATE (erweitert) ─── */
  let _worker = null;          // Improvement #2: Worker-Singleton
  const _ocrCache = new Map(); // Improvement #6: OCR-Ergebnis-Cache
  const OCR_CACHE_MAX = 5;
  const CSS_ID = 'ic-styles';  // ID für das injizierte Stylesheet

  function injectStyles() {
    if (document.getElementById(CSS_ID)) return;
    const link = document.createElement('link');
    link.id = CSS_ID;
    link.rel = 'stylesheet';
    link.href = 'image-compare.css';
    document.head.appendChild(link);
  }

  /* ═══ 2b. WORKER-SINGLETON ═══════════════════════════════════════════ */

  let _ocrProgressCallback = null;

  /** Erstellt oder gibt den persistenten Tesseract-Worker zurück */
  async function getWorker() {
    await ensureTesseract();

    if (_worker) return _worker;

    _worker = await Tesseract.createWorker('deu+eng', 1, {
      logger: (info) => {
        if (info.status === 'recognizing text' && _ocrProgressCallback) {
          _ocrProgressCallback(info.progress);
        }
      }
    });

    await _worker.setParameters({
      tessedit_char_whitelist: '0123456789., OolI|Ss\n',
      tessedit_pageseg_mode: Tesseract.PSM.SPARSE_TEXT
    });

    return _worker;
  }

  /** Worker explizit beenden (für Cleanup) */
  async function terminateWorker() {
    if (_worker) {
      await _worker.terminate();
      _worker = null;
    }
  }

  /* ═══ 3. ROI-ERKENNUNG ═══════════════════════════════════════════════ */

  /**
   * Erkennt die Region of Interest (Bereich mit höchster Textdichte).
   * Analysiert zeilenweise Schwarzpixel-Verteilung nach dem Threshold.
   * @returns {{ x, y, w, h } | null} — Crop-Koordinaten oder null
   */
  function detectROI(imageData, w, h) {
    const d = imageData.data;
    const MIN_DENSITY = 0.03;     // Min. 3% schwarze Pixel pro Zeile
    const MIN_BLOCK_HEIGHT = 0.05; // Min. 5% der Bildhöhe

    // Zeilenweise Schwarzpixel-Dichte berechnen
    const rowDensity = new Float32Array(h);
    for (let y = 0; y < h; y++) {
      let blackCount = 0;
      for (let x = 0; x < w; x++) {
        if (d[(y * w + x) * 4] === 0) blackCount++;
      }
      rowDensity[y] = blackCount / w;
    }

    // Dichtesten zusammenhängenden Block finden
    let bestStart = 0, bestEnd = h - 1, bestScore = 0;
    let blockStart = -1;

    for (let y = 0; y < h; y++) {
      if (rowDensity[y] >= MIN_DENSITY) {
        if (blockStart === -1) blockStart = y;
      } else {
        if (blockStart !== -1) {
          const blockH = y - blockStart;
          if (blockH >= h * MIN_BLOCK_HEIGHT) {
            // Score = Höhe × durchschnittliche Dichte
            let totalDensity = 0;
            for (let i = blockStart; i < y; i++) totalDensity += rowDensity[i];
            const score = blockH * (totalDensity / blockH);
            if (score > bestScore) {
              bestScore = score;
              bestStart = blockStart;
              bestEnd = y;
            }
          }
          blockStart = -1;
        }
      }
    }
    // Letzten Block prüfen
    if (blockStart !== -1) {
      const blockH = h - blockStart;
      if (blockH >= h * MIN_BLOCK_HEIGHT) {
        let totalDensity = 0;
        for (let i = blockStart; i < h; i++) totalDensity += rowDensity[i];
        const score = blockH * (totalDensity / blockH);
        if (score > bestScore) {
          bestStart = blockStart;
          bestEnd = h;
        }
      }
    }

    // Nur croppen wenn ROI deutlich kleiner als Gesamtbild
    const roiH = bestEnd - bestStart;
    if (roiH < h * 0.8 && roiH > h * MIN_BLOCK_HEIGHT) {
      // 10% Padding hinzufügen
      const pad = Math.round(roiH * 0.1);
      return {
        x: 0,
        y: Math.max(0, bestStart - pad),
        w: w,
        h: Math.min(h, roiH + 2 * pad)
      };
    }

    return null; // Kein sinnvoller ROI gefunden
  }

  /* ═══ 6. OCR-CACHE ═══════════════════════════════════════════════════ */

  /** Cache-Key aus File-Metadaten generieren */
  function getCacheKey(file) {
    return `${file.name}_${file.size}_${file.lastModified} `;
  }

  /* ─── TESSERACT.JS LAZY LOADING ──────────── */
  function ensureTesseract() {
    return new Promise((resolve, reject) => {
      if (typeof Tesseract !== 'undefined') {
        resolve();
        return;
      }
      // Check if script is already loading
      if (document.querySelector('script[data-ic-tesseract]')) {
        const check = setInterval(() => {
          if (typeof Tesseract !== 'undefined') {
            clearInterval(check);
            resolve();
          }
        }, 200);
        // Timeout after 30s
        setTimeout(() => { clearInterval(check); reject(new Error('Tesseract-Timeout')); }, 30000);
        return;
      }
      const sc = document.createElement('script');
      sc.src = 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js';
      sc.dataset.icTesseract = '1';
      sc.onload = () => {
        const check = setInterval(() => {
          if (typeof Tesseract !== 'undefined') {
            clearInterval(check);
            resolve();
          }
        }, 100);
        setTimeout(() => { clearInterval(check); reject(new Error('Tesseract-Timeout')); }, 15000);
      };
      sc.onerror = () => reject(new Error('Tesseract konnte nicht geladen werden.'));
      document.head.appendChild(sc);
    });
  }

  /* ═══ 2. PREPROCESSING — Modulare Bild-Pipeline (Web Worker Variante) ═════ */

  let _prepWorker = null;
  let _prepCallbacks = {};
  let _prepMsgId = 0;

  function getPrepWorker() {
    if (_prepWorker) return _prepWorker;
    const workerCode = `
  function toGrayscale(d) {
    for (let i = 0; i < d.length; i += 4) {
      const gray = Math.round(d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114);
      d[i] = d[i + 1] = d[i + 2] = gray;
    }
  }
  function applyGamma(d, gamma) {
    if (!gamma || gamma === 1.0) return;
    const inv = 1 / gamma;
    for (let i = 0; i < d.length; i += 4) {
      const corrected = Math.round(255 * Math.pow(d[i] / 255, inv));
      d[i] = d[i + 1] = d[i + 2] = corrected;
    }
  }
  // Moiré-Reduktion: Leichter Gaußscher Weichzeichner (sigma≈1.0) vor dem Schärfen
  // eliminiert Scan-Lines alter Röhrenmonitore und LCD-Flachbildschirme
  function gaussianBlur(d, w, h, sigma = 1.0) {
    const r = Math.ceil(sigma * 2);
    const kernel = [];
    let sum = 0;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const v = Math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
        kernel.push({ dx, dy, v });
        sum += v;
      }
    }
    for (let k = 0; k < kernel.length; k++) kernel[k].v /= sum;
    const out = new Uint8ClampedArray(d.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let acc = 0;
        for (let k = 0; k < kernel.length; k++) {
          const nx = Math.max(0, Math.min(w - 1, x + kernel[k].dx));
          const ny = Math.max(0, Math.min(h - 1, y + kernel[k].dy));
          acc += d[(ny * w + nx) * 4] * kernel[k].v;
        }
        const i = (y * w + x) * 4;
        const v = Math.round(Math.max(0, Math.min(255, acc)));
        out[i] = out[i + 1] = out[i + 2] = v;
        out[i + 3] = d[i + 3];
      }
    }
    for (let i = 0; i < d.length; i++) d[i] = out[i];
  }
  function sharpen(d, w, h) {
    const out = new Uint8ClampedArray(d.length);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = (y * w + x) * 4;
        const val = Math.max(0, Math.min(255,
          5 * d[i] - d[((y - 1) * w + x) * 4] - d[((y + 1) * w + x) * 4] - d[(y * w + x - 1) * 4] - d[(y * w + x + 1) * 4]
        ));
        out[i] = out[i + 1] = out[i + 2] = val;
        out[i + 3] = 255;
      }
    }
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = (y * w + x) * 4;
        d[i] = out[i]; d[i + 1] = out[i + 1]; d[i + 2] = out[i + 2];
      }
    }
  }
  // Adaptiver Kontrast (CLAHE-Prinzip): Lokale Histogramm-Anpassung pro Tile
  // Hilft bei spiegelnden Röhrenmonitoren und ungleichmäßiger Beleuchtung
  function claheContrast(d, w, h, tileSize = 64, clipLimit = 2.0) {
    const ts = Math.min(tileSize, Math.min(w, h) >> 1);
    const tw = Math.ceil(w / ts), th = Math.ceil(h / ts);
    const LUT = [];
    for (let ty = 0; ty < th; ty++) {
      for (let tx = 0; tx < tw; tx++) {
        const x0 = tx * ts, y0 = ty * ts;
        const x1 = Math.min(x0 + ts, w), y1 = Math.min(y0 + ts, h);
        const hist = new Uint32Array(256);
        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++)
            hist[d[(y * w + x) * 4]]++;
        const total = (x1 - x0) * (y1 - y0);
        const clip = Math.max(1, Math.floor(total / 256 * clipLimit));
        let excess = 0;
        for (let i = 0; i < 256; i++) {
          if (hist[i] > clip) { excess += hist[i] - clip; hist[i] = clip; }
        }
        const perBin = Math.floor(excess / 256);
        for (let i = 0; i < 256; i++) hist[i] += perBin;
        let sum = 0;
        const lut = new Uint8Array(256);
        for (let i = 0; i < 256; i++) {
          sum += hist[i];
          lut[i] = Math.round((sum / total) * 255);
        }
        LUT.push(lut);
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const tx = Math.min(Math.floor(x / ts), tw - 1);
        const ty = Math.min(Math.floor(y / ts), th - 1);
        const idx = ty * tw + tx;
        const i = (y * w + x) * 4;
        const v = LUT[idx][d[i]];
        d[i] = d[i + 1] = d[i + 2] = v;
      }
    }
  }
  function stretchContrast(d) {
    let lo = 255, hi = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] < lo) lo = d[i];
      if (d[i] > hi) hi = d[i];
    }
    const range = hi - lo || 1;
    for (let i = 0; i < d.length; i += 4) {
      const v = Math.round(((d[i] - lo) / range) * 255);
      d[i] = d[i + 1] = d[i + 2] = v;
    }
  }
  function invert(d) {
    for (let i = 0; i < d.length; i += 4) {
      d[i] = 255 - d[i]; d[i + 1] = 255 - d[i + 1]; d[i + 2] = 255 - d[i + 2];
    }
  }
  function adaptiveThreshold(d, w, h, windowRatio = 8, sensitivity = 0.15) {
    const S = Math.max(1, Math.round(w / windowRatio));
    const intImg = new Uint32Array(w * h);
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let y = 0; y < h; y++) {
        const idx = y * w + x;
        sum += d[idx * 4];
        intImg[idx] = (x === 0) ? sum : intImg[idx - 1] + sum;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const x1 = Math.max(x - S, 0), x2 = Math.min(x + S, w - 1);
        const y1 = Math.max(y - S, 0), y2 = Math.min(y + S, h - 1);
        const count = (x2 - x1 + 1) * (y2 - y1 + 1);
        const a = (y1 > 0 && x1 > 0) ? intImg[(y1 - 1) * w + (x1 - 1)] : 0;
        const b = (y1 > 0) ? intImg[(y1 - 1) * w + x2] : 0;
        const c = (x1 > 0) ? intImg[y2 * w + (x1 - 1)] : 0;
        const sum = intImg[y2 * w + x2] - b - c + a;
        const idx = (y * w + x) * 4;
        const val = (d[idx] * count <= sum * (1.0 - sensitivity)) ? 0 : 255;
        d[idx] = d[idx + 1] = d[idx + 2] = val;
      }
    }
  }
  function removeNoise(d, w, h, minNeighbors = 2) {
    const remove = [];
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = (y * w + x) * 4;
        if (d[idx] === 0) {
          let neighbors = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              if (d[((y + dy) * w + (x + dx)) * 4] === 0) neighbors++;
            }
          }
          if (neighbors < minNeighbors) remove.push(idx);
        }
      }
    }
    for (let i = 0; i < remove.length; i++) {
      const idx = remove[i];
      d[idx] = d[idx + 1] = d[idx + 2] = 255;
    }
  }
  // Pixel-Connect: Morphologische Dilation (1px) — verbindet unterbrochene
  // Segmente von alten LCD/LED-Anzeigen zu durchgehenden Ziffern
  function dilate(d, w, h, radius = 1) {
    const r = radius;
    const out = new Uint8ClampedArray(d.length);
    for (let i = 0; i < d.length; i++) out[i] = d[i];
    for (let y = r; y < h - r; y++) {
      for (let x = r; x < w - r; x++) {
        const idx = (y * w + x) * 4;
        let hasBlack = false;
        for (let dy = -r; dy <= r && !hasBlack; dy++)
          for (let dx = -r; dx <= r && !hasBlack; dx++)
            if (d[((y + dy) * w + (x + dx)) * 4] === 0) hasBlack = true;
        if (hasBlack) out[idx] = out[idx + 1] = out[idx + 2] = 0;
      }
    }
    for (let i = 0; i < d.length; i++) d[i] = out[i];
  }

  self.onmessage = function (e) {
    const { id, imageData, w, h, options } = e.data;
    try {
      const d = imageData.data;
      toGrayscale(d);
      applyGamma(d, options.gamma);
      gaussianBlur(d, w, h, 1.0);
      sharpen(d, w, h);
      claheContrast(d, w, h);
      if (options.invert) invert(d);
      adaptiveThreshold(d, w, h);
      removeNoise(d, w, h);
      dilate(d, w, h, 1);
      self.postMessage({ id, imageData }, [imageData.data.buffer]);
    } catch (err) {
      self.postMessage({ id, error: err.message });
    }
  };
  `;
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    _prepWorker = new Worker(URL.createObjectURL(blob));
    _prepWorker.onmessage = function (e) {
      const { id, imageData, error } = e.data;
      if (_prepCallbacks[id]) {
        if (error) _prepCallbacks[id].reject(new Error(error));
        else _prepCallbacks[id].resolve(imageData);
        delete _prepCallbacks[id];
      }
    };
    return _prepWorker;
  }

  function runPrepWorkerAsync(imageData, w, h, options) {
    return new Promise((resolve, reject) => {
      const worker = getPrepWorker();
      const id = ++_prepMsgId;
      _prepCallbacks[id] = { resolve, reject };
      worker.postMessage({ id, imageData, w, h, options }, [imageData.data.buffer]);
    });
  }

  const PREPROCESS = {
    MAX_DIM: 1200,

    prepareCanvas(imgEl, crop) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      let sx = 0, sy = 0;
      let srcW = imgEl.naturalWidth || imgEl.width;
      let srcH = imgEl.naturalHeight || imgEl.height;
      if (crop) { sx = crop.x; sy = crop.y; srcW = crop.w; srcH = crop.h; }

      let w = srcW, h = srcH;
      if (w > this.MAX_DIM || h > this.MAX_DIM) {
        const scale = this.MAX_DIM / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(imgEl, sx, sy, srcW, srcH, 0, 0, w, h);
      return { canvas, ctx, w, h, imageData: ctx.getImageData(0, 0, w, h) };
    },

    async run(imgEl, options = {}) {
      const { canvas, ctx, w, h, imageData } = this.prepareCanvas(imgEl, options.crop);

      const processedImageData = await runPrepWorkerAsync(imageData, w, h, options);
      ctx.putImageData(processedImageData, 0, 0);

      const dataUrl = canvas.toDataURL('image/png');
      canvas.width = 0;
      canvas.height = 0;

      return { dataUrl, width: w, height: h, imageData: processedImageData };
    },
  };

  /* ═══ 4. SCORE-PARSING — Regelbasierte Punktzahl-Extraktion ═════════ */

  /**
   * Berechnet die gewichtete Konfidenz eines einzelnen Score-Kandidaten.
   * Nutzt räumliche Lage, Keyword-Nähe, Typ und Geometrie.
   */
  function calculateCandidateWeight(valStr, bbox, type, isKK, canvasW, canvasH, cleanText) {
    const W = SCORE_CONFIG.WEIGHTS;
    let weight = 1.0;

    // 1. Räumliche Gewichtung: Bonus für Nähe zur Bildmitte
    if (bbox && canvasW && canvasH) {
      const cx = canvasW / 2, cy = canvasH / 2;
      const wordCx = bbox.x0 + (bbox.x1 - bbox.x0) / 2;
      const wordCy = bbox.y0 + (bbox.y1 - bbox.y0) / 2;
      const dist = Math.hypot(wordCx - cx, wordCy - cy);
      const maxDist = Math.hypot(cx, cy);
      weight += (1 - dist / maxDist) * W.CENTER_FACTOR;
    }

    // 2. Keyword-Nähe: Bonus wenn ein Schlüsselwort in der Nähe steht
    const textLower = cleanText.toLowerCase();
    for (const kw of SCORE_CONFIG.KEYWORDS) {
      const kwIdx = textLower.indexOf(kw);
      if (kwIdx === -1) continue;
      const valIdx = cleanText.indexOf(valStr);
      if (Math.abs(kwIdx - valIdx) < W.KEYWORD_MAX_DIST) {
        weight += W.KEYWORD_NEAR_BONUS;
        break; // Nur einmal bonussen
      }
    }

    // 3. Typ-Gewichtung
    if (type === 'labeled' || type === 'total') weight += W.LABELED_TYPE_BONUS;
    if (isKK && !valStr.includes('.')) weight += W.FORMAT_MATCH_BONUS;
    if (!isKK && valStr.includes('.')) weight += W.FORMAT_MATCH_BONUS;

    return weight;
  }

  /**
   * Prüft das Seitenverhältnis einer erkannten Bounding Box.
   * Ziffern haben typischerweise ein Höhe/Breite-Verhältnis von 1.2–2.8.
   */
  function calculateGeometryWeight(bbox) {
    if (!bbox) return 1.0;
    const G = SCORE_CONFIG.GEOMETRY;
    const ratio = (bbox.y1 - bbox.y0) / Math.max(1, bbox.x1 - bbox.x0);
    if (ratio < G.MIN || ratio > G.MAX) return SCORE_CONFIG.WEIGHTS.BAD_GEOMETRY_PENALTY;
    if (ratio > G.GOOD_MIN && ratio < G.GOOD_MAX) return SCORE_CONFIG.WEIGHTS.GOOD_GEOMETRY_BONUS;
    return 1.0;
  }

  /**
   * Extrahiert Punktzahl-Kandidaten aus OCR-Ergebnis.
   * Nutzt Word-Level-Daten für exakte Bounding Boxes und Konfidenz.
   * @returns {{ bestMatch, alternatives, allScores } | null}
   */
  function parseShootingScore(ocrResult, isKK, canvasW, canvasH, discipline = null) {
    if (!ocrResult || !ocrResult.data || !ocrResult.data.words) return null;

    const cleanText = cleanOCRText(ocrResult.data.text);
    const words = ocrResult.data.words || [];
    const { min, max } = SCORE_CONFIG.VALID_RANGE;
    const candidates = [];

    // Quelle 1: Individuelle Wörter mit Bounding-Box-Daten
    for (const w of words) {
      const cleaned = cleanOCRText(w.text);
      const valDec = parseFloat(cleaned);
      // Validation Check (Discipline Context)
      const discConfig = discipline ? SCORE_CONFIG.DISCIPLINES[discipline] : null;

      if (!isNaN(valDec) && valDec >= SCORE_CONFIG.VALID_RANGE.min && valDec <= SCORE_CONFIG.VALID_RANGE.max) {
        const typeStr = cleaned.includes('.') ? 'decimal' : 'integer';
        const geometryConf = calculateGeometryWeight(w.bbox);
        const rawConf = w.confidence / 100;

        let conf = rawConf * geometryConf;
        conf *= calculateCandidateWeight(cleaned, w.bbox, typeStr, isKK, canvasW, canvasH, cleanText);

        let isValid = true;
        if (discConfig) {
          // 1. Check Range constraints
          if (valDec < discConfig.min || valDec > discConfig.max) {
            isValid = false; // Completely ignore values outside expected discipline range
          }
          // 2. Format constraint enforcement
          if (isValid) {
            if (discConfig.isInteger && typeStr === 'decimal') {
              conf *= 0.2; // Massive penalty for decimals in KK_30
            } else if (!discConfig.isInteger && typeStr === 'integer') {
              conf *= 0.6; // Moderate penalty for integers in LG formats
            }
          }
        }

        if (isValid) {
          candidates.push({
            value: valDec,
            type: typeStr,
            confidence: conf,
            bbox: w.bbox,
            rawWord: cleaned
          });
        }
      }
    }
    // Quelle 2: Regex-Suche nach Dezimalzahlen im Gesamttext (Fallback)
    const decRegex = /(\d{2,3})[.,](\d)\b/g;
    let m;
    while ((m = decRegex.exec(cleanText)) !== null) {
      const val = parseFloat(m[1] + '.' + m[2]);
      if (val < min || val > max) continue;
      const weight = calculateCandidateWeight(m[0], null, 'decimal', isKK, canvasW, canvasH, cleanText);
      candidates.push({ value: val, type: 'decimal', confidence: 0.8 * weight });
    }

    if (candidates.length === 0) return null;

    // Sortieren nach Konfidenz (höchste zuerst)
    candidates.sort((a, b) => b.confidence - a.confidence);

    // Deduplizierung: Werte die ≤0.1 voneinander abweichen zusammenfassen
    const unique = [];
    for (const c of candidates) {
      if (!unique.some(u => Math.abs(u.value - c.value) < 0.1)) {
        unique.push(c);
      }
    }

    return { bestMatch: unique[0], alternatives: unique.slice(1, 4), allScores: unique };
  }

  /* ═══ 5. UI-RENDERING ══════════════════════════════════════════════ */

  // Build the upload overlay
  function createOverlay(botScore, isKK) {
    // Remove existing overlay
    const existing = document.getElementById('icOverlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.className = 'ic-overlay';
    overlay.id = 'icOverlay';

    overlay.innerHTML = `
    < div class="ic-sheet" >
        <div class="ic-handle"></div>
        <div class="ic-header">
          <div class="ic-title">📷 FOTO-ANALYSE</div>
          <div class="ic-close" id="icClose">✕</div>
        </div>
        <div class="ic-body">
          <!-- Upload Zone -->
          <div class="ic-upload-zone" id="icUploadZone">
            <input type="file" class="ic-upload-input" id="icFileInput"
                   accept="image/*" capture="environment">
            <span class="ic-upload-icon">📸</span>
            <div class="ic-upload-text">Foto der Ergebnisanzeige<br>hochladen oder aufnehmen</div>
            <div class="ic-upload-sub">JPG, PNG · Kamera oder Galerie</div>
          </div>

          <!-- Progress -->
          <div class="ic-progress" id="icProgress">
            <div class="ic-progress-label">🤖 KI-Analyse</div>
            <div class="ic-progress-bar">
              <div class="ic-progress-fill" id="icProgressFill"></div>
            </div>
            <div class="ic-progress-status" id="icProgressStatus">Wird vorbereitet…</div>
          </div>

          <!-- Result -->
          <div class="ic-result-card" id="icResultCard">
            <div class="ic-result-header">◈ Erkanntes Ergebnis</div>
            <div class="ic-detected-score">
              <span class="ic-detected-icon">🎯</span>
              <div class="ic-detected-info">
                <div class="ic-detected-value" id="icDetectedValue">–</div>
                <div class="ic-detected-label" id="icDetectedLabel">Wird analysiert…</div>
              </div>
            </div>
            <div class="ic-edit-score">
              <input type="number" class="ic-score-input" id="icScoreInput"
                     placeholder="${isKK ? 'z.B. 392' : 'z.B. 405.2'}"
                     step="${isKK ? '1' : '0.1'}" min="0" max="660"
                     inputmode="${isKK ? 'numeric' : 'decimal'}">
            </div>
            <div class="ic-edit-hint">✏️ Ergebnis oben korrigieren, falls die KI ungenau war</div>

            <!-- Raw OCR text toggle -->
            <div class="ic-raw-toggle" id="icRawToggle">▶ OCR-Rohtext anzeigen</div>
            <div class="ic-raw-text" id="icRawText"></div>
          </div>

          <!-- Compare Button -->
          <button class="ic-compare-btn" id="icCompareBtn" disabled>
            ⚡ VERGLEICH STARTEN
          </button>

          <!-- Comparison Result (hidden until compare) -->
          <div class="ic-comparison" id="icComparison"></div>

          <div class="ic-info">
            🔒 Dein Foto wird <b>nur lokal</b> im Browser analysiert.<br>
            Es wird nichts hochgeladen. 100% offline & kostenlos.
          </div>
        </div>
      </div >
    `;

    document.body.appendChild(overlay);
    setupOverlayEvents(overlay, botScore, isKK);
    return overlay;
  }

  function setupOverlayEvents(overlay, botScore, isKK) {
    const closeBtn = overlay.querySelector('#icClose');
    const fileInput = overlay.querySelector('#icFileInput');
    const uploadZone = overlay.querySelector('#icUploadZone');
    const compareBtn = overlay.querySelector('#icCompareBtn');
    const rawToggle = overlay.querySelector('#icRawToggle');
    const rawText = overlay.querySelector('#icRawText');
    const scoreInput = overlay.querySelector('#icScoreInput');

    // Close
    closeBtn.addEventListener('click', () => closeOverlay());
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeOverlay();
    });

    // File input
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) handleImageFile(file, overlay, botScore, isKK);
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('dragover');
    });
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      const file = e.dataTransfer?.files[0];
      if (file && file.type.startsWith('image/')) {
        handleImageFile(file, overlay, botScore, isKK);
      }
    });

    // Compare button
    compareBtn.addEventListener('click', () => {
      const val = parseFloat(scoreInput.value);
      if (isNaN(val) || val < 0) {
        scoreInput.style.borderColor = 'rgba(240,80,60,.6)';
        setTimeout(() => { scoreInput.style.borderColor = ''; }, 1200);
        return;
      }
      showComparison(overlay, val, botScore, isKK);
    });

    // Score input → enable compare button
    scoreInput.addEventListener('input', () => {
      const val = parseFloat(scoreInput.value);
      compareBtn.disabled = isNaN(val) || val < 0;
    });
    // Enter key → compare
    scoreInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') compareBtn.click();
    });

    // Raw OCR toggle
    rawToggle.addEventListener('click', () => {
      rawText.classList.toggle('visible');
      rawToggle.textContent = rawText.classList.contains('visible')
        ? '▼ OCR-Rohtext ausblenden'
        : '▶ OCR-Rohtext anzeigen';
    });

    // Swipe-down to close
    let startY = 0;
    const sheet = overlay.querySelector('.ic-sheet');
    sheet.addEventListener('touchstart', (e) => { startY = e.touches[0].clientY; }, { passive: true });
    sheet.addEventListener('touchend', (e) => {
      const dy = e.changedTouches[0].clientY - startY;
      if (dy > 80) closeOverlay();
    }, { passive: true });
  }

  function closeOverlay() {
    const overlay = document.getElementById('icOverlay');
    if (overlay) {
      overlay.style.opacity = '0';
      overlay.style.transition = 'opacity .2s';
      setTimeout(() => overlay.remove(), 200);
    }
    _isProcessing = false;
  }

  /* ─── IMAGE PROCESSING FLOW ─────────────── */
  async function handleImageFile(file, overlay, botScore, isKK) {
    if (_isProcessing) return;
    _isProcessing = true;

    // Grab discipline attached to overlay by open()
    const discipline = overlay.dataset.discipline || null;

    // Improvement #6: OCR-Ergebnis-Cache
    const cacheKey = getCacheKey(file);
    const cachedResult = _ocrCache.get(cacheKey);

    const uploadZone = overlay.querySelector('#icUploadZone');
    const progress = overlay.querySelector('#icProgress');
    const progressFill = overlay.querySelector('#icProgressFill');
    const progressStatus = overlay.querySelector('#icProgressStatus');
    const resultCard = overlay.querySelector('#icResultCard');
    const detectedValue = overlay.querySelector('#icDetectedValue');
    const detectedLabel = overlay.querySelector('#icDetectedLabel');
    const scoreInput = overlay.querySelector('#icScoreInput');
    const compareBtn = overlay.querySelector('#icCompareBtn');
    const rawText = overlay.querySelector('#icRawText');

    const objectUrl = URL.createObjectURL(file);

    uploadZone.classList.add('has-image');
    uploadZone.innerHTML = `
    < div class="ic-preview-wrap" >
      <img class="ic-preview-img" src="${objectUrl}" alt="Upload" id="icPreviewImg">
        <div class="ic-remove-img" id="icRemoveImg" title="Bild entfernen">✕</div>
      </div>
  `;
    const removeBtn = overlay.querySelector('#icRemoveImg');
    if (removeBtn) {
      removeBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        URL.revokeObjectURL(objectUrl);
        resetUploadZone(overlay, isKK);
        _isProcessing = false;
      });
    }

    if (cachedResult) {
      URL.revokeObjectURL(objectUrl);
      renderOCRResult(cachedResult, cachedResult.rawText, overlay, isKK);
      _isProcessing = false;
      return;
    }

    progress.classList.add('active');
    resultCard.classList.remove('active');
    progressFill.style.width = '5%';
    progressStatus.textContent = 'Bild wird vorbereitet…';

    try {
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = objectUrl;
      });

      progressFill.style.width = '10%';
      progressStatus.textContent = '🧠 TensorFlow: Prüfe Monitor-Erkennung…';

      // ═══ TENSORFLOW MONITOR-ERKENNUNG & AUTO-CROP ═══════════════════
      // Vor der OCR prüfen wir, ob das Bild überhaupt einen Monitor zeigt.
      // Wenn ja UND eine Bounding Box zurückkommt → Auto-Crop für bessere OCR.
      let tfCrop = null;
      const monitorResult = await detectMonitor(img);

      if (!monitorResult.isMonitor) {
        // Kein Monitor erkannt → Benutzer informieren, aber trotzdem weitermachen
        progressStatus.textContent = '⚠ Kein Monitor erkannt — versuche trotzdem OCR…';
        await delay(800);
      } else if (monitorResult.boundingBox) {
        // ── Auto-Crop: TF-Modell hat Ergebnis-Bereich erkannt ──
        // Bounding Box ist normalisiert (0–1), umrechnen auf Pixel
        const imgW = img.naturalWidth || img.width;
        const imgH = img.naturalHeight || img.height;
        const bb = monitorResult.boundingBox;
        tfCrop = {
          x: Math.round(bb.x * imgW),
          y: Math.round(bb.y * imgH),
          w: Math.round(bb.w * imgW),
          h: Math.round(bb.h * imgH)
        };
        progressStatus.textContent = `✓ Monitor erkannt (${Math.round(monitorResult.confidence * 100)}%) — Auto-Crop aktiv`;
        await delay(400);
      } else if (monitorResult.confidence > 0) {
        progressStatus.textContent = `✓ Monitor erkannt (${Math.round(monitorResult.confidence * 100)}%)`;
        await delay(300);
      }

      progressFill.style.width = '15%';
      progressStatus.textContent = 'Lade Tesseract.js OCR-Engine…';

      _ocrProgressCallback = (prog) => {
        const pct = Math.round(30 + prog * 50);
        progressFill.style.width = pct + '%';
        progressStatus.textContent = `Texterkennung… ${Math.round(prog * 100)}% `;
      };

      const worker = await getWorker();

      progressFill.style.width = '25%';
      progressStatus.textContent = 'Erkenne interessanten Bereich (ROI)…';

      try {
        let bestParsed = null;
        let bestResult = null;
        let bestConf = 0;

        // ── Crop-Logik: TF-Crop hat Vorrang, sonst ROI-Fallback ──
        let crop = tfCrop;

        if (!crop) {
          // Fallback: Klassische ROI-Erkennung (Pixel-Analyse)
          const roiPrep = await PREPROCESS.run(img, { gamma: 1.0 });
          const roiCanvas = document.createElement('canvas');
          const roiCtx = roiCanvas.getContext('2d');
          const roiImg = new Image();
          roiImg.src = roiPrep.dataUrl;
          await new Promise(r => { roiImg.onload = r; });
          roiCanvas.width = roiPrep.width;
          roiCanvas.height = roiPrep.height;
          roiCtx.drawImage(roiImg, 0, 0);
          const roiCrop = detectROI(roiCtx.getImageData(0, 0, roiPrep.width, roiPrep.height), roiPrep.width, roiPrep.height);

          // ── Speicher: Canvas sofort freigeben ──
          roiCanvas.width = 0;
          roiCanvas.height = 0;

          if (roiCrop) {
            const scaleX = (img.naturalWidth || img.width) / roiPrep.width;
            const scaleY = (img.naturalHeight || img.height) / roiPrep.height;
            crop = {
              x: roiCrop.x * scaleX,
              y: roiCrop.y * scaleY,
              w: roiCrop.w * scaleX,
              h: roiCrop.h * scaleY
            };
          }
        }

        for (let i = 0; i < OCR_PASSES.length; i++) {
          const pass = OCR_PASSES[i];
          if (bestConf >= pass.triggerBelow) continue;

          progressStatus.textContent = `Analysiere Bild(${pass.name}, Pass ${i + 1} / ${OCR_PASSES.length})…`;
          progressFill.style.width = Math.round(30 + (i / OCR_PASSES.length) * 60) + '%';

          const prepOptions = { ...pass.options, crop };
          const prep = await PREPROCESS.run(img, prepOptions);
          const result = await worker.recognize(prep.dataUrl);
          const parsed = parseShootingScore(result, isKK, prep.width, prep.height, discipline);

          if (parsed?.bestMatch && parsed.bestMatch.confidence > bestConf) {
            bestParsed = parsed;
            bestResult = result;
            bestConf = parsed.bestMatch.confidence;
          }
        }

        // ── Speicher: Object-URL revoken ──
        URL.revokeObjectURL(objectUrl);

        progressFill.style.width = '95%';
        progressStatus.textContent = 'Ergebnis wird finalisiert…';
        await delay(200);

        progressFill.style.width = '100%';
        progressStatus.textContent = '✓ Analyse abgeschlossen';

        const rawTextStr = bestResult?.data?.text || '(kein Text erkannt)';
        bestParsed = bestParsed || {};
        bestParsed.rawText = rawTextStr;

        if (bestParsed && bestParsed.bestMatch) {
          _ocrCache.set(cacheKey, bestParsed);
          if (_ocrCache.size > OCR_CACHE_MAX) {
            const firstKey = _ocrCache.keys().next().value;
            _ocrCache.delete(firstKey);
          }
        }

        renderOCRResult(bestParsed, rawTextStr, overlay, isKK);

        // Worker remains persistent (Improvement #2)

      } catch (innerErr) {
        throw innerErr;
      }

    } catch (err) {
      console.error('ImageCompare OCR error:', err);
      progressFill.style.width = '100%';
      progressFill.style.background = 'linear-gradient(90deg, #d04030, #f08070)';
      progressStatus.textContent = '⚠ Fehler: ' + (err.message || 'OCR fehlgeschlagen');

      await delay(1500);
      progress.classList.remove('active');
      resultCard.classList.add('active');
      detectedValue.textContent = '?';
      detectedLabel.textContent = 'OCR fehlgeschlagen – bitte Ergebnis manuell eingeben';
      scoreInput.value = '';
      scoreInput.focus();
    }

    _isProcessing = false;
  }

  function renderOCRResult(bestParsed, rawTextStr, overlay, isKK) {
    const progress = overlay.querySelector('#icProgress');
    const resultCard = overlay.querySelector('#icResultCard');
    const detectedValue = overlay.querySelector('#icDetectedValue');
    const detectedLabel = overlay.querySelector('#icDetectedLabel');
    const scoreInput = overlay.querySelector('#icScoreInput');
    const compareBtn = overlay.querySelector('#icCompareBtn');
    const rawText = overlay.querySelector('#icRawText');

    rawText.textContent = rawTextStr;

    progress.classList.remove('active');
    resultCard.classList.add('active');

    let exAlt = overlay.querySelector('.ic-alt-chips');
    if (exAlt) exAlt.remove();

    if (bestParsed?.bestMatch) {
      const best = bestParsed.bestMatch;
      const displayVal = isKK ? Math.floor(best.value) : best.value.toFixed(1);
      detectedValue.textContent = displayVal;
      const typeLabel = (best.type === 'decimal') ? 'Dezimalzahl' : 'Ganzzahl';
      detectedLabel.innerHTML = `Typ: ${typeLabel} · Konfidenz: ${Math.round(best.confidence * 100)}% `;
      scoreInput.value = displayVal;
      compareBtn.disabled = false;

      // Improvement #4: Alternative-Chips
      if (bestParsed.alternatives && bestParsed.alternatives.length > 0) {
        const altContainer = document.createElement('div');
        altContainer.className = 'ic-alt-chips';

        const lbl = document.createElement('div');
        lbl.className = 'ic-alt-label';
        lbl.textContent = 'Alternativen:';
        lbl.style.width = '100%';
        altContainer.appendChild(lbl);

        bestParsed.alternatives.forEach((alt) => {
          const altVal = isKK ? Math.floor(alt.value) : alt.value.toFixed(1);
          const btn = document.createElement('button');
          btn.className = 'ic-alt-chip';
          btn.textContent = altVal;
          btn.addEventListener('click', () => {
            scoreInput.value = altVal;
            detectedValue.textContent = altVal;
            compareBtn.disabled = false;
          });
          altContainer.appendChild(btn);
        });

        const editHint = overlay.querySelector('.ic-edit-hint');
        editHint.parentNode.insertBefore(altContainer, editHint);
      }
    } else {
      detectedValue.textContent = '?';
      detectedLabel.innerHTML = 'Keine Punktzahl erkannt – bitte manuell eingeben';
      scoreInput.value = '';
      scoreInput.focus();
      compareBtn.disabled = true;
    }
  }

  function resetUploadZone(overlay, isKK) {
    const uploadZone = overlay.querySelector('#icUploadZone');
    const progress = overlay.querySelector('#icProgress');
    const resultCard = overlay.querySelector('#icResultCard');
    const comparison = overlay.querySelector('#icComparison');
    const compareBtn = overlay.querySelector('#icCompareBtn');

    uploadZone.classList.remove('has-image');
    uploadZone.innerHTML = `
  < input type = "file" class="ic-upload-input" id = "icFileInput"
accept = "image/*" capture = "environment" >
      <span class="ic-upload-icon">📸</span>
      <div class="ic-upload-text">Foto der Ergebnisanzeige<br>hochladen oder aufnehmen</div>
      <div class="ic-upload-sub">JPG, PNG · Kamera oder Galerie</div>
`;
    progress.classList.remove('active');
    resultCard.classList.remove('active');
    comparison.classList.remove('active');
    compareBtn.disabled = true;

    // Re-attach file input listener
    const newInput = overlay.querySelector('#icFileInput');
    const botScore = parseFloat(overlay.dataset.botScore) || 0;
    newInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) handleImageFile(file, overlay, botScore, isKK);
    });
  }

  /* ─── COMPARISON RESULT ─────────────────── */
  function showComparison(overlay, playerScore, botScore, isKK) {
    const comparison = overlay.querySelector('#icComparison');
    if (!comparison) return;

    const diff = playerScore - botScore;
    const absDiff = Math.abs(diff);
    let resultClass, resultEmoji, resultText, diffText;

    if (diff > 0.05) {
      resultClass = 'win';
      resultEmoji = '🏆';
      resultText = 'DU GEWINNST!';
      diffText = isKK
        ? `+ ${Math.round(absDiff)} Ringe Vorsprung`
        : `+ ${absDiff.toFixed(1)} Punkte Vorsprung`;
    } else if (diff < -0.05) {
      resultClass = 'lose';
      resultEmoji = '😔';
      resultText = 'BOT GEWINNT';
      diffText = isKK
        ? `−${Math.round(absDiff)} Ringe Rückstand`
        : `−${absDiff.toFixed(1)} Punkte Rückstand`;
    } else {
      resultClass = 'draw';
      resultEmoji = '🤝';
      resultText = 'UNENTSCHIEDEN!';
      diffText = 'Punktgleich!';
    }

    // Calculate bar widths
    const total = playerScore + botScore;
    const playerPct = total > 0 ? Math.round((playerScore / total) * 100) : 50;
    const botPct = 100 - playerPct;

    const playerDisplay = isKK ? Math.floor(playerScore) : playerScore.toFixed(1);
    const botDisplay = isKK ? Math.floor(botScore) : botScore.toFixed(1);

    comparison.innerHTML = `
  < div style = "text-align:center;font-size:2rem;margin-bottom:-4px;" > ${resultEmoji}</div >
      <div class="ic-comp-title ${resultClass}">${resultText}</div>
      <div class="ic-comp-scores">
        <div class="ic-comp-side">
          <div class="ic-comp-who">👧 Du</div>
          <div class="ic-comp-pts player">${playerDisplay}</div>
        </div>
        <div class="ic-comp-vs">VS</div>
        <div class="ic-comp-side">
          <div class="ic-comp-who">🤖 Bot</div>
          <div class="ic-comp-pts bot">${botDisplay}</div>
        </div>
      </div>
      <div class="ic-bar-wrap">
        <div class="ic-bar-player" style="width:${playerPct}%"></div>
        <div class="ic-bar-bot" style="width:${botPct}%"></div>
      </div>
      <div class="ic-comp-diff ${resultClass}">${diffText}</div>

      <!--Submit Button-- >
  <button class="ic-submit-btn" id="icSubmitBtn">
    ✅ ERGEBNIS ÜBERNEHMEN
  </button>
`;

    comparison.classList.add('active');

    // Scroll comparison into view
    setTimeout(() => {
      comparison.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    // Submit Button Event
    const submitBtn = comparison.querySelector('#icSubmitBtn');
    submitBtn.addEventListener('click', () => {
      closeOverlay();

      // Get inputs directly from document to be safe
      const playerInp = document.getElementById('playerInp');
      const playerInpInt = document.getElementById('playerInpInt');
      const calcFunc = window.calcResult || (typeof calcResult === 'function' ? calcResult : null);

      if (playerInp || playerInpInt) {
        if (isKK && playerInpInt) {
          playerInpInt.value = Math.floor(playerScore);
        } else if (!isKK) {
          if (playerInp) playerInp.value = playerScore.toFixed(1);
          if (playerInpInt) playerInpInt.value = Math.floor(playerScore);
        }

        // Directly trigger the result calculation in index.html
        if (typeof calcFunc === 'function') {
          calcFunc();
        } else if (typeof window.showGameOver === 'function') {
          window.showGameOver(playerScore, botScore, null, Math.floor(playerScore));
        }
      }
    });

    // Play sound if available
    if (typeof Sounds !== 'undefined') {
      setTimeout(() => {
        if (resultClass === 'win') Sounds.win();
        else if (resultClass === 'lose') Sounds.lose();
        else Sounds.draw();
      }, 300);
    }
    if (typeof Haptics !== 'undefined') {
      setTimeout(() => {
        if (resultClass === 'win') Haptics.win();
        else if (resultClass === 'lose') Haptics.lose();
        else Haptics.draw();
      }, 300);
    }
  }

  /* ─── UTILITIES ──────────────────────────── */
  function delay(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  /* ─── PUBLIC API ─────────────────────────── */
  return {
    /**
     * Initialize: inject CSS styles
     */
    init() {
      injectStyles();
    },

    /**
     * Open the image upload & comparison overlay.
     * @param {number} botScore - The bot's total score from the current game
     * @param {boolean} isKK - Whether the current weapon is KK (integer scoring)
     * @param {string} discipline - Context string ('LG_40', 'KK_30') to enforce validation limits
     */
    open(botScore, isKK, discipline = null) {
      // Pre-warm: Tesseract + TF.js parallel vorladen
      ensureTesseract().catch(e => console.warn('[ImageCompare] Tesseract pre-warm fehlgeschlagen:', e));
      loadTFModel().catch(() => { }); // Modell im Hintergrund laden (Fehler = OK)
      injectStyles();
      const overlay = createOverlay(botScore || 0, !!isKK);
      overlay.dataset.botScore = botScore || 0;
      if (discipline) {
        overlay.dataset.discipline = discipline;
      }

      const fileInput = overlay.querySelector('#icFileInput');
      if (fileInput) {
        fileInput.addEventListener('change', (e) => {
          if (e.target.files && e.target.files.length > 0) {
            handleImageFile(e.target.files[0], overlay, botScore, isKK);
          }
        });
      }
    },

    /**
     * Create a small upload button for the game-over screen
     * @param {HTMLElement} container - The DOM element to insert the button into
     * @param {number} botScore - The bot's total score
     * @param {boolean} isKK - Whether scoring is integer (KK)
     * @param {string} discipline - Context string ('LG_40', 'KK_30') to enforce validation limits
     */
    createGameOverButton(container, botScore, isKK, discipline = null) {
      if (!container) return;
      injectStyles();

      // Don't add duplicate buttons
      if (container.querySelector('.ic-go-upload-btn')) return;

      const btn = document.createElement('button');
      btn.className = 'ic-go-upload-btn';
      btn.innerHTML = '<span class="ic-go-upload-ico">�</span> Wettkampf-Foto vergleichen';
      btn.addEventListener('click', () => {
        this.open(botScore, isKK, discipline);
      });

      container.appendChild(btn);
    }
  };
})();
