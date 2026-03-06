/* ═══════════════════════════════════════════════════════════════════════
   IMAGE-COMPARE.JS — KI-gestützte Bilderkennung & Ergebnis-Vergleich
   100% kostenlos · Tesseract.js (Open-Source OCR) · Läuft im Browser
   ═══════════════════════════════════════════════════════════════════════
   Architektur:
     1. KONFIGURATION   — Alle Konstanten & Schwellwerte zentral
     2. PREPROCESSING    — Modulare Bild-Pipeline (Graustufen → Schärfung → Threshold)
     3. OCR-ENGINE       — Tesseract.js Lazy-Loading & Multi-Pass-Erkennung
     4. SCORE-PARSING    — Regelbasierte Punktzahl-Extraktion mit Gewichtung
     5. UI-RENDERING     — Overlay, Upload, Fortschritt, Vergleich
     6. PUBLIC API       — init(), open(), createGameOverButton()
   ═══════════════════════════════════════════════════════════════════════ */

window.ImageCompare = (function () {
  'use strict';

  /* ─── PRIVATE STATE ──────────────────────── */
  let _isProcessing = false;

  /* ═══ 1. KONFIGURATION ═══════════════════════════════════════════════ */

  /** Zentrale Score-Konfiguration — alle Schwellwerte an einem Ort */
  const SCORE_CONFIG = {
    VALID_RANGE: { min: 10, max: 660 },

    /* Gewichtungsfaktoren für die Konfidenz-Berechnung */
    WEIGHTS: {
      CENTER_FACTOR: 0.5,   // Bonus für Nähe zur Bildmitte
      KEYWORD_NEAR_BONUS: 1.0,   // Bonus wenn Score nah an Schlüsselwort
      KEYWORD_MAX_DIST: 15,    // Max. Zeichenabstand für Keyword-Bonus
      LABELED_TYPE_BONUS: 0.5,   // Bonus für beschriftete Werte
      FORMAT_MATCH_BONUS: 0.3,   // Bonus wenn Format zur Waffe passt
      GOOD_GEOMETRY_BONUS: 1.2,   // Bonus für gutes Seitenverhältnis
      BAD_GEOMETRY_PENALTY: 0.5,  // Malus für unplausibles Seitenverhältnis
    },

    /* Schlüsselwörter die auf einen Gesamtscore hindeuten */
    KEYWORDS: ['gesamt', 'total', 'summe', 'ergebnis', 'result', 'ringe', 'pkt'],

    /* Plausibles Seitenverhältnis (Höhe/Breite) für Ziffern */
    GEOMETRY: { MIN: 0.8, MAX: 4.0, GOOD_MIN: 1.2, GOOD_MAX: 2.8 },
  };

  /** OCR-Zeichenkorrekturen: häufige Fehllesungen von Ziffern */
  const OCR_CHAR_FIXES = [
    [/[oO]/g, '0'],
    [/[lI|]/g, '1'],
    [/[sS](?=\d)/g, '5'],
    [/[,]/g, '.'],
  ];

  /** Multi-Pass-OCR Strategie — jeder Pass hat eigene Vorverarbeitungsparameter */
  const OCR_PASSES = [
    { name: 'Standard', options: {}, triggerBelow: 1.0 },
    { name: 'Gamma-Boost', options: { gamma: 1.5 }, triggerBelow: 0.85 },
    { name: 'Invertiert', options: { invert: true }, triggerBelow: 0.7 },
  ];

  /* ─── Textbereinigung (zentral, einmalig definiert) ─── */
  function cleanOCRText(text) {
    let clean = text;
    for (const [pattern, replacement] of OCR_CHAR_FIXES) {
      clean = clean.replace(pattern, replacement);
    }
    return clean.replace(/\s+/g, ' ');
  }

  /* ─── PRIVATE STATE (erweitert) ─── */
  let _worker = null;          // Improvement #2: Worker-Singleton
  const _ocrCache = new Map(); // Improvement #6: OCR-Ergebnis-Cache
  const OCR_CACHE_MAX = 5;

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
      tessedit_char_whitelist: '0123456789., \n',
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
    return `${file.name}_${file.size}_${file.lastModified}`;
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
          const gray = Math.round(d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114);
          d[i] = d[i+1] = d[i+2] = gray;
        }
      }
      function applyGamma(d, gamma) {
        if (!gamma || gamma === 1.0) return;
        const inv = 1 / gamma;
        for (let i = 0; i < d.length; i += 4) {
          const corrected = Math.round(255 * Math.pow(d[i] / 255, inv));
          d[i] = d[i+1] = d[i+2] = corrected;
        }
      }
      function sharpen(d, w, h) {
        const out = new Uint8ClampedArray(d.length);
        for (let y = 1; y < h - 1; y++) {
          for (let x = 1; x < w - 1; x++) {
            const i = (y * w + x) * 4;
            const val = Math.max(0, Math.min(255,
              5 * d[i] - d[((y - 1) * w + x) * 4] - d[((y + 1) * w + x) * 4] - d[(y * w + x - 1) * 4] - d[(y * w + x + 1) * 4]
            ));
            out[i] = out[i+1] = out[i+2] = val;
            out[i+3] = 255;
          }
        }
        for (let y = 1; y < h - 1; y++) {
          for (let x = 1; x < w - 1; x++) {
            const i = (y * w + x) * 4;
            d[i] = out[i]; d[i+1] = out[i+1]; d[i+2] = out[i+2];
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
          d[i] = d[i+1] = d[i+2] = v;
        }
      }
      function invert(d) {
        for (let i = 0; i < d.length; i += 4) {
          d[i] = 255 - d[i]; d[i+1] = 255 - d[i+1]; d[i+2] = 255 - d[i+2];
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
            d[idx] = d[idx+1] = d[idx+2] = val;
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
          d[idx] = d[idx+1] = d[idx+2] = 255;
        }
      }

      self.onmessage = function(e) {
        const { id, imageData, w, h, options } = e.data;
        try {
          const d = imageData.data;
          toGrayscale(d);
          applyGamma(d, options.gamma);
          sharpen(d, w, h);
          stretchContrast(d);
          if (options.invert) invert(d);
          adaptiveThreshold(d, w, h);
          removeNoise(d, w, h);
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
  function parseShootingScore(ocrResult, isKK, canvasW, canvasH) {
    if (!ocrResult?.data?.text) return null;

    const cleanText = cleanOCRText(ocrResult.data.text);
    const words = ocrResult.data.words || [];
    const { min, max } = SCORE_CONFIG.VALID_RANGE;
    const candidates = [];

    // Quelle 1: Individuelle Wörter mit Bounding-Box-Daten
    for (const word of words) {
      const cleaned = cleanOCRText(word.text);
      const val = parseFloat(cleaned);
      if (isNaN(val) || val < min || val > max) continue;

      const type = cleaned.includes('.') ? 'decimal' : 'integer';
      const geoWeight = calculateGeometryWeight(word.bbox);
      const ctxWeight = calculateCandidateWeight(word.text, word.bbox, type, isKK, canvasW, canvasH, cleanText);
      const confidence = (word.confidence / 100) * ctxWeight * geoWeight;

      candidates.push({ value: val, type, confidence, bbox: word.bbox });
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
      <div class="ic-sheet">
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
      </div>
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
      <div class="ic-preview-wrap">
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

      progressFill.style.width = '15%';
      progressStatus.textContent = 'Lade Tesseract.js OCR-Engine…';

      _ocrProgressCallback = (prog) => {
        const pct = Math.round(30 + prog * 50);
        progressFill.style.width = pct + '%';
        progressStatus.textContent = `Texterkennung… ${Math.round(prog * 100)}%`;
      };

      const worker = await getWorker();

      progressFill.style.width = '25%';
      progressStatus.textContent = 'Erkenne interessanten Bereich (ROI)…';

      try {
        let bestParsed = null;
        let bestResult = null;
        let bestConf = 0;

        // Improvement #3: ROI-Erkennung
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

        let crop = null;
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

        for (let i = 0; i < OCR_PASSES.length; i++) {
          const pass = OCR_PASSES[i];
          if (bestConf >= pass.triggerBelow) continue;

          progressStatus.textContent = `Analysiere Bild (${pass.name}, Pass ${i + 1}/${OCR_PASSES.length})…`;
          progressFill.style.width = Math.round(30 + (i / OCR_PASSES.length) * 60) + '%';

          const prepOptions = { ...pass.options, crop };
          const prep = await PREPROCESS.run(img, prepOptions);
          const result = await worker.recognize(prep.dataUrl);
          const parsed = parseShootingScore(result, isKK, prep.width, prep.height);

          if (parsed?.bestMatch && parsed.bestMatch.confidence > bestConf) {
            bestParsed = parsed;
            bestResult = result;
            bestConf = parsed.bestMatch.confidence;
          }
        }

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
      detectedLabel.innerHTML = `Typ: ${typeLabel} · Konfidenz: ${Math.round(best.confidence * 100)}%`;
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
      <input type="file" class="ic-upload-input" id="icFileInput"
             accept="image/*" capture="environment">
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
        ? `+${Math.round(absDiff)} Ringe Vorsprung`
        : `+${absDiff.toFixed(1)} Punkte Vorsprung`;
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
      <div style="text-align:center;font-size:2rem;margin-bottom:-4px;">${resultEmoji}</div>
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

      <!-- Submit Button -->
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
     */
    open(botScore, isKK) {
      injectStyles();
      const overlay = createOverlay(botScore || 0, !!isKK);
      overlay.dataset.botScore = botScore || 0;
    },

    /**
     * Create a small upload button for the game-over screen
     * @param {HTMLElement} container - The DOM element to insert the button into
     * @param {number} botScore - The bot's total score
     * @param {boolean} isKK - Whether scoring is integer (KK)
     */
    createGameOverButton(container, botScore, isKK) {
      if (!container) return;
      injectStyles();

      // Don't add duplicate buttons
      if (container.querySelector('.ic-go-upload-btn')) return;

      const btn = document.createElement('button');
      btn.className = 'ic-go-upload-btn';
      btn.innerHTML = '<span class="ic-go-upload-ico">📷</span> Wettkampf-Foto vergleichen';
      btn.addEventListener('click', () => {
        ImageCompare.open(botScore, isKK);
      });

      container.appendChild(btn);
    }
  };
})();
