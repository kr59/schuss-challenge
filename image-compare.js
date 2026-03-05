/* ═══════════════════════════════════════════════════════════════════════
   IMAGE-COMPARE.JS — KI-gestützte Bilderkennung & Ergebnis-Vergleich
   100% kostenlos · Tesseract.js (Open-Source OCR) · Läuft im Browser
   ═══════════════════════════════════════════════════════════════════════ */

window.ImageCompare = (function () {
    'use strict';

    /* ─── PRIVATE STATE ──────────────────────── */
    let _tesseractReady = false;
    let _worker = null;
    let _isProcessing = false;

    /* ─── STYLES (injected once) ─────────────── */
    const CSS_ID = 'ic-styles-injected';

    function injectStyles() {
        if (document.getElementById(CSS_ID)) return;
        const style = document.createElement('style');
        style.id = CSS_ID;
        style.textContent = `
      /* ═══ IMAGE COMPARE OVERLAY ═══ */
      .ic-overlay {
        position: fixed;
        inset: 0;
        z-index: 950;
        background: rgba(0,0,0,.7);
        backdrop-filter: blur(6px);
        display: flex;
        align-items: flex-end;
        justify-content: center;
        animation: ic-fadeIn .2s ease;
      }
      @keyframes ic-fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
      }
      .ic-sheet {
        width: 100%;
        max-width: 480px;
        max-height: 92vh;
        background: linear-gradient(180deg, #0f1a08 0%, #0a1205 100%);
        border: 1px solid rgba(120,180,40,.22);
        border-bottom: none;
        border-radius: 20px 20px 0 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 -8px 48px rgba(0,0,0,.7), inset 0 1px 0 rgba(160,220,60,.08);
        animation: ic-sheetUp .28s cubic-bezier(.32,1.2,.64,1);
      }
      @keyframes ic-sheetUp {
        from { transform: translateY(60px); opacity: 0; }
        to   { transform: translateY(0); opacity: 1; }
      }
      .ic-handle {
        width: 36px;
        height: 4px;
        background: rgba(120,180,40,.2);
        border-radius: 2px;
        margin: 10px auto 0;
        flex-shrink: 0;
      }
      .ic-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px 10px;
        flex-shrink: 0;
        border-bottom: 1px solid rgba(120,180,40,.12);
      }
      .ic-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.3rem;
        letter-spacing: .12em;
        color: #8ecf40;
        text-shadow: 0 0 12px rgba(120,200,50,.25);
      }
      .ic-close {
        background: rgba(120,180,40,.1);
        border: 1px solid rgba(120,180,40,.2);
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: rgba(160,200,80,.6);
        font-size: .9rem;
        transition: background .2s, color .2s;
      }
      .ic-close:hover {
        background: rgba(120,180,40,.2);
        color: #fff;
      }
      .ic-body {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 16px 18px 24px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        -webkit-overflow-scrolling: touch;
      }

      /* Upload Zone */
      .ic-upload-zone {
        border: 2px dashed rgba(120,180,40,.3);
        border-radius: 16px;
        padding: 28px 16px;
        text-align: center;
        cursor: pointer;
        transition: all .25s ease;
        background: rgba(120,180,40,.04);
        position: relative;
      }
      .ic-upload-zone:hover, .ic-upload-zone.dragover {
        border-color: rgba(120,180,40,.6);
        background: rgba(120,180,40,.1);
      }
      .ic-upload-zone.has-image {
        padding: 8px;
        border-style: solid;
        border-color: rgba(120,180,40,.25);
      }
      .ic-upload-icon {
        font-size: 2.4rem;
        margin-bottom: 8px;
        display: block;
      }
      .ic-upload-text {
        font-size: .82rem;
        color: rgba(200,230,120,.65);
        font-weight: 500;
        line-height: 1.5;
      }
      .ic-upload-sub {
        font-size: .62rem;
        color: rgba(160,200,80,.35);
        margin-top: 4px;
        letter-spacing: .05em;
      }
      .ic-upload-input {
        position: absolute;
        inset: 0;
        opacity: 0;
        cursor: pointer;
        width: 100%;
        height: 100%;
      }
      .ic-preview-wrap {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        max-height: 240px;
      }
      .ic-preview-img {
        width: 100%;
        height: auto;
        max-height: 240px;
        object-fit: contain;
        display: block;
        border-radius: 12px;
      }
      .ic-remove-img {
        position: absolute;
        top: 6px;
        right: 6px;
        background: rgba(0,0,0,.6);
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: #fff;
        font-size: .8rem;
        transition: background .2s;
      }
      .ic-remove-img:hover {
        background: rgba(200,60,60,.7);
      }

      /* Progress */
      .ic-progress {
        display: none;
        flex-direction: column;
        gap: 8px;
        padding: 12px 14px;
        background: rgba(120,180,40,.06);
        border: 1px solid rgba(120,180,40,.12);
        border-radius: 12px;
      }
      .ic-progress.active { display: flex; }
      .ic-progress-label {
        font-size: .65rem;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: rgba(160,200,80,.5);
        font-weight: 600;
      }
      .ic-progress-bar {
        height: 6px;
        background: rgba(120,180,40,.12);
        border-radius: 3px;
        overflow: hidden;
      }
      .ic-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3a8010, #8ecf40);
        border-radius: 3px;
        width: 0%;
        transition: width .3s ease;
      }
      .ic-progress-status {
        font-size: .72rem;
        color: rgba(200,230,120,.6);
      }

      /* Score Result */
      .ic-result-card {
        display: none;
        flex-direction: column;
        gap: 10px;
        padding: 14px;
        background: rgba(120,180,40,.06);
        border: 1px solid rgba(120,180,40,.15);
        border-radius: 14px;
      }
      .ic-result-card.active { display: flex; }
      .ic-result-header {
        font-size: .6rem;
        letter-spacing: .22em;
        text-transform: uppercase;
        color: rgba(160,200,80,.45);
        font-weight: 600;
      }
      .ic-detected-score {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .ic-detected-icon {
        font-size: 1.6rem;
        flex-shrink: 0;
      }
      .ic-detected-info {
        flex: 1;
        min-width: 0;
      }
      .ic-detected-value {
        font-family: 'Bebas Neue', cursive;
        font-size: 2rem;
        color: #8ecf40;
        line-height: 1;
        text-shadow: 0 0 15px rgba(120,200,50,.3);
      }
      .ic-detected-label {
        font-size: .58rem;
        color: rgba(160,200,80,.4);
        margin-top: 2px;
        letter-spacing: .1em;
      }
      .ic-edit-score {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 4px;
      }
      .ic-score-input {
        flex: 1;
        background: rgba(0,0,0,.3);
        border: 1px solid rgba(120,180,40,.25);
        border-radius: 8px;
        padding: 8px 12px;
        color: #e8e8e0;
        font-family: 'DM Mono', monospace;
        font-size: 1rem;
        outline: none;
        transition: border-color .2s;
      }
      .ic-score-input:focus {
        border-color: rgba(120,180,40,.5);
      }
      .ic-score-input::placeholder {
        color: rgba(160,200,80,.25);
      }
      .ic-edit-hint {
        font-size: .58rem;
        color: rgba(160,200,80,.3);
        font-style: italic;
      }

      /* Compare Button */
      .ic-compare-btn {
        width: 100%;
        padding: 13px 20px;
        background: radial-gradient(circle at center, #8ecf40, #5a8c1e);
        border: none;
        border-radius: 12px;
        color: #fff;
        font-family: 'Outfit', sans-serif;
        font-size: .9rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        cursor: pointer;
        text-shadow: 0 1px 3px rgba(0,0,0,.4);
        box-shadow: 0 4px 20px rgba(100,160,30,.25);
        transition: transform .15s, box-shadow .15s;
      }
      .ic-compare-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 28px rgba(100,160,30,.35);
      }
      .ic-compare-btn:active {
        transform: translateY(0);
      }
      .ic-compare-btn:disabled {
        opacity: .4;
        cursor: not-allowed;
        transform: none;
      }

      /* Comparison Result */
      .ic-comparison {
        display: none;
        flex-direction: column;
        gap: 12px;
        padding: 14px;
        background: rgba(120,180,40,.06);
        border: 1px solid rgba(120,180,40,.15);
        border-radius: 14px;
      }
      .ic-comparison.active { display: flex; }
      .ic-comp-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.1rem;
        letter-spacing: .1em;
        text-align: center;
      }
      .ic-comp-title.win { color: #8ecf40; text-shadow: 0 0 10px rgba(120,200,50,.25); }
      .ic-comp-title.lose { color: #f08070; text-shadow: 0 0 10px rgba(240,120,100,.25); }
      .ic-comp-title.draw { color: #ffc840; text-shadow: 0 0 10px rgba(255,200,60,.25); }

      .ic-comp-scores {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
      }
      .ic-comp-side {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 3px;
        flex: 1;
      }
      .ic-comp-who {
        font-size: .58rem;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: rgba(200,220,150,.4);
        font-weight: 600;
      }
      .ic-comp-pts {
        font-family: 'Bebas Neue', cursive;
        font-size: 2.2rem;
        line-height: 1;
      }
      .ic-comp-pts.player { color: #a0e060; text-shadow: 0 0 12px rgba(140,220,60,.25); }
      .ic-comp-pts.bot { color: #f08070; text-shadow: 0 0 12px rgba(240,120,100,.25); }
      .ic-comp-vs {
        font-family: 'Bebas Neue', cursive;
        font-size: 1rem;
        color: rgba(255,255,255,.2);
        letter-spacing: .1em;
      }
      .ic-comp-diff {
        text-align: center;
        font-size: .78rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 20px;
        display: inline-block;
        align-self: center;
      }
      .ic-comp-diff.win {
        background: rgba(120,200,50,.1);
        border: 1px solid rgba(120,200,50,.3);
        color: #a0e060;
      }
      .ic-comp-diff.lose {
        background: rgba(240,100,80,.1);
        border: 1px solid rgba(240,100,80,.3);
        color: #f08070;
      }
      .ic-comp-diff.draw {
        background: rgba(255,200,60,.1);
        border: 1px solid rgba(255,200,60,.3);
        color: #ffc840;
      }

      /* Visual Bar */
      .ic-bar-wrap {
        display: flex;
        align-items: center;
        gap: 6px;
        height: 24px;
      }
      .ic-bar-player {
        height: 100%;
        background: linear-gradient(90deg, #3a8010, #8ecf40);
        border-radius: 6px 0 0 6px;
        transition: width .6s cubic-bezier(.34,1.56,.64,1);
        min-width: 4px;
      }
      .ic-bar-bot {
        height: 100%;
        background: linear-gradient(90deg, #d04030, #f08070);
        border-radius: 0 6px 6px 0;
        transition: width .6s cubic-bezier(.34,1.56,.64,1);
        min-width: 4px;
      }

      /* OCR Raw Text (collapsed by default) */
      .ic-raw-toggle {
        font-size: .62rem;
        color: rgba(160,200,80,.35);
        cursor: pointer;
        text-align: center;
        padding: 4px;
        letter-spacing: .08em;
        transition: color .2s;
      }
      .ic-raw-toggle:hover { color: rgba(160,200,80,.6); }
      .ic-raw-text {
        display: none;
        background: rgba(0,0,0,.3);
        border: 1px solid rgba(120,180,40,.1);
        border-radius: 8px;
        padding: 10px 12px;
        font-family: 'DM Mono', monospace;
        font-size: .68rem;
        color: rgba(200,220,150,.45);
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-all;
        max-height: 160px;
        overflow-y: auto;
      }
      .ic-raw-text.visible { display: block; }

      /* Info hint */
      .ic-info {
        font-size: .62rem;
        color: rgba(160,200,80,.3);
        text-align: center;
        line-height: 1.5;
        padding: 0 8px;
      }

      /* Game Over Upload Button */
      .ic-go-upload-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        max-width: 320px;
        margin: 8px auto 4px;
        padding: 11px 18px;
        background: rgba(120,180,40,.1);
        border: 1px solid rgba(120,180,40,.25);
        border-radius: 12px;
        color: rgba(180,230,80,.8);
        font-family: 'Outfit', sans-serif;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .06em;
        cursor: pointer;
        transition: all .2s ease;
        position: relative;
        z-index: 1;
      }
      .ic-go-upload-btn:hover {
        background: rgba(120,180,40,.16);
        border-color: rgba(120,180,40,.4);
        color: #a0e060;
      }
      .ic-go-upload-btn:active {
        background: rgba(120,180,40,.22);
      }
      .ic-go-upload-ico { font-size: 1.1rem; }
    `;
        document.head.appendChild(style);
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

    /* ─── IMAGE PREPROCESSING ────────────────── */
    function preprocessImage(imgEl) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Scale down large images for faster OCR
        const MAX_DIM = 1200;
        let w = imgEl.naturalWidth || imgEl.width;
        let h = imgEl.naturalHeight || imgEl.height;
        if (w > MAX_DIM || h > MAX_DIM) {
            const scale = MAX_DIM / Math.max(w, h);
            w = Math.round(w * scale);
            h = Math.round(h * scale);
        }

        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(imgEl, 0, 0, w, h);

        // Get pixel data
        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data;

        // Step 1: Convert to grayscale
        for (let i = 0; i < data.length; i += 4) {
            const gray = Math.round(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
            data[i] = gray;
            data[i + 1] = gray;
            data[i + 2] = gray;
        }

        // Step 2: Enhance contrast (histogram stretch)
        let minVal = 255, maxVal = 0;
        for (let i = 0; i < data.length; i += 4) {
            if (data[i] < minVal) minVal = data[i];
            if (data[i] > maxVal) maxVal = data[i];
        }
        const range = maxVal - minVal || 1;
        for (let i = 0; i < data.length; i += 4) {
            const stretched = Math.round(((data[i] - minVal) / range) * 255);
            data[i] = stretched;
            data[i + 1] = stretched;
            data[i + 2] = stretched;
        }

        // Step 3: Adaptive thresholding (Otsu's method simplified)
        // Calculate histogram
        const hist = new Array(256).fill(0);
        for (let i = 0; i < data.length; i += 4) {
            hist[data[i]]++;
        }
        const totalPixels = data.length / 4;

        // Find optimal threshold via Otsu
        let sumTotal = 0;
        for (let i = 0; i < 256; i++) sumTotal += i * hist[i];

        let sumBg = 0, wBg = 0, wFg = 0;
        let bestVar = 0, bestThresh = 128;

        for (let t = 0; t < 256; t++) {
            wBg += hist[t];
            if (wBg === 0) continue;
            wFg = totalPixels - wBg;
            if (wFg === 0) break;

            sumBg += t * hist[t];
            const meanBg = sumBg / wBg;
            const meanFg = (sumTotal - sumBg) / wFg;
            const variance = wBg * wFg * (meanBg - meanFg) * (meanBg - meanFg);

            if (variance > bestVar) {
                bestVar = variance;
                bestThresh = t;
            }
        }

        // Apply threshold
        for (let i = 0; i < data.length; i += 4) {
            const val = data[i] > bestThresh ? 255 : 0;
            data[i] = val;
            data[i + 1] = val;
            data[i + 2] = val;
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas.toDataURL('image/png');
    }

    /* ─── SCORE PARSING ──────────────────────── */
    function parseShootingScore(rawText) {
        if (!rawText || typeof rawText !== 'string') return null;

        // Clean up OCR artifacts
        const text = rawText
            .replace(/[oO]/g, '0')    // Common OCR: o→0
            .replace(/[lI|]/g, '1')   // Common OCR: l,I,|→1
            .replace(/[sS](?=\d)/g, '5') // S before digit → 5
            .replace(/[,]/g, '.')     // Comma → decimal point
            .replace(/\s+/g, ' ');

        const scores = [];

        // Pattern 1: Decimal scores like "405.2", "392.5", "10.5"
        const decRegex = /(\d{2,3})[.,](\d)\b/g;
        let m;
        while ((m = decRegex.exec(text)) !== null) {
            const val = parseFloat(m[1] + '.' + m[2]);
            if (val >= 10 && val <= 660) {
                scores.push({ value: val, type: 'decimal', confidence: 0.9 });
            }
        }

        // Pattern 2: Whole numbers (typical for KK or total scores)
        const intRegex = /\b(\d{2,3})\b/g;
        while ((m = intRegex.exec(text)) !== null) {
            const val = parseInt(m[1], 10);
            // Filter for plausible shooting scores
            if (val >= 50 && val <= 600) {
                // Check it's not already captured as part of a decimal
                const alreadyCaptured = scores.some(s =>
                    Math.abs(s.value - val) < 1 || Math.abs(Math.floor(s.value) - val) === 0
                );
                if (!alreadyCaptured) {
                    scores.push({ value: val, type: 'integer', confidence: 0.7 });
                }
            }
        }

        // Pattern 3: Scores with "Ringe" / "Pkt" / "Punkte" label
        const labelRegex = /(\d{2,3}(?:[.,]\d)?)\s*(?:Ringe|Pkt|Punkte|Treffer|rings?|pts?|points?)/gi;
        while ((m = labelRegex.exec(text)) !== null) {
            const val = parseFloat(m[1].replace(',', '.'));
            if (val >= 10 && val <= 660) {
                // High confidence score since it has a label
                scores.push({ value: val, type: 'labeled', confidence: 0.95 });
            }
        }

        // Pattern 4: "Ergebnis" / "Gesamt" / "Total" followed by number
        const totalRegex = /(?:ergebnis|gesamt|total|summe|result)[:\s]*(\d{2,3}(?:[.,]\d)?)/gi;
        while ((m = totalRegex.exec(text)) !== null) {
            const val = parseFloat(m[1].replace(',', '.'));
            if (val >= 50 && val <= 660) {
                scores.push({ value: val, type: 'total', confidence: 0.95 });
            }
        }

        if (scores.length === 0) return null;

        // Sort by confidence, then by value (higher = more likely total score)
        scores.sort((a, b) => {
            if (b.confidence !== a.confidence) return b.confidence - a.confidence;
            return b.value - a.value;
        });

        // Remove duplicates (values within 0.5 of each other)
        const unique = [];
        for (const s of scores) {
            if (!unique.some(u => Math.abs(u.value - s.value) < 0.5)) {
                unique.push(s);
            }
        }

        return {
            bestMatch: unique[0],
            alternatives: unique.slice(1, 4),
            allScores: unique
        };
    }

    /* ─── UI RENDERING ───────────────────────── */

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

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadZone.classList.add('has-image');
            uploadZone.innerHTML = `
        <div class="ic-preview-wrap">
          <img class="ic-preview-img" src="${e.target.result}" alt="Upload" id="icPreviewImg">
          <div class="ic-remove-img" id="icRemoveImg" title="Bild entfernen">✕</div>
        </div>
      `;
            const removeBtn = overlay.querySelector('#icRemoveImg');
            if (removeBtn) {
                removeBtn.addEventListener('click', (ev) => {
                    ev.stopPropagation();
                    resetUploadZone(overlay, isKK);
                    _isProcessing = false;
                });
            }
        };
        reader.readAsDataURL(file);

        // Show progress
        progress.classList.add('active');
        resultCard.classList.remove('active');
        progressFill.style.width = '5%';
        progressStatus.textContent = 'Bild wird vorbereitet…';

        try {
            // Wait for image to load
            const img = new Image();
            const imgUrl = URL.createObjectURL(file);
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = imgUrl;
            });

            progressFill.style.width = '15%';
            progressStatus.textContent = 'Bildvorverarbeitung (Kontrast, Threshold)…';
            await delay(300);

            // Preprocess
            const processedDataUrl = preprocessImage(img);
            URL.revokeObjectURL(imgUrl);

            progressFill.style.width = '25%';
            progressStatus.textContent = 'Lade Tesseract.js OCR-Engine…';

            // Load Tesseract
            await ensureTesseract();

            progressFill.style.width = '35%';
            progressStatus.textContent = 'OCR-Engine wird initialisiert…';

            // Create worker
            const worker = await Tesseract.createWorker('deu+eng', 1, {
                logger: (info) => {
                    if (info.status === 'recognizing text') {
                        const pct = Math.round(35 + info.progress * 55);
                        progressFill.style.width = pct + '%';
                        progressStatus.textContent = `Texterkennung… ${Math.round(info.progress * 100)}%`;
                    }
                }
            });

            progressFill.style.width = '40%';
            progressStatus.textContent = 'Texterkennung läuft…';

            // Recognize
            const result = await worker.recognize(processedDataUrl);
            const ocrText = result.data.text;

            progressFill.style.width = '95%';
            progressStatus.textContent = 'Ergebnis wird analysiert…';
            await delay(200);

            // Terminate worker to free memory
            await worker.terminate();

            // Parse scores
            const parsed = parseShootingScore(ocrText);

            progressFill.style.width = '100%';
            progressStatus.textContent = '✓ Analyse abgeschlossen';

            // Show raw OCR text
            rawText.textContent = ocrText || '(kein Text erkannt)';

            // Show result
            await delay(400);
            progress.classList.remove('active');
            resultCard.classList.add('active');

            if (parsed && parsed.bestMatch) {
                const best = parsed.bestMatch;
                const displayVal = isKK ? Math.floor(best.value) : best.value.toFixed(1);
                detectedValue.textContent = displayVal;
                detectedLabel.textContent = `Typ: ${best.type === 'labeled' || best.type === 'total' ? 'Beschrifteter Wert' : best.type === 'decimal' ? 'Dezimalzahl' : 'Ganzzahl'} · Konfidenz: ${Math.round(best.confidence * 100)}%`;
                scoreInput.value = displayVal;
                compareBtn.disabled = false;

                // Show alternatives if any
                if (parsed.alternatives.length > 0) {
                    detectedLabel.textContent += ` · ${parsed.alternatives.length} Alternative(n)`;
                }
            } else {
                detectedValue.textContent = '?';
                detectedLabel.textContent = 'Keine Punktzahl erkannt – bitte manuell eingeben';
                scoreInput.value = '';
                scoreInput.focus();
                compareBtn.disabled = true;
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
    `;

        comparison.classList.add('active');

        // Scroll comparison into view
        setTimeout(() => {
            comparison.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);

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
