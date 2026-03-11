window.ImageCompare = (function () {
  'use strict';

  const Brain = window.ImageCompareBrain || null;
  const SCORE_CONFIG = Brain && Brain.SCORE_CONFIG ? Brain.SCORE_CONFIG : null;

  const CSS_ID = 'ic-styles';
  const TESSERACT_SRC = 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js';
  const OCR_CACHE_MAX = 5;

  let _isProcessing = false;
  let _worker = null;
  let _ocrProgressCallback = null;
  const _ocrCache = new Map();

  function injectStyles() {
    if (document.getElementById(CSS_ID)) return;
    const link = document.createElement('link');
    link.id = CSS_ID;
    link.rel = 'stylesheet';
    link.href = 'image-compare.css';
    document.head.appendChild(link);
  }

  function ensureTesseract() {
    return new Promise((resolve, reject) => {
      if (typeof Tesseract !== 'undefined') {
        resolve();
        return;
      }

      const existing = document.querySelector('script[data-ic-tesseract]');
      if (existing) {
        const check = setInterval(() => {
          if (typeof Tesseract !== 'undefined') {
            clearInterval(check);
            resolve();
          }
        }, 120);
        setTimeout(() => {
          clearInterval(check);
          reject(new Error('Tesseract load timeout'));
        }, 30000);
        return;
      }

      const sc = document.createElement('script');
      sc.src = TESSERACT_SRC;
      sc.dataset.icTesseract = '1';
      sc.onload = () => resolve();
      sc.onerror = () => reject(new Error('Tesseract could not be loaded'));
      document.head.appendChild(sc);
    });
  }

  async function getWorker() {
    if (_worker) return _worker;

    await ensureTesseract();

    _worker = await Tesseract.createWorker('deu+eng', 1, {
      logger: (info) => {
        if (info && info.status === 'recognizing text' && _ocrProgressCallback) {
          _ocrProgressCallback(info.progress || 0);
        }
      }
    });

    if (_worker && _worker.setParameters) {
      await _worker.setParameters({
        tessedit_char_whitelist: '0123456789., OolI|Ss\n'
      });
    }

    return _worker;
  }

  function getDisciplineConfig(isKK, discipline) {
    const fallback = isKK
      ? { min: 50, max: 600, isInteger: true }
      : { min: 50, max: 654, isInteger: false };

    if (!SCORE_CONFIG || !SCORE_CONFIG.DISCIPLINES || !discipline) return fallback;
    return SCORE_CONFIG.DISCIPLINES[discipline] || fallback;
  }

  function normalizeOCRText(text) {
    let clean = (text || '').replace(/\r/g, ' ').replace(/\n/g, ' ');
    clean = clean.replace(/[oO]/g, '0');
    clean = clean.replace(/[lI|]/g, '1');
    clean = clean.replace(/,/g, '.');
    clean = clean.replace(/\s+/g, ' ').trim();
    return clean;
  }

  function parseScoreFromText(text, isKK, discipline) {
    const clean = normalizeOCRText(text);
    const cfg = getDisciplineConfig(isKK, discipline);
    const min = cfg.min;
    const max = cfg.max;

    const candidates = [];
    const decimalRegex = /(\d{2,3})\.(\d)\b/g;
    let m;

    while ((m = decimalRegex.exec(clean)) !== null) {
      const value = parseFloat(m[1] + '.' + m[2]);
      if (value >= min && value <= max) {
        candidates.push({ value, confidence: 0.9, type: 'decimal' });
      }
    }

    const intRegex = /\b(\d{2,3})\b/g;
    while ((m = intRegex.exec(clean)) !== null) {
      const value = parseInt(m[1], 10);
      if (value >= min && value <= max) {
        candidates.push({ value, confidence: isKK ? 0.92 : 0.65, type: 'integer' });
      }
    }

    if (candidates.length === 0) {
      return { bestMatch: null, alternatives: [] };
    }

    candidates.sort((a, b) => b.confidence - a.confidence);
    const unique = [];
    for (const c of candidates) {
      if (!unique.some(u => Math.abs(u.value - c.value) < 0.1)) {
        unique.push(c);
      }
    }

    let best = unique[0];
    if (!isKK) {
      const preferredDec = unique.find(c => c.type === 'decimal');
      if (preferredDec) best = preferredDec;
    }

    return { bestMatch: best, alternatives: unique.filter(c => c !== best).slice(0, 3) };
  }

  function getCacheKey(file) {
    return [file.name, file.size, file.lastModified].join('|');
  }

  function updateProgress(overlay, pct, statusText) {
    const progress = overlay.querySelector('#icProgress');
    const progressFill = overlay.querySelector('#icProgressFill');
    const progressStatus = overlay.querySelector('#icProgressStatus');
    if (progress) progress.classList.add('active');
    if (progressFill) progressFill.style.width = Math.max(0, Math.min(100, pct)) + '%';
    if (progressStatus) progressStatus.textContent = statusText;
  }

  function createOverlay(botScore, isKK) {
    const overlay = document.getElementById('icOverlay');
    if (!overlay) {
      console.error('[ImageCompare] #icOverlay not found in index.html');
      return null;
    }

    resetUploadZone(overlay, isKK);

    overlay.style.opacity = '1';
    overlay.style.pointerEvents = 'auto';
    overlay.style.transition = 'opacity .2s';
    overlay.dataset.botScore = String(botScore || 0);
    overlay.dataset.isKK = isKK ? 'true' : 'false';

    if (!overlay.dataset.eventsAttached) {
      setupOverlayEvents(overlay);
      overlay.dataset.eventsAttached = 'true';
    }

    return overlay;
  }

  function setupOverlayEvents(overlay) {
    const closeBtn = overlay.querySelector('#icClose');
    const uploadZone = overlay.querySelector('#icUploadZone');
    const compareBtn = overlay.querySelector('#icCompareBtn');
    const rawToggle = overlay.querySelector('#icRawToggle');
    const rawText = overlay.querySelector('#icRawText');
    const scoreInput = overlay.querySelector('#icScoreInput');
    const btnWrong = overlay.querySelector('#icBtnWrong');
    const editScoreBlock = overlay.querySelector('#icEditScoreBlock');
    const sheet = overlay.querySelector('.ic-sheet');

    if (!uploadZone || !compareBtn || !rawToggle || !rawText || !scoreInput || !sheet) return;

    if (closeBtn) {
      closeBtn.addEventListener('click', () => closeOverlay());
    }

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeOverlay();
    });

    uploadZone.addEventListener('change', (e) => {
      if (e.target && e.target.id === 'icFileInput') {
        const file = e.target.files && e.target.files[0];
        if (file) {
          handleImageFile(file, overlay);
        }
      }
    });

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
      const file = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files[0] : null;
      if (file && file.type && file.type.startsWith('image/')) {
        handleImageFile(file, overlay);
      }
    });

    if (btnWrong && editScoreBlock) {
      btnWrong.addEventListener('click', () => {
        btnWrong.style.display = 'none';
        editScoreBlock.style.display = 'block';
        scoreInput.focus();
      });
    }

    scoreInput.addEventListener('input', () => {
      const val = parseFloat(String(scoreInput.value).replace(',', '.'));
      compareBtn.disabled = Number.isNaN(val) || val < 0;
    });

    scoreInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        compareBtn.click();
      }
    });

    rawToggle.addEventListener('click', () => {
      rawText.classList.toggle('visible');
      rawToggle.textContent = rawText.classList.contains('visible')
        ? '▼ OCR-Rohtext ausblenden'
        : '▶ OCR-Rohtext anzeigen';
    });

    compareBtn.addEventListener('click', () => {
      const isKK = overlay.dataset.isKK === 'true';
      const botScore = parseFloat(overlay.dataset.botScore) || 0;
      const playerScore = parseFloat(String(scoreInput.value).replace(',', '.'));
      if (Number.isNaN(playerScore) || playerScore < 0) {
        scoreInput.style.borderColor = 'rgba(240,80,60,.6)';
        setTimeout(() => { scoreInput.style.borderColor = ''; }, 1200);
        return;
      }

      const detected = overlay.dataset.detectedScore ? parseFloat(overlay.dataset.detectedScore) : NaN;
      if (Brain && Brain.FEEDBACK_ENABLED && overlay._currentFile && !Number.isNaN(detected) && Math.abs(detected - playerScore) > 0.0001) {
        sendToFormspree(overlay._currentFile, playerScore, detected);
      }

      closeOverlay();

      const playerInp = document.getElementById('playerInp');
      const playerInpInt = document.getElementById('playerInpInt');

      if (isKK) {
        if (playerInpInt) playerInpInt.value = String(Math.floor(playerScore));
      } else {
        if (playerInp) playerInp.value = playerScore.toFixed(1);
        if (playerInpInt) playerInpInt.value = String(Math.floor(playerScore));
      }

      if (typeof window.calcResult === 'function') {
        window.calcResult();
      } else if (typeof window.showGameOver === 'function') {
        window.showGameOver(playerScore, botScore, null, Math.floor(playerScore));
      }
    });

    let startY = 0;
    sheet.addEventListener('touchstart', (e) => {
      startY = e.touches[0].clientY;
    }, { passive: true });

    sheet.addEventListener('touchend', (e) => {
      const dy = e.changedTouches[0].clientY - startY;
      if (dy > 80) closeOverlay();
    }, { passive: true });
  }

  function closeOverlay() {
    const overlay = document.getElementById('icOverlay');
    if (overlay) {
      overlay.style.opacity = '0';
      overlay.style.pointerEvents = 'none';
      const isKK = overlay.dataset.isKK === 'true';
      resetUploadZone(overlay, isKK);
    }
    _isProcessing = false;
  }

  async function handleImageFile(file, overlay) {
    if (_isProcessing) return;
    _isProcessing = true;

    const isKK = overlay.dataset.isKK === 'true';
    const discipline = overlay.dataset.discipline || null;
    const cacheKey = getCacheKey(file);

    const objectUrl = URL.createObjectURL(file);
    let revoked = false;
    const safeRevoke = () => {
      if (revoked) return;
      revoked = true;
      URL.revokeObjectURL(objectUrl);
    };

    try {
      updateProgress(overlay, 10, 'Bild wird vorbereitet...');

      // ROI Detection & Preprocessing
      const croppedImage = await detectAndCropROI(objectUrl);
      const processingUrl = croppedImage || objectUrl;

      uploadZone.classList.add('has-image');
      const icon = uploadZone.querySelector('.ic-upload-icon');
      const text = uploadZone.querySelector('.ic-upload-text');
      const sub = uploadZone.querySelector('.ic-upload-sub');
      const input = uploadZone.querySelector('#icFileInput');
      if (icon) icon.style.display = 'none';
      if (text) text.style.display = 'none';
      if (sub) sub.style.display = 'none';
      if (input) input.style.display = 'none';

      const previewWrap = document.createElement('div');
      previewWrap.className = 'ic-preview-wrap';
      previewWrap.innerHTML = `
        <img class="ic-preview-img" src="${processingUrl}" alt="Upload" id="icPreviewImg">
        <div class="ic-remove-img" id="icRemoveImg" title="Bild entfernen">✕</div>
      `;
      uploadZone.appendChild(previewWrap);

      const removeBtn = previewWrap.querySelector('#icRemoveImg');
      if (removeBtn) {
        removeBtn.addEventListener('click', (ev) => {
          ev.stopPropagation();
          safeRevoke();
          resetUploadZone(overlay, isKK);
          _isProcessing = false;
        });
      }

      const cached = _ocrCache.get(cacheKey);
      if (cached) {
        renderOCRResult(cached, cached.rawText || '', overlay, isKK);
        safeRevoke();
        _isProcessing = false;
        return;
      }

      overlay._currentFile = file;

      resultCard.classList.remove('active');

      const worker = await getWorker();
      if (!worker) throw new Error('OCR worker unavailable');

      updateProgress(overlay, 30, 'OCR Engine wird geladen...');
      _ocrProgressCallback = (prog) => {
        const pct = Math.round(35 + (prog || 0) * 60);
        updateProgress(overlay, pct, 'Texterkennung läuft...');
      };

      const result = await worker.recognize(processingUrl);
      _ocrProgressCallback = null;

      const rawText = result && result.data && result.data.text ? result.data.text : '';
      const parsed = parseScoreFromText(rawText, isKK, discipline);
      parsed.rawText = rawText;

      if (parsed.bestMatch) {
        _ocrCache.set(cacheKey, parsed);
        while (_ocrCache.size > OCR_CACHE_MAX) {
          const first = _ocrCache.keys().next().value;
          _ocrCache.delete(first);
        }
      }

      updateProgress(overlay, 100, 'Analyse abgeschlossen');
      renderOCRResult(parsed, rawText, overlay, isKK);
    } catch (err) {
      console.warn('[ImageCompare] OCR failed:', err);
      updateProgress(overlay, 100, 'OCR fehlgeschlagen - manuelle Eingabe');
      renderOCRResult({ bestMatch: null, alternatives: [] }, '', overlay, isKK);
    } finally {
      safeRevoke();
      _isProcessing = false;
    }
  }

  /**
   * ROI Detection & Auto-Cropping
   * Erkennt den hellen Monitor-Bereich und schneidet das Bild darauf zu.
   */
  async function detectAndCropROI(imageUrl) {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d', { willReadFrequently: true });

          // 1. Analyse-Größe (Downsampling für Performance)
          const scale = Math.min(1, 450 / Math.max(img.width, img.height));
          const w = Math.floor(img.width * scale);
          const h = Math.floor(img.height * scale);
          canvas.width = w;
          canvas.height = h;
          ctx.drawImage(img, 0, 0, w, h);

          const imageData = ctx.getImageData(0, 0, w, h);
          const data = imageData.data;

          // 2. ADAPTIVE SCHWELLENWERT-BERECHNUNG (V3 Peak-Detection)
          let maxBrightness = 0;
          const histogram = new Uint32Array(256);
          for (let i = 0; i < data.length; i += 4) {
            const b = Math.floor((data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114)); // Luma
            histogram[b]++;
            if (b > maxBrightness) maxBrightness = b;
          }

          // Wir suchen den Schwellenwert: 85% der maximalen Helligkeit
          // (typisch für leuchtende Zahlen/Monitore auf dunklem Hintergrund)
          const threshold = Math.max(160, Math.floor(maxBrightness * 0.85));

          let minX = w, minY = h, maxX = 0, maxY = 0;
          let found = false;

          // 3. Pixel-Cluster finden
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const idx = (y * w + x) * 4;
              const b = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;

              if (b >= threshold) {
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                found = true;
              }
            }
          }

          if (!found) { resolve(null); return; }

          // 4. Plausibilitäts-Check & Geometrie (V3)
          let roiW = maxX - minX;
          let roiH = maxY - minY;

          // Monitor-Ratio Check (Monitore sind meist breiter als hoch)
          const ratio = roiW / roiH;
          // Abbruch falls Fläche zu klein (<3% des Bildes) oder Ratio extrem unrealistisch
          if (ratio < 0.6 || ratio > 3.5 || (roiW * roiH) < (w * h * 0.03)) {
            resolve(null);
            return;
          }

          // Padding: 8% horizontal, 5% vertikal
          const padW = roiW * 0.08;
          const padH = roiH * 0.05;
          minX = Math.max(0, minX - padW);
          minY = Math.max(0, minY - padH);
          maxX = Math.min(w, maxX + padW);
          maxY = Math.min(h, maxY + padH);

          // 5. Finales Rendering (V3 Optimierungen)
          const finalX = minX / scale;
          const finalY = minY / scale;
          const finalW = (maxX - minX) / scale;
          const finalH = (maxY - minY) / scale;

          const cropCanvas = document.createElement('canvas');
          cropCanvas.width = finalW;
          cropCanvas.height = finalH;
          const cropCtx = cropCanvas.getContext('2d');

          // V3: Hochpräzises Image-Processing für Tesseract OCR
          // - Kontrast massiv hoch (für klare Ziffernkanten)
          // - Sättigung ganz raus (Graustufen optimieren OCR)
          cropCtx.filter = 'contrast(1.6) brightness(1.05) grayscale(1) saturate(0)';
          cropCtx.drawImage(img, finalX, finalY, finalW, finalH, 0, 0, finalW, finalH);

          resolve(cropCanvas.toDataURL('image/jpeg', 0.88));
        } catch (e) {
          console.warn('[ImageCompare] ROI V3 failed:', e);
          resolve(null);
        }
      };
      img.onerror = () => resolve(null);
      img.src = imageUrl;
    });
  }

  function renderOCRResult(parsed, rawTextStr, overlay, isKK) {
    const progress = overlay.querySelector('#icProgress');
    const resultCard = overlay.querySelector('#icResultCard');
    const detectedValue = overlay.querySelector('#icDetectedValue');
    const detectedLabel = overlay.querySelector('#icDetectedLabel');
    const scoreInput = overlay.querySelector('#icScoreInput');
    const compareBtn = overlay.querySelector('#icCompareBtn');
    const rawText = overlay.querySelector('#icRawText');

    if (!resultCard || !scoreInput || !compareBtn) return;

    if (rawText) rawText.textContent = rawTextStr || '(kein Text erkannt)';
    if (progress) progress.classList.remove('active');
    resultCard.classList.add('active');

    if (parsed && parsed.bestMatch) {
      const value = parsed.bestMatch.value;
      const displayValue = isKK ? String(Math.floor(value)) : Number(value).toFixed(1);

      if (detectedValue) detectedValue.textContent = displayValue;
      if (detectedLabel) {
        const conf = Math.round((parsed.bestMatch.confidence || 0) * 100);
        detectedLabel.textContent = 'Erkannt (' + conf + '% Konfidenz)';
      }

      scoreInput.value = displayValue;
      compareBtn.disabled = false;
      overlay.dataset.detectedScore = displayValue;
    } else {
      if (detectedValue) detectedValue.textContent = '?';
      if (detectedLabel) detectedLabel.textContent = 'Keine Punktzahl erkannt - bitte manuell eingeben';
      scoreInput.value = '';
      compareBtn.disabled = true;
      scoreInput.focus();
      delete overlay.dataset.detectedScore;
    }
  }

  function resetUploadZone(overlay, isKK) {
    const uploadZone = overlay.querySelector('#icUploadZone');
    const progress = overlay.querySelector('#icProgress');
    const resultCard = overlay.querySelector('#icResultCard');
    const compareBtn = overlay.querySelector('#icCompareBtn');
    const btnWrong = overlay.querySelector('#icBtnWrong');
    const editScoreBlock = overlay.querySelector('#icEditScoreBlock');
    const scoreInput = overlay.querySelector('#icScoreInput');
    const detectedValue = overlay.querySelector('#icDetectedValue');
    const detectedLabel = overlay.querySelector('#icDetectedLabel');
    const rawText = overlay.querySelector('#icRawText');
    const rawToggle = overlay.querySelector('#icRawToggle');

    if (!uploadZone) return;

    const previewWrap = uploadZone.querySelector('.ic-preview-wrap');
    if (previewWrap) previewWrap.remove();

    uploadZone.classList.remove('has-image');

    const icon = uploadZone.querySelector('.ic-upload-icon');
    const text = uploadZone.querySelector('.ic-upload-text');
    const sub = uploadZone.querySelector('.ic-upload-sub');
    const input = uploadZone.querySelector('#icFileInput');
    if (icon) icon.style.display = '';
    if (text) text.style.display = '';
    if (sub) sub.style.display = '';
    if (input) {
      input.style.display = '';
      input.value = '';
    }

    if (progress) progress.classList.remove('active');
    if (resultCard) resultCard.classList.remove('active');
    if (compareBtn) compareBtn.disabled = true;
    if (btnWrong) btnWrong.style.display = 'block';
    if (editScoreBlock) editScoreBlock.style.display = 'none';

    if (rawText) {
      rawText.classList.remove('visible');
      rawText.textContent = '';
    }
    if (rawToggle) rawToggle.textContent = '▶ OCR-Rohtext anzeigen';

    if (detectedValue) detectedValue.textContent = '–';
    if (detectedLabel) detectedLabel.textContent = 'Wird analysiert...';

    if (scoreInput) {
      scoreInput.value = '';
      scoreInput.placeholder = isKK ? 'z.B. 392' : 'z.B. 405.2';
      scoreInput.step = isKK ? '1' : '0.1';
      scoreInput.inputMode = isKK ? 'numeric' : 'decimal';
    }

    delete overlay.dataset.detectedScore;
    delete overlay._currentFile;
  }

  async function sendToFormspree(file, expectedScore, detectedScore) {
    if (!Brain || !Brain.FEEDBACK_ENABLED || !file || !Brain.FORMSPREE_ENDPOINT) return;

    try {
      const url = 'https://formspree.io/f/' + Brain.FORMSPREE_ENDPOINT;
      const formData = new FormData();
      formData.append('Fehlerbericht', 'KI lag falsch');
      formData.append('KI_dachte', String(detectedScore));
      formData.append('Wahrer_Score', String(expectedScore));
      formData.append('Foto_Upload', file, file.name || 'feedback.jpg');

      await fetch(url, {
        method: 'POST',
        body: formData,
        headers: { 'Accept': 'application/json' }
      });
    } catch (e) {
      console.warn('[ImageCompare] Formspree upload failed:', e);
    }
  }

  return {
    init() {
      injectStyles();
    },

    open(botScore, isKK, discipline = null) {
      injectStyles();
      ensureTesseract().catch(() => { /* lazy failure fallback */ });

      const overlay = createOverlay(botScore || 0, !!isKK);
      if (!overlay) return;
      overlay.dataset.discipline = discipline || '';
    },

    createGameOverButton(container, botScore, isKK, discipline = null) {
      if (!container) return;
      injectStyles();

      if (container.querySelector('.ic-go-upload-btn')) return;

      const btn = document.createElement('button');
      btn.className = 'ic-go-upload-btn';
      btn.innerHTML = '<span class="ic-go-upload-ico">📷</span> Foto schiessen';
      btn.addEventListener('click', () => {
        this.open(botScore, isKK, discipline);
      });

      container.appendChild(btn);
    }
  };
})();
