/**
 * YGO Remote Lens - Step 2 Browser AI Recognition Edition
 *
 * 従来の16×16自作ピクセル比較を廃止し、ブラウザ内AIモデル（DINOv2-small / MobileNet v2）による
 * 高精度な深層特徴ベクトル検索へ完全移行:
 *   1) Transformers.js (Xenova/dinov2-small) および TensorFlow.js (MobileNet v2) のデュアルAI対応
 *   2) 暗色系カード（オルフェゴール・ディヴェル等）やスリーブの光反射に強い適応的コントラスト＆ガンマ補正
 *   3) クリック瞬間のフレームをオフスクリーンCanvasへスナップショットしてから非同期AI解析（カメラ映像ブレ防止）
 *   4) IndexedDB によるカードDB特徴ベクトルの永続キャッシュ（2回目以降の起動を大幅高速化）
 *   5) ビルド不要・GitHub Pages でそのまま動作
 */

const yieldToUI = () => new Promise((resolve) => setTimeout(resolve, 0));

class YgoRemoteLensApp {
  constructor() {
    this.aiEngine = new window.YgoAIEngine();
    this.siftServerBase = null;
    this.useSiftServer = false;
    this.browserAiInitialized = false;
    this.cardsDB = [];
    this.dbMeanVec = null;
    this.dinoMeanVec = null;
    this.history = [];
    this.pinnedCards = [];
    this.currentMatch = null;
    this.currentCardId = null;
    this.lastRecognizedCardId = null;
    this.historyFilter = '';
    this.mode = 'demo'; // 'demo' | 'video' | 'image'
    this.loadedImage = null;
    this.videoStream = null;

    this.demoBoardCards = [];

    // 定位置スキャンエリア＆自動監視状態
    this.scanZone = null; // { x, y, w, h } in mainCanvas coords
    this.isSettingScanZone = false;
    this.autoScanEnabled = false;
    this.lastAutoScanCardId = null;
    this.scanZonePrevGray = null;
    this.scanZoneDirty = false;
    this.scanZoneStillTicks = 0;
    this.scanZoneFlashUntil = 0;
    this.scanZoneFlashLabel = '';
    this.scanZoneProbeCanvas = document.createElement('canvas');
    this.scanZoneProbeCanvas.width = 48;
    this.scanZoneProbeCanvas.height = 48;
    this.scanZoneProbeCtx = this.scanZoneProbeCanvas.getContext('2d', { willReadFrequently: true });

    this.isDragging = false;
    this.isRecognizing = false;
    this.isSwitchingEngine = false;
    this.dragStart = { x: 0, y: 0 };
    this.dragCurrent = { x: 0, y: 0 };
    this.lastSelectionBox = null;
    this.bestMatchedSubBox = null;
    this.lastClickWasDemoCard = false;

    this.initElements();
    this.initLayoutState();
    this.loadSavedScanZone();
    this.bindEvents();
    this.refreshCameraList();
    this.initDatabase();
    this.startAutoScanLoop();
  }

  initElements() {
    this.mainLayout = document.getElementById('mainLayout');
    this.mainCanvas = document.getElementById('mainCanvas');
    this.ctx = this.mainCanvas.getContext('2d');
    this.video = document.getElementById('captureVideo');
    this.cropPreviewCanvas = document.getElementById('cropPreviewCanvas');
    this.cropPreviewCtx = this.cropPreviewCanvas.getContext('2d');

    this.dbStatusBadge = document.getElementById('dbStatusBadge');
    this.aiEngineSelect = document.getElementById('aiEngineSelect');
    this.enhanceLightingCheck = document.getElementById('enhanceLightingCheck');
    this.cropSizeSelect = document.getElementById('cropSizeSelect');
    this.cameraDeviceSelect = document.getElementById('cameraDeviceSelect');
    this.matchStatusText = document.getElementById('matchStatusText');
    this.matchBadgeContainer = document.getElementById('matchBadgeContainer');
    this.cardDetailContainer = document.getElementById('cardDetailContainer');
    this.candidatesSection = document.getElementById('candidatesSection');
    this.candidatesList = document.getElementById('candidatesList');

    this.btnSetScanZone = document.getElementById('btnSetScanZone');
    this.btnToggleAutoScan = document.getElementById('btnToggleAutoScan');
    this.btnClearScanZone = document.getElementById('btnClearScanZone');
    this.scanZoneHint = document.getElementById('scanZoneHint');

    this.historyList = document.getElementById('historyList');
    this.historyCount = document.getElementById('historyCount');
    this.historyFilterInput = document.getElementById('historyFilterInput');
    this.quickHistoryList = document.getElementById('quickHistoryList');
    this.quickHistoryCount = document.getElementById('quickHistoryCount');
    this.btnToggleHistoryRail = document.getElementById('btnToggleHistoryRail');
    this.btnCollapseRail = document.getElementById('btnCollapseRail');
    this.btnExpandRailFromStrip = document.getElementById('btnExpandRailFromStrip');
    this.btnClearHistory = document.getElementById('btnClearHistory');

    this.pinnedCompareSection = document.getElementById('pinnedCompareSection');
    this.pinnedCardsList = document.getElementById('pinnedCardsList');
    this.pinnedCount = document.getElementById('pinnedCount');
    this.btnClearPins = document.getElementById('btnClearPins');

    this.dbCardList = document.getElementById('dbCardList');
    this.dbCount = document.getElementById('dbCount');
    this.btnClearAiCache = document.getElementById('btnClearAiCache');
  }

  loadSavedScanZone() {
    try {
      const raw = localStorage.getItem('ygo_remote_lens_scan_zone');
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && parsed.w >= 24 && parsed.h >= 24) {
          this.scanZone = parsed;
        }
      }
    } catch (_) {
      this.scanZone = null;
    }
    this.updateScanZoneUI();
  }

  saveScanZone() {
    try {
      if (this.scanZone) {
        localStorage.setItem('ygo_remote_lens_scan_zone', JSON.stringify(this.scanZone));
      } else {
        localStorage.removeItem('ygo_remote_lens_scan_zone');
      }
    } catch (_) {
      // 無視
    }
  }

  updateScanZoneUI() {
    if (this.btnSetScanZone) {
      this.btnSetScanZone.classList.toggle('setting', this.isSettingScanZone);
      this.btnSetScanZone.textContent = this.isSettingScanZone
        ? '🟡 盤面をドラッグして枠を指定中...'
        : this.scanZone
        ? '🎯 スキャン枠を再設定'
        : '🎯 定位置スキャン枠を設定';
    }
    if (this.btnToggleAutoScan) {
      this.btnToggleAutoScan.classList.toggle('active', this.autoScanEnabled && Boolean(this.scanZone));
      this.btnToggleAutoScan.textContent = this.autoScanEnabled ? '📡 自動認識: ON' : '📡 自動認識: OFF';
    }
    if (this.btnClearScanZone) {
      this.btnClearScanZone.style.display = this.scanZone ? 'inline-flex' : 'none';
    }
    if (this.scanZoneHint) {
      if (this.isSettingScanZone) {
        this.scanZoneHint.textContent = 'カードを置く場所をマウスで四角く囲んでください';
      } else if (this.autoScanEnabled && this.scanZone) {
        this.scanZoneHint.textContent = '枠内にカードを置いて手を離すと自動認識します（クリック認識も併用可）';
      } else if (this.scanZone) {
        this.scanZoneHint.textContent = 'スキャン枠設定済み（「自動認識: ON」で監視開始）';
      } else {
        this.scanZoneHint.textContent = '盤面クリックでも随時認識できます';
      }
    }
  }

  initLayoutState() {
    let collapsed = false;
    try {
      const saved = localStorage.getItem('ygo_remote_lens_history_collapsed');
      if (saved !== null) {
        collapsed = saved === 'true';
      } else if (window.innerWidth < 1160) {
        collapsed = true;
      }
    } catch (_) {
      collapsed = window.innerWidth < 1160;
    }
    this.setHistoryCollapsed(collapsed, false);
  }

  setHistoryCollapsed(collapsed, persist = true) {
    this.isHistoryCollapsed = Boolean(collapsed);
    if (this.mainLayout) {
      this.mainLayout.classList.toggle('history-collapsed', this.isHistoryCollapsed);
    }
    if (this.btnToggleHistoryRail) {
      this.btnToggleHistoryRail.textContent = this.isHistoryCollapsed ? '📜 履歴を開く' : '📜 履歴を畳む';
    }
    if (persist) {
      try {
        localStorage.setItem('ygo_remote_lens_history_collapsed', String(this.isHistoryCollapsed));
      } catch (_) {
        // ストレージ無効環境は無視
      }
    }
  }

  bindEvents() {
    document.getElementById('btnDemoBoard').addEventListener('click', () => this.startDemoBoard());
    document.getElementById('btnCaptureScreen').addEventListener('click', () => this.startScreenCapture());
    document.getElementById('btnStartCamera').addEventListener('click', () => this.startCamera());
    document.getElementById('fileInput').addEventListener('change', (e) => this.loadImageFile(e));

    if (this.btnSetScanZone) {
      this.btnSetScanZone.addEventListener('click', () => {
        this.isSettingScanZone = !this.isSettingScanZone;
        this.updateScanZoneUI();
      });
    }
    if (this.btnToggleAutoScan) {
      this.btnToggleAutoScan.addEventListener('click', () => {
        if (!this.scanZone) {
          this.isSettingScanZone = true;
          this.updateScanZoneUI();
          return;
        }
        this.autoScanEnabled = !this.autoScanEnabled;
        this.scanZonePrevGray = null;
        this.scanZoneDirty = false;
        this.scanZoneStillTicks = 0;
        this.lastAutoScanCardId = null;
        this.updateScanZoneUI();
        if (this.autoScanEnabled && this.mode === 'video' && this.video.readyState >= 2) {
          this.recognizeRegion(this.scanZone, true, true);
        }
      });
    }
    if (this.btnClearScanZone) {
      this.btnClearScanZone.addEventListener('click', () => {
        this.scanZone = null;
        this.autoScanEnabled = false;
        this.isSettingScanZone = false;
        this.lastAutoScanCardId = null;
        this.scanZonePrevGray = null;
        this.saveScanZone();
        this.updateScanZoneUI();
      });
    }

    if (this.btnToggleHistoryRail) {
      this.btnToggleHistoryRail.addEventListener('click', () => {
        this.setHistoryCollapsed(!this.isHistoryCollapsed);
      });
    }
    if (this.btnCollapseRail) {
      this.btnCollapseRail.addEventListener('click', () => this.setHistoryCollapsed(true));
    }
    if (this.btnExpandRailFromStrip) {
      this.btnExpandRailFromStrip.addEventListener('click', () => this.setHistoryCollapsed(false));
    }
    if (this.btnClearHistory) {
      this.btnClearHistory.addEventListener('click', () => this.clearHistory());
    }
    if (this.historyFilterInput) {
      this.historyFilterInput.addEventListener('input', (e) => {
        this.historyFilter = e.target.value.trim().toLowerCase();
        this.renderHistory();
      });
    }
    if (this.btnClearPins) {
      this.btnClearPins.addEventListener('click', () => {
        this.pinnedCards = [];
        this.renderPinnedCards();
        this.renderHistory();
        this.refreshDetailPinButton();
      });
    }

    if (this.cameraDeviceSelect) {
      this.cameraDeviceSelect.addEventListener('change', () => {
        if (this.mode === 'video' && this.videoStream) {
          this.startCamera();
        }
      });
    }

    if (this.aiEngineSelect) {
      this.aiEngineSelect.addEventListener('change', async () => {
        const selected = this.aiEngineSelect.value;
        await this.switchAIEngine(selected);
      });
    }

    if (this.btnClearAiCache) {
      this.btnClearAiCache.addEventListener('click', async () => {
        if (this.isSwitchingEngine || this.isRecognizing) return;
        this.btnClearAiCache.disabled = true;
        try {
          await this.aiEngine.clearCache();
          await this.rebuildAllCardEmbeddings(true);
        } finally {
          this.btnClearAiCache.disabled = false;
        }
      });
    }

    document.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.getAttribute('data-tab');
        ['detail', 'database'].forEach((t) => {
          const el = document.getElementById(`tab-${t}`);
          if (el) el.style.display = t === target ? 'flex' : 'none';
        });
      });
    });

    document.getElementById('btnAddFromApi').addEventListener('click', () => this.addCardFromApi());

    this.mainCanvas.addEventListener('mousedown', (e) => this.onCanvasMouseDown(e));
    window.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
    window.addEventListener('mouseup', (e) => this.onCanvasMouseUp(e));
  }

  /**
   * 接続されているWebカメラ（Camo等）の一覧を取得してプルダウンに反映
   */
  async refreshCameraList() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices || !this.cameraDeviceSelect) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((d) => d.kind === 'videoinput');
      const currentVal = this.cameraDeviceSelect.value;
      this.cameraDeviceSelect.innerHTML = '<option value="">📷 カメラ自動選択</option>';

      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `カメラ ${idx + 1}`;
        if (/camo|iriun|iphone/i.test(opt.textContent) && !currentVal) {
          opt.selected = true;
        } else if (dev.deviceId === currentVal) {
          opt.selected = true;
        }
        this.cameraDeviceSelect.appendChild(opt);
      });
    } catch (_) {
      // 権限取得前は無視
    }
  }

  setStatusBadge(text, isLoading = false) {
    this.dbStatusBadge.textContent = text;
    if (isLoading) {
      this.dbStatusBadge.classList.add('loading');
    } else {
      this.dbStatusBadge.classList.remove('loading');
    }
  }

  /**
   * Python OpenCV SIFT サーバーの稼働を自動検出
   */
  async checkSiftServer() {
    const candidates =
      window.location.protocol === 'http:' || window.location.protocol === 'https:'
        ? ['', 'http://localhost:8000', 'http://127.0.0.1:8000']
        : ['http://localhost:8000', 'http://127.0.0.1:8000'];

    for (const base of candidates) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 1400);
        const res = await fetch(`${base}/api/health`, { signal: controller.signal });
        clearTimeout(timer);
        if (res.ok) {
          const data = await res.json();
          if (data && data.ok && data.engine === 'opencv-sift') {
            this.siftServerBase = base;
            return data;
          }
        }
      } catch (_) {
        // 次の候補へ
      }
    }
    this.siftServerBase = null;
    return null;
  }

  /**
   * 初期カードデータベースと認識エンジンの初期化（Python SIFTサーバーがあれば即座に接続・最速起動）
   */
  async initDatabase() {
    const initialCards = window.INITIAL_CARDS_DATA || [];

    this.cardsDB = initialCards.map((card) => {
      const fallbackCanvas = this.createFallbackCardArt(card);
      return {
        ...card,
        frameType: this.aiEngine.getCardFrameTypeFromMeta(card),
        safeCanvasElement: fallbackCanvas,
        thumbUrl: card.image_crop,
        feature: null,
        centerFeature: null,
        tightFeature: null,
        isFallback: true,
        aiIndexed: false
      };
    });

    this.renderDBList();
    this.startDemoBoard();
    this.startRenderLoop();

    this.setStatusBadge('🔍 認識エンジンを確認中...', true);
    if (this.aiEngineSelect) this.aiEngineSelect.disabled = true;

    const siftHealth = await this.checkSiftServer();
    if (siftHealth) {
      this.useSiftServer = true;
      if (this.aiEngineSelect) {
        this.aiEngineSelect.value = 'sift';
        this.aiEngineSelect.disabled = false;
      }
      this.cardsDB.forEach((c) => {
        c.aiIndexed = true;
        c.isFallback = false;
      });
      this.renderDBList();
      this.setStatusBadge(
        `✅ OpenCV SIFT 稼働中 (${siftHealth.card_count || this.cardsDB.length}枚・Python高精度)`,
        false
      );
      this.matchStatusText.textContent =
        'カメラを起動して定位置スキャン枠を設定するか、盤面のカードをクリックしてください';
      return;
    }

    // Pythonサーバー未起動時はブラウザ内AIモードへ自動フォールバック
    this.useSiftServer = false;
    if (this.aiEngineSelect && this.aiEngineSelect.value === 'sift') {
      this.aiEngineSelect.value = 'hybrid';
    }
    const preferredEngine = this.aiEngineSelect ? this.aiEngineSelect.value : 'hybrid';
    await this.initBrowserAI(preferredEngine);
  }

  async initBrowserAI(preferredEngine = 'hybrid') {
    this.setStatusBadge('🤖 ブラウザAIモデル＆カード画像を取得中...', true);
    if (this.aiEngineSelect) this.aiEngineSelect.disabled = true;

    let aiProgressText = 'AIモデル初期化中...';
    let imgProgressText = `画像(0/${this.cardsDB.length})`;
    const updateCombinedStatus = () => {
      this.setStatusBadge(`⏳ ${aiProgressText} | ${imgProgressText}`, true);
    };

    const aiInitPromise = this.aiEngine.init(preferredEngine, (msg) => {
      aiProgressText = msg;
      updateCombinedStatus();
    });

    const imageLoadPromise = this.fetchAllCardImages((loaded, total, realCount) => {
      imgProgressText = `画像(${loaded}/${total}・成功${realCount})`;
      updateCombinedStatus();
    });

    const [actualEngine] = await Promise.all([aiInitPromise, imageLoadPromise]);

    if (this.aiEngineSelect) {
      if (actualEngine === 'hybrid' || actualEngine === 'dinov2' || actualEngine === 'mobilenet') {
        this.aiEngineSelect.value = actualEngine;
      }
      this.aiEngineSelect.disabled = false;
    }

    await this.rebuildAllCardEmbeddings(false);
    this.browserAiInitialized = true;
    this.matchStatusText.textContent = '盤面のカードをクリックまたはドラッグで選択してください';
  }

  /**
   * 6枚ずつのバッチ並列＋リトライで全カードの実画像を取得
   */
  async fetchAllCardImages(onProgress) {
    let loadedCount = 0;
    let realImageCount = 0;
    const total = this.cardsDB.length;

    const BATCH_SIZE = 6;
    for (let i = 0; i < total; i += BATCH_SIZE) {
      const batch = this.cardsDB.slice(i, i + BATCH_SIZE);
      await Promise.all(
        batch.map(async (cardEntry) => {
          const ok = await this.loadRealImageForCard(cardEntry);
          loadedCount++;
          if (ok) realImageCount++;
          onProgress(loadedCount, total, realImageCount);
        })
      );
      this.renderDBList();
    }

    // 失敗分を1回リトライ
    const failedEntries = this.cardsDB.filter((c) => c.isFallback);
    for (const cardEntry of failedEntries) {
      const ok = await this.loadRealImageForCard(cardEntry);
      if (ok) {
        realImageCount++;
        onProgress(loadedCount, total, realImageCount);
      }
    }

    this.renderDBList();
    return realImageCount;
  }

  async loadRealImageForCard(cardEntry) {
    try {
      const safeCanvas = await this.fetchImageAsSafeCanvas(cardEntry.image_crop);
      cardEntry.safeCanvasElement = safeCanvas;
      cardEntry.isFallback = false;
      return true;
    } catch (_) {
      return false;
    }
  }

  /**
   * 現在のAIエンジンで全カードの特徴ベクトルを構築（IndexedDBキャッシュ優先）
   */
  async rebuildAllCardEmbeddings(forceRecompute = false) {
    this.isSwitchingEngine = true;
    if (this.aiEngineSelect) this.aiEngineSelect.disabled = true;

    const total = this.cardsDB.length;
    let cacheHits = 0;
    let realCount = 0;
    const engineLabel = this.aiEngine.getEngineLabel();

    for (let i = 0; i < total; i++) {
      const cardEntry = this.cardsDB[i];
      if (!cardEntry.isFallback) realCount++;

      try {
        if (forceRecompute) {
          cardEntry.dinoFeature = null;
          cardEntry.dinoCenterFeature = null;
          cardEntry.dinoTightFeature = null;
        }
        if (forceRecompute || cardEntry.isFallback) {
          const W = cardEntry.safeCanvasElement.width;
          const H = cardEntry.safeCanvasElement.height;
          const colorFeats = this.aiEngine.computeCardColorFeatures(cardEntry.safeCanvasElement);
          cardEntry.feature = await this.aiEngine.extractEmbedding(cardEntry.safeCanvasElement, 0, null, true);
          cardEntry.centerFeature = await this.aiEngine.extractEmbedding(
            cardEntry.safeCanvasElement,
            0,
            { x: W * 0.08, y: H * 0.08, w: W * 0.84, h: H * 0.84 },
            true
          );
          cardEntry.tightFeature = await this.aiEngine.extractEmbedding(
            cardEntry.safeCanvasElement,
            0,
            { x: W * 0.15, y: H * 0.15, w: W * 0.70, h: H * 0.70 },
            true
          );
          cardEntry.colorFeature = colorFeats.colorFeature;
          cardEntry.centerColorFeature = colorFeats.centerColorFeature;
          cardEntry.tightColorFeature = colorFeats.tightColorFeature;
        } else {
          const emb = await this.aiEngine.getOrComputeCardEmbeddings(cardEntry.id, cardEntry.safeCanvasElement);
          cardEntry.feature = emb.feature;
          cardEntry.centerFeature = emb.centerFeature;
          cardEntry.tightFeature = emb.tightFeature;
          cardEntry.colorFeature = emb.colorFeature;
          cardEntry.centerColorFeature = emb.centerColorFeature;
          cardEntry.tightColorFeature = emb.tightColorFeature;
          if (emb.fromCache) cacheHits++;
        }
        cardEntry.aiIndexed = true;
      } catch (err) {
        console.warn(`特徴抽出エラー (${cardEntry.name_ja}):`, err);
        cardEntry.aiIndexed = false;
      }

      if ((i + 1) % 2 === 0 || i === total - 1) {
        this.setStatusBadge(
          `🧠 AIベクトル構築中 [${engineLabel}] (${i + 1}/${total}${cacheHits > 0 ? ` / キャッシュ:${cacheHits}` : ''})...`,
          true
        );
        await yieldToUI();
      }
    }

    this.recomputeHubnessStats(false);

    this.setStatusBadge(
      `✅ AI認識稼働中 [${this.aiEngine.getEngineLabel()}] (${total}枚 / 実画像:${realCount}枚)`,
      false
    );
    this.renderDBList();

    if (this.aiEngineSelect) this.aiEngineSelect.disabled = false;
    this.isSwitchingEngine = false;

    if (this.aiEngine.activeEngine === 'hybrid') {
      this.warmUpHybridDinoInBackground();
    }
  }

  /**
   * ハブ現象（特定カードが何に対しても高スコアで割り込む現象）を数学的に解消するため、
   * 全カード平均ベクトルの減算（Mean-Centering）と各カードのCSLSハブ基準値を事前算出（約5ms）
   */
  recomputeHubnessStats(useDino = false) {
    const validCards = this.cardsDB.filter((c) => (useDino ? c.dinoFeature : c.feature));
    if (validCards.length < 4) {
      if (useDino) this.dinoMeanVec = null;
      else this.dbMeanVec = null;
      return;
    }

    const baseVecs = validCards.map((c) => (useDino ? c.dinoFeature : c.feature));
    const meanVec = this.aiEngine.computeMeanVector(baseVecs);
    if (useDino) {
      this.dinoMeanVec = meanVec;
    } else {
      this.dbMeanVec = meanVec;
    }

    for (const c of validCards) {
      if (useDino) {
        c.cDinoFeature = this.aiEngine.centerAndNormalize(c.dinoFeature, meanVec);
        c.cDinoCenterFeature = this.aiEngine.centerAndNormalize(c.dinoCenterFeature, meanVec);
        c.cDinoTightFeature = this.aiEngine.centerAndNormalize(c.dinoTightFeature, meanVec);
      } else {
        c.cFeature = this.aiEngine.centerAndNormalize(c.feature, meanVec);
        c.cCenterFeature = this.aiEngine.centerAndNormalize(c.centerFeature, meanVec);
        c.cTightFeature = this.aiEngine.centerAndNormalize(c.tightFeature, meanVec);
      }
    }

    // CSLS: 各カードの「他のカード上位K枚に対する平均類似度（ハブ度）」を算出
    const K = Math.min(10, validCards.length - 1);
    const n = validCards.length;
    const centeredList = validCards.map((c) => (useDino ? c.cDinoFeature : c.cFeature));

    for (let i = 0; i < n; i++) {
      const sims = [];
      const vi = centeredList[i];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        sims.push(this.aiEngine.cosineSimilarity(vi, centeredList[j]));
      }
      sims.sort((a, b) => b - a);
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += sims[k];
      }
      const hubBias = sum / K;
      if (useDino) {
        validCards[i].dinoHubBias = hubBias;
      } else {
        validCards[i].hubBias = hubBias;
      }
    }
  }

  /**
   * ハイブリッドモード時、アイドル時間を使ってバックグラウンドでDINOv2モデルと全カードの特徴をIndexedDBへ準備
   */
  async warmUpHybridDinoInBackground() {
    if (this._warmingUpDino) return;
    this._warmingUpDino = true;
    try {
      const ok = await this.aiEngine.ensureDinoLoaded();
      if (!ok || this.aiEngine.activeEngine !== 'hybrid') return;

      const total = this.cardsDB.length;
      const realCount = this.cardsDB.filter((c) => !c.isFallback).length;
      this.setStatusBadge(
        `✅ AI認識稼働中 [${this.aiEngine.getEngineLabel()}] (${total}枚 / 実画像:${realCount}枚)`,
        false
      );

      for (let i = 0; i < this.cardsDB.length; i++) {
        if (this.aiEngine.activeEngine !== 'hybrid') break;
        while (this.isRecognizing || this.isSwitchingEngine) {
          await new Promise((r) => setTimeout(r, 120));
        }
        const cardEntry = this.cardsDB[i];
        if (!cardEntry.isFallback && !cardEntry.dinoFeature) {
          try {
            const emb = await this.aiEngine.getOrComputeCardEmbeddings(
              cardEntry.id,
              cardEntry.safeCanvasElement,
              'dinov2'
            );
            cardEntry.dinoFeature = emb.feature;
            cardEntry.dinoCenterFeature = emb.centerFeature;
            cardEntry.dinoTightFeature = emb.tightFeature;
          } catch (_) {
            // 個別エラーは無視
          }
          await yieldToUI();
        }
      }

      if (this.aiEngine.activeEngine === 'hybrid') {
        this.recomputeHubnessStats(true);
      }
    } finally {
      this._warmingUpDino = false;
    }
  }

  /**
   * 認識エンジンを動的に切り替え（OpenCV SIFT / Hybrid / DINOv2-small / MobileNet v2）
   */
  async switchAIEngine(engineId) {
    if (this.isSwitchingEngine || this.isRecognizing) return;
    this.isSwitchingEngine = true;
    if (this.aiEngineSelect) this.aiEngineSelect.disabled = true;

    try {
      if (engineId === 'sift') {
        this.setStatusBadge('🔄 OpenCV SIFT サーバーに接続中...', true);
        const health = await this.checkSiftServer();
        if (health) {
          this.useSiftServer = true;
          this.setStatusBadge(
            `✅ OpenCV SIFT 稼働中 (${health.card_count || this.cardsDB.length}枚・Python高精度)`,
            false
          );
          if (this.aiEngineSelect) this.aiEngineSelect.value = 'sift';
        } else {
          alert('Python OpenCV SIFT サーバー（http://localhost:8000）が見つかりません。\n./start_server.sh を起動してから選択してください。');
          this.useSiftServer = false;
          if (this.aiEngineSelect) this.aiEngineSelect.value = this.aiEngine.activeEngine || 'hybrid';
        }
        return;
      }

      this.useSiftServer = false;
      if (!this.browserAiInitialized) {
        this.isSwitchingEngine = false;
        await this.initBrowserAI(engineId);
        return;
      }

      this.setStatusBadge(`🔄 AIエンジン切替中 (${engineId})...`, true);
      const actualEngine = await this.aiEngine.init(engineId, (msg) => {
        this.setStatusBadge(`🔄 ${msg}`, true);
      });

      if (
        this.aiEngineSelect &&
        (actualEngine === 'hybrid' || actualEngine === 'dinov2' || actualEngine === 'mobilenet')
      ) {
        this.aiEngineSelect.value = actualEngine;
      }

      await this.rebuildAllCardEmbeddings(false);
    } catch (err) {
      console.error('Engine switch error:', err);
      this.setStatusBadge('⚠️ エンジン切替エラー: ' + err.message, false);
    } finally {
      if (this.aiEngineSelect) this.aiEngineSelect.disabled = false;
      this.isSwitchingEngine = false;
    }

    // 直前の選択範囲があれば新しいエンジンで自動再認識してすぐ比較できる
    if (this.lastSelectionBox) {
      await this.recognizeRegion(this.lastSelectionBox, Boolean(this.lastSelectionWasDrag));
    }
  }

  async fetchImageAsSafeCanvas(url) {
    const tryLoadUrl = (targetUrl, timeoutMs = 6000) =>
      new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';
        const timer = setTimeout(() => {
          img.src = '';
          reject(new Error('timeout'));
        }, timeoutMs);

        img.onload = () => {
          clearTimeout(timer);
          try {
            const c = document.createElement('canvas');
            c.width = img.naturalWidth || 224;
            c.height = img.naturalHeight || 224;
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0);
            ctx.getImageData(0, 0, 1, 1);
            resolve(c);
          } catch (err) {
            reject(err);
          }
        };

        img.onerror = (err) => {
          clearTimeout(timer);
          reject(err);
        };

        img.src = targetUrl;
      });

    const urlsToTry = [
      `https://wsrv.nl/?url=${encodeURIComponent(url)}&w=256&h=256&fit=cover&output=jpg`,
      url,
      `https://corsproxy.io/?${encodeURIComponent(url)}`
    ];

    for (const candidateUrl of urlsToTry) {
      try {
        return await tryLoadUrl(candidateUrl);
      } catch (_) {
        // 次の候補を試す
      }
    }
    throw new Error('Image load failed');
  }

  createFallbackCardArt(card) {
    const c = document.createElement('canvas');
    c.width = 224;
    c.height = 224;
    const ctx = c.getContext('2d');
    const hue = (card.id * 137) % 360;
    const grad = ctx.createLinearGradient(0, 0, 224, 224);
    grad.addColorStop(0, `hsl(${hue}, 70%, 35%)`);
    grad.addColorStop(1, `hsl(${(hue + 60) % 360}, 80%, 20%)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 18px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(card.name_ja.slice(0, 7), 112, 118);
    return c;
  }

  startDemoBoard() {
    this.stopVideoStream();
    this.mode = 'demo';
    this.lastSelectionBox = null;
    this.bestMatchedSubBox = null;

    if (this.cardsDB.length === 0) return;

    // ディヴェル(57835716)とギルス(69811710)を含む代表6枚を配置
    const priorityIds = [57835716, 69811710, 14558127, 48835607, 93854893, 2501624];
    const selected = [];
    for (const pid of priorityIds) {
      const found = this.cardsDB.find((c) => c.id === pid);
      if (found) selected.push(found);
    }
    for (const c of this.cardsDB) {
      if (selected.length >= 6) break;
      if (!selected.includes(c)) selected.push(c);
    }

    this.demoBoardCards = [];
    const cardW = 126;
    const cardH = 184;

    for (let i = 0; i < 3; i++) {
      const card = selected[i];
      if (!card) continue;
      this.demoBoardCards.push({
        card,
        x: 200 + i * 220,
        y: 65,
        w: cardW,
        h: cardH,
        upsideDown: true
      });
    }

    for (let i = 0; i < 3; i++) {
      const card = selected[i + 3];
      if (!card) continue;
      this.demoBoardCards.push({
        card,
        x: 200 + i * 220,
        y: 330,
        w: cardW,
        h: cardH,
        upsideDown: false
      });
    }
  }

  async startScreenCapture() {
    try {
      this.stopVideoStream();
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { cursor: 'always' },
        audio: false
      });
      this.videoStream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      this.mode = 'video';
      this.lastSelectionBox = null;
      this.bestMatchedSubBox = null;
    } catch (err) {
      alert('画面キャプチャがキャンセルされたか、許可されませんでした。\n' + err.message);
    }
  }

  async startCamera() {
    try {
      this.stopVideoStream();
      const selectedDeviceId = this.cameraDeviceSelect ? this.cameraDeviceSelect.value : '';
      const videoConstraints = {
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      };
      if (selectedDeviceId) {
        videoConstraints.deviceId = { exact: selectedDeviceId };
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false
      });
      this.videoStream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      this.mode = 'video';
      this.lastSelectionBox = null;
      this.bestMatchedSubBox = null;

      await this.refreshCameraList();
    } catch (err) {
      alert('Webカメラの起動に失敗しました。\n' + err.message);
    }
  }

  loadImageFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    this.stopVideoStream();

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        this.loadedImage = img;
        this.mode = 'image';
        this.lastSelectionBox = null;
        this.bestMatchedSubBox = null;
        this.matchStatusText.textContent = '📸 画像を読み込みました！カードのイラスト部分をドラッグで囲むかクリックしてください';
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
    event.target.value = '';
  }

  stopVideoStream() {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((t) => t.stop());
      this.videoStream = null;
    }
  }

  startRenderLoop() {
    const render = () => {
      this.drawFrame();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  drawFrame() {
    const ctx = this.ctx;
    const W = this.mainCanvas.width;
    const H = this.mainCanvas.height;

    ctx.clearRect(0, 0, W, H);

    if (this.mode === 'demo') {
      this.drawDemoPlaymat(ctx, W, H);
    } else if (this.mode === 'video' && this.video.readyState >= 2) {
      this.drawMediaContain(ctx, this.video, this.video.videoWidth, this.video.videoHeight, W, H);
    } else if (this.mode === 'image' && this.loadedImage) {
      this.drawMediaContain(ctx, this.loadedImage, this.loadedImage.width, this.loadedImage.height, W, H);
    }

    // 定位置スキャン枠の描画（常に視覚的に把握可能）
    if (this.scanZone && !this.isSettingScanZone) {
      const isFlashing = performance.now() < this.scanZoneFlashUntil;
      const zoneColor = isFlashing
        ? '#34d399'
        : this.autoScanEnabled
        ? '#38bdf8'
        : 'rgba(148, 163, 184, 0.75)';
      const zoneLabel = isFlashing
        ? this.scanZoneFlashLabel
        : this.autoScanEnabled
        ? '📡 自動スキャン監視中'
        : '📡 定位置スキャン枠 (停止中)';
      ctx.save();
      ctx.setLineDash([8, 5]);
      this.drawSelectionOverlay(ctx, this.scanZone, zoneColor, zoneLabel);
      ctx.restore();
    }

    if (this.isDragging) {
      const rect = this.getNormalizedDragRect();
      const dragColor = this.isSettingScanZone ? '#fbbf24' : '#38bdf8';
      const dragLabel = this.isSettingScanZone
        ? '📡 ここに定位置スキャン枠を設定します...'
        : 'カードを四角く囲んでください...';
      this.drawSelectionOverlay(ctx, rect, dragColor, dragLabel);
    } else {
      if (this.lastSelectionBox && this.lastSelectionBox !== this.scanZone) {
        this.drawSelectionOverlay(ctx, this.lastSelectionBox, 'rgba(251, 191, 36, 0.55)', '選択範囲');
      }
      if (this.bestMatchedSubBox && this.bestMatchedSubBox !== this.scanZone) {
        this.drawSelectionOverlay(ctx, this.bestMatchedSubBox, '#10b981', '🎯 AI認識エリア');
      }
    }
  }

  drawMediaContain(ctx, media, srcW, srcH, dstW, dstH) {
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, dstW, dstH);
    if (!srcW || !srcH) return;
    const scale = Math.min(dstW / srcW, dstH / srcH);
    const w = srcW * scale;
    const h = srcH * scale;
    const x = (dstW - w) / 2;
    const y = (dstH - h) / 2;
    ctx.drawImage(media, x, y, w, h);
  }

  drawDemoPlaymat(ctx, W, H) {
    const grad = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, 550);
    grad.addColorStop(0, '#1e293b');
    grad.addColorStop(1, '#090d16');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(40, H / 2);
    ctx.lineTo(W - 40, H / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(148, 163, 184, 0.75)';
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText('【相手フィールド（180°逆さま配置：ディヴェル・ギルス・うらら）】 ← クリックしてAI認識！', 50, 35);
    ctx.fillText('【自分フィールド（正位置配置）】 ← クリックまたはイラスト部分をドラッグで囲んでAI認識！', 50, H - 20);

    for (const item of this.demoBoardCards) {
      ctx.save();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(item.x - 6, item.y - 6, item.w + 12, item.h + 12);

      ctx.translate(item.x + item.w / 2, item.y + item.h / 2);
      if (item.upsideDown) {
        ctx.rotate(Math.PI);
      }

      const cx = -item.w / 2;
      const cy = -item.h / 2;

      ctx.fillStyle = '#a16207';
      ctx.fillRect(cx, cy, item.w, item.h);
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(cx + 6, cy + 6, item.w - 12, item.h - 12);

      const artSize = item.w - 24;
      const artX = cx + 12;
      const artY = cy + 32;
      if (item.card.safeCanvasElement) {
        ctx.drawImage(item.card.safeCanvasElement, artX, artY, artSize, artSize);
      }

      ctx.fillStyle = '#e2e8f0';
      ctx.fillRect(cx + 10, cy + 36 + artSize, item.w - 20, item.h - (44 + artSize));

      ctx.restore();
    }
  }

  drawSelectionOverlay(ctx, rect, color, label) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

    ctx.fillStyle = color;
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(label, rect.x + 4, Math.max(14, rect.y - 6));
    ctx.restore();
  }

  getCanvasCoords(e) {
    const rect = this.mainCanvas.getBoundingClientRect();
    const scaleX = this.mainCanvas.width / rect.width;
    const scaleY = this.mainCanvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  onCanvasMouseDown(e) {
    if (e.button !== 0 || this.isRecognizing || this.isSwitchingEngine) return;
    const pos = this.getCanvasCoords(e);
    this.isDragging = true;
    this.dragStart = pos;
    this.dragCurrent = pos;
  }

  onCanvasMouseMove(e) {
    if (!this.isDragging) return;
    this.dragCurrent = this.getCanvasCoords(e);
  }

  onCanvasMouseUp(e) {
    if (!this.isDragging) return;
    this.isDragging = false;
    this.dragCurrent = this.getCanvasCoords(e);

    const dx = Math.abs(this.dragCurrent.x - this.dragStart.x);
    const dy = Math.abs(this.dragCurrent.y - this.dragStart.y);

    const isDrag = dx >= 12 || dy >= 12;

    // 定位置スキャン枠の設定モード中なら、ドラッグ範囲をスキャン枠として保存＆自動認識ON
    if (this.isSettingScanZone) {
      this.isSettingScanZone = false;
      if (isDrag) {
        this.scanZone = this.getNormalizedDragRect();
        this.autoScanEnabled = true;
        this.lastAutoScanCardId = null;
        this.scanZonePrevGray = null;
        this.scanZoneDirty = false;
        this.scanZoneStillTicks = 0;
        this.saveScanZone();
        this.updateScanZoneUI();
        // 設定直後に1回スキャン実行
        this.recognizeRegion(this.scanZone, true, true);
      } else {
        this.updateScanZoneUI();
      }
      return;
    }

    this.lastClickWasDemoCard = false;
    this.lastSelectionWasDrag = isDrag;
    let targetBox;
    if (!isDrag) {
      targetBox = this.determineClickCropBox(this.dragStart.x, this.dragStart.y);
    } else {
      targetBox = this.getNormalizedDragRect();
    }

    this.lastSelectionBox = targetBox;
    this.recognizeRegion(targetBox, isDrag, false);
  }

  determineClickCropBox(clickX, clickY) {
    if (this.mode === 'demo') {
      for (const item of this.demoBoardCards) {
        if (
          clickX >= item.x &&
          clickX <= item.x + item.w &&
          clickY >= item.y &&
          clickY <= item.y + item.h
        ) {
          this.lastClickWasDemoCard = true;
          const artW = item.w * 0.78;
          const artH = artW;
          const artX = item.x + (item.w - artW) / 2;
          const artY = item.upsideDown
            ? item.y + item.h - item.h * 0.175 - artH
            : item.y + item.h * 0.175;
          return { x: artX, y: artY, w: artW, h: artH };
        }
      }
    }

    const size = parseInt(this.cropSizeSelect.value, 10) || 85;
    // OpenCV SIFTモードではカード全体（枠・星・イラスト）が入った方が一致点が倍増するため広めに切り抜く
    const boxW = this.useSiftServer ? Math.round(size * 1.6) : size;
    const boxH = this.useSiftServer ? Math.round(size * 2.0) : size;
    const x = Math.max(0, Math.min(this.mainCanvas.width - boxW, clickX - boxW / 2));
    const y = Math.max(0, Math.min(this.mainCanvas.height - boxH, clickY - boxH / 2));
    return { x, y, w: boxW, h: boxH };
  }

  getNormalizedDragRect() {
    const x = Math.min(this.dragStart.x, this.dragCurrent.x);
    const y = Math.min(this.dragStart.y, this.dragCurrent.y);
    const w = Math.max(16, Math.abs(this.dragCurrent.x - this.dragStart.x));
    const h = Math.max(16, Math.abs(this.dragCurrent.y - this.dragStart.y));
    return { x, y, w, h };
  }

  /**
   * AI推論向けに厳選したマルチスケール探索ウィンドウを生成
   * 深層AI特徴量は微小な位置ズレに強いため、不要な細かすぎるスライドを省いて高速＆高精度化
   */
  generateAISearchWindows(box, isDragBox) {
    const windows = [];
    const W = this.mainCanvas.width;
    const H = this.mainCanvas.height;

    const addWin = (x, y, w, h, uprightOnly = false) => {
      const cw = Math.max(24, Math.min(W, w));
      const ch = Math.max(24, Math.min(H, h));
      const cx = Math.max(0, Math.min(W - cw, x));
      const cy = Math.max(0, Math.min(H - ch, y));
      windows.push({ x: cx, y: cy, w: cw, h: ch, uprightOnly });
    };

    // デモ盤面のカードクリック時は正確なイラスト領域が確定しているため1枠のみ（4方向=4推論で最速）
    if (this.lastClickWasDemoCard) {
      addWin(box.x, box.y, box.w, box.h);
      return windows;
    }

    const centerX = box.x + box.w / 2;
    const centerY = box.y + box.h / 2;
    const baseSide = Math.min(box.w, box.h);

    if (isDragBox) {
      // 1) ユーザーが囲んだ枠そのまま
      addWin(box.x, box.y, box.w, box.h);

      // 2) 縦長のカード全体をドラッグで囲んだ場合、正位置・逆位置のイラスト領域を自動抽出
      if (box.h > box.w * 1.18) {
        const artSide = box.w * 0.78;
        addWin(box.x + (box.w - artSide) / 2, box.y + box.h * 0.175, artSide, artSide);
        addWin(box.x + (box.w - artSide) / 2, box.y + box.h - box.h * 0.175 - artSide, artSide, artSide);
      } else if (box.w > box.h * 1.18) {
        // 横向き（守備表示）のカード全体を囲んだ場合
        const artSide = box.h * 0.78;
        addWin(box.x + box.w * 0.175, box.y + (box.h - artSide) / 2, artSide, artSide);
        addWin(box.x + box.w - box.w * 0.175 - artSide, box.y + (box.h - artSide) / 2, artSide, artSide);
      } else {
        // 正方形に近くイラストを囲んだ場合：外枠の茶色いフチを除外した内側84%もチェック
        const innerSide = baseSide * 0.84;
        addWin(centerX - innerSide / 2, centerY - innerSide / 2, innerSide, innerSide);
      }
    } else {
      // ワンクリック時：標準(100%) + 小(78%) + 大(124%) の3スケール
      const scales = [1.0, 0.78, 1.24];
      for (const s of scales) {
        const side = baseSide * s;
        addWin(centerX - side / 2, centerY - side / 2, side, side, false);
      }
      // カード中央をクリックした際にテキスト欄が混ざるのを防ぐ上下シフト枠（高速なMobileNet/Hybrid時に有効化）
      if (this.aiEngine.activeEngine !== 'dinov2') {
        const shiftSide = baseSide * 0.92;
        const offset = baseSide * 0.22;
        addWin(centerX - shiftSide / 2, centerY - offset - shiftSide / 2, shiftSide, shiftSide, true);
        addWin(centerX - shiftSide / 2, centerY + offset - shiftSide / 2, shiftSide, shiftSide, true);
      }
    }

    return windows;
  }

  /**
   * 1枚のカード（3種クロップ）とクエリ特徴（Mean-Centering済みAIベクトル＋空間カラー特徴＋カード枠色判定）の統合スコアを計算
   * 戻り値: { rankScore: 順位付け用の高精度スコア, rawSim: 画面表示用の直感的な一致率(0..1) }
   */
  scoreCardAgainstQuery(card, q, useDinoFeatures = false, aiWeight = 0.86, colorWeight = 0.14, applySidewaysGuard = true) {
    const useCentered = useDinoFeatures
      ? Boolean(this.dinoMeanVec && card.cDinoFeature && q.centeredVec)
      : Boolean(this.dbMeanVec && card.cFeature && q.centeredVec);

    const qVec = useCentered ? q.centeredVec : q.vec;
    const fFull = useDinoFeatures
      ? (useCentered ? card.cDinoFeature : card.dinoFeature)
      : (useCentered ? card.cFeature : card.feature);
    const fCenter = useDinoFeatures
      ? (useCentered ? card.cDinoCenterFeature : card.dinoCenterFeature)
      : (useCentered ? card.cCenterFeature : card.centerFeature);
    const fTight = useDinoFeatures
      ? (useCentered ? card.cDinoTightFeature : card.dinoTightFeature)
      : (useCentered ? card.cTightFeature : card.tightFeature);
    if (!fFull || !qVec) return { rankScore: -1, rawSim: 0 };

    const rawFull = useDinoFeatures ? card.dinoFeature : card.feature;
    const rawCenter = useDinoFeatures ? card.dinoCenterFeature : card.centerFeature;
    const rawTight = useDinoFeatures ? card.dinoTightFeature : card.tightFeature;

    const calcCropScore = (aiVec, cardColorVec) => {
      if (!aiVec) return -1;
      const aiSim = this.aiEngine.cosineSimilarity(qVec, aiVec);
      if (q.colorVec && cardColorVec) {
        const colorSim = this.aiEngine.cosineSimilarity(q.colorVec, cardColorVec);
        return aiSim * aiWeight + colorSim * colorWeight;
      }
      return aiSim;
    };

    const sFull = calcCropScore(fFull, card.colorFeature);
    const sCenter = calcCropScore(fCenter, card.centerColorFeature);
    const sTight = calcCropScore(fTight, card.tightColorFeature);

    // 単一クロップの偶然一致（外れ値）を抑えるため、1位クロップ(82%) + 2位クロップ(18%) のコンセンサススコアを採用
    const arr = [sFull, sCenter, sTight].sort((a, b) => b - a);
    let rankScore = arr[1] > -0.5 ? arr[0] * 0.82 + arr[1] * 0.18 : arr[0];

    // 画面表示用の素のコサイン類似度（直感的な%表示用）
    const rFull = rawFull ? this.aiEngine.cosineSimilarity(q.vec, rawFull) : 0;
    const rCenter = rawCenter ? this.aiEngine.cosineSimilarity(q.vec, rawCenter) : 0;
    const rTight = rawTight ? this.aiEngine.cosineSimilarity(q.vec, rawTight) : 0;
    let rawSim = Math.max(rFull, rCenter, rTight);

    // CSLS ハブ現象補正（他カード全般と似やすいハブカードの底上げ分を減算）
    if (useCentered) {
      const hubBias = useDinoFeatures ? (card.dinoHubBias || 0) : (card.hubBias || 0);
      const hubPenaltyWeight = useDinoFeatures ? 0.40 : 0.45;
      rankScore -= hubBias * hubPenaltyWeight;
    }

    // カード枠色判定によるボーナス・ペナルティ（黒枠X vs 青枠L などの別種別誤爆を強力に抑止）
    const frameInfo = q.frameInfo;
    if (
      frameInfo &&
      frameInfo.frameType !== 'unknown' &&
      frameInfo.confidence >= 0.42 &&
      card.frameType &&
      card.frameType !== 'unknown'
    ) {
      if (card.frameType === frameInfo.frameType) {
        rankScore += 0.095 * frameInfo.confidence;
        rawSim = Math.min(0.99, rawSim + 0.04 * frameInfo.confidence);
      } else {
        rankScore -= 0.075 * frameInfo.confidence;
        rawSim = Math.max(0, rawSim - 0.05 * frameInfo.confidence);
      }
    }

    // 横向き（90°/270°）への偶然誤爆ガード（本当に横向きのカードなら大差で勝つため影響せず、縦カードの誤爆だけ防ぐ）
    if (applySidewaysGuard && (q.deg === 90 || q.deg === 270)) {
      rankScore -= 0.018;
    }

    return { rankScore, rawSim };
  }

  /**
   * 現在の表示ソースから劣化のない最高解像度スナップショットを取得し、
   * mainCanvas座標（960x580）→ 元映像/元画像座標への変換関数を返す
   */
  captureHighResSnapshot() {
    const dstW = this.mainCanvas.width;
    const dstH = this.mainCanvas.height;

    const buildContainMapper = (srcW, srcH, snapW, snapH) => {
      const displayScale = Math.min(dstW / srcW, dstH / srcH);
      const dispW = srcW * displayScale;
      const dispH = srcH * displayScale;
      const offsetX = (dstW - dispW) / 2;
      const offsetY = (dstH - dispH) / 2;
      const snapRatioX = snapW / srcW;
      const snapRatioY = snapH / srcH;

      return (rect) => {
        const rawX = ((rect.x - offsetX) / displayScale) * snapRatioX;
        const rawY = ((rect.y - offsetY) / displayScale) * snapRatioY;
        const rawW = (rect.w / displayScale) * snapRatioX;
        const rawH = (rect.h / displayScale) * snapRatioY;

        const w = Math.max(16, Math.min(snapW, rawW));
        const h = Math.max(16, Math.min(snapH, rawH));
        const x = Math.max(0, Math.min(snapW - w, rawX));
        const y = Math.max(0, Math.min(snapH - h, rawY));
        return { x, y, w, h };
      };
    };

    // 1) Webカメラ / Discord画面キャプチャ：縮小前のネイティブ解像度videoフレームから直接切り抜く
    if (this.mode === 'video' && this.video.readyState >= 2 && this.video.videoWidth > 0 && this.video.videoHeight > 0) {
      const srcW = this.video.videoWidth;
      const srcH = this.video.videoHeight;
      const snap = document.createElement('canvas');
      snap.width = srcW;
      snap.height = srcH;
      snap.getContext('2d').drawImage(this.video, 0, 0, srcW, srcH);
      return {
        snapshotCanvas: snap,
        mapBoxToSnapshot: buildContainMapper(srcW, srcH, srcW, srcH)
      };
    }

    // 2) 画像ファイル読込：元画像のネイティブ解像度（最大2560px）から直接切り抜く
    if (this.mode === 'image' && this.loadedImage && this.loadedImage.width > 0 && this.loadedImage.height > 0) {
      const srcW = this.loadedImage.width;
      const srcH = this.loadedImage.height;
      const maxDim = 2560;
      const capScale = Math.min(1, maxDim / Math.max(srcW, srcH));
      const snapW = Math.max(1, Math.round(srcW * capScale));
      const snapH = Math.max(1, Math.round(srcH * capScale));
      const snap = document.createElement('canvas');
      snap.width = snapW;
      snap.height = snapH;
      snap.getContext('2d').drawImage(this.loadedImage, 0, 0, snapW, snapH);
      return {
        snapshotCanvas: snap,
        mapBoxToSnapshot: buildContainMapper(srcW, srcH, snapW, snapH)
      };
    }

    // 3) デモ盤面モード：オーバーレイ抜きのmainCanvasからスナップショット
    this.lastSelectionBox = null;
    this.bestMatchedSubBox = null;
    this.drawFrame();

    const snap = document.createElement('canvas');
    snap.width = dstW;
    snap.height = dstH;
    snap.getContext('2d').drawImage(this.mainCanvas, 0, 0);
    return {
      snapshotCanvas: snap,
      mapBoxToSnapshot: (rect) => rect
    };
  }

  /**
   * 定位置スキャンエリアの常時監視ループ（350ms間隔・動きを検知して静止した瞬間に自動認識）
   */
  startAutoScanLoop() {
    if (this._autoScanTimer) clearInterval(this._autoScanTimer);
    this._autoScanTimer = setInterval(() => {
      this.tickAutoScan();
    }, 350);
  }

  tickAutoScan() {
    if (
      !this.autoScanEnabled ||
      !this.scanZone ||
      this.mode !== 'video' ||
      this.video.readyState < 2 ||
      this.isRecognizing ||
      this.isSwitchingEngine ||
      this.isDragging ||
      this.isSettingScanZone
    ) {
      return;
    }

    const srcW = this.video.videoWidth;
    const srcH = this.video.videoHeight;
    if (!srcW || !srcH) return;

    const dstW = this.mainCanvas.width;
    const dstH = this.mainCanvas.height;
    const displayScale = Math.min(dstW / srcW, dstH / srcH);
    const offsetX = (dstW - srcW * displayScale) / 2;
    const offsetY = (dstH - srcH * displayScale) / 2;

    const sx = Math.max(0, Math.min(srcW - 16, (this.scanZone.x - offsetX) / displayScale));
    const sy = Math.max(0, Math.min(srcH - 16, (this.scanZone.y - offsetY) / displayScale));
    const sw = Math.max(16, Math.min(srcW - sx, this.scanZone.w / displayScale));
    const sh = Math.max(16, Math.min(srcH - sy, this.scanZone.h / displayScale));

    const ctx = this.scanZoneProbeCtx;
    ctx.drawImage(this.video, sx, sy, sw, sh, 0, 0, 48, 48);
    const data = ctx.getImageData(0, 0, 48, 48).data;
    const gray = new Uint8Array(48 * 48);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = Math.round(0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]);
    }

    if (!this.scanZonePrevGray) {
      this.scanZonePrevGray = gray;
      return;
    }

    let diffSum = 0;
    for (let i = 0; i < gray.length; i++) {
      diffSum += Math.abs(gray[i] - this.scanZonePrevGray[i]);
    }
    const meanDiff = diffSum / gray.length;
    this.scanZonePrevGray = gray;

    if (meanDiff > 7.5) {
      // 手やカードが動いている間は待機フラグを立てる
      this.scanZoneDirty = true;
      this.scanZoneStillTicks = 0;
    } else if (this.scanZoneDirty && meanDiff < 4.5) {
      this.scanZoneStillTicks++;
      // 約0.7秒静止したら（手が離れたら）自動で認識実行
      if (this.scanZoneStillTicks >= 2) {
        this.scanZoneDirty = false;
        this.scanZoneStillTicks = 0;
        this.recognizeRegion(this.scanZone, true, true);
      }
    }
  }

  /**
   * Python OpenCV SIFT + RANSAC サーバーで高速・超高精度認識（送信画像軽量化＆トンネル切断時自動リトライ対応）
   */
  async recognizeWithSiftServer(snapshotCanvas, srcRect, displayBox, isAutoScan = false) {
    const startTime = performance.now();

    // サーバー側の上限520pxに合わせて事前に軽量リサイズし、細い回線でも一瞬で送信できるようにする
    const maxDim = 520;
    const srcMax = Math.max(1, srcRect.w, srcRect.h);
    const scale = Math.min(1, maxDim / srcMax);
    const outW = Math.max(32, Math.round(srcRect.w * scale));
    const outH = Math.max(32, Math.round(srcRect.h * scale));

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = outW;
    cropCanvas.height = outH;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.imageSmoothingEnabled = true;
    cropCtx.imageSmoothingQuality = 'high';
    cropCtx.drawImage(snapshotCanvas, srcRect.x, srcRect.y, srcRect.w, srcRect.h, 0, 0, outW, outH);

    if (!isAutoScan) {
      this.matchStatusText.textContent = '⚡ OpenCV SIFT サーバーで解析中...';
      this.matchBadgeContainer.innerHTML = `<span class="match-badge processing">OpenCV SIFT + RANSAC で照合中</span>`;
    }

    const blob = await new Promise((resolve) => cropCanvas.toBlob(resolve, 'image/jpeg', 0.85));

    let data = null;
    let lastErr = null;
    const MAX_ATTEMPTS = 3;

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      const formData = new FormData();
      formData.append('file', blob, 'crop.jpg');
      formData.append('mode', 'auto');

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 12000);
      try {
        const res = await fetch(`${this.siftServerBase}/api/recognize`, {
          method: 'POST',
          body: formData,
          signal: controller.signal
        });
        clearTimeout(timer);

        if (res.status === 502 || res.status === 503 || res.status === 504) {
          throw new Error(`トンネル再接続中 (${res.status})`);
        }
        if (!res.ok) {
          throw new Error(`SIFTサーバーエラー (${res.status})`);
        }
        data = await res.json();
        break;
      } catch (err) {
        clearTimeout(timer);
        lastErr = err;
        if (attempt < MAX_ATTEMPTS) {
          if (!isAutoScan) {
            this.matchStatusText.textContent = `🔄 通信再試行中 (${attempt + 1}/${MAX_ATTEMPTS})...`;
          }
          await new Promise((r) => setTimeout(r, 1100 * attempt));
        }
      }
    }

    if (!data) {
      throw lastErr || new Error('SIFTサーバーへ接続できませんでした');
    }
    if (!data.ok) {
      throw new Error(data.error || 'SIFT認識に失敗しました');
    }

    // 自動スキャン時は、カードが置かれていない（一致特徴点が足りない）フレームを静かにスキップ
    if (isAutoScan && !data.recognized) {
      this.lastAutoScanCardId = null;
      return;
    }

    const mapped = (data.matches || [])
      .map((m) => {
        const card = this.cardsDB.find((c) => c.id === m.id);
        if (!card) return null;
        return {
          card,
          score: m.score,
          displayScore: m.score,
          inliers: m.inliers,
          orientation: isAutoScan ? `自動スキャン (一致点:${m.inliers})` : `SIFT全方向対応 (一致点:${m.inliers})`,
          deg: 0,
          matchedBox: displayBox
        };
      })
      .filter(Boolean);

    if (mapped.length === 0) {
      if (!isAutoScan) {
        this.matchStatusText.textContent = '⚠️ 一致するカードが見つかりませんでした';
        this.matchBadgeContainer.innerHTML = '';
      }
      return;
    }

    const topMatch = mapped[0];
    // 自動スキャンで直前とまったく同じカードが置かれたままなら履歴を重複更新しない
    if (isAutoScan && this.lastAutoScanCardId === topMatch.card.id) {
      return;
    }
    if (isAutoScan) {
      this.lastAutoScanCardId = topMatch.card.id;
      this.scanZoneFlashUntil = performance.now() + 2200;
      this.scanZoneFlashLabel = `✅ ${topMatch.card.name_ja}`;
    }

    // プレビュー表示更新
    this.cropPreviewCtx.clearRect(0, 0, 64, 64);
    this.cropPreviewCtx.drawImage(cropCanvas, 0, 0, 64, 64);

    if (!isAutoScan) {
      this.lastSelectionBox = displayBox;
      this.bestMatchedSubBox = displayBox;
    }

    const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(2);
    const secondMatch = mapped[1];
    const margin = secondMatch ? topMatch.score - secondMatch.score : 1;

    const meta = {
      elapsedSec,
      margin: data.recognized ? Math.max(margin, 0.1) : 0,
      engineLabel: `OpenCV SIFT (${topMatch.inliers}pt)`,
      frameLabel: ''
    };

    this.lastRecognizedCardId = topMatch.card.id;
    this.addToHistory(topMatch);
    this.displayRecognitionResult(topMatch, mapped.slice(0, 5), meta);
  }

  /**
   * 選択領域を認識（OpenCV SIFTサーバー優先、未接続時はブラウザ内AIへフォールバック）
   */
  async recognizeRegion(box, isDragBox = false, isAutoScan = false) {
    if (this.isRecognizing || this.isSwitchingEngine) return;

    if (this.useSiftServer && this.siftServerBase !== null) {
      this.isRecognizing = true;
      try {
        const { snapshotCanvas, mapBoxToSnapshot } = this.captureHighResSnapshot();
        const srcRect = mapBoxToSnapshot(box);
        await this.recognizeWithSiftServer(snapshotCanvas, srcRect, box, isAutoScan);
      } catch (err) {
        console.error('SIFT recognition error:', err);
        if (!isAutoScan) {
          this.matchStatusText.textContent = '⚠️ SIFT認識エラー: ' + err.message;
          this.matchBadgeContainer.innerHTML = '';
        }
      } finally {
        this.isRecognizing = false;
      }
      return;
    }

    const indexedCards = this.cardsDB.filter((c) => c.feature);
    if (!this.aiEngine.isReady || indexedCards.length === 0) {
      if (!isAutoScan) {
        this.matchStatusText.textContent = '⏳ AIモデルとカードDBの準備完了をお待ちください...';
      }
      return;
    }

    this.isRecognizing = true;
    const startTime = performance.now();

    try {
      // 1) 縮小前の元映像・元画像から劣化のない高解像度フレームをスナップショット
      const { snapshotCanvas, mapBoxToSnapshot } = this.captureHighResSnapshot();

      // 選択枠を即座に戻して視覚フィードバックを維持
      if (!isAutoScan) {
        this.lastSelectionBox = box;
        this.bestMatchedSubBox = null;
      }

      const enhanceLighting = this.enhanceLightingCheck ? this.enhanceLightingCheck.checked : true;
      const searchWindows = this.generateAISearchWindows(box, isDragBox);
      const allOrientations = [
        { deg: 0, label: '正位置 (0°)' },
        { deg: 180, label: '相手視点・逆さま (180°)' },
        { deg: 90, label: '横向き・守備 (90°)' },
        { deg: 270, label: '横向き・守備 (270°)' }
      ];
      const uprightOrientations = allOrientations.slice(0, 2);

      const isHybrid = this.aiEngine.activeEngine === 'hybrid';
      const stage1Engine = isHybrid ? 'mobilenet' : this.aiEngine.activeEngine;
      const stage1AiWeight = stage1Engine === 'dinov2' ? 0.90 : 0.86;
      const stage1ColorWeight = 1 - stage1AiWeight;
      const applySidewaysGuard = !(isDragBox && box.w > box.h * 1.15);

      const queryPlan = [];
      for (const win of searchWindows) {
        const oris = win.uprightOnly ? uprightOrientations : allOrientations;
        for (const ori of oris) {
          queryPlan.push({ win, ori });
        }
      }

      const totalSteps = queryPlan.length + (isHybrid ? 2 : 0);
      let doneSteps = 0;

      this.matchStatusText.textContent = `🤖 AI解析中... (0/${totalSteps})`;
      this.matchBadgeContainer.innerHTML = `<span class="match-badge processing">${this.aiEngine.getEngineLabel()} で照合中</span>`;

      const queries = [];
      for (const step of queryPlan) {
        const srcWin = mapBoxToSnapshot(step.win);
        const { vec, colorVec } = await this.aiEngine.extractQueryFeatures(
          snapshotCanvas,
          step.ori.deg,
          srcWin,
          enhanceLighting,
          stage1Engine
        );
        const centeredVec = this.dbMeanVec
          ? this.aiEngine.centerAndNormalize(vec, this.dbMeanVec)
          : null;
        const frameInfo = this.aiEngine.detectCardFrameType(snapshotCanvas, srcWin, step.ori.deg);
        queries.push({
          win: step.win,
          srcWin,
          deg: step.ori.deg,
          orientation: step.ori.label,
          vec,
          centeredVec,
          colorVec,
          frameInfo
        });
        doneSteps++;
        this.matchStatusText.textContent = `🤖 AI解析中... (${doneSteps}/${totalSteps})`;
        await yieldToUI();
      }

      // Stage 1: 全登録カードとの類似度を算出
      let scored = indexedCards.map((card) => {
        let bestRank = -1;
        let bestRaw = 0;
        let bestOrientation = '正位置 (0°)';
        let bestDeg = 0;
        let bestWin = box;
        let bestQueryIdx = 0;
        let bestFrameInfo = null;

        for (let qi = 0; qi < queries.length; qi++) {
          const q = queries[qi];
          const { rankScore, rawSim } = this.scoreCardAgainstQuery(
            card,
            q,
            false,
            stage1AiWeight,
            stage1ColorWeight,
            applySidewaysGuard
          );

          if (rankScore > bestRank) {
            bestRank = rankScore;
            bestRaw = rawSim;
            bestOrientation = q.orientation;
            bestDeg = q.deg;
            bestWin = q.win;
            bestQueryIdx = qi;
            bestFrameInfo = q.frameInfo;
          }
        }

        return {
          card,
          score: bestRank,
          displayScore: bestRaw,
          stage1Score: bestRank,
          stage1Raw: bestRaw,
          orientation: bestOrientation,
          deg: bestDeg,
          matchedBox: bestWin,
          bestQueryIdx,
          frameInfo: bestFrameInfo
        };
      });

      scored.sort((a, b) => b.score - a.score);

      // Stage 2 (Hybridモード): 最良の最大2枠だけを DINOv2-small で推論し、キャッシュ済みの全カード＋上位12枚を精密再ランキング
      if (isHybrid && scored.length > 1) {
        const dinoReady = await this.aiEngine.ensureDinoLoaded((msg) => {
          this.matchStatusText.textContent = `🎯 ${msg}`;
        });

        if (dinoReady) {
          this.matchStatusText.textContent = `🎯 DINOv2 精密判定中... (${doneSteps}/${totalSteps})`;
          await yieldToUI();

          // 上位カードが選んだベスト枠（最大2パターン）だけをDINOv2で推論（12〜16回→最大2回に削減）
          const selectedQueryIndices = [];
          for (const item of scored.slice(0, 6)) {
            if (!selectedQueryIndices.includes(item.bestQueryIdx)) {
              selectedQueryIndices.push(item.bestQueryIdx);
              if (selectedQueryIndices.length >= 2) break;
            }
          }

          const dinoQueries = [];
          for (const qi of selectedQueryIndices) {
            const baseQ = queries[qi];
            const { vec } = await this.aiEngine.extractQueryFeatures(
              snapshotCanvas,
              baseQ.deg,
              baseQ.srcWin,
              enhanceLighting,
              'dinov2'
            );
            const centeredVec = this.dinoMeanVec
              ? this.aiEngine.centerAndNormalize(vec, this.dinoMeanVec)
              : null;
            dinoQueries.push({
              ...baseQ,
              vec,
              centeredVec
            });
            doneSteps++;
            this.matchStatusText.textContent = `🎯 DINOv2 精密判定中... (${Math.min(doneSteps, totalSteps)}/${totalSteps})`;
            await yieldToUI();
          }

          // Top8足切りを撤廃：DINOv2キャッシュ済みの全カード＋上位12枚すべてをDINOv2で再採点
          for (let idx = 0; idx < scored.length; idx++) {
            const item = scored[idx];
            const c = item.card;
            if (!c.dinoFeature) {
              if (idx >= 12) continue;
              const emb = await this.aiEngine.getOrComputeCardEmbeddings(c.id, c.safeCanvasElement, 'dinov2');
              c.dinoFeature = emb.feature;
              c.dinoCenterFeature = emb.centerFeature;
              c.dinoTightFeature = emb.tightFeature;
              if (this.dinoMeanVec) {
                c.cDinoFeature = this.aiEngine.centerAndNormalize(c.dinoFeature, this.dinoMeanVec);
                c.cDinoCenterFeature = this.aiEngine.centerAndNormalize(c.dinoCenterFeature, this.dinoMeanVec);
                c.cDinoTightFeature = this.aiEngine.centerAndNormalize(c.dinoTightFeature, this.dinoMeanVec);
              }
            }

            let bestDinoRank = -1;
            let bestDinoRaw = 0;
            let bestQ = dinoQueries[0];
            for (const dq of dinoQueries) {
              const { rankScore, rawSim } = this.scoreCardAgainstQuery(c, dq, true, 0.90, 0.10, applySidewaysGuard);
              if (rankScore > bestDinoRank) {
                bestDinoRank = rankScore;
                bestDinoRaw = rawSim;
                bestQ = dq;
              }
            }

            if (bestDinoRank > -0.5) {
              // MobileNet+色+枠(28%) + DINOv2+色+枠(72%) のハイブリッド統合スコア
              item.score = item.stage1Score * 0.28 + bestDinoRank * 0.72;
              item.displayScore = item.stage1Raw * 0.28 + bestDinoRaw * 0.72;
              item.orientation = bestQ.orientation;
              item.deg = bestQ.deg;
              item.matchedBox = bestQ.win;
              item.frameInfo = bestQ.frameInfo;
            }
          }

          scored.sort((a, b) => b.score - a.score);
        }
      }

      const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(2);
      // 順位と表示用%の逆転を防ぎつつ、1位と2位以下のスコア差を直感的に反映
      if (scored.length > 0) {
        const topAnchor = Math.max(0.1, Math.min(0.99, scored[0].displayScore ?? scored[0].score));
        const topRank = scored[0].score;
        scored[0].displayScore = topAnchor;
        for (let i = 1; i < scored.length; i++) {
          const gap = Math.max(0.001, topRank - scored[i].score);
          scored[i].displayScore = Math.max(0.05, topAnchor - gap);
        }
      }
      const topMatch = scored[0];
      const secondMatch = scored[1];
      const margin = topMatch && secondMatch ? topMatch.score - secondMatch.score : 1;
      const top5 = scored.slice(0, 5);

      // 実際にAIが見た（補正・回転済みの）ベストマッチ領域をプレビューに表示
      const finalCropBox = topMatch ? topMatch.matchedBox : box;
      const previewCanvas = this.aiEngine.preprocessToCanvas(
        snapshotCanvas,
        topMatch ? topMatch.deg : 0,
        mapBoxToSnapshot(finalCropBox),
        enhanceLighting
      );
      this.cropPreviewCtx.clearRect(0, 0, 64, 64);
      this.cropPreviewCtx.drawImage(previewCanvas, 0, 0, 64, 64);

      this.lastSelectionBox = box;
      this.bestMatchedSubBox = finalCropBox;

      const meta = {
        elapsedSec,
        margin,
        engineLabel: this.aiEngine.getEngineLabel(),
        frameLabel:
          topMatch && topMatch.frameInfo && topMatch.frameInfo.confidence >= 0.42
            ? topMatch.frameInfo.label
            : ''
      };

      this.lastRecognizedCardId = topMatch ? topMatch.card.id : null;
      this.addToHistory(topMatch);
      this.displayRecognitionResult(topMatch, top5, meta);
    } catch (err) {
      console.error('Recognition error:', err);
      this.matchStatusText.textContent = '⚠️ AI認識エラー: ' + err.message;
      this.matchBadgeContainer.innerHTML = '';
    } finally {
      this.isRecognizing = false;
    }
  }

  escapeHtml(text) {
    return String(text ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  isCardPinned(cardId) {
    return this.pinnedCards.some((p) => p.card.id === cardId);
  }

  togglePinCard(card, orientation = 'ピン留め') {
    if (!card) return;
    const idx = this.pinnedCards.findIndex((p) => p.card.id === card.id);
    if (idx !== -1) {
      this.pinnedCards.splice(idx, 1);
    } else {
      this.pinnedCards.push({ card, orientation });
      if (this.pinnedCards.length > 2) {
        this.pinnedCards.shift();
      }
    }
    this.renderPinnedCards();
    this.renderHistory();
    this.refreshDetailPinButton();
  }

  refreshDetailPinButton() {
    const btn = document.getElementById('btnPinActiveCard');
    if (!btn || !this.currentMatch) return;
    const pinned = this.isCardPinned(this.currentMatch.card.id);
    btn.classList.toggle('pinned', pinned);
    btn.textContent = pinned ? '📌 ピン留め中（クリックで解除）' : '📌 比較用にピン留め';
  }

  renderPinnedCards() {
    if (!this.pinnedCompareSection || !this.pinnedCardsList || !this.pinnedCount) return;

    this.pinnedCount.textContent = this.pinnedCards.length;
    if (this.pinnedCards.length === 0) {
      this.pinnedCompareSection.style.display = 'none';
      this.pinnedCardsList.innerHTML = '';
      return;
    }

    this.pinnedCompareSection.style.display = 'flex';
    this.pinnedCardsList.innerHTML = '';

    this.pinnedCards.forEach((item) => {
      const { card } = item;
      const box = document.createElement('div');
      box.className = 'pinned-card-box';
      box.innerHTML = `
        <div class="pinned-card-head">
          <img src="${this.escapeHtml(card.image_crop)}" alt="${this.escapeHtml(card.name_ja)}" />
          <div class="pinned-card-meta">
            <div class="pinned-card-name">${this.escapeHtml(card.name_ja)}</div>
            <div class="pinned-card-sub">${this.escapeHtml(card.stats_ja)}</div>
          </div>
          <button class="icon-btn btn-pinned-focus" title="このカードをメイン詳細に表示">拡大</button>
          <button class="icon-btn btn-pinned-remove" title="ピン留めを解除">✕</button>
        </div>
        <div class="pinned-card-effect">${this.escapeHtml(card.desc_ja)}</div>
      `;

      box.querySelector('.btn-pinned-focus').addEventListener('click', () => {
        this.displayRecognitionResult({ card, score: 1, orientation: item.orientation || 'ピン参照' }, [{ card, score: 1 }]);
      });
      box.querySelector('.btn-pinned-remove').addEventListener('click', () => {
        this.togglePinCard(card);
      });

      this.pinnedCardsList.appendChild(box);
    });
  }

  displayRecognitionResult(match, topCandidates, meta = null) {
    const { card, score, displayScore, orientation } = match;
    this.currentMatch = match;
    this.currentCardId = card.id;

    // 画面表示には素のコサイン類似度（displayScore）を使用し、直感的な%表示を維持
    const shownScore = displayScore !== undefined ? displayScore : score;
    const clampedScore = Math.max(0, Math.min(1, shownScore));
    const percent = (clampedScore * 100).toFixed(1);

    const frameSuffix = meta && meta.frameLabel ? ` / 枠: ${meta.frameLabel}` : '';
    this.matchStatusText.textContent = `向き判定: ${orientation}${frameSuffix}`;

    const isHighConfidence = !meta || meta.margin >= 0.045 || clampedScore >= 0.70;
    const badgeClass = isHighConfidence ? 'match-badge' : 'match-badge medium';
    const confidenceLabel = isHighConfidence ? '✅ 高信頼' : '⚠️ 候補確認推奨';
    const engineSuffix = meta
      ? `<span class="engine-badge">${this.escapeHtml(meta.engineLabel)} / ${meta.elapsedSec}s</span>`
      : '';

    this.matchBadgeContainer.innerHTML = `<span class="${badgeClass}">${confidenceLabel} 類似度: ${percent}%</span>${engineSuffix}`;

    const imgSrc = card.image_full;
    const pinned = this.isCardPinned(card.id);

    this.cardDetailContainer.innerHTML = `
      <div class="card-detail">
        <div class="card-top">
          <div class="card-image-box">
            <img src="${this.escapeHtml(imgSrc)}" alt="${this.escapeHtml(card.name_ja)}" onerror="this.src='${this.escapeHtml(card.image_crop)}'" />
          </div>
          <div class="card-meta">
            <div class="card-name-ja">${this.escapeHtml(card.name_ja)}</div>
            <div class="card-name-en">${this.escapeHtml(card.name_en)} (ID: ${card.id})</div>
            <div class="card-type-badge">${this.escapeHtml(card.type_ja)}</div>
            <div class="card-stats">${this.escapeHtml(card.stats_ja)}</div>
            <div class="card-actions-row">
              <button id="btnPinActiveCard" class="btn-pin-card ${pinned ? 'pinned' : ''}" title="このカードを下にピン留めして別カードと効果を見比べます">
                ${pinned ? '📌 ピン留め中（クリックで解除）' : '📌 比較用にピン留め'}
              </button>
            </div>
          </div>
        </div>
        <div class="card-effect-box">
          <div class="card-effect-title">
            <span>📖 カード効果・テキスト</span>
            <span>ID: ${card.id}</span>
          </div>
          <div class="card-effect-text">${this.escapeHtml(card.desc_ja)}</div>
        </div>
      </div>
    `;

    const pinBtn = document.getElementById('btnPinActiveCard');
    if (pinBtn) {
      pinBtn.addEventListener('click', () => {
        this.togglePinCard(card, orientation);
      });
    }

    if (topCandidates && topCandidates.length > 1) {
      this.candidatesSection.style.display = 'block';
      this.candidatesList.innerHTML = '';
      topCandidates.forEach((item, idx) => {
        const shown = item.displayScore !== undefined ? item.displayScore : item.score;
        const p = (Math.max(0, Math.min(1, shown)) * 100).toFixed(1);
        const thumbSrc = item.card.image_crop;

        const div = document.createElement('div');
        div.className = `candidate-item ${item.card.id === card.id ? 'active' : ''}`;
        div.innerHTML = `
          <img src="${this.escapeHtml(thumbSrc)}" alt="${this.escapeHtml(item.card.name_ja)}" />
          <div class="candidate-name">${this.escapeHtml(item.card.name_ja)}</div>
          <div class="candidate-score">${idx + 1}位 (${p}%)</div>
        `;
        div.addEventListener('click', () => {
          // 直前のAI認識で誤った1位を別候補へ修正したときは履歴も入れ替え
          if (this.lastRecognizedCardId && this.lastRecognizedCardId !== item.card.id) {
            const wrongIdx = this.history.findIndex((h) => h.card.id === this.lastRecognizedCardId);
            if (wrongIdx === 0) {
              if ((this.history[0].count || 1) <= 1) {
                this.history.shift();
              } else {
                this.history[0].count -= 1;
              }
            }
            this.lastRecognizedCardId = item.card.id;
            this.addToHistory(item);
          }
          this.displayRecognitionResult(item, topCandidates, meta);
        });
        this.candidatesList.appendChild(div);
      });
    } else {
      this.candidatesSection.style.display = 'none';
      this.candidatesList.innerHTML = '';
    }

    this.renderPinnedCards();
    this.renderHistory();
  }

  addToHistory(match) {
    if (!match || !match.card) return;
    const timeStr = new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const existingIdx = this.history.findIndex((h) => h.card.id === match.card.id);

    if (existingIdx !== -1) {
      const existing = this.history.splice(existingIdx, 1)[0];
      this.history.unshift({
        ...match,
        timeStr,
        count: (existing.count || 1) + 1
      });
    } else {
      this.history.unshift({
        ...match,
        timeStr,
        count: 1
      });
    }

    if (this.history.length > 30) this.history.pop();
    this.renderHistory();
  }

  clearHistory() {
    this.history = [];
    this.renderHistory();
  }

  renderHistory() {
    if (!this.historyList || !this.historyCount) return;

    const totalCount = this.history.length;
    this.historyCount.textContent = totalCount;
    if (this.quickHistoryCount) {
      this.quickHistoryCount.textContent = totalCount;
    }

    const filtered = this.historyFilter
      ? this.history.filter(
          (item) =>
            item.card.name_ja.toLowerCase().includes(this.historyFilter) ||
            item.card.name_en.toLowerCase().includes(this.historyFilter)
        )
      : this.history;

    this.historyList.innerHTML = '';
    if (filtered.length === 0) {
      const msg =
        totalCount === 0
          ? 'まだ認識されたカードはありません。<br />盤面をクリックするとここに追加されます。'
          : '一致するカードがありません。';
      this.historyList.innerHTML = `<div class="empty-state">${msg}</div>`;
    } else {
      filtered.forEach((item) => {
        const isActive = item.card.id === this.currentCardId;
        const isPinned = this.isCardPinned(item.card.id);
        const countBadge = item.count > 1 ? `<span class="history-count-badge">×${item.count}</span>` : '';

        const div = document.createElement('div');
        div.className = `history-item ${isActive ? 'active' : ''}`;
        div.innerHTML = `
          <img src="${this.escapeHtml(item.card.image_crop)}" alt="${this.escapeHtml(item.card.name_ja)}" />
          <div class="history-meta">
            <div class="history-name-row">
              <div class="history-name" title="${this.escapeHtml(item.card.name_ja)}">${this.escapeHtml(item.card.name_ja)}</div>
              ${countBadge}
            </div>
            <div class="history-sub">${this.escapeHtml(item.timeStr)} / ${this.escapeHtml(item.orientation)}</div>
          </div>
          <button class="pin-mini-btn ${isPinned ? 'pinned' : ''}" title="${isPinned ? 'ピン留め解除' : '比較用にピン留め'}">📌</button>
        `;

        div.querySelector('.pin-mini-btn').addEventListener('click', (e) => {
          e.stopPropagation();
          this.togglePinCard(item.card, item.orientation);
        });

        div.addEventListener('click', () => {
          document.querySelector('[data-tab="detail"]').click();
          this.displayRecognitionResult(item, [item]);
        });

        this.historyList.appendChild(div);
      });
    }

    if (this.quickHistoryList) {
      this.quickHistoryList.innerHTML = '';
      if (this.history.length === 0) {
        this.quickHistoryList.innerHTML =
          '<span style="font-size: 0.74rem; color: var(--text-sub);">まだ履歴はありません</span>';
      } else {
        this.history.forEach((item) => {
          const isActive = item.card.id === this.currentCardId;
          const isPinned = this.isCardPinned(item.card.id);
          const countBadge = item.count > 1 ? `<span class="history-count-badge">×${item.count}</span>` : '';

          const chip = document.createElement('div');
          chip.className = `quick-chip ${isActive ? 'active' : ''}`;
          chip.innerHTML = `
            <img src="${this.escapeHtml(item.card.image_crop)}" alt="${this.escapeHtml(item.card.name_ja)}" />
            <span class="quick-chip-name">${this.escapeHtml(item.card.name_ja)}</span>
            ${countBadge}
            <button class="pin-mini-btn ${isPinned ? 'pinned' : ''}" title="${isPinned ? 'ピン留め解除' : '比較用にピン留め'}">📌</button>
          `;

          chip.querySelector('.pin-mini-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            this.togglePinCard(item.card, item.orientation);
          });

          chip.addEventListener('click', () => {
            this.displayRecognitionResult(item, [item]);
          });

          this.quickHistoryList.appendChild(chip);
        });
      }
    }
  }

  renderDBList() {
    this.dbCount.textContent = this.cardsDB.length;
    this.dbCardList.innerHTML = '';
    this.cardsDB.forEach((card) => {
      const statusIcon = card.aiIndexed && !card.isFallback ? '🤖' : card.isFallback ? '⏳' : '🖼️';
      const div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML = `
        <img src="${card.image_crop}" alt="${card.name_ja}" />
        <div class="history-meta">
          <div class="history-name">${statusIcon} ${card.name_ja}</div>
          <div class="history-sub">${card.name_en} (${card.stats_ja})</div>
        </div>
      `;
      div.addEventListener('click', () => {
        document.querySelector('[data-tab="detail"]').click();
        this.displayRecognitionResult({ card, score: 1, orientation: 'DB参照' }, [{ card, score: 1 }]);
      });
      this.dbCardList.appendChild(div);
    });
  }

  async addCardFromApi() {
    const input = document.getElementById('addCardQuery');
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('btnAddFromApi');
    btn.disabled = true;
    btn.textContent = '検索中...';

    try {
      const res = await fetch(`https://db.ygoprodeck.com/api/v7/cardinfo.php?fname=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error('カードが見つかりませんでした');
      const data = await res.json();
      const cards = data.data.slice(0, 8);

      let added = 0;
      for (const c of cards) {
        if (this.cardsDB.some((existing) => existing.id === c.id)) continue;
        const newCard = {
          id: c.id,
          name_ja: c.name,
          name_en: c.name,
          type_ja: `${c.type} / ${c.race || ''}`,
          stats_ja: c.atk !== undefined ? `ATK ${c.atk} / DEF ${c.def ?? '-'}` : c.type,
          desc_ja: c.desc,
          image_full: c.card_images[0].image_url,
          image_crop: c.card_images[0].image_url_cropped
        };
        const fallbackCanvas = this.createFallbackCardArt(newCard);
        const entry = {
          ...newCard,
          frameType: this.aiEngine.getCardFrameTypeFromMeta(newCard),
          safeCanvasElement: fallbackCanvas,
          feature: null,
          centerFeature: null,
          tightFeature: null,
          isFallback: true,
          aiIndexed: false
        };
        this.cardsDB.push(entry);
        await this.loadRealImageForCard(entry);
        if (this.useSiftServer && this.siftServerBase !== null) {
          try {
            await fetch(`${this.siftServerBase}/api/cards/add`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                id: entry.id,
                name_ja: entry.name_ja,
                image_full: entry.image_full,
                image_crop: entry.image_crop
              })
            });
            entry.aiIndexed = true;
            entry.isFallback = false;
          } catch (_) {
            // SIFT追加失敗時は無視
          }
        } else if (this.aiEngine.isReady) {
          const emb = await this.aiEngine.getOrComputeCardEmbeddings(entry.id, entry.safeCanvasElement);
          entry.feature = emb.feature;
          entry.centerFeature = emb.centerFeature;
          entry.tightFeature = emb.tightFeature;
          entry.colorFeature = emb.colorFeature;
          entry.centerColorFeature = emb.centerColorFeature;
          entry.tightColorFeature = emb.tightColorFeature;
          entry.aiIndexed = true;
        }
        added++;
      }

      this.recomputeHubnessStats(false);
      this.renderDBList();
      this.setStatusBadge(
        `✅ AI認識稼働中 [${this.aiEngine.getEngineLabel()}] (${this.cardsDB.length}枚登録済)`,
        false
      );
      alert(`${added}枚の新しいカードをAI認識データベースに追加しました！`);
      input.value = '';
    } catch (err) {
      alert('API検索エラー: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '追加';
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new YgoRemoteLensApp();
});
