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
    this.cardsDB = [];
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
    this.bindEvents();
    this.refreshCameraList();
    this.initDatabase();
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
   * 初期カードデータベースとAIエンジンの並列初期化
   */
  async initDatabase() {
    const initialCards = window.INITIAL_CARDS_DATA || [];

    this.cardsDB = initialCards.map((card) => {
      const fallbackCanvas = this.createFallbackCardArt(card);
      return {
        ...card,
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

    this.setStatusBadge('🤖 AIモデル＆カード画像を取得中...', true);
    if (this.aiEngineSelect) this.aiEngineSelect.disabled = true;

    const preferredEngine = this.aiEngineSelect ? this.aiEngineSelect.value : 'mobilenet';

    // AIモデル読み込みとカード実画像取得を並列で進める
    let aiProgressText = 'AIモデル初期化中...';
    let imgProgressText = `画像(0/${initialCards.length})`;
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

    // フォールバックが発生した場合はプルダウン表示も同期
    if (this.aiEngineSelect) {
      if (actualEngine === 'dinov2' || actualEngine === 'mobilenet') {
        this.aiEngineSelect.value = actualEngine;
      }
      this.aiEngineSelect.disabled = false;
    }

    // 全カードのAI特徴ベクトルを構築（IndexedDBキャッシュがあれば即座に完了）
    await this.rebuildAllCardEmbeddings(false);

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
        if (forceRecompute || cardEntry.isFallback) {
          const W = cardEntry.safeCanvasElement.width;
          const H = cardEntry.safeCanvasElement.height;
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
        } else {
          const emb = await this.aiEngine.getOrComputeCardEmbeddings(cardEntry.id, cardEntry.safeCanvasElement);
          cardEntry.feature = emb.feature;
          cardEntry.centerFeature = emb.centerFeature;
          cardEntry.tightFeature = emb.tightFeature;
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

    this.setStatusBadge(
      `✅ AI認識稼働中 [${engineLabel}] (${total}枚 / 実画像:${realCount}枚)`,
      false
    );
    this.renderDBList();

    if (this.aiEngineSelect) this.aiEngineSelect.disabled = false;
    this.isSwitchingEngine = false;
  }

  /**
   * AIエンジンを動的に切り替え（DINOv2-small <-> MobileNet v2）
   */
  async switchAIEngine(engineId) {
    if (this.isSwitchingEngine || this.isRecognizing) return;
    this.isSwitchingEngine = true;
    if (this.aiEngineSelect) this.aiEngineSelect.disabled = true;

    try {
      this.setStatusBadge(`🔄 AIエンジン切替中 (${engineId})...`, true);
      const actualEngine = await this.aiEngine.init(engineId, (msg) => {
        this.setStatusBadge(`🔄 ${msg}`, true);
      });

      if (this.aiEngineSelect && (actualEngine === 'dinov2' || actualEngine === 'mobilenet')) {
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
        width: { ideal: 1280 },
        height: { ideal: 720 }
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

    if (this.isDragging) {
      const rect = this.getNormalizedDragRect();
      this.drawSelectionOverlay(ctx, rect, '#38bdf8', 'イラスト部分を四角く囲んでください...');
    } else {
      if (this.lastSelectionBox) {
        this.drawSelectionOverlay(ctx, this.lastSelectionBox, 'rgba(251, 191, 36, 0.55)', '選択範囲');
      }
      if (this.bestMatchedSubBox) {
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
    this.lastClickWasDemoCard = false;
    this.lastSelectionWasDrag = isDrag;
    let targetBox;
    if (!isDrag) {
      targetBox = this.determineClickCropBox(this.dragStart.x, this.dragStart.y);
    } else {
      targetBox = this.getNormalizedDragRect();
    }

    this.lastSelectionBox = targetBox;
    this.recognizeRegion(targetBox, isDrag);
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

    const size = parseInt(this.cropSizeSelect.value, 10) || 130;
    const x = Math.max(0, Math.min(this.mainCanvas.width - size, clickX - size / 2));
    const y = Math.max(0, Math.min(this.mainCanvas.height - size, clickY - size / 2));
    return { x, y, w: size, h: size };
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

    const addWin = (x, y, w, h) => {
      const cw = Math.max(24, Math.min(W, w));
      const ch = Math.max(24, Math.min(H, h));
      const cx = Math.max(0, Math.min(W - cw, x));
      const cy = Math.max(0, Math.min(H - ch, y));
      windows.push({ x: cx, y: cy, w: cw, h: ch });
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
      // ワンクリック時：標準(100%) + 小(76%) + 大(128%) の3スケールでカバー
      const scales = [1.0, 0.76, 1.28];
      for (const s of scales) {
        const side = baseSide * s;
        addWin(centerX - side / 2, centerY - side / 2, side, side);
      }
    }

    return windows;
  }

  /**
   * 選択領域をブラウザ内AIで非同期認識
   */
  async recognizeRegion(box, isDragBox = false) {
    if (this.isRecognizing || this.isSwitchingEngine) return;

    const indexedCards = this.cardsDB.filter((c) => c.feature);
    if (!this.aiEngine.isReady || indexedCards.length === 0) {
      this.matchStatusText.textContent = '⏳ AIモデルとカードDBの準備完了をお待ちください...';
      return;
    }

    this.isRecognizing = true;
    const startTime = performance.now();

    try {
      // 1) 選択枠オーバーレイを含まないクリーンな現在フレームをオフスクリーンにスナップショット
      this.lastSelectionBox = null;
      this.bestMatchedSubBox = null;
      this.drawFrame();

      const snapshotCanvas = document.createElement('canvas');
      snapshotCanvas.width = this.mainCanvas.width;
      snapshotCanvas.height = this.mainCanvas.height;
      snapshotCanvas.getContext('2d').drawImage(this.mainCanvas, 0, 0);

      // 選択枠を即座に戻して視覚フィードバックを維持
      this.lastSelectionBox = box;

      const enhanceLighting = this.enhanceLightingCheck ? this.enhanceLightingCheck.checked : true;
      const searchWindows = this.generateAISearchWindows(box, isDragBox);
      const orientations = [
        { deg: 0, label: '正位置 (0°)' },
        { deg: 180, label: '相手視点・逆さま (180°)' },
        { deg: 90, label: '横向き・守備 (90°)' },
        { deg: 270, label: '横向き・守備 (270°)' }
      ];

      const totalSteps = searchWindows.length * orientations.length;
      let doneSteps = 0;

      this.matchStatusText.textContent = `🤖 AI解析中... (0/${totalSteps})`;
      this.matchBadgeContainer.innerHTML = `<span class="match-badge processing">${this.aiEngine.getEngineLabel()} で照合中</span>`;

      const queries = [];
      for (const win of searchWindows) {
        for (const ori of orientations) {
          const vec = await this.aiEngine.extractEmbedding(snapshotCanvas, ori.deg, win, enhanceLighting);
          queries.push({
            win,
            deg: ori.deg,
            orientation: ori.label,
            vec
          });
          doneSteps++;
          this.matchStatusText.textContent = `🤖 AI解析中... (${doneSteps}/${totalSteps})`;
          await yieldToUI();
        }
      }

      // 全登録カード（3クロップ）とのコサイン類似度を算出
      const scored = indexedCards.map((card) => {
        let bestSim = -1;
        let bestOrientation = '正位置 (0°)';
        let bestDeg = 0;
        let bestWin = box;

        for (const q of queries) {
          const simFull = this.aiEngine.cosineSimilarity(q.vec, card.feature);
          const simCenter = card.centerFeature
            ? this.aiEngine.cosineSimilarity(q.vec, card.centerFeature)
            : -1;
          const simTight = card.tightFeature
            ? this.aiEngine.cosineSimilarity(q.vec, card.tightFeature)
            : -1;
          const sim = Math.max(simFull, simCenter, simTight);

          if (sim > bestSim) {
            bestSim = sim;
            bestOrientation = q.orientation;
            bestDeg = q.deg;
            bestWin = q.win;
          }
        }

        return {
          card,
          score: bestSim,
          orientation: bestOrientation,
          deg: bestDeg,
          matchedBox: bestWin
        };
      });

      scored.sort((a, b) => b.score - a.score);

      const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(2);
      const topMatch = scored[0];
      const secondMatch = scored[1];
      const margin = topMatch && secondMatch ? topMatch.score - secondMatch.score : 1;
      const top5 = scored.slice(0, 5);

      // 実際にAIが見た（補正・回転済みの）ベストマッチ領域をプレビューに表示
      const finalCropBox = topMatch ? topMatch.matchedBox : box;
      const previewCanvas = this.aiEngine.preprocessToCanvas(
        snapshotCanvas,
        topMatch ? topMatch.deg : 0,
        finalCropBox,
        enhanceLighting
      );
      this.cropPreviewCtx.clearRect(0, 0, 64, 64);
      this.cropPreviewCtx.drawImage(previewCanvas, 0, 0, 64, 64);

      this.lastSelectionBox = box;
      this.bestMatchedSubBox = finalCropBox;

      const meta = {
        elapsedSec,
        margin,
        engineLabel: this.aiEngine.getEngineLabel()
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
    const { card, score, orientation } = match;
    this.currentMatch = match;
    this.currentCardId = card.id;

    // AI埋め込みのコサイン類似度（0〜1）を直感的なパーセント表示に変換
    const clampedScore = Math.max(0, Math.min(1, score));
    const percent = (clampedScore * 100).toFixed(1);

    this.matchStatusText.textContent = `向き判定: ${orientation}`;

    const isHighConfidence = !meta || meta.margin >= 0.04 || clampedScore >= 0.72;
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
        const p = (Math.max(0, Math.min(1, item.score)) * 100).toFixed(1);
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
          safeCanvasElement: fallbackCanvas,
          feature: null,
          centerFeature: null,
          tightFeature: null,
          isFallback: true,
          aiIndexed: false
        };
        this.cardsDB.push(entry);
        await this.loadRealImageForCard(entry);
        if (this.aiEngine.isReady) {
          const emb = await this.aiEngine.getOrComputeCardEmbeddings(entry.id, entry.safeCanvasElement);
          entry.feature = emb.feature;
          entry.centerFeature = emb.centerFeature;
          entry.tightFeature = emb.tightFeature;
          entry.aiIndexed = true;
        }
        added++;
      }

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
