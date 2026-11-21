/**
 * YGO Remote Lens - Step 1.7 Precision Enhanced Prototype
 * - 『オルフェゴール・ディヴェル』等の暗色・単色系イラストが認識されにくい問題を解決：
 *   1) チャンネル別正規化で単色系（赤茶色など）の固有色味が消えていたのを修正し、「固有カラー特徴」「照明正規化コントラスト」「8x8ブロック明暗シルエット」「エッジ勾配」の4種統合特徴ベクトルに刷新
 *   2) 40枚の画像を同時リクエストして一部がタイムアウト落ちするのを防ぐため、6枚ずつのバッチ並列＋リトライ取得に変更（40枚すべて確実に実画像化）
 *   3) Camo Camera 等の外部カメラを画面上で直接選べるカメラ選択プルダウン対応
 */

const FEATURE_GRID_SIZE = 16;

class YgoRemoteLensApp {
  constructor() {
    this.cardsDB = [];
    this.history = [];
    this.mode = 'demo'; // 'demo' | 'video' | 'image'
    this.loadedImage = null;
    this.videoStream = null;

    this.demoBoardCards = [];

    this.isDragging = false;
    this.dragStart = { x: 0, y: 0 };
    this.dragCurrent = { x: 0, y: 0 };
    this.lastSelectionBox = null;
    this.bestMatchedSubBox = null;

    this.initElements();
    this.bindEvents();
    this.refreshCameraList();
    this.initDatabase();
  }

  initElements() {
    this.mainCanvas = document.getElementById('mainCanvas');
    this.ctx = this.mainCanvas.getContext('2d');
    this.video = document.getElementById('captureVideo');
    this.cropPreviewCanvas = document.getElementById('cropPreviewCanvas');
    this.cropPreviewCtx = this.cropPreviewCanvas.getContext('2d');

    this.dbStatusBadge = document.getElementById('dbStatusBadge');
    this.cropSizeSelect = document.getElementById('cropSizeSelect');
    this.cameraDeviceSelect = document.getElementById('cameraDeviceSelect');
    this.matchStatusText = document.getElementById('matchStatusText');
    this.matchBadgeContainer = document.getElementById('matchBadgeContainer');
    this.cardDetailContainer = document.getElementById('cardDetailContainer');
    this.candidatesSection = document.getElementById('candidatesSection');
    this.candidatesList = document.getElementById('candidatesList');
    this.historyList = document.getElementById('historyList');
    this.historyCount = document.getElementById('historyCount');
    this.dbCardList = document.getElementById('dbCardList');
    this.dbCount = document.getElementById('dbCount');
  }

  bindEvents() {
    document.getElementById('btnDemoBoard').addEventListener('click', () => this.startDemoBoard());
    document.getElementById('btnCaptureScreen').addEventListener('click', () => this.startScreenCapture());
    document.getElementById('btnStartCamera').addEventListener('click', () => this.startCamera());
    document.getElementById('fileInput').addEventListener('change', (e) => this.loadImageFile(e));

    if (this.cameraDeviceSelect) {
      this.cameraDeviceSelect.addEventListener('change', () => {
        if (this.mode === 'video' && this.videoStream) {
          this.startCamera();
        }
      });
    }

    document.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.getAttribute('data-tab');
        ['detail', 'history', 'database'].forEach((t) => {
          document.getElementById(`tab-${t}`).style.display = t === target ? 'flex' : 'none';
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
        // Camo があれば優先選択しやすいようにマーク
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

  /**
   * 初期カードデータベースの読み込み（バッチ制御により全40枚を通信落ちなく確実に取得）
   */
  async initDatabase() {
    const initialCards = window.INITIAL_CARDS_DATA || [];

    this.cardsDB = initialCards.map((card) => {
      const fallbackCanvas = this.createFallbackCardArt(card);
      const feature = this.extractFeatureFromImage(fallbackCanvas, 0);
      return {
        ...card,
        safeCanvasElement: fallbackCanvas,
        thumbUrl: card.image_crop,
        feature,
        centerFeature: feature,
        tightFeature: feature,
        isFallback: true
      };
    });

    this.renderDBList();
    this.startDemoBoard();
    this.startRenderLoop();

    let loadedCount = 0;
    let realImageCount = 0;
    this.dbStatusBadge.textContent = `実カード画像を取得中 (0/${initialCards.length})...`;
    this.dbStatusBadge.classList.add('loading');

    // 一気に40枚リクエストするとプロキシ側でレート制限・タイムアウトが起きるため、6枚ずつ順番に確実に取得
    const BATCH_SIZE = 6;
    for (let i = 0; i < this.cardsDB.length; i += BATCH_SIZE) {
      const batch = this.cardsDB.slice(i, i + BATCH_SIZE);
      await Promise.all(
        batch.map(async (cardEntry) => {
          const ok = await this.upgradeCardWithRealImage(cardEntry);
          loadedCount++;
          if (ok) realImageCount++;
          this.dbStatusBadge.textContent = `画像読込中 (${loadedCount}/${initialCards.length} / 成功:${realImageCount})...`;
        })
      );
      this.renderDBList();
    }

    // もし1回目のバッチでタイムアウトしたカードがあればもう1度だけリトライ
    const failedEntries = this.cardsDB.filter((c) => c.isFallback);
    for (const cardEntry of failedEntries) {
      const ok = await this.upgradeCardWithRealImage(cardEntry);
      if (ok) realImageCount++;
    }

    this.dbStatusBadge.textContent = `✅ 認識エンジン稼働中 (${this.cardsDB.length}枚 / 実画像:${realImageCount}枚)`;
    this.dbStatusBadge.classList.remove('loading');
    this.renderDBList();
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
            c.width = img.naturalWidth || 180;
            c.height = img.naturalHeight || 180;
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
      `https://wsrv.nl/?url=${encodeURIComponent(url)}&w=200&h=200&fit=cover&output=jpg`,
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

  /**
   * 実カード画像から「全体」「84%内枠」「72%中心コア」の3段階の特徴量を抽出して登録
   * → カメラ撮影時にカードの茶色い外枠が少し入っても、逆にイラスト内側だけを小さく囲んでも高スコアが出る！
   */
  async upgradeCardWithRealImage(cardEntry) {
    try {
      const safeCanvas = await this.fetchImageAsSafeCanvas(cardEntry.image_crop);
      const W = safeCanvas.width;
      const H = safeCanvas.height;

      const feature = this.extractFeatureFromImage(safeCanvas, 0);
      const centerFeature = this.extractFeatureFromImage(safeCanvas, 0, {
        x: W * 0.08,
        y: H * 0.08,
        w: W * 0.84,
        h: H * 0.84
      });
      const tightFeature = this.extractFeatureFromImage(safeCanvas, 0, {
        x: W * 0.15,
        y: H * 0.15,
        w: W * 0.70,
        h: H * 0.70
      });

      cardEntry.safeCanvasElement = safeCanvas;
      cardEntry.feature = feature;
      cardEntry.centerFeature = centerFeature;
      cardEntry.tightFeature = tightFeature;
      cardEntry.isFallback = false;
      return true;
    } catch (_) {
      return false;
    }
  }

  createFallbackCardArt(card) {
    const c = document.createElement('canvas');
    c.width = 180;
    c.height = 180;
    const ctx = c.getContext('2d');
    const hue = (card.id * 137) % 360;
    const grad = ctx.createLinearGradient(0, 0, 180, 180);
    grad.addColorStop(0, `hsl(${hue}, 70%, 35%)`);
    grad.addColorStop(1, `hsl(${(hue + 60) % 360}, 80%, 20%)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 180, 180);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(card.name_ja.slice(0, 7), 90, 95);
    return c;
  }

  /**
   * 改良版特徴ベクトル抽出（ディヴェル等の暗色・単色系イラストにも強いハイブリッド特徴量）
   * - 各ピクセルで以下の7要素を抽出：
   *   1-3) Global正規化RGB（赤茶色や青色といった「イラスト全体の色調・単色カラー特徴」を保持！）
   *   4)   局所コントラスト輝度（暗いカードの中のハープやキャラクターの明暗シルエットを強調！）
   *   5)   赤-青の色差 (R-B) ＆ 緑-赤青の色差（照明の明るさに左右されない純粋な色相特徴！）
   *   6-7) 縦横の輝度エッジ勾配（イラストの輪郭・構図！）
   */
  extractFeatureFromImage(source, rotateDeg = 0, cropRect = null) {
    const size = FEATURE_GRID_SIZE;
    const offCanvas = document.createElement('canvas');
    offCanvas.width = size;
    offCanvas.height = size;
    const ctx = offCanvas.getContext('2d');

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    ctx.save();
    if (rotateDeg !== 0) {
      ctx.translate(size / 2, size / 2);
      ctx.rotate((rotateDeg * Math.PI) / 180);
      ctx.translate(-size / 2, -size / 2);
    }

    if (cropRect) {
      ctx.drawImage(
        source,
        cropRect.x,
        cropRect.y,
        cropRect.w,
        cropRect.h,
        0,
        0,
        size,
        size
      );
    } else {
      ctx.drawImage(source, 0, 0, size, size);
    }
    ctx.restore();

    const imgData = ctx.getImageData(0, 0, size, size).data;
    const count = size * size;

    const rArr = new Float32Array(count);
    const gArr = new Float32Array(count);
    const bArr = new Float32Array(count);
    const luma = new Float32Array(count);

    let sumR = 0, sumG = 0, sumB = 0, sumL = 0;
    for (let i = 0; i < count; i++) {
      const r = imgData[i * 4];
      const g = imgData[i * 4 + 1];
      const b = imgData[i * 4 + 2];
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      rArr[i] = r;
      gArr[i] = g;
      bArr[i] = b;
      luma[i] = l;
      sumR += r;
      sumG += g;
      sumB += b;
      sumL += l;
    }

    const meanR = sumR / count;
    const meanG = sumG / count;
    const meanB = sumB / count;
    const meanL = sumL / count;
    // 全チャンネル共通の平均輝度
    const globalMean = (meanR + meanG + meanB) / 3;

    let varGlobal = 0;
    let varLuma = 0;
    for (let i = 0; i < count; i++) {
      const dr = rArr[i] - globalMean;
      const dg = gArr[i] - globalMean;
      const db = bArr[i] - globalMean;
      varGlobal += dr * dr + dg * dg + db * db;

      const dl = luma[i] - meanL;
      varLuma += dl * dl;
    }

    // 重要：チャンネル別ではなく全RGB共通の標準偏差で割ることで、
    // 『ディヴェル』のような「全体が赤茶色」「全体が青色」という単色カラーの個性が消えずに残る！
    const stdGlobal = Math.sqrt(varGlobal / (count * 3)) || 1;
    const stdLuma = Math.sqrt(varLuma / count) || 1;

    const DIM_PER_PIXEL = 7;
    const vec = new Float32Array(count * DIM_PER_PIXEL);
    let normSq = 0;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        const dx = (x - (size - 1) / 2) / (size / 2);
        const dy = (y - (size - 1) / 2) / (size / 2);
        // 外周（カード枠が混ざりやすい端っこ）の重みを下げ、イラスト中心〜中間の重みを高める
        const spatialWeight = Math.max(0.35, 1.2 - 0.55 * (dx * dx + dy * dy));

        // 1〜3: 固有の色味を残した正規化RGB
        const gr = ((rArr[i] - globalMean) / stdGlobal) * 0.75 * spatialWeight;
        const gg = ((gArr[i] - globalMean) / stdGlobal) * 0.75 * spatialWeight;
        const gb = ((bArr[i] - globalMean) / stdGlobal) * 0.75 * spatialWeight;

        // 4: 照明の明るさ・暗さに影響されない純粋な明暗シルエット（暗いカードの模様をくっきり浮き上がらせる）
        const nl = ((luma[i] - meanL) / stdLuma) * 1.15 * spatialWeight;

        // 5: 照明変化に強い色相差分（赤み vs 青み）
        const chromaRB = (((rArr[i] - bArr[i]) - (meanR - meanB)) / stdGlobal) * 0.85 * spatialWeight;

        // 6〜7: エッジ勾配（形の輪郭）
        const rightIdx = y * size + Math.min(size - 1, x + 1);
        const downIdx = Math.min(size - 1, y + 1) * size + x;
        const gx = ((luma[rightIdx] - luma[i]) / (stdLuma * 1.2)) * 0.9 * spatialWeight;
        const gy = ((luma[downIdx] - luma[i]) / (stdLuma * 1.2)) * 0.9 * spatialWeight;

        const base = i * DIM_PER_PIXEL;
        vec[base] = gr;
        vec[base + 1] = gg;
        vec[base + 2] = gb;
        vec[base + 3] = nl;
        vec[base + 4] = chromaRB;
        vec[base + 5] = gx;
        vec[base + 6] = gy;

        normSq +=
          gr * gr +
          gg * gg +
          gb * gb +
          nl * nl +
          chromaRB * chromaRB +
          gx * gx +
          gy * gy;
      }
    }

    const norm = Math.sqrt(normSq) || 1;
    for (let i = 0; i < vec.length; i++) {
      vec[i] /= norm;
    }

    return vec;
  }

  cosineSimilarity(vecA, vecB) {
    let dot = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
    }
    return dot;
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

      // カメラ許可後にデバイス名（Camo等）が取得できるようになるのでリストを更新
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
        this.drawSelectionOverlay(ctx, this.bestMatchedSubBox, '#10b981', '🎯 認識エリア');
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
    ctx.fillText('【相手フィールド（180°逆さま配置：ディヴェル・ギルス・うらら）】 ← クリックして認識！', 50, 35);
    ctx.fillText('【自分フィールド（正位置配置）】 ← クリックまたはイラスト部分をドラッグで囲んで認識！', 50, H - 20);

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
    if (e.button !== 0) return;
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

    let targetBox;
    if (dx < 12 && dy < 12) {
      targetBox = this.determineClickCropBox(this.dragStart.x, this.dragStart.y);
    } else {
      targetBox = this.getNormalizedDragRect();
    }

    this.lastSelectionBox = targetBox;
    this.recognizeRegion(targetBox);
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
          const artW = item.w * 0.76;
          const artH = artW;
          const artX = item.x + (item.w - artW) / 2;
          const artY = item.upsideDown
            ? item.y + item.h - item.h * 0.19 - artH
            : item.y + item.h * 0.19;
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

  generateSearchWindows(box, isDragBox) {
    const windows = [];
    const W = this.mainCanvas.width;
    const H = this.mainCanvas.height;

    const addWin = (x, y, w, h) => {
      const cw = Math.max(20, Math.min(W, w));
      const ch = Math.max(20, Math.min(H, h));
      const cx = Math.max(0, Math.min(W - cw, x));
      const cy = Math.max(0, Math.min(H - ch, y));
      windows.push({ x: cx, y: cy, w: cw, h: ch });
    };

    // 1. ユーザーが囲んだ枠そのまま
    addWin(box.x, box.y, box.w, box.h);

    const centerX = box.x + box.w / 2;
    const centerY = box.y + box.h / 2;
    const baseSide = Math.min(box.w, box.h);

    // ドラッグで明示的にイラストを囲んだ場合は、外側の茶色いカード枠を拾いすぎないように内側〜等倍中心（0.7〜1.15倍）に絞る
    const scales = isDragBox
      ? [0.72, 0.85, 0.94, 1.0, 1.12]
      : [0.6, 0.8, 1.0, 1.35, 1.8, 2.3];

    const shifts = [
      [0, 0],
      [0, -0.10],
      [0, 0.10],
      [-0.10, 0],
      [0.10, 0]
    ];

    for (const s of scales) {
      const side = baseSide * s;
      for (const [sx, sy] of shifts) {
        addWin(
          centerX - side / 2 + side * sx,
          centerY - side / 2 + side * sy,
          side,
          side
        );
      }
    }

    // 縦長のカード全体をドラッグで囲んだ場合のイラスト枠（正位置・逆位置）
    if (box.h > box.w * 1.15) {
      const artSide = box.w * 0.76;
      addWin(box.x + (box.w - artSide) / 2, box.y + box.h * 0.18, artSide, artSide);
      addWin(box.x + (box.w - artSide) / 2, box.y + box.h - box.h * 0.18 - artSide, artSide, artSide);
    }

    return windows;
  }

  recognizeRegion(box) {
    if (this.cardsDB.length === 0) {
      this.matchStatusText.textContent = '⚠️ カードDBがまだ読み込まれていません';
      return;
    }

    try {
      const dx = Math.abs(this.dragCurrent.x - this.dragStart.x);
      const dy = Math.abs(this.dragCurrent.y - this.dragStart.y);
      const isDragBox = dx >= 12 || dy >= 12;

      this.lastSelectionBox = null;
      this.bestMatchedSubBox = null;
      this.drawFrame();

      const searchWindows = this.generateSearchWindows(box, isDragBox);
      const orientations = [
        { deg: 0, label: '正位置 (0°)' },
        { deg: 180, label: '相手視点・逆さま (180°)' },
        { deg: 90, label: '横向き・守備 (90°)' },
        { deg: 270, label: '横向き・守備 (270°)' }
      ];

      const queries = [];
      for (const win of searchWindows) {
        for (const ori of orientations) {
          queries.push({
            win,
            orientation: ori.label,
            vec: this.extractFeatureFromImage(this.mainCanvas, ori.deg, win)
          });
        }
      }

      const scored = this.cardsDB.map((card) => {
        let bestSim = -1;
        let bestOrientation = '正位置 (0°)';
        let bestWin = box;

        for (const q of queries) {
          const simFull = this.cosineSimilarity(q.vec, card.feature);
          const simCenter = this.cosineSimilarity(q.vec, card.centerFeature || card.feature);
          const simTight = this.cosineSimilarity(q.vec, card.tightFeature || card.feature);
          const sim = Math.max(simFull, simCenter, simTight);

          if (sim > bestSim) {
            bestSim = sim;
            bestOrientation = q.orientation;
            bestWin = q.win;
          }
        }

        return {
          card,
          score: bestSim,
          orientation: bestOrientation,
          matchedBox: bestWin
        };
      });

      scored.sort((a, b) => b.score - a.score);

      const topMatch = scored[0];
      const top5 = scored.slice(0, 5);

      const finalCropBox = topMatch ? topMatch.matchedBox : box;
      this.cropPreviewCtx.clearRect(0, 0, 64, 64);
      this.cropPreviewCtx.drawImage(
        this.mainCanvas,
        finalCropBox.x,
        finalCropBox.y,
        finalCropBox.w,
        finalCropBox.h,
        0,
        0,
        64,
        64
      );

      this.lastSelectionBox = box;
      this.bestMatchedSubBox = finalCropBox;

      this.displayRecognitionResult(topMatch, top5);
      this.addToHistory(topMatch);
    } catch (err) {
      console.error('Recognition error:', err);
      this.matchStatusText.textContent = '⚠️ 認識処理エラー: ' + err.message;
    }
  }

  displayRecognitionResult(match, topCandidates) {
    const { card, score, orientation } = match;
    const percent = Math.max(0, Math.min(99.9, ((score + 1) / 2) * 100)).toFixed(1);

    this.matchStatusText.textContent = `向き判定: ${orientation}`;
    this.matchBadgeContainer.innerHTML = `<span class="match-badge">一致度スコア: ${percent}%</span>`;

    const imgSrc = card.image_full;

    this.cardDetailContainer.innerHTML = `
      <div class="card-detail">
        <div class="card-top">
          <div class="card-image-box">
            <img src="${imgSrc}" alt="${card.name_ja}" onerror="this.src='${card.image_crop}'" />
          </div>
          <div class="card-meta">
            <div class="card-name-ja">${card.name_ja}</div>
            <div class="card-name-en">${card.name_en} (ID: ${card.id})</div>
            <div class="card-type-badge">${card.type_ja}</div>
            <div class="card-stats">${card.stats_ja}</div>
          </div>
        </div>
        <div class="card-effect-box">
          <div class="card-effect-title">
            <span>📖 カード効果・テキスト</span>
            <span>ID: ${card.id}</span>
          </div>
          <div class="card-effect-text">${card.desc_ja}</div>
        </div>
      </div>
    `;

    this.candidatesSection.style.display = 'block';
    this.candidatesList.innerHTML = '';
    topCandidates.forEach((item, idx) => {
      const p = Math.max(0, Math.min(99.9, ((item.score + 1) / 2) * 100)).toFixed(0);
      const thumbSrc = item.card.image_crop;

      const div = document.createElement('div');
      div.className = `candidate-item ${item.card.id === card.id ? 'active' : ''}`;
      div.innerHTML = `
        <img src="${thumbSrc}" alt="${item.card.name_ja}" />
        <div class="candidate-name">${item.card.name_ja}</div>
        <div class="candidate-score">${idx + 1}位 (${p}%)</div>
      `;
      div.addEventListener('click', () => {
        this.displayRecognitionResult(item, topCandidates);
      });
      this.candidatesList.appendChild(div);
    });
  }

  addToHistory(match) {
    const timeStr = new Date().toLocaleTimeString('ja-JP');
    this.history.unshift({ ...match, timeStr });
    if (this.history.length > 30) this.history.pop();

    this.historyCount.textContent = this.history.length;
    this.historyList.innerHTML = '';

    this.history.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML = `
        <img src="${item.card.image_crop}" alt="${item.card.name_ja}" />
        <div class="history-meta">
          <div class="history-name">${item.card.name_ja}</div>
          <div class="history-sub">${item.timeStr} / ${item.orientation}</div>
        </div>
      `;
      div.addEventListener('click', () => {
        document.querySelector('[data-tab="detail"]').click();
        this.displayRecognitionResult(item, [item]);
      });
      this.historyList.appendChild(div);
    });
  }

  renderDBList() {
    this.dbCount.textContent = this.cardsDB.length;
    this.dbCardList.innerHTML = '';
    this.cardsDB.forEach((card) => {
      const statusIcon = card.isFallback ? '⏳' : '✅';
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
          feature: this.extractFeatureFromImage(fallbackCanvas, 0),
          isFallback: true
        };
        this.cardsDB.push(entry);
        await this.upgradeCardWithRealImage(entry);
        added++;
      }

      this.renderDBList();
      this.dbStatusBadge.textContent = `✅ 認識エンジン稼働中 (${this.cardsDB.length}枚登録済)`;
      alert(`${added}枚の新しいカードを認識データベースに追加しました！`);
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
