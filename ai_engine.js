/**
 * YGO Remote Lens - Browser AI Recognition Engine
 *
 * 2種類のブラウザ内AIモデルを切り替え可能（ビルド不要・GitHub Pages対応）:
 *   1) dinov2    : Transformers.js v3 + Xenova/dinov2-small (ViT-S/14, 量子化q8)
 *                  自己教師あり視覚基盤モデル。暗色系カードや光の反射・構図ズレに極めて強い。
 *   2) mobilenet : TensorFlow.js + MobileNet v2 (WebGL GPU推論)
 *                  超高速・軽量。DINOv2ロード失敗時の自動フォールバック先としても機能。
 *
 * 主な工夫:
 *   - 暗所・光反射耐性のための適応的コントラスト補正（パーセンタイルクリップ＋暗部ガンマ補正）
 *   - DINOv2では CLSトークン + パッチ平均トークンの結合L2正規化ベクトル（768次元）を採用し、
 *     全体構図と細部ディテールの両方を高精度に照合
 *   - IndexedDB によるカードDB特徴ベクトルの永続キャッシュ（2回目以降の起動を高速化）
 */

(function (global) {
  const INPUT_SIZE = 224;
  const CACHE_DB_NAME = 'ygo_remote_lens_ai_cache_v2';
  const CACHE_STORE_NAME = 'embeddings';
  const CACHE_VERSION = 'v2_enh1';

  /**
   * IndexedDB ベースの軽量ベクトルキャッシュ
   */
  class EmbeddingCache {
    constructor() {
      this.dbPromise = null;
    }

    open() {
      if (this.dbPromise) return this.dbPromise;
      this.dbPromise = new Promise((resolve) => {
        if (typeof indexedDB === 'undefined') {
          resolve(null);
          return;
        }
        try {
          const req = indexedDB.open(CACHE_DB_NAME, 1);
          req.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(CACHE_STORE_NAME)) {
              db.createObjectStore(CACHE_STORE_NAME, { keyPath: 'key' });
            }
          };
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => resolve(null);
        } catch (_) {
          resolve(null);
        }
      });
      return this.dbPromise;
    }

    makeKey(engineId, cardId, variant) {
      return `${engineId}:${CACHE_VERSION}:${cardId}:${variant}`;
    }

    async get(engineId, cardId, variant) {
      const db = await this.open();
      if (!db) return null;
      const key = this.makeKey(engineId, cardId, variant);
      return new Promise((resolve) => {
        try {
          const tx = db.transaction(CACHE_STORE_NAME, 'readonly');
          const req = tx.objectStore(CACHE_STORE_NAME).get(key);
          req.onsuccess = () => {
            const record = req.result;
            if (record && record.data) {
              resolve(new Float32Array(record.data));
            } else {
              resolve(null);
            }
          };
          req.onerror = () => resolve(null);
        } catch (_) {
          resolve(null);
        }
      });
    }

    async set(engineId, cardId, variant, float32Vec) {
      const db = await this.open();
      if (!db || !float32Vec) return;
      const key = this.makeKey(engineId, cardId, variant);
      return new Promise((resolve) => {
        try {
          const tx = db.transaction(CACHE_STORE_NAME, 'readwrite');
          // ArrayBuffer を安全にコピーして保存
          const copied = new Float32Array(float32Vec);
          tx.objectStore(CACHE_STORE_NAME).put({
            key,
            data: copied.buffer,
            updatedAt: Date.now()
          });
          tx.oncomplete = () => resolve();
          tx.onerror = () => resolve();
        } catch (_) {
          resolve();
        }
      });
    }

    async clearAll() {
      const db = await this.open();
      if (!db) return;
      return new Promise((resolve) => {
        try {
          const tx = db.transaction(CACHE_STORE_NAME, 'readwrite');
          tx.objectStore(CACHE_STORE_NAME).clear();
          tx.oncomplete = () => resolve();
          tx.onerror = () => resolve();
        } catch (_) {
          resolve();
        }
      });
    }
  }

  /**
   * 外部スクリプトを動的ロードするヘルパー
   */
  function loadScriptOnce(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) {
        if (existing.dataset.loaded === 'true') {
          resolve();
          return;
        }
        existing.addEventListener('load', () => resolve(), { once: true });
        existing.addEventListener('error', (e) => reject(e), { once: true });
        return;
      }
      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = () => {
        script.dataset.loaded = 'true';
        resolve();
      };
      script.onerror = (e) => reject(new Error(`スクリプト読み込み失敗: ${src}`));
      document.head.appendChild(script);
    });
  }

  /**
   * L2正規化（コサイン類似度を単純な内積で計算可能にする）
   */
  function l2Normalize(vec) {
    let normSq = 0;
    for (let i = 0; i < vec.length; i++) {
      normSq += vec[i] * vec[i];
    }
    const norm = Math.sqrt(normSq) || 1;
    const out = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      out[i] = vec[i] / norm;
    }
    return out;
  }

  /**
   * コサイン類似度（L2正規化済みベクトル同士の内積）
   */
  function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return -1;
    let dot = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
    }
    return dot;
  }

  /**
   * ブラウザ内AI認識エンジン本体
   */
  class YgoAIEngine {
    constructor() {
      this.activeEngine = 'dinov2'; // 'dinov2' | 'mobilenet' | 'legacy'
      this.backendInfo = '';
      this.isReady = false;
      this.cache = new EmbeddingCache();

      // Transformers.js (DINOv2)
      this.transformersModule = null;
      this.dinoPipe = null;

      // TensorFlow.js (MobileNet v2)
      this.mobilenetModel = null;

      // 使い回し用オフスクリーンCanvas
      this.workCanvas = document.createElement('canvas');
      this.workCanvas.width = INPUT_SIZE;
      this.workCanvas.height = INPUT_SIZE;
      this.workCtx = this.workCanvas.getContext('2d', { willReadFrequently: true });
    }

    getEngineLabel() {
      if (this.activeEngine === 'dinov2') {
        return `DINOv2-small (${this.backendInfo || 'AI'})`;
      }
      if (this.activeEngine === 'mobilenet') {
        return `MobileNet v2 (${this.backendInfo || 'WebGL'})`;
      }
      return '軽量フォールバック (16x16)';
    }

    /**
     * 指定エンジンを初期化（失敗時は自動フォールバック）
     */
    async init(engineId = 'dinov2', onProgress = () => {}) {
      this.isReady = false;

      if (engineId === 'dinov2') {
        try {
          await this.loadDinoV2(onProgress);
          this.activeEngine = 'dinov2';
          this.isReady = true;
          return this.activeEngine;
        } catch (err) {
          console.warn('[YgoAIEngine] DINOv2ロード失敗、MobileNet v2へフォールバック:', err);
          onProgress('DINOv2読込失敗 → MobileNet v2へ自動切替中...');
        }
      }

      if (engineId === 'mobilenet' || engineId === 'dinov2') {
        try {
          await this.loadMobileNet(onProgress);
          this.activeEngine = 'mobilenet';
          this.isReady = true;
          return this.activeEngine;
        } catch (err) {
          console.warn('[YgoAIEngine] MobileNet v2ロード失敗、フォールバックエンジンを使用:', err);
          onProgress('AIモデル読込失敗 → 軽量モードで起動');
        }
      }

      this.activeEngine = 'legacy';
      this.backendInfo = 'CPU';
      this.isReady = true;
      return this.activeEngine;
    }

    /**
     * Transformers.js v3 + Xenova/dinov2-small のロード
     */
    async loadDinoV2(onProgress) {
      if (this.dinoPipe) {
        onProgress('DINOv2モデル準備完了');
        return;
      }

      onProgress('Transformers.js 読み込み中...');
      if (!this.transformersModule) {
        this.transformersModule = await import(
          'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1'
        );
      }

      const { pipeline, env } = this.transformersModule;
      // GitHub Pages 等（COOP/COEP未設定）でも安全に動くよう設定
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
        env.backends.onnx.wasm.numThreads = 1;
      }

      // WebGPU が利用可能かチェック
      let device = 'wasm';
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) device = 'webgpu';
        } catch (_) {
          device = 'wasm';
        }
      }

      const progressCallback = (p) => {
        if (!p) return;
        if (p.status === 'progress' && typeof p.progress === 'number') {
          onProgress(`DINOv2モデル取得中 (${Math.round(p.progress)}%)...`);
        } else if (p.status === 'initiate') {
          onProgress('DINOv2モデル準備中...');
        } else if (p.status === 'ready') {
          onProgress('DINOv2モデル初期化完了');
        }
      };

      onProgress(`DINOv2-small 読み込み中 (${device.toUpperCase()})...`);
      try {
        this.dinoPipe = await pipeline('image-feature-extraction', 'Xenova/dinov2-small', {
          device,
          dtype: 'q8',
          progress_callback: progressCallback
        });
        this.backendInfo = device.toUpperCase();
      } catch (gpuErr) {
        if (device === 'webgpu') {
          console.warn('[YgoAIEngine] WebGPU初期化失敗、WASMで再試行:', gpuErr);
          onProgress('DINOv2-small (WASM) で再試行中...');
          this.dinoPipe = await pipeline('image-feature-extraction', 'Xenova/dinov2-small', {
            device: 'wasm',
            dtype: 'q8',
            progress_callback: progressCallback
          });
          this.backendInfo = 'WASM';
        } else {
          throw gpuErr;
        }
      }
    }

    /**
     * TensorFlow.js + MobileNet v2 のロード
     */
    async loadMobileNet(onProgress) {
      if (this.mobilenetModel) {
        onProgress('MobileNet v2 準備完了');
        return;
      }

      onProgress('TensorFlow.js 読み込み中...');
      await loadScriptOnce('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
      await loadScriptOnce(
        'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js'
      );

      if (!global.tf || !global.mobilenet) {
        throw new Error('TensorFlow.js / MobileNet のグローバル読み込みに失敗しました');
      }

      await global.tf.ready();
      this.backendInfo = (global.tf.getBackend() || 'webgl').toUpperCase();

      onProgress(`MobileNet v2 読み込み中 (${this.backendInfo})...`);
      this.mobilenetModel = await global.mobilenet.load({
        version: 2,
        alpha: 1.0
      });

      // ウォームアップ推論（初回シェーダーコンパイルの遅延を解消）
      onProgress('MobileNet v2 ウォームアップ中...');
      global.tf.tidy(() => {
        const dummy = global.tf.zeros([INPUT_SIZE, INPUT_SIZE, 3]);
        this.mobilenetModel.infer(dummy, true);
      });
    }

    /**
     * 切り抜き・回転・暗所＆光反射補正を行った 224x224 Canvas を生成
     *
     * 暗色系カード（オルフェゴール・ディヴェル等）対策:
     *   - 輝度ヒストグラムの上位2%・下位2%でパーセンタイルストレッチを実施
     *     → スリーブの強い光反射（上位2%）に引っ張られず、暗部ディテールを鮮明化
     *   - 全体が暗い場合は輝度比ベースの自動ガンマ補正で固有の色味を保ったまま暗部を持ち上げ
     */
    preprocessToCanvas(source, rotateDeg = 0, cropRect = null, enhanceLighting = true, targetCanvas = null) {
      const canvas = targetCanvas || this.workCanvas;
      canvas.width = INPUT_SIZE;
      canvas.height = INPUT_SIZE;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.clearRect(0, 0, INPUT_SIZE, INPUT_SIZE);

      ctx.save();
      if (rotateDeg !== 0) {
        ctx.translate(INPUT_SIZE / 2, INPUT_SIZE / 2);
        ctx.rotate((rotateDeg * Math.PI) / 180);
        ctx.translate(-INPUT_SIZE / 2, -INPUT_SIZE / 2);
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
          INPUT_SIZE,
          INPUT_SIZE
        );
      } else {
        ctx.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
      }
      ctx.restore();

      if (enhanceLighting) {
        this.applyAdaptiveLightingEnhancement(ctx, INPUT_SIZE, INPUT_SIZE);
      }

      return canvas;
    }

    /**
     * 暗所・反射耐性のための適応的コントラスト＆ガンマ補正
     * RGB比率を保持しながら輝度レンジを最適化するため、赤茶色や青色などの固有色味を壊さない
     */
    applyAdaptiveLightingEnhancement(ctx, width, height) {
      const imgData = ctx.getImageData(0, 0, width, height);
      const data = imgData.data;
      const pixelCount = width * height;

      // 輝度ヒストグラム構築
      const hist = new Uint32Array(256);
      let lumaSum = 0;
      for (let i = 0; i < data.length; i += 4) {
        const l = Math.min(
          255,
          Math.max(0, Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]))
        );
        hist[l]++;
        lumaSum += l;
      }

      const meanLuma = lumaSum / pixelCount;
      const lowTarget = Math.floor(pixelCount * 0.02);
      const highTarget = Math.floor(pixelCount * 0.98);

      let acc = 0;
      let pLow = 0;
      let pHigh = 255;
      for (let v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc >= lowTarget) {
          pLow = v;
          break;
        }
      }
      acc = 0;
      for (let v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc >= highTarget) {
          pHigh = v;
          break;
        }
      }

      // ノイズの過剰増幅を防ぐため、最小スパンを確保
      const span = Math.max(55, pHigh - pLow);

      // 暗いカード（例：ディヴェル等の暗所撮影）ほど暗部を穏やかに持ち上げるガンマ値を選択
      let gamma = 1.0;
      if (meanLuma < 85) {
        gamma = 0.72;
      } else if (meanLuma < 115) {
        gamma = 0.84;
      } else if (meanLuma > 185) {
        // 全体的に白っぽく光が反射している場合はわずかに引き締める
        gamma = 1.15;
      }

      // ルックアップテーブルで高速変換
      const lut = new Uint8ClampedArray(256);
      for (let v = 0; v < 256; v++) {
        let norm = (v - pLow) / span;
        if (norm < 0) norm = 0;
        if (norm > 1) norm = 1;
        const corrected = Math.pow(norm, gamma);
        lut[v] = Math.round(corrected * 255);
      }

      // 補正を元画像と75:25でブレンド（極端な破綻を防ぎつつ暗部・反射を強力に改善）
      const alpha = 0.75;
      const invAlpha = 1 - alpha;
      for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.round(lut[data[i]] * alpha + data[i] * invAlpha);
        data[i + 1] = Math.round(lut[data[i + 1]] * alpha + data[i + 1] * invAlpha);
        data[i + 2] = Math.round(lut[data[i + 2]] * alpha + data[i + 2] * invAlpha);
      }

      ctx.putImageData(imgData, 0, 0);
    }

    /**
     * 画像ソースから L2正規化済みのAI特徴ベクトルを抽出
     */
    async extractEmbedding(source, rotateDeg = 0, cropRect = null, enhanceLighting = true) {
      const canvas = this.preprocessToCanvas(source, rotateDeg, cropRect, enhanceLighting);

      if (this.activeEngine === 'dinov2' && this.dinoPipe) {
        return await this.extractDinoV2Embedding(canvas);
      }
      if (this.activeEngine === 'mobilenet' && this.mobilenetModel) {
        return this.extractMobileNetEmbedding(canvas);
      }
      return this.extractLegacyFeature(canvas);
    }

    /**
     * DINOv2-small 特徴抽出
     * CLSトークン（全体構図・意味）とパッチ平均トークン（細部テクスチャ）を結合し、
     * 暗色系カードや部分的な光反射にも強い768次元ベクトルを生成
     */
    async extractDinoV2Embedding(canvas) {
      const { RawImage } = this.transformersModule;
      const input =
        RawImage && typeof RawImage.fromCanvas === 'function'
          ? RawImage.fromCanvas(canvas)
          : canvas.toDataURL('image/jpeg', 0.92);
      const output = await this.dinoPipe(input);

      const data = output.data;
      const dims = output.dims; // 通常 [1, 257, 384] または [1, 384]

      if (dims.length === 3) {
        const numTokens = dims[1];
        const hiddenDim = dims[2];

        // 1) CLSトークン (index 0)
        const clsVec = new Float32Array(hiddenDim);
        for (let d = 0; d < hiddenDim; d++) {
          clsVec[d] = data[d];
        }
        const clsNorm = l2Normalize(clsVec);

        // 2) パッチトークン平均 (index 1 .. numTokens-1)
        const patchMean = new Float32Array(hiddenDim);
        const patchCount = Math.max(1, numTokens - 1);
        for (let t = 1; t < numTokens; t++) {
          const offset = t * hiddenDim;
          for (let d = 0; d < hiddenDim; d++) {
            patchMean[d] += data[offset + d];
          }
        }
        for (let d = 0; d < hiddenDim; d++) {
          patchMean[d] /= patchCount;
        }
        const patchNorm = l2Normalize(patchMean);

        // CLS (重み 0.65) + パッチ平均 (重み 0.35) を結合してL2正規化
        const combined = new Float32Array(hiddenDim * 2);
        for (let d = 0; d < hiddenDim; d++) {
          combined[d] = clsNorm[d] * 0.65;
          combined[hiddenDim + d] = patchNorm[d] * 0.35;
        }
        return l2Normalize(combined);
      }

      // 2次元出力 [1, D] の場合
      return l2Normalize(new Float32Array(data));
    }

    /**
     * MobileNet v2 特徴抽出（1280次元・WebGL高速推論）
     */
    extractMobileNetEmbedding(canvas) {
      const rawArray = global.tf.tidy(() => {
        const activation = this.mobilenetModel.infer(canvas, true);
        return activation.dataSync();
      });
      return l2Normalize(new Float32Array(rawArray));
    }

    /**
     * 万が一AIモデルが全て使えない場合のフォールバック（16x16特徴量）
     */
    extractLegacyFeature(canvas) {
      const size = 16;
      const small = document.createElement('canvas');
      small.width = size;
      small.height = size;
      const ctx = small.getContext('2d');
      ctx.drawImage(canvas, 0, 0, size, size);
      const imgData = ctx.getImageData(0, 0, size, size).data;
      const count = size * size;
      const vec = new Float32Array(count * 3);
      let mean = 0;
      for (let i = 0; i < count; i++) {
        const r = imgData[i * 4];
        const g = imgData[i * 4 + 1];
        const b = imgData[i * 4 + 2];
        vec[i * 3] = r;
        vec[i * 3 + 1] = g;
        vec[i * 3 + 2] = b;
        mean += r + g + b;
      }
      mean /= count * 3;
      for (let i = 0; i < vec.length; i++) {
        vec[i] -= mean;
      }
      return l2Normalize(vec);
    }

    /**
     * カード1枚分のマルチクロップ特徴ベクトルを取得（IndexedDBキャッシュ優先）
     * - full   : イラスト全体 (100%)
     * - center : 少し内枠 (84%) — 外側のカード枠が映り込んだ／逆に内側だけ囲んだズレ対策
     * - tight  : 中心コア (70%) — 大きくトリミングされた／端が光で飛んだ場合の対策
     */
    async getOrComputeCardEmbeddings(cardId, safeCanvas) {
      const engineId = this.activeEngine;

      if (engineId !== 'legacy') {
        const [cFull, cCenter, cTight] = await Promise.all([
          this.cache.get(engineId, cardId, 'full'),
          this.cache.get(engineId, cardId, 'center'),
          this.cache.get(engineId, cardId, 'tight')
        ]);
        if (cFull && cCenter && cTight) {
          return {
            feature: cFull,
            centerFeature: cCenter,
            tightFeature: cTight,
            fromCache: true
          };
        }
      }

      const W = safeCanvas.width;
      const H = safeCanvas.height;

      const feature = await this.extractEmbedding(safeCanvas, 0, null, true);
      const centerFeature = await this.extractEmbedding(
        safeCanvas,
        0,
        { x: W * 0.08, y: H * 0.08, w: W * 0.84, h: H * 0.84 },
        true
      );
      const tightFeature = await this.extractEmbedding(
        safeCanvas,
        0,
        { x: W * 0.15, y: H * 0.15, w: W * 0.70, h: H * 0.70 },
        true
      );

      if (engineId !== 'legacy') {
        await Promise.all([
          this.cache.set(engineId, cardId, 'full', feature),
          this.cache.set(engineId, cardId, 'center', centerFeature),
          this.cache.set(engineId, cardId, 'tight', tightFeature)
        ]);
      }

      return { feature, centerFeature, tightFeature, fromCache: false };
    }

    async clearCache() {
      await this.cache.clearAll();
    }

    cosineSimilarity(vecA, vecB) {
      return cosineSimilarity(vecA, vecB);
    }
  }

  global.YgoAIEngine = YgoAIEngine;
})(window);
