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
  const CACHE_VERSION = 'v3_luma1';

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
      this.activeEngine = 'hybrid'; // 'hybrid' | 'dinov2' | 'mobilenet' | 'legacy'
      this.backendInfo = '';
      this.dinoBackendInfo = '';
      this.isReady = false;
      this.cache = new EmbeddingCache();

      // Transformers.js (DINOv2)
      this.transformersModule = null;
      this.dinoPipe = null;
      this.dinoLoadPromise = null;
      this._dinoQueue = Promise.resolve();

      // TensorFlow.js (MobileNet v2)
      this.mobilenetModel = null;

      // 使い回し用オフスクリーンCanvas
      this.workCanvas = document.createElement('canvas');
      this.workCanvas.width = INPUT_SIZE;
      this.workCanvas.height = INPUT_SIZE;
      this.workCtx = this.workCanvas.getContext('2d', { willReadFrequently: true });

      // 空間カラー特徴用の10x10オフスクリーンCanvas
      this.colorCanvas = document.createElement('canvas');
      this.colorCanvas.width = 10;
      this.colorCanvas.height = 10;
      this.colorCtx = this.colorCanvas.getContext('2d', { willReadFrequently: true });
    }

    getEngineLabel() {
      if (this.activeEngine === 'hybrid') {
        return this.dinoPipe
          ? `Hybrid (MobileNet+DINOv2)`
          : `Hybrid (MobileNet+色特徴)`;
      }
      if (this.activeEngine === 'dinov2') {
        return `DINOv2-small (${this.dinoBackendInfo || this.backendInfo || 'AI'})`;
      }
      if (this.activeEngine === 'mobilenet') {
        return `MobileNet v2 (${this.backendInfo || 'WebGL'})`;
      }
      return '軽量フォールバック (16x16)';
    }

    isDinoReady() {
      return Boolean(this.dinoPipe);
    }

    async ensureDinoLoaded(onProgress = () => {}) {
      if (this.dinoPipe) return true;
      if (!this.dinoLoadPromise) {
        this.dinoLoadPromise = this.loadDinoV2(onProgress).catch((err) => {
          console.warn('[YgoAIEngine] DINOv2バックグラウンド読込失敗:', err);
          this.dinoLoadPromise = null;
          return false;
        });
      }
      await this.dinoLoadPromise;
      return Boolean(this.dinoPipe);
    }

    /**
     * 指定エンジンを初期化（失敗時は自動フォールバック）
     */
    async init(engineId = 'hybrid', onProgress = () => {}) {
      this.isReady = false;

      if (engineId === 'hybrid') {
        try {
          await this.loadMobileNet(onProgress);
          this.activeEngine = 'hybrid';
          this.isReady = true;
          // DINOv2はバックグラウンドで非同期プリロード開始（起動をブロックしない）
          this.ensureDinoLoaded();
          return this.activeEngine;
        } catch (err) {
          console.warn('[YgoAIEngine] Hybrid用MobileNetロード失敗、DINOv2単体へ切替:', err);
        }
      }

      if (engineId === 'dinov2' || engineId === 'hybrid') {
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

      if (engineId === 'mobilenet' || engineId === 'dinov2' || engineId === 'hybrid') {
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
        this.dinoBackendInfo = device.toUpperCase();
        if (this.activeEngine === 'dinov2') this.backendInfo = this.dinoBackendInfo;
      } catch (gpuErr) {
        if (device === 'webgpu') {
          console.warn('[YgoAIEngine] WebGPU初期化失敗、WASMで再試行:', gpuErr);
          onProgress('DINOv2-small (WASM) で再試行中...');
          this.dinoPipe = await pipeline('image-feature-extraction', 'Xenova/dinov2-small', {
            device: 'wasm',
            dtype: 'q8',
            progress_callback: progressCallback
          });
          this.dinoBackendInfo = 'WASM';
          if (this.activeEngine === 'dinov2') this.backendInfo = 'WASM';
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

      // 暗いカード（例：ディヴェル・ティ・フォン等）の暗部を持ち上げつつ、ノイズやホイル反射の過剰増幅を抑える
      let gamma = 1.0;
      if (meanLuma < 85) {
        gamma = 0.78;
      } else if (meanLuma < 115) {
        gamma = 0.88;
      } else if (meanLuma > 185) {
        // 全体的に白っぽく光が反射している場合はわずかに引き締める
        gamma = 1.12;
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

      // 輝度（明るさ）のみにLUTを適用し、R/G/Bに同じ倍率をかけることで元の色味（紫・青・黒など）を絶対に壊さない
      const alpha = 0.55;
      const invAlpha = 1 - alpha;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const l = 0.299 * r + 0.587 * g + 0.114 * b;
        const lIdx = Math.min(255, Math.max(0, Math.round(l)));
        const targetL = lut[lIdx] * alpha + l * invAlpha;
        const scale = l > 4 ? Math.min(2.1, targetL / l) : 1.0;

        data[i] = Math.min(255, Math.round(r * scale));
        data[i + 1] = Math.min(255, Math.round(g * scale));
        data[i + 2] = Math.min(255, Math.round(b * scale));
      }

      ctx.putImageData(imgData, 0, 0);
    }

    /**
     * 画像ソースから L2正規化済みのAI特徴ベクトルを抽出
     */
    async extractEmbedding(source, rotateDeg = 0, cropRect = null, enhanceLighting = true, targetEngine = null) {
      const dedicatedCanvas = document.createElement('canvas');
      const canvas = this.preprocessToCanvas(source, rotateDeg, cropRect, enhanceLighting, dedicatedCanvas);
      const engine = targetEngine || (this.activeEngine === 'hybrid' ? 'mobilenet' : this.activeEngine);

      if (engine === 'dinov2' && this.dinoPipe) {
        return await this.extractDinoV2Embedding(canvas);
      }
      if (engine === 'mobilenet' && this.mobilenetModel) {
        return this.extractMobileNetEmbedding(canvas);
      }
      return this.extractLegacyFeature(canvas);
    }

    /**
     * AI特徴ベクトルと空間カラー配色特徴を同時に抽出（+0.1ms未満で色誤爆を抑制）
     */
    async extractQueryFeatures(source, rotateDeg = 0, cropRect = null, enhanceLighting = true, targetEngine = null) {
      const dedicatedCanvas = document.createElement('canvas');
      const canvas = this.preprocessToCanvas(source, rotateDeg, cropRect, enhanceLighting, dedicatedCanvas);
      const engine = targetEngine || (this.activeEngine === 'hybrid' ? 'mobilenet' : this.activeEngine);

      let vec;
      if (engine === 'dinov2' && this.dinoPipe) {
        vec = await this.extractDinoV2Embedding(canvas);
      } else if (engine === 'mobilenet' && this.mobilenetModel) {
        vec = this.extractMobileNetEmbedding(canvas);
      } else {
        vec = this.extractLegacyFeature(canvas);
      }

      const colorVec = this.extractColorFeature(canvas);
      return { vec, colorVec, canvas };
    }

    /**
     * 10x10グリッドの照明不変・空間カラー＆エッジ勾配特徴（400次元）
     * MobileNet等の深層モデルが最後に失いやすい「イラスト各部の色相・明暗配置」を補完する
     */
    extractColorFeature(canvas) {
      const G = 10;
      const ctx = this.colorCtx;
      ctx.clearRect(0, 0, G, G);
      ctx.drawImage(canvas, 0, 0, G, G);
      const data = ctx.getImageData(0, 0, G, G).data;

      const count = G * G;
      const luma = new Float32Array(count);
      const rg = new Float32Array(count);
      const by = new Float32Array(count);

      let lumaMean = 0;
      for (let i = 0; i < count; i++) {
        const r = data[i * 4] / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;
        const l = 0.299 * r + 0.587 * g + 0.114 * b;
        luma[i] = l;
        lumaMean += l;
        // 反対色空間（照明の明るさ変化に強く、赤/緑・青/黄の固有色差を保持）
        rg[i] = r - g;
        by[i] = b - 0.5 * (r + g);
      }
      lumaMean /= count;

      let lumaVar = 0;
      for (let i = 0; i < count; i++) {
        const d = luma[i] - lumaMean;
        lumaVar += d * d;
      }
      const lumaStd = Math.sqrt(lumaVar / count) || 0.15;

      const out = new Float32Array(count * 4);
      for (let y = 0; y < G; y++) {
        for (let x = 0; x < G; x++) {
          const idx = y * G + x;
          const xPrev = y * G + Math.max(0, x - 1);
          const xNext = y * G + Math.min(G - 1, x + 1);
          const yPrev = Math.max(0, y - 1) * G + x;
          const yNext = Math.min(G - 1, y + 1) * G + x;

          const gx = luma[xNext] - luma[xPrev];
          const gy = luma[yNext] - luma[yPrev];
          const grad = Math.sqrt(gx * gx + gy * gy);

          // 外周1マスは赤いスリーブやカード枠が映り込みやすいため重みを下げ、中心イラスト色を重視
          const isBorder = x === 0 || x === G - 1 || y === 0 || y === G - 1;
          const spatialWeight = isBorder ? 0.25 : 1.0;

          const base = idx * 4;
          out[base] = ((luma[idx] - lumaMean) / lumaStd) * 0.24 * spatialWeight;
          out[base + 1] = rg[idx] * 1.35 * spatialWeight;
          out[base + 2] = by[idx] * 1.35 * spatialWeight;
          out[base + 3] = grad * 0.80 * spatialWeight;
        }
      }

      return l2Normalize(out);
    }

    /**
     * 複数ベクトルの平均ベクトル（重心）を算出（ハブ現象対策のMean-Centering用）
     */
    computeMeanVector(vectors) {
      if (!vectors || vectors.length === 0) return null;
      const dim = vectors[0].length;
      const mean = new Float32Array(dim);
      let valid = 0;
      for (const v of vectors) {
        if (!v || v.length !== dim) continue;
        for (let i = 0; i < dim; i++) {
          mean[i] += v[i];
        }
        valid++;
      }
      if (valid === 0) return null;
      for (let i = 0; i < dim; i++) {
        mean[i] /= valid;
      }
      return mean;
    }

    /**
     * 平均ベクトルを差し引いてL2正規化（全カード共通成分を除去して弁別力を大幅向上）
     */
    centerAndNormalize(vec, meanVec) {
      if (!vec) return null;
      if (!meanVec || vec.length !== meanVec.length) return vec;
      const out = new Float32Array(vec.length);
      for (let i = 0; i < vec.length; i++) {
        out[i] = vec[i] - meanVec[i];
      }
      return l2Normalize(out);
    }

    /**
     * DINOv2-small 特徴抽出（WASM同時実行衝突を防ぐ直列キュー付き）
     * CLSトークン（全体構図・意味）とパッチ平均トークン（細部テクスチャ）を結合し、
     * 暗色系カードや部分的な光反射にも強い768次元ベクトルを生成
     */
    async extractDinoV2Embedding(canvas) {
      const runInference = async () => {
        const { RawImage } = this.transformersModule;
        const input =
          RawImage && typeof RawImage.fromCanvas === 'function'
            ? RawImage.fromCanvas(canvas)
            : canvas.toDataURL('image/jpeg', 0.92);
        return await this.dinoPipe(input);
      };

      const nextPromise = this._dinoQueue.then(runInference, runInference);
      this._dinoQueue = nextPromise.catch(() => {});
      const output = await nextPromise;

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
     * カード1枚分のマルチクロップ空間カラー特徴を高速抽出（<0.2ms）
     */
    computeCardColorFeatures(safeCanvas) {
      const W = safeCanvas.width;
      const H = safeCanvas.height;
      const tmp = document.createElement('canvas');

      const cFull = this.preprocessToCanvas(safeCanvas, 0, null, true, tmp);
      const colorFeature = this.extractColorFeature(cFull);

      const cCenter = this.preprocessToCanvas(
        safeCanvas,
        0,
        { x: W * 0.08, y: H * 0.08, w: W * 0.84, h: H * 0.84 },
        true,
        tmp
      );
      const centerColorFeature = this.extractColorFeature(cCenter);

      const cTight = this.preprocessToCanvas(
        safeCanvas,
        0,
        { x: W * 0.15, y: H * 0.15, w: W * 0.70, h: H * 0.70 },
        true,
        tmp
      );
      const tightColorFeature = this.extractColorFeature(cTight);

      return { colorFeature, centerColorFeature, tightColorFeature };
    }

    /**
     * カード1枚分のマルチクロップ特徴ベクトルを取得（IndexedDBキャッシュ優先）
     * - full   : イラスト全体 (100%)
     * - center : 少し内枠 (84%) — 外側のカード枠が映り込んだ／逆に内側だけ囲んだズレ対策
     * - tight  : 中心コア (70%) — 大きくトリミングされた／端が光で飛んだ場合の対策
     */
    async getOrComputeCardEmbeddings(cardId, safeCanvas, targetEngine = null) {
      const engineId = targetEngine || (this.activeEngine === 'hybrid' ? 'mobilenet' : this.activeEngine);
      const colorFeats = this.computeCardColorFeatures(safeCanvas);

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
            ...colorFeats,
            fromCache: true
          };
        }
      }

      const W = safeCanvas.width;
      const H = safeCanvas.height;

      const feature = await this.extractEmbedding(safeCanvas, 0, null, true, engineId);
      const centerFeature = await this.extractEmbedding(
        safeCanvas,
        0,
        { x: W * 0.08, y: H * 0.08, w: W * 0.84, h: H * 0.84 },
        true,
        engineId
      );
      const tightFeature = await this.extractEmbedding(
        safeCanvas,
        0,
        { x: W * 0.15, y: H * 0.15, w: W * 0.70, h: H * 0.70 },
        true,
        engineId
      );

      if (engineId !== 'legacy') {
        await Promise.all([
          this.cache.set(engineId, cardId, 'full', feature),
          this.cache.set(engineId, cardId, 'center', centerFeature),
          this.cache.set(engineId, cardId, 'tight', tightFeature)
        ]);
      }

      return {
        feature,
        centerFeature,
        tightFeature,
        ...colorFeats,
        fromCache: false
      };
    }

    async clearCache() {
      await this.cache.clearAll();
    }

    cosineSimilarity(vecA, vecB) {
      return cosineSimilarity(vecA, vecB);
    }

    /**
     * カードDBのtype_jaからカード枠種別を返す
     */
    getCardFrameTypeFromMeta(card) {
      const t = String((card && card.type_ja) || '');
      if (t.includes('エクシーズ')) return 'xyz';
      if (t.includes('リンク')) return 'link';
      if (t.includes('シンクロ')) return 'synchro';
      if (t.includes('融合')) return 'fusion';
      if (t.includes('魔法')) return 'spell';
      if (t.includes('罠')) return 'trap';
      if (t.includes('通常モンスター')) return 'normal';
      if (t.includes('モンスター')) return 'effect';
      return 'unknown';
    }

    /**
     * イラスト枠のすぐ外側の細い領域（左右・上下）から遊戯王カードの枠色（黒X/青L/白S/紫融合/緑魔法/桃罠/茶効果）を推定
     */
    detectCardFrameType(snapshotCanvas, artRect, rotateDeg = 0) {
      if (!snapshotCanvas || !artRect) {
        return { frameType: 'unknown', label: '枠未判定', confidence: 0 };
      }

      const W = snapshotCanvas.width;
      const H = snapshotCanvas.height;
      const ctx = snapshotCanvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return { frameType: 'unknown', label: '枠未判定', confidence: 0 };

      // 0°/180°（縦向きカード）はイラストの左右が確実にカード枠、90°/270°（横向き）は上下が確実にカード枠
      const isSideways = rotateDeg === 90 || rotateDeg === 270;
      const marginIn = Math.max(2, Math.round(Math.min(artRect.w, artRect.h) * 0.04));
      const marginOut = Math.max(5, Math.round(Math.min(artRect.w, artRect.h) * 0.12));
      const stripThickness = Math.max(3, marginOut - marginIn);

      const strips = isSideways
        ? [
            { x: artRect.x + artRect.w * 0.15, y: artRect.y - marginOut, w: artRect.w * 0.7, h: stripThickness },
            { x: artRect.x + artRect.w * 0.15, y: artRect.y + artRect.h + marginIn, w: artRect.w * 0.7, h: stripThickness }
          ]
        : [
            { x: artRect.x - marginOut, y: artRect.y + artRect.h * 0.15, w: stripThickness, h: artRect.h * 0.7 },
            { x: artRect.x + artRect.w + marginIn, y: artRect.y + artRect.h * 0.15, w: stripThickness, h: artRect.h * 0.7 }
          ];

      const stripStats = [];
      for (const s of strips) {
        const sx = Math.max(0, Math.min(W - 1, Math.round(s.x)));
        const sy = Math.max(0, Math.min(H - 1, Math.round(s.y)));
        const sw = Math.max(1, Math.min(W - sx, Math.round(s.w)));
        const sh = Math.max(1, Math.min(H - sy, Math.round(s.h)));
        if (sw < 2 || sh < 2) continue;

        const imgData = ctx.getImageData(sx, sy, sw, sh).data;
        const rs = [];
        const gs = [];
        const bs = [];
        for (let i = 0; i < imgData.length; i += 4) {
          const r = imgData[i];
          const g = imgData[i + 1];
          const b = imgData[i + 2];
          // 極端な白飛び光反射を除外
          if (r > 238 && g > 238 && b > 238) continue;
          rs.push(r);
          gs.push(g);
          bs.push(b);
        }
        if (rs.length < 6) continue;
        rs.sort((a, b) => a - b);
        gs.sort((a, b) => a - b);
        bs.sort((a, b) => a - b);
        const mid = Math.floor(rs.length / 2);
        stripStats.push({ r: rs[mid], g: gs[mid], b: bs[mid] });
      }

      if (stripStats.length === 0) {
        return { frameType: 'unknown', label: '枠未判定', confidence: 0 };
      }

      let r = 0;
      let g = 0;
      let b = 0;
      for (const st of stripStats) {
        r += st.r;
        g += st.g;
        b += st.b;
      }
      r /= stripStats.length;
      g /= stripStats.length;
      b /= stripStats.length;

      // 左右（または上下）の枠色の一致度から信頼度を算出（片方が机やスリーブなら一致度が落ちる）
      let consistency = 0.7;
      if (stripStats.length === 2) {
        const diff =
          (Math.abs(stripStats[0].r - stripStats[1].r) +
            Math.abs(stripStats[0].g - stripStats[1].g) +
            Math.abs(stripStats[0].b - stripStats[1].b)) /
          3;
        consistency = Math.max(0, Math.min(1, 1 - diff / 65));
      }

      const rn = r / 255;
      const gn = g / 255;
      const bn = b / 255;
      const max = Math.max(rn, gn, bn);
      const min = Math.min(rn, gn, bn);
      const delta = max - min;
      const v = max;
      const s = max === 0 ? 0 : delta / max;

      let h = 0;
      if (delta > 0.001) {
        if (max === rn) h = 60 * (((gn - bn) / delta) % 6);
        else if (max === gn) h = 60 * ((bn - rn) / delta + 2);
        else h = 60 * ((rn - gn) / delta + 4);
        if (h < 0) h += 360;
      }

      let frameType = 'unknown';
      let label = '枠未判定';
      let typeConf = 0;

      if (v < 0.34 && s < 0.38) {
        frameType = 'xyz';
        label = '黒枠 (エクシーズ)';
        typeConf = 0.92;
      } else if (v > 0.62 && s < 0.22) {
        frameType = 'synchro';
        label = '白枠 (シンクロ)';
        typeConf = 0.85;
      } else if (s >= 0.16 && h >= 130 && h <= 198) {
        frameType = 'spell';
        label = '緑枠 (魔法)';
        typeConf = 0.9;
      } else if (s >= 0.18 && h > 198 && h <= 248) {
        frameType = 'link';
        label = '青枠 (リンク)';
        typeConf = 0.9;
      } else if (s >= 0.16 && h > 248 && h <= 298) {
        frameType = 'fusion';
        label = '紫枠 (融合)';
        typeConf = 0.82;
      } else if (s >= 0.18 && h > 298 && h <= 346) {
        frameType = 'trap';
        label = '桃紫枠 (罠)';
        typeConf = 0.88;
      } else if (s >= 0.18 && h >= 10 && h < 42) {
        frameType = 'effect';
        label = '茶枠 (効果モンスター)';
        typeConf = 0.82;
      } else if (s >= 0.18 && h >= 42 && h <= 65) {
        frameType = 'normal';
        label = '黄枠 (通常モンスター)';
        typeConf = 0.75;
      }

      return {
        frameType,
        label,
        confidence: typeConf * consistency,
        rgb: [Math.round(r), Math.round(g), Math.round(b)]
      };
    }
  }

  global.YgoAIEngine = YgoAIEngine;
})(window);
