"""
YGO Remote Lens - OpenCV SIFT + RANSAC High-Precision Recognition Server (FastAPI)

起動方法:
  ./start_server.sh
  または:
  uv run --with-requirements requirements.txt uvicorn server:app --host 0.0.0.0 --port 8000

ブラウザで http://localhost:8000 を開くと、フロントエンドが自動的にこのSIFTエンジンに接続します。
"""

import concurrent.futures
import json
import pickle
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MASTER_DB_PATH = DATA_DIR / "cards_master_ja.json"
CACHE_DIR = BASE_DIR / ".cache" / "cards"
INDEX_CACHE_V1_PATH = BASE_DIR / ".cache" / "sift_index_v1.pkl"
INDEX_CACHE_PATH = BASE_DIR / ".cache" / "sift_index_v2.pkl"
BOVW_CACHE_PATH = BASE_DIR / ".cache" / "sift_bovw_v2.pkl"
(CACHE_DIR / "full").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "crop").mkdir(parents=True, exist_ok=True)

BOVW_K = 512
STAGE1_EXHAUSTIVE_THRESHOLD = 120
STAGE1_TOP_K = 60

app = FastAPI(title="YGO Remote Lens SIFT Server")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CardAddRequest(BaseModel):
    id: int
    name_ja: str = ""
    image_full: str
    image_crop: str


class SiftRecognitionEngine:
    def __init__(self) -> None:
        self._tls = threading.local()
        self._lock = threading.RLock()
        self._download_lock = threading.Lock()
        # card_id -> {"id", "name_ja", "full": (pts_f32, desc_u8), "crop": (pts_f32, desc_u8)}
        self.index: Dict[int, dict] = {}
        self.cards_meta: Dict[int, dict] = {}
        self.card_urls: Dict[int, dict] = {}
        # 2段階検索用のStage1 BoVWインデックス
        self.bovw_centers: Optional[np.ndarray] = None  # (K, 128) float32
        self.bovw_data: Dict[str, dict] = {}  # kind -> {"cids": np.ndarray, "mat": np.ndarray, "idf": np.ndarray}
        self.ready: bool = False

    def _get_cv_tools(self):
        tools = getattr(self._tls, "cv_tools", None)
        if tools is None:
            tools = (
                cv2.SIFT_create(nfeatures=550),
                cv2.BFMatcher(cv2.NORM_L2),
                cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)),
            )
            self._tls.cv_tools = tools
        return tools

    def parse_cards_js(self) -> List[dict]:
        js_path = BASE_DIR / "cards_data.js"
        if not js_path.exists():
            return []
        text = js_path.read_text(encoding="utf-8")
        cards = []
        pattern = re.compile(
            r'id:\s*(\d+)\s*,\s*'
            r'name_ja:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
            r'name_en:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
            r'type_ja:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
            r'stats_ja:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
            r'desc_ja:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
            r'image_full:\s*"((?:[^"\\]|\\.)*)"\s*,\s*'
            r'image_crop:\s*"((?:[^"\\]|\\.)*)"',
            re.S,
        )
        unescape = lambda s: s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        for m in pattern.finditer(text):
            cards.append(
                {
                    "id": int(m.group(1)),
                    "name_ja": unescape(m.group(2)),
                    "name_en": unescape(m.group(3)),
                    "type_ja": unescape(m.group(4)),
                    "stats_ja": unescape(m.group(5)),
                    "desc_ja": unescape(m.group(6)),
                    "image_full": unescape(m.group(7)),
                    "image_crop": unescape(m.group(8)),
                }
            )
        return cards

    def load_all_card_metadata(self) -> List[dict]:
        merged: Dict[int, dict] = {}
        # 1) data/cards_master_ja.json が存在すれば読み込み
        if MASTER_DB_PATH.exists():
            try:
                raw = json.loads(MASTER_DB_PATH.read_text(encoding="utf-8"))
                items = raw.get("cards", raw) if isinstance(raw, dict) else raw
                if isinstance(items, list):
                    for c in items:
                        if isinstance(c, dict) and c.get("id"):
                            cid = int(c["id"])
                            merged[cid] = c
            except Exception as e:
                print(f"[DB] cards_master_ja.json 読込警告: {e}")

        # 2) cards_data.js の手作業キュレーション済みデータを優先マージ
        for c in self.parse_cards_js():
            merged[int(c["id"])] = c

        return list(merged.values())

    def _download_image(self, url: str, dest: Path) -> bool:
        if dest.exists() and dest.stat().st_size > 1000:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    tmp = dest.with_name(f"{dest.name}.{threading.get_ident()}.tmp")
                    tmp.write_bytes(r.content)
                    tmp.replace(dest)
                    return True
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        return False

    def get_card_image_path(self, kind: str, card_id: int) -> Optional[Path]:
        if kind not in ("full", "crop") or card_id <= 0:
            return None
        dest = CACHE_DIR / kind / f"{card_id}.jpg"
        if dest.exists() and dest.stat().st_size > 1000:
            return dest

        with self._download_lock:
            if dest.exists() and dest.stat().st_size > 1000:
                return dest
            urls = self.card_urls.get(card_id)
            if urls and urls.get(kind):
                url = urls[kind]
            elif kind == "crop":
                url = f"https://images.ygoprodeck.com/images/cards_cropped/{card_id}.jpg"
            else:
                url = f"https://images.ygoprodeck.com/images/cards/{card_id}.jpg"
            if self._download_image(url, dest):
                return dest
        return None

    def _extract_sift_from_gray(self, gray: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        sift, _, _ = self._get_cv_tools()
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kps) < 4:
            return None
        pts = np.float32([kp.pt for kp in kps])
        desc_u8 = np.clip(desc, 0, 255).astype(np.uint8)
        return pts, desc_u8

    def _normalize_entry_uint8(self, entry: dict) -> dict:
        out = dict(entry)
        for kind in ("full", "crop"):
            feat = out.get(kind)
            if feat is not None:
                pts, desc = feat
                if desc.dtype != np.uint8:
                    out[kind] = (pts.astype(np.float32), np.clip(desc, 0, 255).astype(np.uint8))
        return out

    def _compute_card_features(self, card: dict) -> Optional[dict]:
        cid = int(card["id"])
        full_path = CACHE_DIR / "full" / f"{cid}.jpg"
        crop_path = CACHE_DIR / "crop" / f"{cid}.jpg"

        self._download_image(card["image_full"], full_path)
        self._download_image(card["image_crop"], crop_path)

        entry = {"id": cid, "name_ja": card.get("name_ja", str(cid)), "full": None, "crop": None}

        for kind, path in [("full", full_path), ("crop", crop_path)]:
            if not path.exists():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(1.0, 320.0 / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            entry[kind] = self._extract_sift_from_gray(img)

        if entry["full"] is None and entry["crop"] is None:
            return None
        return entry

    def _compute_bovw_vec(self, desc: np.ndarray, centers_norm: np.ndarray) -> np.ndarray:
        d = desc.astype(np.float32)
        d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
        sims = d_norm @ centers_norm.T
        k = centers_norm.shape[0]
        top2 = np.argpartition(sims, -2, axis=1)[:, -2:]
        hist = np.zeros(k, dtype=np.float32)
        np.add.at(hist, top2[:, 0], 1.0)
        np.add.at(hist, top2[:, 1], 0.5)
        hist = np.sqrt(hist)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def build_bovw_index(self, save: bool = True) -> None:
        with self._lock:
            items = list(self.index.items())

        if len(items) < 8:
            self.bovw_centers = None
            self.bovw_data = {}
            return

        rng = np.random.default_rng(42)
        sampled_chunks = []
        for _, entry in items:
            for kind in ("full", "crop"):
                feat = entry.get(kind)
                if feat is not None:
                    sampled_chunks.append(feat[1])
        if not sampled_chunks:
            return

        stacked = np.vstack(sampled_chunks).astype(np.float32)
        max_samples = 24000
        if len(stacked) > max_samples:
            sel = rng.choice(len(stacked), max_samples, replace=False)
            sample = stacked[sel]
        else:
            sample = stacked

        k = min(BOVW_K, max(16, len(sample) // 8))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, _, centers = cv2.kmeans(sample, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

        bovw_data = {}
        for kind in ("full", "crop"):
            cids = []
            vecs = []
            for cid, entry in items:
                feat = entry.get(kind)
                if feat is not None:
                    cids.append(cid)
                    vecs.append(self._compute_bovw_vec(feat[1], centers_norm))
            if vecs:
                mat = np.vstack(vecs).astype(np.float32)
                df = (mat > 0).sum(axis=0).astype(np.float32)
                idf = np.log((len(cids) + 1.0) / (df + 1.0)) + 1.0
                mat_tfidf = mat * idf
                mat_tfidf /= np.linalg.norm(mat_tfidf, axis=1, keepdims=True) + 1e-8
                bovw_data[kind] = {
                    "cids": np.array(cids, dtype=np.int64),
                    "mat": mat_tfidf.astype(np.float32),
                    "idf": idf.astype(np.float32),
                }

        with self._lock:
            self.bovw_centers = centers_norm.astype(np.float32)
            self.bovw_data = bovw_data

        if save:
            self._save_bovw_cache()

    def _save_bovw_cache(self) -> None:
        try:
            with self._lock:
                payload = {
                    "index_count": len(self.index),
                    "centers": self.bovw_centers,
                    "data": self.bovw_data,
                }
            tmp = BOVW_CACHE_PATH.with_name(f"{BOVW_CACHE_PATH.name}.tmp")
            with tmp.open("wb") as f:
                pickle.dump(payload, f)
            tmp.replace(BOVW_CACHE_PATH)
        except Exception as e:
            print(f"[BoVW] キャッシュ保存失敗: {e}")

    def _load_bovw_cache(self) -> bool:
        if not BOVW_CACHE_PATH.exists():
            return False
        try:
            with BOVW_CACHE_PATH.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("index_count") != len(self.index):
                return False
            self.bovw_centers = payload.get("centers")
            self.bovw_data = payload.get("data", {})
            return self.bovw_centers is not None and bool(self.bovw_data)
        except Exception:
            return False

    def load_or_build(self, allow_network_build: bool = True) -> None:
        t0 = time.perf_counter()
        cards = self.load_all_card_metadata()
        self.cards_meta = {int(c["id"]): c for c in cards}
        self.card_urls = {
            int(c["id"]): {"full": c["image_full"], "crop": c["image_crop"]} for c in cards
        }
        card_ids = set(self.card_urls.keys())

        cached: Dict[int, dict] = {}
        migrated_from_v1 = False
        if INDEX_CACHE_PATH.exists():
            try:
                with INDEX_CACHE_PATH.open("rb") as f:
                    cached = pickle.load(f)
            except Exception:
                cached = {}
        elif INDEX_CACHE_V1_PATH.exists():
            try:
                with INDEX_CACHE_V1_PATH.open("rb") as f:
                    raw_v1 = pickle.load(f)
                cached = {cid: self._normalize_entry_uint8(val) for cid, val in raw_v1.items()}
                migrated_from_v1 = True
            except Exception:
                cached = {}

        # キャッシュに存在するカードを再利用
        with self._lock:
            for cid, val in cached.items():
                if cid in card_ids:
                    self.index[cid] = self._normalize_entry_uint8(val)

        if allow_network_build:
            missing = [
                c for c in cards
                if int(c["id"]) not in self.index or self.index[int(c["id"])].get("full") is None
            ]
            # マスターDBに数千枚ある場合でも、初回起動が長時間ブロックされないよう、キャッシュ済みのカードのみ先にロードするか、
            # cards_master_ja.json が未作成（93枚構成）のときのみ自動DLを行う
            if missing and not MASTER_DB_PATH.exists():
                print(f"[SIFT] 新規/未キャッシュカード {len(missing)} 枚をインデックス構築中...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
                    for entry in ex.map(self._compute_card_features, missing):
                        if entry is not None:
                            with self._lock:
                                self.index[entry["id"]] = entry
                self._save_cache()
            elif migrated_from_v1:
                self._save_cache()
        elif migrated_from_v1:
            self._save_cache()

        if not self._load_bovw_cache() and len(self.index) >= 8:
            self.build_bovw_index(save=True)

        self.ready = True
        elapsed = time.perf_counter() - t0
        print(
            f"[SIFT] 準備完了: インデックス {len(self.index)}/{len(cards)} 枚 "
            f"(BoVW={'ON' if self.bovw_centers is not None else 'OFF'}, {elapsed:.2f}s)"
        )

    def _save_cache(self) -> None:
        try:
            with self._lock:
                snapshot = dict(self.index)
            tmp = INDEX_CACHE_PATH.with_name(f"{INDEX_CACHE_PATH.name}.tmp")
            with tmp.open("wb") as f:
                pickle.dump(snapshot, f)
            tmp.replace(INDEX_CACHE_PATH)
        except Exception as e:
            print(f"[SIFT] キャッシュ保存失敗: {e}")

    def add_card(self, card: dict) -> bool:
        cid = int(card["id"])
        self.card_urls[cid] = {
            "full": card.get("image_full", ""),
            "crop": card.get("image_crop", ""),
        }
        if card.get("name_ja") or card.get("desc_ja"):
            self.cards_meta[cid] = card
        entry = self._compute_card_features(card)
        if entry is None:
            return False
        with self._lock:
            self.index[entry["id"]] = entry
        self._save_cache()
        self.build_bovw_index(save=True)
        return True

    def _stage1_candidates(
        self, q_desc: np.ndarray, kind: str, top_k: int
    ) -> Optional[List[Tuple[int, dict]]]:
        with self._lock:
            centers = self.bovw_centers
            bovw_kind = self.bovw_data.get(kind)
            index_ref = self.index
            total_count = len(index_ref)

        if (
            total_count <= STAGE1_EXHAUSTIVE_THRESHOLD
            or centers is None
            or bovw_kind is None
        ):
            return None

        qv = self._compute_bovw_vec(q_desc, centers) * bovw_kind["idf"]
        qnorm = np.linalg.norm(qv)
        if qnorm > 0:
            qv /= qnorm
        scores = bovw_kind["mat"] @ qv
        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        cids = bovw_kind["cids"][top_idx]

        candidates = []
        with self._lock:
            for cid in cids:
                entry = index_ref.get(int(cid))
                if entry is not None:
                    candidates.append((int(cid), entry))
        return candidates

    def _match_against_kind(
        self, q_pts: np.ndarray, q_desc: np.ndarray, kind: str, top_k: int = STAGE1_TOP_K
    ) -> List[Tuple[int, int, int, str]]:
        _, bf, _ = self._get_cv_tools()
        items = self._stage1_candidates(q_desc, kind, top_k)
        if items is None:
            with self._lock:
                items = list(self.index.items())

        scored = []
        for cid, entry in items:
            feat = entry.get(kind)
            if feat is None:
                continue
            db_pts, db_desc_u8 = feat
            db_desc = db_desc_u8.astype(np.float32) if db_desc_u8.dtype != np.float32 else db_desc_u8
            matches = bf.knnMatch(q_desc.astype(np.float32), db_desc, k=2)
            good = []
            for pair in matches:
                if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance:
                    good.append(pair[0])

            inliers = 0
            if len(good) >= 6:
                src = q_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
                dst = db_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(mask.sum())

            scored.append((inliers, len(good), cid, kind))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored

    def recognize(self, image_bytes: bytes, mode: str = "auto") -> dict:
        t0 = time.perf_counter()
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"ok": False, "error": "画像をデコードできませんでした"}

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        long_side = max(h, w)
        if long_side < 280:
            s = 280.0 / long_side
            gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        elif long_side > 520:
            s = 520.0 / long_side
            gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        _, _, clahe = self._get_cv_tools()
        gray = clahe.apply(gray)
        q_feat = self._extract_sift_from_gray(gray)
        if q_feat is None:
            return {
                "ok": True,
                "elapsed_sec": round(time.perf_counter() - t0, 3),
                "engine": "OpenCV SIFT + RANSAC",
                "recognized": False,
                "top_inliers": 0,
                "matches": [],
            }

        q_pts, q_desc = q_feat

        if mode == "crop":
            results = self._match_against_kind(q_pts, q_desc, "crop")
        elif mode == "full":
            results = self._match_against_kind(q_pts, q_desc, "full")
        else:
            # auto: まずカード全体(full)で照合し、一致点が少なければイラストのみ(crop)も照合して良い方を採用
            res_full = self._match_against_kind(q_pts, q_desc, "full")
            best_full = res_full[0][0] if res_full else 0
            if best_full >= 16:
                results = res_full
            else:
                res_crop = self._match_against_kind(q_pts, q_desc, "crop")
                best_crop = res_crop[0][0] if res_crop else 0
                results = res_crop if best_crop > best_full else res_full

        top5 = results[:5]
        top_inliers = top5[0][0] if top5 else 0
        second_inliers = top5[1][0] if len(top5) > 1 else 0

        # 一致特徴点数（inliers）から直感的な信頼度(0.0〜0.99)を算出
        formatted = []
        for inl, gd, cid, mkind in top5:
            # inliers=12 で約70%、25で約88%、45以上で98%になる緩やかな曲線
            conf = min(0.99, max(0.05, 1.0 - np.exp(-inl / 16.0)))
            formatted.append(
                {
                    "id": cid,
                    "inliers": inl,
                    "good_matches": gd,
                    "score": round(float(conf), 4),
                    "match_kind": mkind,
                }
            )

        recognized = top_inliers >= 12 and (top_inliers >= second_inliers + 4 or top_inliers >= 20)
        elapsed = round(time.perf_counter() - t0, 3)

        return {
            "ok": True,
            "elapsed_sec": elapsed,
            "engine": "OpenCV SIFT + RANSAC",
            "recognized": recognized,
            "top_inliers": top_inliers,
            "matches": formatted,
        }


engine = SiftRecognitionEngine()


@app.on_event("startup")
def on_startup() -> None:
    engine.load_or_build()


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "engine": "opencv-sift",
        "ready": engine.ready,
        "card_count": len(engine.index),
        "meta_count": len(engine.cards_meta),
        "two_stage_bovw": engine.bovw_centers is not None and len(engine.index) > STAGE1_EXHAUSTIVE_THRESHOLD,
    }


@app.get("/api/cards")
def list_cards():
    # インデックス済みのカードを優先して返す（マスターDBがある場合はフロント側を自動同期）
    with engine._lock:
        indexed_ids = set(engine.index.keys())
    cards = []
    for cid, meta in engine.cards_meta.items():
        if cid in indexed_ids or not indexed_ids:
            cards.append(meta)
    return JSONResponse({"ok": True, "count": len(cards), "cards": cards})


@app.get("/api/card-image/{kind}/{card_id}.jpg")
def serve_card_image(kind: str, card_id: int):
    if kind not in ("full", "crop") or card_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid card image request")
    path = engine.get_card_image_path(kind, card_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Card image not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.post("/api/recognize")
def recognize(file: UploadFile = File(...), mode: str = Form("auto")):
    # 同期defにすることでFastAPIのスレッドプール上で並列実行され、同時リクエストでもイベントループを止めない
    data = file.file.read()
    result = engine.recognize(data, mode=mode)
    return JSONResponse(result)


@app.post("/api/cards/add")
def add_card(req: CardAddRequest):
    ok = engine.add_card(req.model_dump())
    return {"ok": ok, "card_count": len(engine.index)}


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/{filename:path}")
def serve_static(filename: str):
    allowed = {"index.html", "app.js", "ai_engine.js", "cards_data.js", "robots.txt"}
    if filename in allowed:
        target = BASE_DIR / filename
        if target.exists():
            return FileResponse(target)
    return JSONResponse({"detail": "Not Found"}, status_code=404)
