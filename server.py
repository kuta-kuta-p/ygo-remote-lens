"""
YGO Remote Lens - OpenCV SIFT + RANSAC High-Precision Recognition Server (FastAPI)

起動方法:
  ./start_server.sh
  または:
  uv run --with-requirements requirements.txt uvicorn server:app --host 0.0.0.0 --port 8000

ブラウザで http://localhost:8000 を開くと、フロントエンドが自動的にこのSIFTエンジンに接続します。
"""

import concurrent.futures
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache" / "cards"
INDEX_CACHE_PATH = BASE_DIR / ".cache" / "sift_index_v1.pkl"
(CACHE_DIR / "full").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "crop").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YGO Remote Lens SIFT Server")
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
        self.sift = cv2.SIFT_create(nfeatures=550)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # card_id -> {"id", "name_ja", "full": (pts_ndarray, desc), "crop": (pts_ndarray, desc)}
        self.index: Dict[int, dict] = {}
        self.ready: bool = False

    def parse_cards_js(self) -> List[dict]:
        js_path = BASE_DIR / "cards_data.js"
        if not js_path.exists():
            return []
        text = js_path.read_text(encoding="utf-8")
        cards = []
        pattern = re.compile(
            r'id:\s*(\d+).*?name_ja:\s*"([^"]+)".*?image_full:\s*"([^"]+)".*?image_crop:\s*"([^"]+)"',
            re.S,
        )
        for m in pattern.finditer(text):
            cards.append(
                {
                    "id": int(m.group(1)),
                    "name_ja": m.group(2),
                    "image_full": m.group(3),
                    "image_crop": m.group(4),
                }
            )
        return cards

    def _download_image(self, url: str, dest: Path) -> bool:
        if dest.exists() and dest.stat().st_size > 1000:
            return True
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    dest.write_bytes(r.content)
                    return True
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        return False

    def _extract_sift_from_gray(self, gray: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kps) < 4:
            return None
        pts = np.float32([kp.pt for kp in kps])
        return pts, desc

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

    def load_or_build(self) -> None:
        t0 = time.perf_counter()
        cards = self.parse_cards_js()
        card_ids = {c["id"] for c in cards}

        cached: Dict[int, dict] = {}
        if INDEX_CACHE_PATH.exists():
            try:
                with INDEX_CACHE_PATH.open("rb") as f:
                    cached = pickle.load(f)
            except Exception:
                cached = {}

        # キャッシュに存在するカードを再利用し、未登録カードだけダウンロード＆特徴抽出
        missing = [c for c in cards if c["id"] not in cached or cached[c["id"]].get("full") is None]
        for cid, val in cached.items():
            if cid in card_ids:
                self.index[cid] = val

        if missing:
            print(f"[SIFT] 新規/未キャッシュカード {len(missing)} 枚をインデックス構築中...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
                for entry in ex.map(self._compute_card_features, missing):
                    if entry is not None:
                        self.index[entry["id"]] = entry
            self._save_cache()

        self.ready = True
        elapsed = time.perf_counter() - t0
        print(f"[SIFT] 準備完了: {len(self.index)}/{len(cards)} 枚 ({elapsed:.2f}s)")

    def _save_cache(self) -> None:
        try:
            with INDEX_CACHE_PATH.open("wb") as f:
                pickle.dump(self.index, f)
        except Exception as e:
            print(f"[SIFT] キャッシュ保存失敗: {e}")

    def add_card(self, card: dict) -> bool:
        entry = self._compute_card_features(card)
        if entry is None:
            return False
        self.index[entry["id"]] = entry
        self._save_cache()
        return True

    def _match_against_kind(
        self, q_pts: np.ndarray, q_desc: np.ndarray, kind: str
    ) -> List[Tuple[int, int, int, str]]:
        scored = []
        for cid, entry in self.index.items():
            feat = entry.get(kind)
            if feat is None:
                continue
            db_pts, db_desc = feat
            matches = self.bf.knnMatch(q_desc, db_desc, k=2)
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

        gray = self.clahe.apply(gray)
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
    }


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...), mode: str = Form("auto")):
    data = await file.read()
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
