"""
YGO Remote Lens - 全カード日本語マスター辞書生成 ＆ レジューム対応SIFTインデックス構築ツール

使い方例:
  1) 全カード日本語マスター辞書 (data/cards_master_ja.json) を生成:
     uv run --with-requirements requirements.txt python build_card_db.py --build-dict

  2) まずは500枚だけ画像取得＆SIFTインデックス構築（秒速制限・中断再開対応）:
     uv run --with-requirements requirements.txt python build_card_db.py --build-index --limit 500 --rate 3.0

  3) 特定の .ydk ファイルや YDKEコード、テーマのカードを優先追加:
     uv run --with-requirements requirements.txt python build_card_db.py --build-index --ydk my_deck.ydk
     uv run --with-requirements requirements.txt python build_card_db.py --build-index --archetype "Blue-Eyes"

  4) 既存SIFTインデックスからStage 1 BoVW検索インデックスだけ再構築（オフライン・高速）:
     uv run --with-requirements requirements.txt python build_card_db.py --rebuild-bovw
"""

import argparse
import base64
import io
import json
import struct
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from server import DATA_DIR, MASTER_DB_PATH, SiftRecognitionEngine

YGOPRODECK_ALL_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
YGOCDB_ZIP_URL = "https://ygocdb.com/api/v0/cards.zip"


def format_stats_ja(en_card: dict, jp_entry: Optional[dict]) -> str:
    ctype = str(en_card.get("type", ""))
    if "Spell" in ctype or "Trap" in ctype:
        race = en_card.get("race", "")
        race_map = {
            "Normal": "通常",
            "Quick-Play": "速攻",
            "Continuous": "永続",
            "Equip": "装備",
            "Field": "フィールド",
            "Ritual": "儀式",
            "Counter": "カウンター",
        }
        kind_ja = "魔法" if "Spell" in ctype else "罠"
        return f"{race_map.get(race, race)}{kind_ja}"

    atk = en_card.get("atk")
    defv = en_card.get("def")
    linkval = en_card.get("linkval")
    if linkval is not None:
        return f"ATK {atk if atk is not None else '?'} / LINK-{linkval}"
    if atk is not None:
        return f"ATK {atk} / DEF {defv if defv is not None else '-'}"
    return ctype


def format_type_ja(en_card: dict, jp_entry: Optional[dict]) -> str:
    if jp_entry and isinstance(jp_entry.get("text"), dict) and jp_entry["text"].get("types"):
        return str(jp_entry["text"]["types"]).replace("\n", " / ").strip()
    parts = [en_card.get("type", "")]
    if en_card.get("attribute"):
        parts.append(en_card["attribute"])
    if en_card.get("race"):
        parts.append(en_card["race"])
    if en_card.get("linkval") is not None:
        parts.append(f"LINK-{en_card['linkval']}")
    elif en_card.get("level") is not None:
        prefix = "ランク" if "XYZ" in str(en_card.get("type", "")) else "★"
        parts.append(f"{prefix}{en_card['level']}")
    return " / ".join(p for p in parts if p)


def build_japanese_master_dict() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("⬇️ YGOPRODeck 全カードデータを取得中...")
    r_en = requests.get(YGOPRODECK_ALL_URL, timeout=60)
    r_en.raise_for_status()
    en_cards = r_en.json().get("data", [])
    print(f"   取得完了: {len(en_cards)} 枚")

    jp_by_id: Dict[int, dict] = {}
    try:
        print("⬇️ ygocdb 日本語カードテキストデータ (cards.zip) を取得中...")
        r_jp = requests.get(YGOCDB_ZIP_URL, timeout=60)
        r_jp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r_jp.content)) as zf:
            json_names = [n for n in zf.namelist() if n.endswith(".json")]
            if json_names:
                raw_jp = json.loads(zf.read(json_names[0]).decode("utf-8"))
                entries = raw_jp.values() if isinstance(raw_jp, dict) else raw_jp
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    cid = item.get("id")
                    if cid:
                        jp_by_id[int(cid)] = item
        print(f"   日本語辞書読込完了: {len(jp_by_id)} 件")
    except Exception as e:
        print(f"⚠️ ygocdb 取得失敗（英語フォールバックを含めて構築します）: {e}")

    master_cards = []
    seen_ids: Set[int] = set()
    for c in en_cards:
        cid = int(c["id"])
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        images = c.get("card_images") or []
        if not images:
            continue

        jp = jp_by_id.get(cid)
        jp_text = jp.get("text", {}) if isinstance(jp, dict) and isinstance(jp.get("text"), dict) else {}
        name_ja = (jp.get("jp_name") if isinstance(jp, dict) else None) or c.get("name", "")
        desc_ja = jp_text.get("desc") or c.get("desc", "")

        master_cards.append(
            {
                "id": cid,
                "name_ja": name_ja,
                "name_en": c.get("name", ""),
                "archetype": c.get("archetype", ""),
                "type_ja": format_type_ja(c, jp),
                "stats_ja": format_stats_ja(c, jp),
                "desc_ja": desc_ja,
                "image_full": images[0].get("image_url", f"https://images.ygoprodeck.com/images/cards/{cid}.jpg"),
                "image_crop": images[0].get("image_url_cropped", f"https://images.ygoprodeck.com/images/cards_cropped/{cid}.jpg"),
            }
        )

    tmp = MASTER_DB_PATH.with_name(f"{MASTER_DB_PATH.name}.tmp")
    tmp.write_text(
        json.dumps({"count": len(master_cards), "cards": master_cards}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(MASTER_DB_PATH)
    print(f"✅ 日本語マスター辞書を保存しました: {MASTER_DB_PATH} ({len(master_cards)} 枚)")
    return MASTER_DB_PATH


def parse_ydk_file(path: Path) -> List[int]:
    ids = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.isdigit():
            ids.append(int(s))
    return ids


def parse_ydke_uri(uri: str) -> List[int]:
    if not uri.startswith("ydke://"):
        return []
    body = uri[len("ydke://") :]
    ids = []
    for part in body.split("!"):
        if not part:
            continue
        pad = "=" * ((4 - len(part) % 4) % 4)
        raw = base64.b64decode(part + pad)
        for i in range(0, len(raw) - 3, 4):
            ids.append(struct.unpack("<I", raw[i : i + 4])[0])
    return ids


def build_index_incremental(
    limit: Optional[int],
    offset: int,
    rate_per_sec: float,
    ydk_path: Optional[str],
    ydke_uri: Optional[str],
    archetype: Optional[str],
    include_all: bool,
) -> None:
    eng = SiftRecognitionEngine()
    eng.load_or_build(allow_network_build=False)
    cards = eng.load_all_card_metadata()
    by_id = {int(c["id"]): c for c in cards}

    target_cards: List[dict] = []
    if ydk_path:
        wanted = set(parse_ydk_file(Path(ydk_path)))
        target_cards.extend(by_id[cid] for cid in wanted if cid in by_id)
    if ydke_uri:
        wanted = set(parse_ydke_uri(ydke_uri))
        target_cards.extend(by_id[cid] for cid in wanted if cid in by_id)
    if archetype:
        q = archetype.lower()
        target_cards.extend(
            c for c in cards if q in str(c.get("archetype", "")).lower() or q in str(c.get("name_en", "")).lower() or q in str(c.get("name_ja", "")).lower()
        )
    if include_all or (not ydk_path and not ydke_uri and not archetype):
        target_cards = cards[offset:]

    # 重複除去しつつ、既にキャッシュ＆インデックス済みのカードはスキップ（レジューム対応）
    seen: Set[int] = set()
    pending: List[dict] = []
    for c in target_cards:
        cid = int(c["id"])
        if cid in seen:
            continue
        seen.add(cid)
        if cid in eng.index and eng.index[cid].get("full") is not None and eng.index[cid].get("crop") is not None:
            continue
        pending.append(c)

    if limit is not None and limit > 0:
        pending = pending[:limit]

    print(
        f"📦 インデックス構築対象: {len(pending)} 枚 "
        f"(既存インデックス: {len(eng.index)} 枚 / レート上限: {rate_per_sec:.1f}枚/秒)"
    )
    if not pending:
        print("✅ すべて構築済みです。")
        return

    interval = 1.0 / max(0.5, rate_per_sec)
    added = 0
    try:
        for i, card in enumerate(pending, 1):
            t_start = time.perf_counter()
            entry = eng._compute_card_features(card)
            if entry is not None:
                with eng._lock:
                    eng.index[entry["id"]] = entry
                added += 1

            if i % 25 == 0 or i == len(pending):
                eng._save_cache()
                print(f"   進捗 {i}/{len(pending)} (新規保存累計: {added} / 総インデックス: {len(eng.index)})")

            elapsed = time.perf_counter() - t_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        print("\n⏸️ 中断を検知しました。ここまでの進捗を保存します...")
        eng._save_cache()

    print("🧠 Stage 1 BoVW 検索インデックスを更新中...")
    eng.build_bovw_index(save=True)
    print(f"✅ 完了: 総インデックス {len(eng.index)} 枚")


def main() -> None:
    parser = argparse.ArgumentParser(description="YGO Remote Lens カードDB・SIFTインデックス構築ツール")
    parser.add_argument("--build-dict", action="store_true", help="全カード日本語マスター辞書 (data/cards_master_ja.json) を生成")
    parser.add_argument("--build-index", action="store_true", help="画像取得＆SIFTインデックスをレジューム対応で段階構築")
    parser.add_argument("--rebuild-bovw", action="store_true", help="既存SIFTインデックスからStage 1 BoVWインデックスを再構築")
    parser.add_argument("--all", action="store_true", help="全カードを対象にする")
    parser.add_argument("--limit", type=int, default=None, help="今回処理する最大カード枚数 (例: 500)")
    parser.add_argument("--offset", type=int, default=0, help="開始オフセット位置")
    parser.add_argument("--rate", type=float, default=3.0, help="1秒あたりの処理カード数上限 (デフォルト: 3.0)")
    parser.add_argument("--ydk", type=str, default=None, help=".ydk デッキファイルのパス")
    parser.add_argument("--ydke", type=str, default=None, help="ydke:// デッキコード")
    parser.add_argument("--archetype", type=str, default=None, help="テーマ名/カード名フィルタ (例: Blue-Eyes)")

    args = parser.parse_args()
    if not (args.build_dict or args.build_index or args.rebuild_bovw):
        parser.print_help()
        return

    if args.build_dict:
        build_japanese_master_dict()

    if args.build_index:
        build_index_incremental(
            limit=args.limit,
            offset=args.offset,
            rate_per_sec=args.rate,
            ydk_path=args.ydk,
            ydke_uri=args.ydke,
            archetype=args.archetype,
            include_all=args.all,
        )
    elif args.rebuild_bovw:
        eng = SiftRecognitionEngine()
        eng.load_or_build(allow_network_build=False)
        eng.build_bovw_index(save=True)
        print(f"✅ BoVW再構築完了 ({len(eng.index)} 枚)")


if __name__ == "__main__":
    main()
