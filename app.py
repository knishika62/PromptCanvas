import argparse
import base64
import io
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ---------------------------------------------------------------------------
# .env 自動読み込み
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    for env_path in [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    line = line.removeprefix("export").strip()
                    if "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
            break

_load_dotenv()

APP_DIR   = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MOCK_MODE  = os.environ.get("IMAGE_TO_PROMPT_MOCK", "").lower() in {"1", "true", "yes"}

# PaddleOCR
OCR_LANG = os.environ.get("IMAGE_TO_PROMPT_OCR_LANG", "japan,en")

# Vision LLM
VISION_API_BASE  = os.environ.get("VISION_API_BASE", "").rstrip("/")
VISION_API_KEY   = os.environ.get("VISION_API_KEY", "EMPTY")
VISION_MODEL     = os.environ.get("VISION_MODEL", "")
VISION_FULL_MAX_PX = int(os.environ.get("VISION_FULL_MAX_PX", "1024"))
VISION_CROP_MAX_PX = int(os.environ.get("VISION_CROP_MAX_PX", "512"))
# thinking対応モデルは思考にトークンを消費するため大きめに確保する
VISION_MAX_TOKENS  = int(os.environ.get("VISION_MAX_TOKENS", "6000"))

# WOMAN=Japanese / Korean 等を設定すると woman/girl/female/lady に属性を付与して
# "{WOMAN} woman" に統一する。未設定なら補正無効
WOMAN_TAG = os.environ.get("WOMAN", "").strip()

app = FastAPI(title="Image to Prompt", version="3.0.0")


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict[str, Any]):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response


paddle_lock = threading.Lock()
paddle_ocr: Any = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_progress(request_id: str | None, message: str) -> None:
    prefix = "[Image to Prompt]"
    if request_id:
        prefix += f"[{request_id}]"
    print(f"{prefix} {message}", flush=True)


@contextmanager
def progress_stage(request_id: str | None, label: str):
    started_at = time.perf_counter()
    log_progress(request_id, f"{label}...")
    try:
        yield
    except Exception:
        log_progress(request_id, f"{label} failed after {time.perf_counter() - started_at:.1f}s")
        raise
    else:
        log_progress(request_id, f"{label} done in {time.perf_counter() - started_at:.1f}s")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def load_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from exc
    if image.width < 1 or image.height < 1:
        raise HTTPException(status_code=400, detail="Upload a non-empty image.")
    return image


def resize_for_vision(image: Image.Image, max_short_side: int) -> Image.Image:
    w, h = image.size
    short = min(w, h)
    if short <= max_short_side:
        return image
    scale = max_short_side / short
    return image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def normalize_bbox_xyxy(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = clamp(x1, 0, width);  x2 = clamp(x2, 0, width)
    y1 = clamp(y1, 0, height); y2 = clamp(y2, 0, height)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [
        round((y1 / height) * 1000),
        round((x1 / width)  * 1000),
        round((y2 / height) * 1000),
        round((x2 / width)  * 1000),
    ]


def bbox_area(bbox: list[int]) -> int:
    y1, x1, y2, x2 = bbox
    return max(0, y2 - y1) * max(0, x2 - x1)


def bbox_iou(a: list[int], b: list[int]) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    inter = bbox_area([iy1, ix1, iy2, ix2])
    denom = bbox_area(a) + bbox_area(b) - inter
    return inter / denom if denom else 0.0


def sample_color(image: Image.Image, bbox: list[int]) -> str:
    y1, x1, y2, x2 = bbox
    left   = int((x1 / 1000) * image.width)
    top    = int((y1 / 1000) * image.height)
    right  = max(left + 1, int((x2 / 1000) * image.width))
    bottom = max(top + 1, int((y2 / 1000) * image.height))
    crop = image.crop((left, top, right, bottom)).resize((1, 1), Image.Resampling.BILINEAR)
    r, g, b = crop.getpixel((0, 0))
    return f"#{r:02X}{g:02X}{b:02X}"


def dominant_palette(image: Image.Image, count: int = 5) -> list[str]:
    small = image.resize((80, 80), Image.Resampling.BILINEAR)
    arr = np.asarray(small).reshape(-1, 3)
    if arr.size == 0:
        return []
    bins = np.clip((arr // 32) * 32 + 16, 0, 255).astype(np.uint8)
    colors, counts = np.unique(bins, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:count]
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors[order]]


def detect_aspect_ratio(width: int, height: int) -> str:
    RATIOS = [
        (1, 1), (4, 3), (3, 4), (3, 2), (2, 3),
        (16, 9), (9, 16), (16, 10), (10, 16),
        (21, 9), (9, 21), (4, 5), (5, 4),
        (2, 1), (1, 2), (3, 1), (1, 3),
    ]
    actual = width / height
    best = min(RATIOS, key=lambda r: abs(r[0] / r[1] - actual))
    return f"{best[0]}:{best[1]}"


# ---------------------------------------------------------------------------
# Vision LLM
# ---------------------------------------------------------------------------

def vision_llm_available() -> bool:
    return bool(VISION_API_BASE and VISION_MODEL)


def call_vision_llm(prompt: str, image: Image.Image | None, max_tokens: int = 256, system_prompt: str | None = None) -> str | None:
    """LLMを呼び出す。imageがNoneならテキストのみのリクエストになる。

    system_promptを指定すると、静的な指示をsystemロールのメッセージとして
    先頭に追加する（promptには動的な内容のみを渡す）。
    """
    if not vision_llm_available():
        return None
    try:
        import urllib.request
        content: list[dict[str, Any]] = []
        if image is not None:
            b64 = image_to_base64(image)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        payload = json.dumps({
            "model": VISION_MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
        }).encode()
        req = urllib.request.Request(
            f"{VISION_API_BASE}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {VISION_API_KEY}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        content = (data["choices"][0]["message"].get("content") or "").strip()
        return content or None
    except Exception as exc:
        log_progress(None, f"Vision LLM call failed: {exc}")
        return None


def parse_llm_json(result: str) -> dict[str, Any]:
    """LLM応答からJSON本体を抽出してパースする。

    thinkingブロック・markdownフェンス・前後の説明文に対応。
    """
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", result, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```[a-z]*\n?|```", "", clean).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end > start:
        clean = clean[start:end + 1]
    return json.loads(clean)


# プロンプト本文(system prompt)はmd/以下のファイルに分離している
PROMPTS_DIR = Path(__file__).parent / "md"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


# 人物要素のdesc詳細記述ルール（画像解析・テキスト生成の両system promptで共用、Textタブのデフォルト）
PERSON_DETAIL_RULES = load_prompt("person_detail_rules")
VISION_ANALYZE_SYSTEM_PROMPT = load_prompt("vision_analyze_system").replace("{{PERSON_DETAIL_RULES}}", PERSON_DETAIL_RULES)
# Textタブはルールセットをリクエスト時に選択するため、プレースホルダ未解決のテンプレートのまま保持
TEXT_GENERATE_SYSTEM_TEMPLATE = load_prompt("text_generate_system")


def load_text_presets() -> dict[str, str]:
    """app/md/_*.md をTextタブ用プリセット（PERSON_DETAIL_RULES差し替え用）として読み込む。"""
    presets = {}
    for path in sorted(PROMPTS_DIR.glob("_*.md")):
        presets[path.stem[1:]] = path.read_text(encoding="utf-8")
    return presets


def preset_label(preset_id: str) -> str:
    return preset_id.replace("_", " ").replace("-", " ").title()


TEXT_PRESETS = load_text_presets()


def vision_analyze(image: Image.Image, ocr_elements: list[dict[str, Any]], request_id: str) -> dict[str, Any]:
    """Vision LLMに画像全体を解析させてIdeogram JSON構造を生成する。

    OCRで検出済みのテキスト要素を参考情報として渡し、
    obj要素のbbox・description と caption・background を生成する。
    """
    resized = resize_for_vision(image, VISION_FULL_MAX_PX)

    # OCR検出済みテキストを参考情報としてuserメッセージに含める
    # 注意: Qwen3系VLMは指示に関わらず0-1000正規化座標で返す（ピクセル座標方式を
    # 試したがy軸が 1000/height に圧縮されるズレが出たため0-1000方式に戻した）
    if ocr_elements:
        lines = []
        for e in ocr_elements:
            # [ymin,xmin,ymax,xmax] → [xmin,ymin,xmax,ymax] に変換してプロンプトに渡す
            ymin, xmin, ymax, xmax = e["bbox"]
            lines.append(f'  - "{e["text"]}" at bbox [{xmin},{ymin},{xmax},{ymax}]')
        user_text = (
            "Analyze the attached image.\n\n"
            "The following text regions have already been detected by OCR "
            "(bbox format is [xmin, ymin, xmax, ymax] in 0-1000 scale, origin top-left):\n"
            + "\n".join(lines)
            + "\nDo NOT include these as elements. They will be added separately."
        )
    else:
        user_text = "Analyze the attached image."

    with progress_stage(request_id, "Vision LLM: analyzing image"):
        result = call_vision_llm(user_text, resized, max_tokens=VISION_MAX_TOKENS, system_prompt=VISION_ANALYZE_SYSTEM_PROMPT)

    if not result:
        return {}

    try:
        data = parse_llm_json(result)

        # LLMは [xmin, ymin, xmax, ymax]（0-1000）で返すので
        # Ideogram形式 [ymin, xmin, ymax, xmax] に変換する
        raw_elements = [e for e in data.get("elements", []) if len(e.get("bbox", [])) == 4]

        # 座標スケール自動判定: 0-1000指示でも絶対ピクセル座標で返すモデルがある
        # （Qwen2.5-VL系はピクセル座標で学習されている）。
        # 1000を超える座標が含まれていればピクセル座標とみなし、リサイズ画像サイズで正規化する
        rw, rh = resized.size
        max_coord = max((float(v) for e in raw_elements for v in e["bbox"]), default=0)
        pixel_scale = max_coord > 1000
        if pixel_scale:
            log_progress(request_id, f"Vision LLM returned pixel-scale bboxes (max={max_coord:.0f}); normalizing by {rw}x{rh}")

        elements = []
        for elem in raw_elements:
            bbox_raw = [float(v) for v in elem["bbox"]]
            if pixel_scale:
                bbox_raw = [bbox_raw[0] / rw * 1000, bbox_raw[1] / rh * 1000, bbox_raw[2] / rw * 1000, bbox_raw[3] / rh * 1000]
            x1 = int(clamp(round(bbox_raw[0]), 0, 1000))
            y1 = int(clamp(round(bbox_raw[1]), 0, 1000))
            x2 = int(clamp(round(bbox_raw[2]), 0, 1000))
            y2 = int(clamp(round(bbox_raw[3]), 0, 1000))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            bbox = [y1, x1, y2, x2]  # [ymin, xmin, ymax, xmax]
            if bbox_area(bbox) <= 10:
                continue
            elements.append({
                "id": f"item-{len(elements) + 1}",
                "type": "obj",
                "label": (elem.get("label") or "object").strip(),
                "description": (elem.get("desc") or "").strip(),
                "bbox": bbox,
                "color": sample_color(image, bbox),
            })

        return {
            "caption": data.get("caption", "").strip(),
            "background": data.get("background", "").strip(),
            "elements": elements,
        }

    except Exception as exc:
        log_progress(request_id, f"Vision LLM JSON parse failed: {exc}\nRaw: {result[:200]}")
        return {}


def text_generate(description: str, aspect_ratio: str, request_id: str, preset_id: str | None = None) -> dict[str, Any]:
    """テキスト説明からIdeogram JSON構造（caption/background/elements）を生成する。

    Text to Promptタブ用。LLMはテキストのみで呼び出す（画像なし）。
    preset_idを指定すると、PERSON_DETAIL_RULESの代わりにapp/md/_<preset_id>.mdの
    内容をルールセットとして埋め込む。
    """
    user_text = f"""## Target canvas
Aspect ratio {aspect_ratio} (width:height).

## User description
{description}"""

    rules = TEXT_PRESETS.get(preset_id, PERSON_DETAIL_RULES) if preset_id else PERSON_DETAIL_RULES
    system_prompt = TEXT_GENERATE_SYSTEM_TEMPLATE.replace("{{PERSON_DETAIL_RULES}}", rules)

    with progress_stage(request_id, "Vision LLM: generating layout from text"):
        result = call_vision_llm(user_text, None, max_tokens=VISION_MAX_TOKENS, system_prompt=system_prompt)

    if not result:
        return {}

    try:
        data = parse_llm_json(result)
        elements = []
        for elem in data.get("elements", []):
            bbox_raw = elem.get("bbox", [])
            if len(bbox_raw) != 4:
                continue
            x1 = int(clamp(round(float(bbox_raw[0])), 0, 1000))
            y1 = int(clamp(round(float(bbox_raw[1])), 0, 1000))
            x2 = int(clamp(round(float(bbox_raw[2])), 0, 1000))
            y2 = int(clamp(round(float(bbox_raw[3])), 0, 1000))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            bbox = [y1, x1, y2, x2]  # [ymin, xmin, ymax, xmax]
            if bbox_area(bbox) <= 10:
                continue
            etype = "text" if elem.get("type") == "text" else "obj"
            entry: dict[str, Any] = {
                "id": f"item-{len(elements) + 1}",
                "type": etype,
                "label": (elem.get("label") or elem.get("text") or "object").strip(),
                "description": (elem.get("desc") or "").strip(),
                "bbox": bbox,
                "color": "#BDBAB2",
            }
            if etype == "text":
                text = (elem.get("text") or elem.get("label") or "").strip()
                entry["text"] = text
                entry["label"] = entry["label"] or text
                if not entry["description"]:
                    entry["description"] = f'text "{text}"'
            elements.append(entry)

        return {
            "caption": data.get("caption", "").strip(),
            "background": data.get("background", "").strip(),
            "elements": elements,
        }

    except Exception as exc:
        log_progress(request_id, f"Text generation JSON parse failed: {exc}\nRaw: {result[:200]}")
        return {}


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

def get_paddle_ocr(request_id: str | None = None) -> Any:
    global paddle_ocr
    if paddle_ocr is not None:
        return paddle_ocr
    with paddle_lock:
        if paddle_ocr is not None:
            return paddle_ocr
        try:
            from paddleocr import PaddleOCR
            import inspect
            lang = [l.strip() for l in OCR_LANG.split(",") if l.strip()]
            lang = lang[0] if lang else "japan"
            with progress_stage(request_id, f"Loading PaddleOCR (lang={lang})"):
                sig = inspect.signature(PaddleOCR.__init__)
                kwargs: dict[str, Any] = {"lang": lang}
                if "use_textline_orientation" in sig.parameters:
                    kwargs["use_textline_orientation"] = True
                elif "use_angle_cls" in sig.parameters:
                    kwargs["use_angle_cls"] = True
                if "show_log" in sig.parameters:
                    kwargs["show_log"] = False
                paddle_ocr = PaddleOCR(**kwargs)
            log_progress(request_id, "PaddleOCR ready")
        except ImportError:
            log_progress(request_id, "PaddleOCR not installed")
            paddle_ocr = None
    return paddle_ocr


def merge_ocr_lines(raw_lines: list[tuple], image_w: int, image_h: int) -> list[tuple]:
    """y座標が近いOCR要素を同一行としてマージする。"""
    if not raw_lines:
        return raw_lines

    items = []
    for quad, text, score in raw_lines:
        pts = np.asarray(quad, dtype=float).reshape(-1, 2)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        cy = (y1 + y2) / 2
        h  = max(abs(y2 - y1), 1.0)  # 座標逆転対応
        items.append({"text": text, "score": score, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cy": cy, "h": h})

    items.sort(key=lambda e: e["cy"])

    groups: list[list[dict]] = []
    for item in items:
        merged = False
        for group in reversed(groups):
            # グループ平均ではなく直近要素のcy・h基準で判定（平均ずれ防止）
            last = max(group, key=lambda e: e["cy"])
            threshold = max(last["h"], item["h"]) * 0.6
            if abs(item["cy"] - last["cy"]) <= threshold:
                group.append(item)
                merged = True
                break
        if not merged:
            groups.append([item])

    result = []
    for group in groups:
        avg_h = sum(e["h"] for e in group) / len(group)

        # x中心（cx）でクラスタリングして列を特定する
        # 同じ列 = cx が近い要素（avg_w の 0.5 倍以内）
        avg_w = sum(e["x2"] - e["x1"] for e in group) / len(group)
        cx_threshold = avg_w * 0.6

        # cx でソートして列クラスタを作成
        group_by_cx = sorted(group, key=lambda e: (e["x1"] + e["x2"]) / 2)
        col_clusters: list[list[dict]] = [[group_by_cx[0]]]
        for e in group_by_cx[1:]:
            ecx = (e["x1"] + e["x2"]) / 2
            ref_cx = sum((r["x1"] + r["x2"]) / 2 for r in col_clusters[-1]) / len(col_clusters[-1])
            if abs(ecx - ref_cx) <= cx_threshold:
                col_clusters[-1].append(e)
            else:
                col_clusters.append([e])

        for col in col_clusters:
            # 列内をy順にソートしてテキスト結合
            col.sort(key=lambda e: e["cy"])
            merged_text  = " ".join(e["text"] for e in col).strip()
            merged_score = sum(e["score"] for e in col) / len(col)
            x1 = min(e["x1"] for e in col)
            y1 = min(e["y1"] for e in col)
            x2 = max(e["x2"] for e in col)
            y2 = max(e["y2"] for e in col)
            quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            result.append((quad, merged_text, merged_score))

    return result


# ---------------------------------------------------------------------------
# WOMAN タグ補正
# ---------------------------------------------------------------------------

# "woman" にマッチするが既にタグ（Japanese 等）が付いていればスキップするパターン
# "easy woman" "old woman" 等の形容詞付きも正しく置換する
_WOMAN_LABEL_RE = re.compile(
    r"(?<![\w])"  # 直前が単語文字でない
    + (rf"(?:{re.escape(WOMAN_TAG)}\s+)?" if WOMAN_TAG else "")  # 既にタグが付いている場合は飲み込んで二重付与を防ぐ
    + r"("
    r"(?:young\s+|old\s+|middle[- ]aged\s+|beautiful\s+|pretty\s+|attractive\s+|sexy\s+|elegant\s+|easy\s+|casual\s+|traditional\s+)*"
    r"woman|girl|female|lady"
    r")"
    r"(?![\w])",
    re.IGNORECASE,
)

def _patch_woman(text: str) -> str:
    """woman/girl/female/lady を "{WOMAN_TAG} woman" に統一する。
    形容詞が付いている場合（easy woman など）も含めて置換する。
    既にタグが付いていればスキップ。
    """
    if not WOMAN_TAG:
        return text
    def _replace(m: re.Match) -> str:
        full = m.group(0)
        # 既にタグが先行していれば置換しない
        start = m.start()
        preceding = text[max(0, start - len(WOMAN_TAG) - 1):start].lower()
        if WOMAN_TAG.lower() in preceding:
            return full
        return f"{WOMAN_TAG} woman"
    return _WOMAN_LABEL_RE.sub(_replace, text)


def apply_woman_tag(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """WOMAN設定時に obj 要素の label / description を補正する。"""
    if not WOMAN_TAG:
        return elements
    for elem in elements:
        if elem.get("type") != "obj":
            continue
        if elem.get("label"):
            elem["label"] = _patch_woman(elem["label"])
        if elem.get("description"):
            elem["description"] = _patch_woman(elem["description"])
    return elements



def run_paddle_ocr(image: Image.Image, request_id: str) -> list[dict[str, Any]]:
    """PaddleOCR でテキスト領域を検出する。2.x / 3.x API 両対応。"""
    ocr = get_paddle_ocr(request_id)
    if ocr is None:
        return []

    img_array = np.asarray(image)
    raw_lines: list[tuple] = []

    with progress_stage(request_id, "Running PaddleOCR"):
        if hasattr(ocr, "predict"):
            result_list = ocr.predict(img_array)
            for res in (result_list or []):
                for text, score, poly in zip(
                    res.get("rec_texts") or [],
                    res.get("rec_scores") or [],
                    res.get("rec_polys") or [],
                ):
                    raw_lines.append((poly, text, float(score)))
        else:
            results = ocr.ocr(img_array)
            for line in (results[0] or []):
                quad, (text, score) = line
                raw_lines.append((quad, text, float(score)))

    raw_lines = merge_ocr_lines(raw_lines, image.width, image.height)

    elements: list[dict[str, Any]] = []
    for quad, text, confidence in raw_lines:
        if confidence < 0.5:
            continue
        text = (text or "").strip()
        if not text:
            continue
        pts = np.asarray(quad, dtype=float).reshape(-1, 2)
        xs, ys = pts[:, 0].tolist(), pts[:, 1].tolist()
        bbox = normalize_bbox_xyxy([min(xs), min(ys), max(xs), max(ys)], image.width, image.height)
        if bbox_area(bbox) <= 20:
            continue
        elements.append({
            "id": f"item-ocr-{len(elements) + 1}",
            "type": "text",
            "label": text,
            "text": text,
            "description": f'text "{text}"',
            "bbox": bbox,
            "color": sample_color(image, bbox),
        })

    log_progress(request_id, f"PaddleOCR found {len(elements)} text regions")
    return elements


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------

def expand_text_bbox(bbox: list[int]) -> list[int]:
    """テキストbboxに余白を付与する（app.jsのexpandTextBboxと同一ロジック）。

    PaddleOCRのbboxはテキストの実描画幅ぴったりを返すため、
    Ideogramがbbox幅にテキストを収縮させて端が切れる問題を防ぐ。
    文字高さに応じた余白を全方向に付与し、左端テキスト（xmin < 50）は
    さらにxmaxを元の値の1.8倍（最大600）まで広げる。
    """
    ymin, xmin, ymax, xmax = bbox
    h = ymax - ymin
    pad_x = max(10, round(h * 0.4))
    pad_y = max(4, round(h * 0.15))
    xmin = max(0, xmin - pad_x)
    xmax = min(1000, xmax + pad_x)
    ymin = max(0, ymin - pad_y)
    ymax = min(1000, ymax + pad_y)
    if xmin < 50:
        xmax = max(xmax, min(int(xmax * 1.8), 600))
    return [ymin, xmin, ymax, xmax]


def build_ideogram_json(
    caption: str,
    background: str,
    elements: list[dict[str, Any]],
    width: int = 0,
    height: int = 0,
) -> dict[str, Any]:
    clean_caption = caption.strip() or "Uploaded image scene."
    bg = background.strip() or "Background and setting inferred from the uploaded image."
    ordered = []
    for idx, item in enumerate(elements, start=1):
        item_type = item.get("type") if item.get("type") in {"obj", "text"} else "obj"
        bbox = [int(v) for v in item["bbox"][:4]]
        desc = (item.get("description") or item.get("label") or f"object {idx}").strip()
        if item_type == "text":
            bbox = expand_text_bbox(bbox)
            text = (item.get("text") or item.get("label") or desc).strip()
            ordered.append({"type": "text", "bbox": bbox, "text": text, "desc": desc or f'text "{text}"'})
        else:
            ordered.append({"type": "obj", "bbox": bbox, "desc": desc})
    result: dict[str, Any] = {}
    if width > 0 and height > 0:
        result["aspect_ratio"] = detect_aspect_ratio(width, height)
    result["high_level_description"] = clean_caption
    result["compositional_deconstruction"] = {"background": bg, "elements": ordered}
    return result


# ---------------------------------------------------------------------------
# Main parse pipeline
# ---------------------------------------------------------------------------

def parse_image(image: Image.Image, request_id: str) -> tuple[str, str, list[dict[str, Any]], bool]:
    # 1. PaddleOCR: テキスト領域検出
    ocr_elements = run_paddle_ocr(image, request_id)

    # 2. Vision LLM: obj検出 + caption + background
    vision_result = vision_analyze(image, ocr_elements, request_id)
    vision_ok = bool(vision_result)

    caption    = vision_result.get("caption", "")
    background = vision_result.get("background", "")
    obj_elements = apply_woman_tag(vision_result.get("elements", []))

    if WOMAN_TAG:
        if caption:
            caption = _patch_woman(caption)
        if background:
            background = _patch_woman(background)

    if not caption:
        caption = "Uploaded image scene."
    if not background:
        background = caption

    # 3. obj + text を結合してソート（上→下、左→右）
    all_elements = obj_elements + ocr_elements
    seen: list[dict[str, Any]] = []
    for element in sorted(all_elements, key=lambda e: (e["bbox"][0], e["bbox"][1])):
        duplicate = any(
            bbox_iou(element["bbox"], other["bbox"]) > 0.85
            and element.get("label") == other.get("label")
            for other in seen
        )
        if not duplicate:
            element["id"] = f"item-{len(seen) + 1}"
            seen.append(element)

    return caption, background, seen[:50], vision_ok


def mock_parse(image: Image.Image) -> tuple[str, str, list[dict[str, Any]]]:
    caption = "Four cats resting together on a sofa in a softly lit room."
    boxes = [[290, 70, 720, 300], [260, 290, 760, 510], [270, 500, 740, 730], [300, 700, 760, 930]]
    elements = []
    for idx, bbox in enumerate(boxes, start=1):
        elements.append({
            "id": f"item-{idx}", "type": "obj",
            "label": f"cat {idx}", "description": f"cat {idx}",
            "bbox": bbox, "color": sample_color(image, bbox),
        })
    return caption, "A sofa and room background behind the four cats.", elements


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mock": MOCK_MODE,
        "ocr_lang": OCR_LANG,
        "paddle_ocr_loaded": paddle_ocr is not None,
        "vision_llm": {
            "enabled": vision_llm_available(),
            "api_base": VISION_API_BASE or None,
            "model": VISION_MODEL or None,
        },
    }


@app.get("/api/text-presets")
def text_presets() -> dict[str, Any]:
    """Textタブ用プリセット一覧（app/md/_*.md）を返す。"""
    return {"presets": [{"id": pid, "label": preset_label(pid)} for pid in TEXT_PRESETS]}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    request_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()
    log_progress(request_id, f"Analysis request received: {file.filename or 'uploaded image'}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        with progress_stage(request_id, "Reading uploaded image"):
            image = load_image(await file.read())
        log_progress(request_id, f"Image size: {image.width}x{image.height}")

        if MOCK_MODE:
            with progress_stage(request_id, "Running mock parser"):
                caption, background, elements = mock_parse(image)
            vision_ok = True
        else:
            caption, background, elements, vision_ok = parse_image(image, request_id)

        with progress_stage(request_id, "Building Ideogram JSON"):
            palette = dominant_palette(image)
            prompt_json = build_ideogram_json(caption, background, elements, image.width, image.height)

    except HTTPException:
        raise
    except Exception as exc:
        log_progress(request_id, f"Analysis failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    elapsed = time.perf_counter() - started_at
    log_progress(request_id, f"Analysis complete: {len(elements)} elements in {elapsed:.1f}s")

    return JSONResponse({
        "image": {"width": image.width, "height": image.height},
        "vision_model": VISION_MODEL or None,
        "vision_ok": vision_ok,
        "caption": caption,
        "background": background,
        "palette": palette,
        "elements": elements,
        "json": prompt_json,
    })


@app.post("/api/generate")
async def generate(payload: dict[str, Any]) -> JSONResponse:
    """テキスト説明からIdeogram JSONを生成する（Text to Promptタブ用）。"""
    request_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()
    description = str(payload.get("description") or "").strip()
    aspect_ratio = str(payload.get("aspect_ratio") or "1:1").strip()
    preset_id = str(payload.get("preset") or "").strip() or None
    if not description:
        raise HTTPException(status_code=400, detail="Enter a description.")
    log_progress(request_id, f"Generate request received ({aspect_ratio}): {description[:60]}")

    if MOCK_MODE:
        result: dict[str, Any] = {
            "caption": f"Mock layout for: {description[:80]}",
            "background": "Mock background and lighting.",
            "elements": [
                {"id": "item-1", "type": "obj", "bbox": [100, 100, 800, 550], "label": "subject",
                 "description": description[:120] or "subject", "color": "#BDBAB2"},
                {"id": "item-2", "type": "text", "bbox": [820, 150, 900, 850], "label": "SAMPLE",
                 "text": "SAMPLE", "description": 'text "SAMPLE"', "color": "#BDBAB2"},
            ],
        }
    else:
        if not vision_llm_available():
            raise HTTPException(status_code=503, detail="Vision LLM is not configured.")
        result = text_generate(description, aspect_ratio, request_id, preset_id=preset_id)
        if not result:
            raise HTTPException(status_code=502, detail="Vision LLM generation failed. Check the server log.")

    # WOMANタグ補正は適用しない（ユーザー自身が書いた説明文が入力のため）
    caption = result.get("caption") or "Generated scene."
    background = result.get("background") or caption
    elements = result.get("elements", [])

    # アスペクト比文字列からbuild_ideogram_json用の仮想サイズを作る
    try:
        ar_w, ar_h = (int(v) for v in aspect_ratio.split(":"))
    except ValueError:
        ar_w, ar_h = 1, 1
    prompt_json = build_ideogram_json(caption, background, elements, ar_w, ar_h)

    elapsed = time.perf_counter() - started_at
    log_progress(request_id, f"Generation complete: {len(elements)} elements in {elapsed:.1f}s")

    return JSONResponse({
        "image": None,
        "vision_model": VISION_MODEL or None,
        "vision_ok": True,
        "caption": caption,
        "background": background,
        "palette": [],
        "elements": elements,
        "json": prompt_json,
    })


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store, max-age=0"})


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("PORT", "7860")), type=int)
    args = parser.parse_args()
    print(f"Image to Prompt v3 running at http://{args.host}:{args.port}", flush=True)
    if vision_llm_available():
        print(f"Vision LLM : {VISION_API_BASE} / {VISION_MODEL}", flush=True)
    else:
        print("Vision LLM : not configured", flush=True)
    print(f"OCR        : PaddleOCR (lang={OCR_LANG})", flush=True)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
