"""TCG Quest optional numeric score API for a Flask AI backend.

Register once in your EXISTING Flask application:
    from grade_scores_blueprint import score_blueprint
    app.register_blueprint(score_blueprint)

POST /api/cards/grade/scores: multipart form front/back JPEG, PNG, or WebP.
Returns validated AI-estimated category scores or null; it does NOT fake results.
Requires OPENAI_API_KEY on Render. Keep that key server-side only.
"""
from __future__ import annotations

import base64
import json
import os
import time
import math
import io
import urllib.error
import urllib.request
from collections import defaultdict, deque
from threading import Lock

from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

score_blueprint = Blueprint("tcg_category_scores", __name__)
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_FORM_BYTES = 16 * 1024 * 1024 + 65536
_RATE_LIMIT_HOURLY = max(1, int(os.getenv("TCG_SCORE_HOURLY_LIMIT", "6")))
_calls = defaultdict(deque)
_lock = Lock()
_KEYS = ("centering", "corners", "edges", "surface")


def _image_data_url(upload):
    """Decode and resize physical-card photos before sending them to AI.

    The image isn't interpreted as a catalog card and never leaves the
    backend except as a grading input to the selected AI model.
    """
    if upload is None:
        raise ValueError("Both photos are required")
    data = upload.read(_MAX_IMAGE_BYTES + 1)
    if not data or len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("Each photo must be 8 MB or smaller")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Unsupported photo format")
            width, height = source.size
            if min(width, height) < 100 or width * height > 20_000_000:
                raise ValueError("Photo resolution is too small or large")
            source.load()
            img = ImageOps.exif_transpose(source)
            img.thumbnail((1536, 1536))
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=88)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid photo") from error
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _validated_score(value):
    """A float in 1-10 at 0.5 increments, or None if AI output is invalid."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not 1 <= value <= 10:
        return None
    if abs(value * 2 - round(value * 2)) > 1e-7:
        return None
    return float(value)


def _score(value):
    # Backwards-compatible normalization name from earlier add-on tests.
    return _validated_score(value)


def _validated_report(raw):
    if not isinstance(raw, dict):
        raise ValueError("AI result was not a JSON object")
    scores = raw.get("category_scores")
    if not isinstance(scores, dict):
        raise ValueError("AI did not include category_scores")
    result = {key: _score(scores.get(key)) for key in _KEYS}
    notes = raw.get("score_notes") or {}
    notes = {key: str(notes.get(key, ""))[:700] for key in _KEYS} if isinstance(notes, dict) else {}
    return {"grading": {"category_scores": result},
            "score_notes": notes,
            "photo_quality": str(raw.get("photo_quality", ""))[:700],
            "source": "AI inspection of collector-uploaded front and back photos",
            "unofficial": True}


def _allowed_origin():
    origin = request.headers.get("Origin", "")
    configured = os.getenv("TCG_SCORE_ALLOWED_ORIGINS", "https://tcg-quest-v8-test.onrender.com")
    allowed = {x.strip().rstrip("/") for x in configured.split(",") if x.strip()}
    return origin if origin in allowed else None


def _rate_limited():
    # Application-level soft limit. Use user auth and managed/global rate limiting
    # before opening a paid API service to the public.
    identity = request.remote_addr or "unknown"
    now = time.monotonic()
    with _lock:
        attempts = _calls[identity]
        while attempts and attempts[0] < now - 3600:
            attempts.popleft()
        if len(attempts) >= _RATE_LIMIT_HOURLY:
            return True
        attempts.append(now)
        return False


@score_blueprint.route("/api/cards/grade/scores", methods=["POST", "OPTIONS"])
def grade_category_scores():
    origin = _allowed_origin()
    headers = {"Vary": "Origin"}
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        headers["Access-Control-Allow-Headers"] = "Content-Type"
    if request.method == "OPTIONS":
        return ("", 204, headers)
    if request.headers.get("Origin") and not origin:
        return jsonify(error="This origin is not permitted"), 403, headers
    if request.content_length and request.content_length > _MAX_FORM_BYTES:
        return jsonify(error="Images exceed total upload limit of 16 MB"), 413, headers
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return jsonify(error="Server-side OpenAI API key not configured"), 503, headers
    try:
        front = _image_data_url(request.files.get("front"))
        back = _image_data_url(request.files.get("back"))
    except (ValueError, AttributeError):
        return jsonify(error="Upload clear front and back photos (JPEG/PNG/WebP, up to 8 MB each)"), 400, headers
    if _rate_limited():
        return jsonify(error="Score requests are temporarily rate-limited"), 429, headers

    instructions = (
        "You are estimating the visible physical condition of ONE collectible trading card "
        "using TWO collector-uploaded photos (FRONT then BACK). This is an experimental "
        "estimate, not an official PSA/BGS/CGC grade, and not a guarantee. "
        "Inspect centering, corners, edges, and surface independently. Evaluate front AND "
        "back for each where visible. Score each category 1 to 10, or null when details "
        "are insufficient (e.g. poor resolution, glare, sleeves, obscured edges, no measurable "
        "print borders). Scores must be evidence-based from these photos, not inferred from "
        "the overall grade and not from catalog artwork. Do not assume unseen scratches "
        "or automatically award 10. Borderless cards often require null for centering. "
        "Return ONLY one JSON object in exactly this shape: "
        '{"category_scores":{"centering":null,"corners":null,"edges":null,"surface":null},'
        '"score_notes":{"centering":"","corners":"","edges":"","surface":""},'
        '"photo_quality":""}. '
        "Use JSON numeric values such as 8 or 8.5, not strings; null if unknown. "
        "Be conservative about uncertainty. Say briefly what physical evidence supports "
        "each non-null score and explain why a null cannot be assessed."
    )
    body = {
        "model": os.getenv("TCG_SCORE_MODEL", os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")),
        "response_format": {"type": "json_object"},
        "max_tokens": 650,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": instructions + " First image is FRONT; second is BACK."},
            {"type": "image_url", "image_url": {"url": front, "detail": "high"}},
            {"type": "image_url", "image_url": {"url": back, "detail": "high"}},
        ]}],
    }
    call = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(call, timeout=35) as reply:
            outer = json.loads(reply.read())
        content = outer["choices"][0]["message"]["content"]
        report = _validated_report(json.loads(content))
        return jsonify(report), 200, headers
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        # Don't leak provider request body or any key to browser logs.
        return jsonify(error="AI provider unavailable; check backend service logs and credits"), 502, headers
    except (ValueError, KeyError, IndexError, TypeError):
        return jsonify(error="AI did not return a reliable structured category report"), 502, headers
