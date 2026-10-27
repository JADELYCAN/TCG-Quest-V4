"""TCG Quest geometry module. Does not claim to grade card condition.

Canonical input: decoded, EXIF-corrected BGR image. Corner order TL, TR, BR, BL.
"""
from __future__ import annotations
import io
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

TARGET = (1000, 1400)  # standard 2.5 x 3.5 card only
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SOURCE_PIXELS = 45_000_000
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS

class CardGeometryError(ValueError):
    pass

@dataclass
class NormalizedCard:
    image: np.ndarray
    corners: np.ndarray
    homography: np.ndarray
    method: str
    source_card_pixels: float


def decode_upload(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise CardGeometryError('Empty image or file exceeds 25 MB')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_SOURCE_PIXELS:
                raise CardGeometryError('Image dimensions exceed limit')
            corrected = ImageOps.exif_transpose(image)
            rgb = corrected.convert('RGB')
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CardGeometryError(f'Image cannot be decoded: {exc}') from exc


def order_corners(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise CardGeometryError('Exactly four finite corners required')
    hull = cv2.convexHull(p).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 100:
        raise CardGeometryError('Corners must form a nondegenerate convex quadrilateral')
    center = p.mean(axis=0)
    theta = np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    p = p[np.argsort(theta)]
    start = np.argmin(p[:, 0] + p[:, 1])
    p = np.roll(p, -int(start), axis=0)
    # Standard image coordinates: TL -> TR -> BR -> BL
    if p[1, 0] < p[3, 0]:
        p = p[[0, 3, 2, 1]]
    return p


def detect_card_corners(image: np.ndarray) -> np.ndarray:
    """Conservative classical-CV detector. Raises to trigger manual corner fallback."""
    if image is None or image.ndim != 3:
        raise CardGeometryError('Expected color image')
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 130)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        coverage = area / (width * height)
        if not 0.06 <= coverage <= 0.96:
            continue
        poly = cv2.approxPolyDP(contour, 0.022 * cv2.arcLength(contour, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        corners = order_corners(poly.reshape(4, 2))
        edge_lengths = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
        if min(edge_lengths) < 35:
            continue
        # Rank by plausible coverage only; do not silently accept ambiguity.
        candidates.append((area, corners))
    if not candidates:
        raise CardGeometryError('No reliable card outline; use four-corner editor')
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > candidates[0][0] * 0.83:
        # Nested artwork rectangles may be more than one candidate. A manual review is safer.
        if np.linalg.norm(candidates[1][1] - candidates[0][1], axis=1).mean() > 12:
            raise CardGeometryError('Ambiguous outlines; confirm corners manually')
    return candidates[0][1]


def normalize_card(image: np.ndarray, corners=None, size=TARGET) -> NormalizedCard:
    if image is None or image.ndim != 3:
        raise CardGeometryError('Expected decoded BGR image')
    width, height = size
    if width < 100 or height < 100:
        raise CardGeometryError('Invalid target dimensions')
    method = 'automatic' if corners is None else 'manual'
    ordered = detect_card_corners(image) if corners is None else order_corners(corners)
    polygon_area = abs(cv2.contourArea(ordered))
    dest = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(ordered, dest)
    if not np.isfinite(H).all():
        raise CardGeometryError('Perspective mapping failed')
    warped = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_LINEAR)
    return NormalizedCard(warped, ordered, H, method, polygon_area)


def map_image_points_to_normalized(points, homography):
    """Map original-image editor coordinates into normalized-card coordinates."""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, homography).reshape(-1, 2)


def border_balance(left: float, right: float) -> dict:
    """Returns L/R border balance without inventing a professional grade."""
    if min(left, right) < 0 or left + right <= 0:
        raise CardGeometryError('Border widths must be nonnegative and nonzero in total')
    l = 100 * left / (left + right)
    return {'left_percent': round(l, 3), 'right_percent': round(100-l, 3)}


def border_measurements(outer_width: float, left: float, right: float) -> dict:
    if outer_width <= 0 or left + right > outer_width:
        raise CardGeometryError('Invalid card geometry')
    return {**border_balance(left, right),
            'left_of_card_percent': round(left / outer_width * 100, 3),
            'right_of_card_percent': round(right / outer_width * 100, 3)}
