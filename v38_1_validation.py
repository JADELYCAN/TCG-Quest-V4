"""TCG Quest V38.1 research-only repeatability and landmark diagnostic.

Compares two independently photographed Pokémon backs using separately reviewed
physical cut corners. Only verifies consistency of *experimental* printed-frame
measurements. Catalog landmark projections on the Giratina front are NOT ideal
print-center positions, independent detections, or centering percentages.
"""
from __future__ import annotations
import math
from typing import Any

from v38_centering_lab import (PILOT_ID, WARP_WIDTH, WARP_HEIGHT, _read_photo,
                               _reviewed_points, _back_measure, _encode_overlay,
                               analyze_v38)

# Approximate catalog *regions*, not calibrated physical coordinates. These
# positions must never be used for deriving a grade or a 50/50 claim.
REFERENCE_REGIONS = (
    ('Upper-left card label', (0.125, 0.073)),
    ('Upper-right HP region', (0.820, 0.073)),
    ('Lower-left card number', (0.205, 0.950)),
    ('Bottom copyright region', (0.510, 0.985)),
)


def _front_landmark_preview(front_raw: bytes, reviewed_corners: Any,
                            projected_template_quad: Any) -> dict:
    import cv2
    import numpy as np
    img = _read_photo(front_raw)
    h, w = img.shape[:2]
    physical = _reviewed_points(reviewed_corners, w, h, 'front')
    template_quad = np.asarray(projected_template_quad, dtype=np.float32)
    if template_quad.shape != (4, 2) or not np.isfinite(template_quad).all():
        raise ValueError('Catalog alignment did not return four usable corners.')
    template_px = (template_quad * np.array([w, h], np.float32)).astype(np.float32)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    to_photo = cv2.getPerspectiveTransform(square, template_px)
    card_coords = np.array([[0, 0], [WARP_WIDTH-1, 0],
                            [WARP_WIDTH-1, WARP_HEIGHT-1], [0, WARP_HEIGHT-1]], np.float32)
    to_card = cv2.getPerspectiveTransform(physical, card_coords)
    overlay = img.copy()
    cv2.polylines(overlay, [np.round(physical).astype(np.int32)], True, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.polylines(overlay, [np.round(template_px).astype(np.int32)], True, (0, 194, 255), 2, cv2.LINE_AA)
    anchors = []
    for i, (label, coords) in enumerate(REFERENCE_REGIONS):
        projected = cv2.perspectiveTransform(np.array([[coords]], np.float32), to_photo)[0, 0]
        in_card = cv2.perspectiveTransform(np.array([[projected]], np.float32), to_card)[0, 0]
        if not np.isfinite(projected).all() or not np.isfinite(in_card).all():
            raise ValueError('Catalog landmark projection was numerically unstable.')
        px, py = map(lambda v: int(round(float(v))), projected)
        cv2.circle(overlay, (px, py), 8, (45, 255, 192), -1, cv2.LINE_AA)
        cv2.putText(overlay, str(i+1), (px+11, py-9), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, str(i+1), (px+11, py-9), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (10, 70, 40), 1, cv2.LINE_AA)
        anchors.append({'name': label, 'catalog_region_normalized': list(coords),
                        'location_in_reviewed_cut_frame': [round(float(in_card[0]) / (WARP_WIDTH-1), 4),
                                                           round(float(in_card[1]) / (WARP_HEIGHT-1), 4)]})
    return {'status': 'catalog_region_projection_only', 'calibrated': False,
            'ratio': None, 'centering_grade': None,
            'explanation': 'Green markers are approximate catalog-region projections derived from the overall image match. '
                           'They are NOT independent detections of printed landmark boundaries, and the catalog is NOT a calibrated ideal-print master.',
            'projected_regions': anchors, 'overlay_data_uri': _encode_overlay(overlay)}


def _first(b: dict | None) -> float | None:
    if b is None: return None
    x = b.get('first_pct')
    return float(x) if isinstance(x, (int,float)) and math.isfinite(x) else None


def compare_v38_1(front_raw: bytes, back_a_raw: bytes, back_b_raw: bytes,
                  front_corners: Any, back_a_corners: Any, back_b_corners: Any,
                  card_id: str = PILOT_ID) -> dict:
    """Compare independent back photos; never returns grades or verified front centering."""
    if card_id != PILOT_ID:
        raise ValueError('V38.1 pilot is limited to Giratina V Lost Origin #186.')
    # Same exact file is not a second independent photograph.
    import hashlib
    if hashlib.sha256(back_a_raw).digest() == hashlib.sha256(back_b_raw).digest():
        raise ValueError('Upload a different original back photograph for the repeatability test; identical files do not count.')
    primary = analyze_v38(front_raw, back_a_raw, front_corners, back_a_corners, card_id)
    if primary['front'].get('status') != 'catalog_artwork_aligned':
        return {'status': 'withheld', 'reason': 'Exact front artwork matching failed.',
                'front': primary['front'], 'back_a': primary['back'], 'back_b': None,
                'repeatability': None, 'calibrated': False, 'grade': None}
    im_b = _read_photo(back_b_raw)
    h, w = im_b.shape[:2]
    quad = _reviewed_points(back_b_corners, w, h, 'back second photograph')
    secondary = _back_measure(im_b, quad)
    a, b = primary['back'], secondary
    repeatability = {'status': 'not_comparable', 'lr_difference_pp': None,
                     'tb_difference_pp': None, 'calibrated': False,
                     'explanation': 'At least one back photo has no supported printed-frame measurement.'}
    if a.get('status') == b.get('status') == 'experimental_measurement':
        lr = abs(_first(a.get('lr')) - _first(b.get('lr')))
        tb = abs(_first(a.get('tb')) - _first(b.get('tb')))
        repeatability = {'status': 'compared', 'lr_difference_pp': round(lr, 2),
                         'tb_difference_pp': round(tb, 2), 'calibrated': False,
                         'interpretation': 'Differences between independently photographed measurements, not proven measurement accuracy. '
                                           'No validated pass/fail threshold or centering grade exists.',
                         'source': 'two independently marked, perspective-rectified blue-frame estimates'}
    front_preview = _front_landmark_preview(front_raw, front_corners, primary['front']['projected_template_quad'])
    return {'status': 'research_only', 'pilot_card_id': PILOT_ID, 'back_a': a, 'back_b': b,
            'front': front_preview, 'repeatability': repeatability, 'calibrated': False,
            'grade': None, 'grade_updated': False, 'verified_front_lr': None,
            'verified_front_tb': None, 'centering_category_score': None,
            'limitations': 'Photographs must be independent and their review points accurate. '
                           'Catalog projections cannot establish an ideal full-art print center; two agreeing back photos can still share systematic error.',
            'privacy_note': 'Images are processed in memory by the research endpoint and not saved by this feature.'}
