"""Geometry-only centering engine for TCG Quest.

Inner border is provided as four reliable points detected by a separate algorithm
or adjusted by the user. No PSA grade or unvalidated AI confidence is produced.
"""
from dataclasses import dataclass
import numpy as np
from card_geometry import (CardGeometryError, order_corners,
                           map_image_points_to_normalized, border_measurements)

@dataclass
class CenteringResult:
    horizontal: dict
    vertical: dict
    inner_corners_normalized: list
    outer_corners_normalized: list
    source: str


def measure_centering(normalized_card, inner_corners, *, coordinate_space='original',
                      source='manual') -> CenteringResult:
    """Measure normalized inner border margin pairs.

    inner_corners: TL,TR,BR,BL points of printed border; either original upload
    pixel coordinates or normalized 1000x1400 image pixel coordinates.
    We evaluate margins at edge midpoints. For strong curved/skewed printing,
    separate edge-model fitting is needed; don't report this as PSA precision.
    """
    h, w = normalized_card.image.shape[:2]
    if coordinate_space == 'original':
        pts = map_image_points_to_normalized(inner_corners, normalized_card.homography)
    elif coordinate_space == 'normalized':
        pts = np.asarray(inner_corners, dtype=np.float32)
    else:
        raise CardGeometryError('coordinate_space must be original or normalized')
    inner = order_corners(pts)
    tl, tr, br, bl = inner
    left_x = (tl[0]+bl[0])/2
    right_x = (tr[0]+br[0])/2
    top_y = (tl[1]+tr[1])/2
    bottom_y = (bl[1]+br[1])/2
    margins = [left_x, (w-1)-right_x, top_y, (h-1)-bottom_y]
    if any(not np.isfinite(v) or v < 0 for v in margins):
        raise CardGeometryError('Printed border lies outside physical card; adjust corners')
    if right_x <= left_x or bottom_y <= top_y:
        raise CardGeometryError('Inner border is reversed')
    if min((right_x-left_x)/w, (bottom_y-top_y)/h) < 0.2:
        raise CardGeometryError('Printed border is implausibly small; review selection')
    horizontal = border_measurements(w-1, margins[0], margins[1])
    vertical = border_measurements(h-1, margins[2], margins[3])
    horizontal.update({'left_px_normalized':round(float(margins[0]),2),
                       'right_px_normalized':round(float(margins[1]),2)})
    vertical.update({'top_px_normalized':round(float(margins[2]),2),
                     'bottom_px_normalized':round(float(margins[3]),2)})
    return CenteringResult(horizontal,vertical,inner.tolist(),
                           [[0.,0.],[float(w-1),0.],[float(w-1),float(h-1)],[0.,float(h-1)]],source)
