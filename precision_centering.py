"""TCG Quest V32: conservative photographed-card centering measurements.

Image-processing, NOT model-generated ratios. Measurements require both physical
card outline and four repeatable printed-border transitions. In ambiguous
full-art / borderless cases we return 'unavailable'. Coordinates use the image
supplied to /api/cards/grade after EXIF orientation and JPEG downscaling.
Dependencies: opencv-python-headless and numpy. Missing dependencies fail closed.
"""
from __future__ import annotations
import base64
import io
import math
from typing import Any


def _unavailable(reason: str) -> dict[str, Any]:
    return {'status': 'unavailable', 'method': 'v32_cv_geometry',
            'reason': str(reason)[:200], 'lr': None, 'tb': None,
            'outer_quad': None, 'inner_quad': None,
            'confidence': 'low', 'border_pixels': None,
            'units': 'rectified_image_pixels', 'image_size': None}


def _ordered(pts, np):
    """Top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, dtype=np.float32)
    sums = p.sum(axis=1)
    diffs = p[:, 0] - p[:, 1]
    ordered = np.array([p[np.argmin(sums)], p[np.argmax(diffs)],
                        p[np.argmax(sums)], p[np.argmin(diffs)]], dtype=np.float32)
    if len({(round(float(x)), round(float(y))) for x, y in ordered}) != 4:
        return None
    return ordered


def _physical_quad(bgr, cv2, np):
    """Find outer card with multiple independent mask thresholds and edge cues.

    A dominant centered rectangular foreground object is required. This is
    NOT a border or defect detector; it declines ambiguous images rather than
    assume that the largest high-contrast item is always a card.
    """
    h, w = bgr.shape[:2]
    if min(h, w) < 350:
        return None, 'Photo resolution too low for boundary measurement.'
    scale = min(1., 1100 / max(h, w))
    small = cv2.resize(bgr, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch = max(9, int(min(sw, sh)*.048))
    patches = [lab[:patch, :patch], lab[:patch, -patch:], lab[-patch:, :patch], lab[-patch:, -patch:]]
    bg_points = np.concatenate([x.reshape(-1,3) for x in patches])
    bg = np.median(bg_points,axis=0)
    dev = np.linalg.norm(lab-bg,axis=2)
    # LAB background variance controls sensitivity: white, gray, colored matte,
    # and slightly uneven backgrounds get several chances at segmentation.
    bg_noise = float(np.median(np.linalg.norm(bg_points-bg,axis=1)))
    thresholds = [max(17, bg_noise*2.0+9), max(27,bg_noise*2.8+12), max(38,bg_noise*3.8+16)]
    kernel = np.ones((5,5),np.uint8)
    masks=[]
    for thresh in thresholds:
        mask=(dev>thresh).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        masks.append(mask)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    # An independent outline candidate is useful when the card and background
    # are similar colors but a narrow edge is still visible.
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),45,120)
    edges=cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=2)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    masks.append(edges)
    options=[]
    for mask_id,mask in enumerate(masks):
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if not (.14*sw*sh<area<.87*sw*sh):continue
            hull=cv2.convexHull(cnt)
            rect=cv2.minAreaRect(hull)
            rw,rh=rect[1]
            if min(rw,rh)<min(sw,sh)*.24:continue
            ratio=min(rw,rh)/max(rw,rh)
            if not (.59<ratio<.85):continue
            rect_area=rw*rh
            if rect_area<=0:continue
            # Reject amorphous high-contrast objects and broad shadows.
            fill=area/rect_area
            if not (.77<fill<1.07):continue
            x,y,bw,bh=cv2.boundingRect(hull)
            if min(x,y,sw-x-bw,sh-y-bh)<min(sw,sh)*.009:continue
            center=np.mean(cv2.boxPoints(rect),axis=0)
            offset=np.linalg.norm((center-np.array([sw/2,sh/2]))/np.array([sw,sh]))
            if offset>.31:continue
            approx=cv2.approxPolyDP(hull,.012*cv2.arcLength(hull,True),True)
            quad=_ordered(approx.reshape(4,2) if len(approx)==4 else cv2.boxPoints(rect),np)
            if quad is None:continue
            # Area and contour rectangularity favor the full card over artwork
            # embedded inside it, but this cannot prove a dark edge is correct.
            quality=(area/(sw*sh))*(.70+.30*fill)*(1-.6*offset)
            options.append((quality,quad,mask_id))
    if not options:
        return None,'Physical card outline not isolated. Use a contrasting matte background with all edges visible.'
    # Only accept a contour that has independent segmentation support when
    # multiple color thresholds generate plausible but conflicting boundaries.
    options.sort(key=lambda x:x[0],reverse=True)
    best=options[0]
    corners=best[1]
    comparable=[item for item in options if item[2]!=best[2]
        and np.mean(np.linalg.norm(item[1]-corners,axis=1))<min(sw,sh)*.025]
    if len(options)>=3 and not comparable and best[2]!=3:
        return None,'Card boundary was inconsistent across image segmentation methods; retake on a contrasting matte surface.'
    return corners/scale,None


def _border_transition(warp, side, cv2, np):
    """Find repeated inner print frame from independent scan lines.

    Returns (position in rectified pixels, reliability). We reject edges that
    do not appear in consistent locations along a majority of the card.
    """
    h, w = warp.shape[:2]
    lab = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB).astype(np.float32)
    vertical = side in ('left', 'right')
    length = w if vertical else h
    sample_count = 21
    transverse = (h if vertical else w)
    samples = np.linspace(.28 * transverse, .72 * transverse, sample_count).astype(int)
    direction = -1 if side in ('right', 'bottom') else 1
    lo, hi = int(.035 * length), int(.185 * length)
    if hi - lo < 15:
        return None
    profiles = []
    for at in samples:
        if vertical:
            stripe = np.median(lab[max(0, at - 2):min(h, at + 3), :, :], axis=0)
        else:
            stripe = np.median(lab[:, max(0, at - 2):min(w, at + 3), :], axis=1)
        if direction == -1:
            stripe = stripe[::-1]
        stripe = cv2.GaussianBlur(stripe.reshape(1, length, 3), (7, 1), 1.2).reshape(length, 3)
        delta = np.linalg.norm(stripe[4:] - stripe[:-4], axis=1)
        signal = np.zeros(length, dtype=np.float32)
        signal[2:-2] = delta
        profiles.append(signal)
    profiles = np.array(profiles)
    mean_energy = np.median(profiles, axis=0)
    search = mean_energy[lo:hi]
    if search.size < 12:
        return None
    # Median across independently sampled scan lines suppresses artwork texture.
    choice = lo + int(search.argmax())
    strength = float(mean_energy[choice])
    baseline = float(np.median(search))
    margin = max(4, round(length * .012))
    valid = 0
    picks = []
    for row in profiles:
        best = lo + int(row[lo:hi].argmax())
        if abs(best - choice) <= margin and row[best] >= max(9.0, baseline * 1.7):
            valid += 1
            picks.append(best)
    # Photographs may show a small amount of scanline jitter, but require a
    # stable straight printed boundary on at least ~2/3 of the central samples.
    if strength < 13 or strength < max(1.8 * baseline, 10) or valid < 14:
        return None
    position = float(np.median(picks))
    consistency = valid / sample_count
    return {'inset': position, 'strength': round(strength, 2),
            'consistency': round(consistency, 2)}


def measure_photo(encoded_jpeg: str) -> dict[str, Any]:
    """Run independent geometry on one prepared JPG; fail closed on uncertain images."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _unavailable('Image geometry module not installed (opencv-python-headless and numpy required).')
    try:
        raw = base64.b64decode(encoded_jpeg, validate=True)
        if len(raw) > 8 * 1024 * 1024:
            return _unavailable('Image exceeds centering processing limit.')
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return _unavailable('Image cannot be decoded.')
        height, width = bgr.shape[:2]
        quad, reason = _physical_quad(bgr, cv2, np)
        if quad is None:
            out = _unavailable(reason)
            out['image_size'] = [width, height]
            return out
        W, H = 700, 980
        target = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(quad, target)
        warp = cv2.warpPerspective(bgr, matrix, (W, H), flags=cv2.INTER_LINEAR)
        sides = {name: _border_transition(warp, name, cv2, np) for name in ('left', 'right', 'top', 'bottom')}
        lr = tb = None
        if sides['left'] and sides['right']:
            l, r = sides['left']['inset'], sides['right']['inset']
            if l > 2 and r > 2:
                lr = round(100 * l / (l + r), 1)
        if sides['top'] and sides['bottom']:
            t, b = sides['top']['inset'], sides['bottom']['inset']
            if t > 2 and b > 2:
                tb = round(100 * t / (t + b), 1)
        if lr is None and tb is None:
            status = 'unavailable'
            reason = 'The physical card was found, but a repeatable printed border was not measurable.'
        elif lr is None or tb is None:
            status = 'partial'
            reason = 'Only one axis had a consistent printed-border transition. Other measurement withheld.'
        else:
            status = 'measured'
            reason = 'Measured from repeated printed-border transitions after perspective correction.'
        if status != 'unavailable':
            # Inverse perspective maps the measured inner corners back to photo coordinates.
            left = sides['left']['inset'] if sides['left'] else None
            right = sides['right']['inset'] if sides['right'] else None
            top = sides['top']['inset'] if sides['top'] else None
            bottom = sides['bottom']['inset'] if sides['bottom'] else None
            if None not in (left, right, top, bottom):
                inner_rect = np.array([[[left, top]], [[W - 1 - right, top]],
                                       [[W - 1 - right, H - 1 - bottom]], [[left, H - 1 - bottom]]], dtype=np.float32)
                inner_native = cv2.perspectiveTransform(inner_rect, np.linalg.inv(matrix)).reshape(4, 2)
            else:
                inner_native = None
        else:
            inner_native = None
        def norm_quad(points):
            if points is None:
                return None
            return [[round(float(x / width), 6), round(float(y / height), 6)] for x, y in points]
        valid_infos = [s for s in sides.values() if s]
        confidence = 'high' if status == 'measured' and all(s['consistency'] >= .80 for s in valid_infos) else ('medium' if status != 'unavailable' else 'low')
        return {'status': status, 'method': 'v32_cv_geometry', 'reason': reason,
                'lr': lr, 'tb': tb, 'outer_quad': norm_quad(quad),
                'inner_quad': norm_quad(inner_native), 'image_size': [width, height],
                'border_pixels': {k: (round(v['inset'], 1) if v else None) for k, v in sides.items()},
                'confidence': confidence, 'units': 'rectified_image_pixels',
                'scanline_consistency': {k: (v['consistency'] if v else None) for k, v in sides.items()}}
    except (ValueError, TypeError, OverflowError) as exc:
        return _unavailable('Invalid photo data for measurement.')
    except Exception:
        # Fail closed: image analysis should never interrupt an otherwise good grade.
        return _unavailable('Image geometry could not be assessed; AI grading may still proceed.')


def measure_pair(front_b64: str, back_b64: str) -> dict[str, Any]:
    return {'front': measure_photo(front_b64), 'back': measure_photo(back_b64),
            'version': 2, 'method': 'v32_cv_geometry',
            'disclaimer': 'Photo-based estimates; not calibrated grader measurements. Full-art designs may not have measurable frame borders.'}
