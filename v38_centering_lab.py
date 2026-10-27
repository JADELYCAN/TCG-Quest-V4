"""V38 staging-only printed-artwork research, not a PSA/AI grading method.

Works ONLY with the pilot card (Giratina V, Lost Origin 186/196) and
collector-reviewed, normalized *physical* cut-edge corners. Front: exact-card
catalog registration, never pretends catalog crops are calibration masters.
Back: experimental blue-frame pixel measurements after perspective correction.
No grading updates, no saved images, no professional centering claim.
"""
from __future__ import annotations
import base64
import math
from typing import Any

PILOT_ID = 'swsh11-186'
MAX_PHOTO_BYTES = 8*1024*1024
MAX_PIXELS = 25_000_000
WARP_WIDTH, WARP_HEIGHT = 700, 980


def _read_photo(raw: bytes):
    import cv2
    import numpy as np
    if not isinstance(raw, bytes) or not (100 < len(raw) <= MAX_PHOTO_BYTES):
        raise ValueError('Each photograph must be a nonempty JPG, PNG, or WebP no larger than 8 MB.')
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or min(img.shape[:2]) < 350 or img.shape[0]*img.shape[1] > MAX_PIXELS:
        raise ValueError('Invalid photo or insufficient resolution (minimum side 350 px, maximum 25 MP).')
    scale = min(1., 1400/max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, (round(img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return img


def _reviewed_points(points: Any, width: int, height: int, side: str):
    """Do not silently reorder / autocorrect collector-marked points."""
    import cv2
    import numpy as np
    p = np.asarray(points, dtype=float)
    if p.shape != (4,2) or not np.isfinite(p).all():
        raise ValueError(f'{side}: supply four normalized cut corners, TL, TR, BR, BL.')
    if (p < .002).any() or (p > .998).any():
        raise ValueError(f'{side}: each corner must be strictly inside the photograph.')
    pix = (p * np.array([width,height], dtype=float)).astype(np.float32)
    top = np.linalg.norm(pix[1]-pix[0]); right = np.linalg.norm(pix[2]-pix[1]); bottom = np.linalg.norm(pix[2]-pix[3]); left = np.linalg.norm(pix[3]-pix[0])
    if min(top,right,bottom,left) < 100:
        raise ValueError(f'{side}: marked card is too small for printed-frame measurement.')
    area = float(cv2.contourArea(pix))
    if area < .13 * width * height or not cv2.isContourConvex(pix):
        raise ValueError(f'{side}: cut-corner outline is not a plausible convex card.')
    # 2.5 x 3.5 inch cards, allowing modest perspective and hand marks.
    aspect=(top+bottom)/max(left+right,1)
    if not .54 <= aspect <= .90:
        raise ValueError(f'{side}: card outline has an implausible shape or corner order.')
    return pix


def _encode_overlay(image):
    import cv2
    success, buf=cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 83])
    if not success:
        raise ValueError('Unable to render analysis overlay.')
    return 'data:image/jpeg;base64,'+base64.b64encode(buf).decode('ascii')


def _centering_balance(left, right):
    total=left+right
    if total <= 0: return None
    return {'first_pct':round(100*left/total,1), 'second_pct':round(100*right/total,1)}


def _back_measure(img, corners):
    import cv2
    import numpy as np
    from precision_centering import _pokemon_back_candidate, _pokemon_back_side, _pokemon_back_gradient_side
    t=np.array([[0,0],[WARP_WIDTH-1,0],[WARP_WIDTH-1,WARP_HEIGHT-1],[0,WARP_HEIGHT-1]],np.float32)
    m=cv2.getPerspectiveTransform(corners,t)
    warp=cv2.warpPerspective(img,m,(WARP_WIDTH,WARP_HEIGHT))
    if not _pokemon_back_candidate(warp,cv2,np):
        return {'status':'unavailable','reason':'This photograph does not confidently show the conventional Pokémon blue back design. No printed-frame ratio provided.',
                'centered_status':'not_assessed','lr':None,'tb':None,'confidence':'unavailable'}
    sides={side:(_pokemon_back_side(warp,side,cv2,np) or _pokemon_back_gradient_side(warp,side,cv2,np)) for side in ('left','right','top','bottom')}
    if not all(sides.values()):
        missing=[k for k,v in sides.items() if v is None]
        return {'status':'unavailable','reason':'Printed blue-frame boundary uncertain on '+', '.join(missing)+'. No ratio inferred.',
                'missing_sides':missing,'centered_status':'not_assessed','lr':None,'tb':None,'confidence':'unavailable'}
    inset={k:float(sides[k]['inset']) for k in sides}
    if not(12<min(inset.values()) and max(inset['left'],inset['right'])<WARP_WIDTH*.14 and max(inset['top'],inset['bottom'])<WARP_HEIGHT*.14):
        return {'status':'unavailable','reason':'Detected printed-frame widths are implausible. Measurement withheld.',
                'centered_status':'not_assessed','lr':None,'tb':None,'confidence':'unavailable'}
    low_confidence=any(sides[k].get('fallback') or sides[k]['consistency']<.73 for k in sides)
    frame=(round(inset['left']),round(inset['top']),round(WARP_WIDTH-1-inset['right']),round(WARP_HEIGHT-1-inset['bottom']))
    # Rectified image preview: photo is already warped to the user-reviewed physical boundary.
    overlay=warp.copy()
    cv2.rectangle(overlay,(2,2),(WARP_WIDTH-3,WARP_HEIGHT-3),(255,255,0),3)
    cv2.rectangle(overlay,frame[:2],frame[2:],(85,255,135),3)
    for side,(a,b) in {'left':((0,490),(frame[0],490)),'right':((frame[2],490),(WARP_WIDTH-1,490)),
                       'top':((350,0),(350,frame[1])),'bottom':((350,frame[3]),(350,WARP_HEIGHT-1))}.items():
        cv2.line(overlay,a,b,(40,205,255),3)
    return {'status':'experimental_measurement','method':'approved_cut_edge_plus_blue_printed_frame',
            'reason':'Experimental blue printed-frame measurement from collector-reviewed physical corners. Not calibrated for official grading.',
            'lr':_centering_balance(inset['left'],inset['right']),
            'tb':_centering_balance(inset['top'],inset['bottom']),
            'border_pixels':{k:round(v,1) for k,v in inset.items()},
            'frame_line_consistency':{k:round(float(v['consistency']),3) for k,v in sides.items()},
            'edge_detection_methods':{k:('gradient_fallback' if v.get('fallback') else 'blue_frame_scanlines') for k,v in sides.items()},
            'confidence':'low' if low_confidence else 'medium',
            'centered_status':'not_professionally_calibrated','units':'rectified_photo_pixels',
            'rectified_dimensions':[WARP_WIDTH,WARP_HEIGHT],
            'overlay_data_uri':_encode_overlay(overlay)}


def analyze_v38(front_raw: bytes, back_raw: bytes, front_corners: Any, back_corners: Any,
                card_id: str=PILOT_ID) -> dict:
    """Research-only. Never return an overall/card grade or validated front ratio."""
    import cv2
    import numpy as np
    if card_id != PILOT_ID:
        raise ValueError('V38 research pilot supports Giratina V, Lost Origin 186/196 only.')
    front=_read_photo(front_raw)
    back=_read_photo(back_raw)
    fh,fw=front.shape[:2];bh,bw=back.shape[:2]
    fquad=_reviewed_points(front_corners,fw,fh,'front')
    bquad=_reviewed_points(back_corners,bw,bh,'back')
    # Verify the exact artwork independently; no need to trust the declared card_id.
    from template_alignment_v34 import inspect_photo
    matching=inspect_photo(front_raw,card_id,include_overlay=False)
    front_view=front.copy()
    cv2.polylines(front_view,[np.round(fquad).astype(np.int32)],True,(255,255,0),3,cv2.LINE_AA)
    front_result={'status':'unavailable','method':'catalog_registration_without_calibrated_master',
                  'lr':None,'tb':None,'calibrated':False,
                  'explanation':'Full-art fronts do not have a verified physical printed-frame reference. No centering percentage or score can be justified by this catalog thumbnail.'}
    if matching.get('status')=='artwork_aligned_only':
        target=np.array(matching['projected_template_quad'],np.float32)*np.array([fw,fh],np.float32)
        cv2.polylines(front_view,[np.round(target).astype(np.int32)],True,(0,202,255),3,cv2.LINE_AA)
        front_result.update(status='catalog_artwork_aligned',
                            exact_card_match=True,feature_inliers=matching.get('inliers'),
                            matching_fraction=matching.get('inlier_fraction'),
                            projected_template_quad=matching['projected_template_quad'])
    else:
        front_result.update(status='template_match_withheld',exact_card_match=False,
                            explanation='Exact-card registration could not be independently confirmed; centering withheld. '+str(matching.get('reason',''))[:160])
    front_result['overlay_data_uri']=_encode_overlay(front_view)
    if matching.get('status')=='artwork_aligned_only':
        back_result=_back_measure(back,bquad)
    else:
        back_result={'status':'unavailable','reason':'Exact Giratina card identity is not confirmed from the front. Back measurement withheld to prevent associating it with a different card.',
                     'centered_status':'not_assessed','lr':None,'tb':None,'confidence':'unavailable'}
    # Never compute or imply a front centering measurement from back ratio.
    return {'status':'research_only','pilot_card_id':PILOT_ID,'calibrated':False,
            'front':front_result,'back':back_result,
            'grade':None,'grade_updated':False,'centering_category_score':None,
            'cut_edge_source':'collector_reviewed_corners',
            'units_note':'Pixel ratios compare blue printed-frame widths *within the same back photograph*. They do not establish a professional grade, a calibrated front-centering percentage, or printing tolerance.',
            'privacy_note':'This research endpoint processes uploaded photos in memory and does not store them.',
            'future_calibration':'Before measuring a full-art front, independently validate ideal printed landmark locations across known centered physical examples.'}
