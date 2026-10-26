"""Integrated V36 research pipeline using existing exact-card alignment.

Uses unchanged V34 identity verification, new template-guided cut-edge
refinement, and quality/review-zone diagnostics. Research ONLY: returns
no numeric centering, flaw scores, professional grade, or provider API calls.
"""
from __future__ import annotations
import base64
import hashlib
import cv2
import numpy as np
from template_alignment_v34 import inspect_photo,PILOT_ID,MAX_IMAGE_BYTES
from edge_refine_v36 import refine_photo_edges
from photo_diagnostics_v36 import photo_quality,inspection_crops,_encode
from condition_candidates_v36 import examine_image_signals
from grading_gate_v36 import research_grading_gate


def _dec(raw):
    if not isinstance(raw,bytes) or not raw or len(raw)>MAX_IMAGE_BYTES:
        raise ValueError('Photo must be between 1 byte and 8 MB.')
    arr=np.frombuffer(raw,np.uint8)
    bgr=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim!=3:raise ValueError('Invalid JPG, PNG, or WebP photo.')
    if bgr.shape[0]*bgr.shape[1]>25_000_000 or min(bgr.shape[:2])<350:
        raise ValueError('Photo must show the full card at reasonable resolution (under 25 MP).')
    # V34 downsizes to longest 1400 for SIFT. Keep same coordinate frame here.
    scale=min(1.,1400/max(bgr.shape[:2]))
    if scale<1:
        bgr=cv2.resize(bgr,(round(bgr.shape[1]*scale),round(bgr.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    return bgr


def _side(raw, side, front_match=None):
    bgr=_dec(raw)
    hh,ww=bgr.shape[:2]
    # No assumption that the front template matches Pokemon back art.
    aligned=front_match if side=='front' else None
    if side=='front' and aligned is None:
        aligned=inspect_photo(raw,PILOT_ID,include_overlay=False)
    if side=='front' and aligned['status']!='artwork_aligned_only':
        return {'status':'unavailable','side':side,'reason':aligned.get('reason','Exact-card template match failed.'),
                'quality':photo_quality(bgr),'centering_ratio':None,'grade':None}
    if side=='front':
        template_q=np.float32(aligned['projected_template_quad'])*np.array([ww,hh],np.float32)
        detection=refine_photo_edges(bgr,template_q,include_overlay=True)
        template_meta={'card_id':PILOT_ID,'inliers':aligned.get('inliers'),
                       'matching_fraction':aligned.get('inlier_fraction'),
                       'competing_artwork_inliers':aligned.get('competing_artwork_inliers')}
        if detection['status']!='candidate_found':
            return {'status':'unavailable','side':side,'reason':detection.get('reason'),
                    'template':template_meta,'edge_diagnostics':detection,
                    'quality':photo_quality(bgr),'centering_ratio':None,'grade':None}
        quad=detection['physical_quad']; overlay=detection.pop('overlay_bgr')
    else:
        # V36.1: use the actual dark-blue cut-stock silhouette, not the old
        # generic dark-object contour. Withhold when this side-specific
        # detector is uncertain; do not silently borrow an older guide.
        from back_edge_v36_1 import detect_pokemon_back_cut_edge
        detection=detect_pokemon_back_cut_edge(bgr)
        if detection['status']!='candidate_found':
            return {'status':'unavailable','side':side,
                    'reason':detection.get('reason','Back cut edge not found.'),
                    'edge_diagnostics':detection,'quality':photo_quality(bgr),
                    'centering_ratio':None,'grade':None}
        quad=detection['physical_quad'];overlay=bgr.copy()
        cv2.polylines(overlay,[np.int32(np.round(quad))],True,(255,255,0),3,cv2.LINE_AA)
        template_meta=None
    quality=photo_quality(bgr,quad)
    # Including all eight crops is optional to reduce payload size.
    crops=inspection_crops(bgr,quad,side=side)
    return {'status':'candidate_cut_edge_found','side':side,'width':ww,'height':hh,
            'template':template_meta,'edge_diagnostics':detection,
            'edge_quad_normalized':[[round(float(x)/ww,5),round(float(y)/hh,5)] for x,y in quad],
            'overlay_data_uri':_encode(overlay,80),
            'quality':quality,'review_crops':crops,
            'research_signal_maps':examine_image_signals(bgr,quad),'centering_ratio':None,
            'damage_claims':[],'grade':None,
            'note':'Edges and crops are research candidates, not calibrated centering or confirmed defects.'}


def inspect_v36(front_bytes,card_id,back_bytes=None):
    if card_id!=PILOT_ID:
        return {'status':'unavailable','reason':'V36 prototype supports confirmed Giratina V Lost Origin 186/196 only.',
                'verified_centering':False,'grade':None}
    front=_side(front_bytes,'front')
    back=None
    if back_bytes is not None:
        back=_side(back_bytes,'back')
    return {'status':'research_result','method':'v36_hybrid_cpu_staging',
            'card_id':PILOT_ID,'front':front,'back':back,
            'verified_centering':False,'centering_ratios':None,'grade':None,
            'calibration_status':'not_calibrated','model_status':'not_trained',
            'grade_gate':research_grading_gate(None,None),
            'note':'Research detection results only. Independent human-verified corner labels and calibrated print masters remain necessary.'}
