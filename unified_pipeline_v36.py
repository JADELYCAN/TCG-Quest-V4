"""Integrated V36 research pipeline using existing exact-card alignment.

Uses unchanged V34 identity verification, new template-guided cut-edge
refinement, and quality/review-zone diagnostics. Research ONLY: returns
no numeric centering, flaw scores, professional grade, or provider API calls.
"""
from __future__ import annotations
import base64
import hashlib
import os
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
        baseline_status=detection.get('status')
        baseline_reason=detection.get('reason')
        # V37.3 integration: V36 remains the first choice. Recover ONLY the
        # confirmed pilot-card front when V36 cannot produce a physical edge.
        # Research flag defaults OFF; this cannot affect regular grading.
        selected='v36_front_baseline'
        if baseline_status!='candidate_found' and os.getenv('TCG_V37_3_ADAPTIVE_ENABLED')=='1':
            # Import lazily so optional research files cannot break production
            # app startup when the feature flag is not enabled.
            try:
                from v37_2_seed_recovery import recover_front_from_template
                recovery=recover_front_from_template(bgr,template_q)
            except (ImportError,ValueError,TypeError,cv2.error) as exc:
                recovery={'status':'unavailable','reason':'Adaptive recovery module unavailable or rejected input.'}
            if recovery.get('status')=='candidate_found':
                detection=recovery
                selected='v37_3_template_seeded_recovery'
            else:
                return {'status':'unavailable','side':side,
                        'reason':('V36: '+str(baseline_reason or 'edge not detected')+'; V37.3: '+str(recovery.get('reason','recovery unavailable'))),
                        'template':template_meta,
                        'edge_diagnostics':{'status':'unavailable','reason':recovery.get('reason'),
                                            'baseline_reason':baseline_reason,'selected_detector':None},
                        'quality':photo_quality(bgr),'centering_ratio':None,'grade':None,
                        'adaptive_attempted':True,'review_required':True}
        if detection['status']!='candidate_found':
            return {'status':'unavailable','side':side,'reason':baseline_reason,
                    'template':template_meta,'edge_diagnostics':{'status':'unavailable','reason':baseline_reason},
                    'quality':photo_quality(bgr),'centering_ratio':None,'grade':None}
        quad=detection['physical_quad']
        if selected=='v36_front_baseline':
            overlay=detection.pop('overlay_bgr')
        else:
            overlay=bgr.copy()
            cv2.polylines(overlay,[np.int32(np.round(np.asarray(quad,np.float32)))],True,(255,255,0),3,cv2.LINE_AA)
        detection['selected_detector']=selected
        detection['baseline_reason']=baseline_reason if selected!='v36_front_baseline' else None
        detection['review_required']=True
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
        # V37.4 staging-only gradient cross-check: V36 always provides the
        # accepted baseline. Refine only when >=3 independently consistent
        # physical-edge transitions and conservative motion checks pass.
        selected_back='v36_pokemon_back'
        if os.getenv('TCG_V37_4_BACK_REFINEMENT_ENABLED')=='1':
            try:
                from back_refinement_v37_4 import analyze_pokemon_back_edges
                micro=analyze_pokemon_back_edges(bgr,detection['physical_quad'])
            except (ImportError,ValueError,TypeError,cv2.error) as exc:
                micro={'status':'baseline_preserved','reason':'Optional V37.4 edge cross-check unavailable; V36 baseline preserved.'}
            detection['v37_4_crosscheck']=micro
            if micro.get('status')=='micro_refinement_candidate':
                # This is a photo candidate, not a validated result. Must lock.
                detection['physical_quad']=micro['refined_quad']
                detection['confidence']='experimental_guarded_refinement'
                detection['weak_sides']=[s for s,d in micro.get('side_evidence',{}).items() if not d.get('usable')]
                selected_back='v37_4_guarded_back_refinement'
            else:
                detection['confidence']='research_candidate_only'
        quad=detection['physical_quad'];overlay=bgr.copy()
        detection['selected_detector']=selected_back
        detection['review_required']=True
        cv2.polylines(overlay,[np.int32(np.round(quad))],True,(255,255,0),3,cv2.LINE_AA)
        template_meta=None
    quality=photo_quality(bgr,quad)
    # Including all eight crops is optional to reduce payload size.
    crops=inspection_crops(bgr,quad,side=side)
    return {'status':'candidate_cut_edge_found','side':side,'width':ww,'height':hh,
            'template':template_meta,'edge_diagnostics':detection,
            'selected_detector':detection.get('selected_detector'),
            'edge_confidence':detection.get('confidence','research_candidate_only'),
            'weak_sides':detection.get('weak_sides',[]),
            'review_required':True, 'verified_card_edges':False,
            'edge_quad_normalized':[[round(float(x)/ww,5),round(float(y)/hh,5)] for x,y in quad],
            'overlay_data_uri':_encode(overlay,80),
            'quality':quality,'review_crops':crops,
            'research_signal_maps':examine_image_signals(bgr,quad),'centering_ratio':None,
            'damage_claims':[],'grade':None,
            'note':'Edges are unverified photo candidates. Review and lock all four cut corners before using in report. No calibrated printed centering or confirmed damage.'}


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
