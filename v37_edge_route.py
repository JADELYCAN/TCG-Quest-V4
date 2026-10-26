"""TCG Quest V37 / V37.1: optional comparative edge-detection endpoints.

No changes to regular AI grading, grades, history, or printed centering.
V37.1 is NEVER auto-promoted without reference labels. The default remains V36.
"""
from __future__ import annotations
import base64
import os
import time
from collections import defaultdict, deque
import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from template_alignment_v34 import inspect_photo, PILOT_ID, MAX_IMAGE_BYTES
from edge_refine_v36 import refine_photo_edges
from back_edge_v36_1 import detect_pokemon_back_cut_edge
from v37_edge_accuracy import detect_front, refine_back
from v37_1_edge_refinement import refine_physical_edges

v37_edges=Blueprint('v37_edges',__name__)
_rate=defaultdict(deque)

def _image(raw):
    if not raw or len(raw)>MAX_IMAGE_BYTES:raise ValueError('Each photo must be JPG, PNG or WebP and <=8 MB.')
    img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.ndim!=3 or min(img.shape[:2])<350 or img.shape[0]*img.shape[1]>25_000_000:
        raise ValueError('Upload a full-card photograph at least 350 pixels and at most 25 MP.')
    h,w=img.shape[:2];scale=min(1.,1400/max(h,w))
    if scale<1:img=cv2.resize(img,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    return img

def _norm(quad,w,h):
    if quad is None:return None
    return [[round(float(x)/w,6),round(float(y)/h,6)] for x,y in quad]

def _candidate(r,w,h):
    return {'status':r.get('status','unavailable'),'method':r.get('method'),
            'reason':r.get('reason'),'quad':_norm(r.get('physical_quad'),w,h),
            'side_evidence':r.get('side_evidence'),'verified':False,
            'baseline_preserved':r.get('status')!='candidate_found',
            'mean_move_px':r.get('mean_move_px'), 'max_move_px':r.get('max_move_px')}

def _compare_front(raw):
    img=_image(raw);h,w=img.shape[:2]
    match=inspect_photo(raw,PILOT_ID,include_overlay=False)
    if match.get('status')!='artwork_aligned_only':
        return {'status':'unavailable','side':'front','reason':match.get('reason','Exact card could not be verified.'),'old':None,'new':None,'guarded':None}
    seed=np.asarray(match['projected_template_quad'],np.float32)*[w,h]
    old=refine_photo_edges(img,seed,include_overlay=False)
    v37=detect_front(img,seed)
    guarded=refine_physical_edges(img,old['physical_quad'],side='front') if old.get('physical_quad') is not None else {'status':'unavailable','reason':'V36 front baseline unavailable; preserving original detector.'}
    oq=old.get('physical_quad');nq=v37.get('physical_quad');gq=guarded.get('physical_quad')
    return {'status':'comparison_ready','side':'front','width':w,'height':h,
            'reference':'Verified Giratina V 186/196 artwork (uncalibrated catalog projection).',
            'template_match_inliers':match.get('inliers'),
            'template_match_fraction':match.get('inlier_fraction'),
            'old':_candidate(old,w,h),'new':_candidate(v37,w,h),'guarded':_candidate(guarded,w,h),
            'comparison_mean_px':round(float(np.mean(np.linalg.norm(np.asarray(oq)-np.asarray(gq),axis=1))),2) if oq is not None and gq is not None else None,
            'measurement_claim':'NONE: research-only candidate physical cut edges; grading and printed centering unchanged.'}

def _compare_back(raw):
    img=_image(raw);h,w=img.shape[:2]
    old=detect_pokemon_back_cut_edge(img)
    v37=refine_back(img,old['physical_quad']) if old.get('status')=='candidate_found' else {'status':'unavailable','reason':'V36 Pokémon back baseline unavailable.'}
    guarded=refine_physical_edges(img,old['physical_quad'],side='back') if old.get('status')=='candidate_found' else {'status':'unavailable','reason':'V36 Pokémon back baseline unavailable.'}
    oq=old.get('physical_quad');gq=guarded.get('physical_quad')
    return {'status':'comparison_ready','side':'back','width':w,'height':h,
            'old':_candidate(old,w,h),'new':_candidate(v37,w,h),'guarded':_candidate(guarded,w,h),
            'comparison_mean_px':round(float(np.mean(np.linalg.norm(np.asarray(oq)-np.asarray(gq),axis=1))),2) if oq is not None and gq is not None else None,
            'measurement_claim':'NONE: research-only candidate physical cut edges; grading and printed centering unchanged.'}

@v37_edges.get('/api/research/v37/status')
def status():
    if os.getenv('TCG_V37_EDGE_LAB_ENABLED')!='1':return jsonify(error='V37 lab disabled'),404
    return jsonify(status='ok',pilot_card_id=PILOT_ID,experimental=True,centering_grades=False,
                   v37_1_refinement_available=os.getenv('TCG_V37_1_EDGE_LAB_ENABLED')=='1',
                   changes_to_existing_grades=False,requires_reviewed_corner_reference=True)

@v37_edges.get('/api/research/v37-1/status')
def status_v371():
    if os.getenv('TCG_V37_1_EDGE_LAB_ENABLED')!='1':return jsonify(error='V37.1 lab disabled'),404
    return jsonify(status='ok',lab='V37.1 research',baseline='V36',
                   can_withhold_unsafe_refinement=True,automatic_promotion=False,
                   calibrated_centering=False,grading_changes=False)

@v37_edges.route('/api/research/v37/edge-compare',methods=['POST','OPTIONS'])
@v37_edges.route('/api/research/v37-1/edge-compare',methods=['POST','OPTIONS'])
def compare():
    if request.method=='OPTIONS':return ('',204)
    v371=request.path.startswith('/api/research/v37-1/')
    if os.getenv('TCG_V37_1_EDGE_LAB_ENABLED' if v371 else 'TCG_V37_EDGE_LAB_ENABLED')!='1':return jsonify(error='Requested research lab disabled'),404
    origin=request.headers.get('Origin','').rstrip('/')
    allowed={s.strip().rstrip('/') for s in os.getenv('ALLOWED_ORIGINS','').split(',') if s.strip()}
    if origin and origin not in allowed:return jsonify(error='Origin not allowed'),403
    ip=request.remote_addr or 'unknown';queue=_rate[ip];now=time.monotonic()
    while queue and now-queue[0]>3600:queue.popleft()
    if len(queue)>=12:return jsonify(error='Research demo limit: 12 comparisons/hour'),429
    if request.form.get('card_id')!=PILOT_ID:return jsonify(error='Pilot supports Giratina V Lost Origin #186/196 only.'),400
    f=request.files.get('photo_front');b=request.files.get('photo_back')
    if f is None:return jsonify(error='Front photograph is required.'),400
    fb=f.stream.read(MAX_IMAGE_BYTES+1);bb=b.stream.read(MAX_IMAGE_BYTES+1) if b is not None else None
    if not fb or len(fb)>MAX_IMAGE_BYTES or bb is not None and (not bb or len(bb)>MAX_IMAGE_BYTES):return jsonify(error='Upload JPG/PNG/WebP photos no larger than 8MB.'),413
    queue.append(now)
    try:
        front=_compare_front(fb);back=_compare_back(bb) if bb is not None else None
        return jsonify(status='research_result',version='v37.1',card_id=PILOT_ID,
                       front=front,back=back,centering_ratio=None,grade=None,
                       winner='not_selected_without_reviewed_reference',
                       notice='V36 is preserved. A V37.1 outline is an unverified candidate even when displayed. Compare against separately reviewed corner locations before considering promotion.')
    except (ValueError,TypeError) as exc:return jsonify(error=str(exc)[:220]),400
    except Exception:
        # Do not reveal internal paths or identity details to the browser.
        return jsonify(error='Research comparison failed; inspect private Render logs.'),500
