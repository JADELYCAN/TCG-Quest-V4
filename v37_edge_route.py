"""Opt-in staging-only endpoint for comparing card edge candidates.

Does not replace /api/cards/grade, upload to storage, call OpenAI, or grade.
Do not enable on public production. Uses V34 pilot card only.
"""
from __future__ import annotations
import base64
import os
import time
from collections import defaultdict,deque

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from template_alignment_v34 import inspect_photo, PILOT_ID, MAX_IMAGE_BYTES
from edge_refine_v36 import refine_photo_edges
from back_edge_v36_1 import detect_pokemon_back_cut_edge
from v37_edge_accuracy import detect_front,refine_back,overlay

v37_edges=Blueprint('v37_edges',__name__)
_rate=defaultdict(deque)

def _image(raw):
    if not raw or len(raw)>MAX_IMAGE_BYTES:raise ValueError('Each photo must be JPG, PNG, or WebP and no larger than 8 MB.')
    img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if img is None or img.ndim!=3 or min(img.shape[:2])<350 or img.shape[0]*img.shape[1]>25_000_000:
        raise ValueError('Upload a readable full-card photograph between 350 pixels and 25 megapixels.')
    h,w=img.shape[:2];scale=min(1.,1400/max(h,w))
    if scale<1:img=cv2.resize(img,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    return img

def _norm(quad,w,h):
    if quad is None:return None
    return [[round(float(x)/w,6),round(float(y)/h,6)] for x,y in quad]

def _candidate(r,w,h):
    data={'status':r.get('status','unavailable'),'method':r.get('method'),
          'reason':r.get('reason'), 'quad':_norm(r.get('physical_quad'),w,h),
          'side_evidence':r.get('side_evidence'), 'verified':False}
    return data

def _encode(im):
    ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,78])
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.tobytes()).decode() if ok else None

def _compare_front(raw):
    img=_image(raw);h,w=img.shape[:2]
    a=inspect_photo(raw,PILOT_ID,include_overlay=False)
    if a.get('status')!='artwork_aligned_only':
        return {'status':'unavailable','side':'front','reason':a.get('reason','Exact identity could not be verified.'),
                'old':None,'new':None,'overlay_data_uri':_encode(img)}
    q=np.asarray(a['projected_template_quad'],np.float32)*[w,h]
    old=refine_photo_edges(img,q,include_overlay=False)
    new=detect_front(img,q)
    oq=old.get('physical_quad');nq=new.get('physical_quad')
    canvas=overlay(img,oq,nq,q)
    return {'status':'comparison_ready','side':'front','width':w,'height':h,
            'reference':'Verified Giratina V Lost Origin 186/196 artwork; catalog not a calibrated cut-edge master.',
            'template_match_inliers':a.get('inliers'),
            'template_match_fraction':a.get('inlier_fraction'),
            'old':_candidate(old,w,h),'new':_candidate(new,w,h),
            'overlay_data_uri':_encode(canvas),
            'agreement_mean_px':round(float(np.mean(np.linalg.norm(np.asarray(oq)-np.asarray(nq),axis=1))),2) if oq is not None and nq is not None else None,
            'measurement_claim':'NONE: candidate physical cut edges only; no verified centering or grading.'}

def _compare_back(raw):
    img=_image(raw);h,w=img.shape[:2]
    old=detect_pokemon_back_cut_edge(img)
    # Do not fall back to a catalog-front projection on the back.
    new=refine_back(img,old['physical_quad']) if old.get('status')=='candidate_found' else {
        'status':'unavailable','reason':'Existing Pokémon-back silhouette not found; no safe refinement seed.'}
    oq=old.get('physical_quad');nq=new.get('physical_quad')
    canvas=overlay(img,oq,nq)
    return {'status':'comparison_ready','side':'back','width':w,'height':h,
            'old':_candidate(old,w,h),'new':_candidate(new,w,h),
            'overlay_data_uri':_encode(canvas),
            'agreement_mean_px':round(float(np.mean(np.linalg.norm(np.asarray(oq)-np.asarray(nq),axis=1))),2) if oq is not None and nq is not None else None,
            'measurement_claim':'NONE: candidate physical cut edges only; no verified centering or grading.'}

@v37_edges.get('/api/research/v37/status')
def status():
    if os.getenv('TCG_V37_EDGE_LAB_ENABLED')!='1':return jsonify(error='V37 lab disabled'),404
    return jsonify(status='ok',pilot_card_id=PILOT_ID,experimental=True,
                   centering_grades=False,changes_to_existing_grades=False,
                   requires_reviewed_corner_reference=True)

@v37_edges.route('/api/research/v37/edge-compare',methods=['POST','OPTIONS'])
def compare():
    if request.method=='OPTIONS':return ('',204)
    if os.getenv('TCG_V37_EDGE_LAB_ENABLED')!='1':return jsonify(error='V37 lab disabled'),404
    origin=request.headers.get('Origin','').rstrip('/')
    allowed={s.strip().rstrip('/') for s in os.getenv('ALLOWED_ORIGINS','').split(',') if s.strip()}
    if origin and origin not in allowed:return jsonify(error='Origin not allowed.'),403
    ip=request.remote_addr or 'unknown';queue=_rate[ip];now=time.monotonic()
    while queue and now-queue[0]>3600:queue.popleft()
    if len(queue)>=12:return jsonify(error='V37 research demo limit: 12 comparisons/hour.'),429
    if request.form.get('card_id')!=PILOT_ID:return jsonify(error='Research pilot supports only verified Giratina V #186/196.'),400
    f=request.files.get('photo_front');b=request.files.get('photo_back')
    if f is None:return jsonify(error='Front photograph required.'),400
    fb=f.stream.read(MAX_IMAGE_BYTES+1);bb=b.stream.read(MAX_IMAGE_BYTES+1) if b is not None else None
    if not fb or len(fb)>MAX_IMAGE_BYTES or bb is not None and (not bb or len(bb)>MAX_IMAGE_BYTES):
        return jsonify(error='Each photo must be nonempty and no larger than 8 MB.'),413
    queue.append(now)
    try:
        return jsonify(status='research_result',card_id=PILOT_ID,front=_compare_front(fb),
                       back=_compare_back(bb) if bb is not None else None,
                       centering_ratio=None,grade=None,
                       notice='Differences between algorithms are NOT accuracy measurements. Click four reviewed physical corners to calculate approximate error per method. Do not use these candidate outlines for grade decisions.')
    except (ValueError,TypeError) as exc:
        return jsonify(error=str(exc)[:220]),400
    except Exception:
        return jsonify(error='V37 image comparison failed. Check private staging backend logs.'),500
