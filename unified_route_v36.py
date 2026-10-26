"""Opt-in V36 research endpoint, not part of card grading."""
from __future__ import annotations
import os,time
from collections import defaultdict,deque
from flask import Blueprint,request,jsonify
from template_alignment_v34 import MAX_IMAGE_BYTES
from unified_pipeline_v36 import inspect_v36

v36_lab=Blueprint('v36_lab',__name__)
_history=defaultdict(deque)

@v36_lab.route('/api/research/v36/unified-inspect',methods=['POST','OPTIONS'])
def unified_inspect():
    if request.method=='OPTIONS':return ('',204)
    if os.getenv('TCG_V36_LAB_ENABLED')!='1':return jsonify(error='V36 research lab is disabled.'),404
    origin=request.headers.get('Origin','').rstrip('/')
    allowed={v.strip().rstrip('/') for v in os.getenv('ALLOWED_ORIGINS','').split(',') if v.strip()}
    if origin and origin not in allowed:return jsonify(error='Origin not allowed.'),403
    ip=request.remote_addr or 'unknown'; q=_history[ip]; now=time.monotonic()
    while q and now-q[0]>3600:q.popleft()
    if len(q)>=8:return jsonify(error='V36 research limit: eight requests per instance per hour.'),429
    f=request.files.get('photo_front');b=request.files.get('photo_back')
    if f is None:return jsonify(error='Upload a physical front photo.'),400
    ba=f.stream.read(MAX_IMAGE_BYTES+1)
    bb=b.stream.read(MAX_IMAGE_BYTES+1) if b is not None else None
    if not ba or len(ba)>MAX_IMAGE_BYTES or (bb is not None and (not bb or len(bb)>MAX_IMAGE_BYTES)):
        return jsonify(error='Each photo must be 8 MB or smaller.'),413
    q.append(now)
    try:
        result=inspect_v36(ba,request.form.get('card_id',''),bb)
        return jsonify(result)
    except (ValueError,TypeError) as e:
        return jsonify(error=str(e)[:250]),400
    except Exception:
        # Never dump path, source, or credentials to client.
        return jsonify(error='V36 inspection failed. See private Render logs.'),500


@v36_lab.get('/api/research/v36/status')
def v36_status():
    # Does not call a model or access user images. Safe staging status check.
    if os.getenv('TCG_V36_LAB_ENABLED')!='1':return jsonify(error='V36 lab disabled.'),404
    return jsonify(status='ok',pilot_card_id='swsh11-186',
                   research_only=True,adaptive_v37_3_enabled=os.getenv('TCG_V37_3_ADAPTIVE_ENABLED')=='1',
                   guarded_back_refinement_v37_4_enabled=os.getenv('TCG_V37_4_BACK_REFINEMENT_ENABLED')=='1',
                   adaptive_mode='V36 front, V37.3 recovered front on failure; V36 back baseline with optional V37.4 guarded gradient cross-check',
                   trained_ml_model=False,
                   verified_centering=False,grade_generation=False,
                   note='Experimental CPU photo detector active; no calibrated centering or grading.')
