"""TCG Quest V35 opt-in, staging-only research endpoint. No OpenAI calls."""
from __future__ import annotations
import os,time
from collections import defaultdict,deque
from flask import Blueprint,request,jsonify
from template_alignment_v34 import MAX_IMAGE_BYTES
from template_geometry_v35 import inspect_repeatability

v35_template_lab=Blueprint('v35_template_lab', __name__)
_recent=defaultdict(deque)

@v35_template_lab.route('/api/research/v35/template-repeatability', methods=['POST','OPTIONS'])
def compare_template_frame():
    if request.method=='OPTIONS':return ('',204)
    if os.environ.get('TCG_V35_GEOMETRY_LAB_ENABLED')!='1':
        return jsonify(error='V35 geometry research lab is disabled.'),404
    origin=request.headers.get('Origin','').rstrip('/')
    permitted={s.strip().rstrip('/') for s in os.getenv('ALLOWED_ORIGINS','').split(',') if s.strip()}
    if origin and origin not in permitted:return jsonify(error='Origin is not allowed on the staging research service.'),403
    ip=request.remote_addr or 'unknown'; hist=_recent[ip]; now=time.monotonic()
    while hist and now-hist[0]>=3600:hist.popleft()
    if len(hist)>=8:return jsonify(error='Research limit: eight analyses per instance per hour.'),429
    a=request.files.get('photo_a'); b=request.files.get('photo_b')
    if a is None:return jsonify(error='Upload a front photograph in photo_a.'),400
    ba=a.stream.read(MAX_IMAGE_BYTES+1)
    if not ba or len(ba)>MAX_IMAGE_BYTES:return jsonify(error='Photo A must be no larger than 8 MB.'),413
    bb=None
    if b is not None:
        bb=b.stream.read(MAX_IMAGE_BYTES+1)
        if not bb or len(bb)>MAX_IMAGE_BYTES:return jsonify(error='Photo B must be no larger than 8 MB.'),413
    hist.append(now)
    result=inspect_repeatability(ba, request.form.get('card_id',''), bb,include_overlay=True)
    return jsonify(result)
