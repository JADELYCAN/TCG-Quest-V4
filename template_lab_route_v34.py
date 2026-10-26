"""Opt-in V34 research endpoint; no provider/API token required, no photo storage.

Only register on an isolated staging service via TCG_V34_TEMPLATE_LAB_ENABLED=1.
Only the exact Giratina Lost Origin #186 template is bundled as a pilot.
"""
from __future__ import annotations
import os, time
from collections import defaultdict, deque
from flask import Blueprint, request, jsonify
from template_alignment_v34 import inspect_photo, MAX_IMAGE_BYTES

v34_template_lab = Blueprint('v34_template_lab',__name__)
_checks = defaultdict(deque)

@v34_template_lab.route('/api/research/v34/template-align',methods=['POST','OPTIONS'])
def align_pilot():
    if request.method=='OPTIONS':
        return ('',204)
    if os.getenv('TCG_V34_TEMPLATE_LAB_ENABLED') != '1':
        return jsonify(error='Experimental template lab is disabled.'),404
    allowed = {v.strip().rstrip('/') for v in os.getenv('ALLOWED_ORIGINS','').split(',') if v.strip()}
    origin=request.headers.get('Origin','').rstrip('/')
    if origin and origin not in allowed:
        return jsonify(error='Origin not allowed on this research service.'),403
    ip=request.remote_addr or 'unknown'
    history=_checks[ip]; now=time.monotonic()
    while history and now-history[0]>3600:history.popleft()
    if len(history)>=12:
        return jsonify(error='Research limit: 12 photo comparisons per hour per instance.'),429
    uploaded=request.files.get('photo')
    if uploaded is None:
        return jsonify(error='Upload a front photograph under the field name photo.'),400
    photo=uploaded.stream.read(MAX_IMAGE_BYTES+1)
    if len(photo)>MAX_IMAGE_BYTES:
        return jsonify(error='Image must be 8 MB or less.'),413
    history.append(now)
    result=inspect_photo(photo, request.form.get('card_id',''),include_overlay=True)
    return jsonify(result)
