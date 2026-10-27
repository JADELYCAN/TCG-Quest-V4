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


# V38 calibration lab: attached to the already-registered V36 staging blueprint.
# Does not touch /api/cards/grade or any existing report route.
@v36_lab.route('/api/research/v38/centering', methods=['POST', 'OPTIONS'])
def v38_centering_research():
    if request.method == 'OPTIONS':
        return ('', 204)
    if os.getenv('TCG_V38_CENTERING_LAB_ENABLED') != '1' or os.getenv('TCG_V36_LAB_ENABLED') != '1':
        return jsonify(error='V38 research lab disabled.'), 404
    origin=request.headers.get('Origin','').rstrip('/')
    allowed={v.strip().rstrip('/') for v in os.getenv('ALLOWED_ORIGINS','').split(',') if v.strip()}
    if origin and origin not in allowed: return jsonify(error='Origin not permitted.'), 403
    ip=request.remote_addr or 'unknown'
    q=_history[ip]; now=time.monotonic()
    while q and now-q[0]>3600: q.popleft()
    if len(q)>=8: return jsonify(error='Staging research request limit reached. Try later.'), 429
    f=request.files.get('photo_front'); b=request.files.get('photo_back')
    if f is None or b is None: return jsonify(error='Both original front and back photos are required.'), 400
    a=f.stream.read(8*1024*1024+1); c=b.stream.read(8*1024*1024+1)
    if not a or not c or len(a)>8*1024*1024 or len(c)>8*1024*1024:
        return jsonify(error='Each photo must be less than 8 MB.'), 413
    try:
        import json
        front=json.loads(request.form.get('front_corners','null'))
        back=json.loads(request.form.get('back_corners','null'))
        if request.form.get('review_confirmed')!='yes':
            return jsonify(error='Review all four physical corners on each side before measuring.'), 400
        from v38_centering_lab import analyze_v38
        result=analyze_v38(a,c,front,back,request.form.get('card_id',''))
        q.append(now)
        return jsonify(result)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return jsonify(error=str(exc)[:240]), 400
    except Exception:
        return jsonify(error='V38 research processing failed. Check private Render logs.'), 500

@v36_lab.get('/api/research/v38/status')
def v38_centering_status():
    if os.getenv('TCG_V38_CENTERING_LAB_ENABLED')!='1' or os.getenv('TCG_V36_LAB_ENABLED')!='1':
        return jsonify(error='V38 research lab disabled.'),404
    return jsonify(status='ok',pilot_card_id='swsh11-186', research_only=True,
                   front_calibrated=False,grade_generation=False,
                   conventional_back_frame_method='blue_border_pixel_scanlines')

# V38.1 two-photo repeatability plus *uncalibrated* front-landmark preview.
# Staging-only extension of the registered V36 research blueprint.
@v36_lab.route('/api/research/v38-1/validate', methods=['POST', 'OPTIONS'])
def v38_1_repeatability():
    if request.method == 'OPTIONS':return ('', 204)
    if os.getenv('TCG_V38_1_VALIDATION_ENABLED')!='1' or os.getenv('TCG_V38_CENTERING_LAB_ENABLED')!='1' or os.getenv('TCG_V36_LAB_ENABLED')!='1':
        return jsonify(error='V38.1 research comparison disabled.'),404
    origin=request.headers.get('Origin','').rstrip('/')
    allowed={v.strip().rstrip('/') for v in os.getenv('ALLOWED_ORIGINS','').split(',') if v.strip()}
    if origin and origin not in allowed:return jsonify(error='Origin not permitted.'),403
    ip=request.remote_addr or 'unknown';q=_history[ip];now=time.monotonic()
    while q and now-q[0]>3600:q.popleft()
    if len(q)>=8:return jsonify(error='Research request limit reached for this instance.'),429
    names=['photo_front','photo_back_a','photo_back_b']
    if any(request.files.get(n) is None for n in names):
        return jsonify(error='Upload the Giratina front and TWO independent back photographs.'),400
    raws=[request.files[n].stream.read(8*1024*1024+1) for n in names]
    if any(not x or len(x)>8*1024*1024 for x in raws):
        return jsonify(error='Each photo must be nonempty and under 8 MB.'),413
    if request.form.get('review_confirmed')!='yes':
        return jsonify(error='Confirm reviewed cut corners for all three photos.'),400
    try:
        import json
        vals=[json.loads(request.form.get(k,'null')) for k in ('front_corners','back_a_corners','back_b_corners')]
        from v38_1_validation import compare_v38_1
        result=compare_v38_1(*raws,*vals,card_id=request.form.get('card_id',''))
        q.append(now)
        return jsonify(result)
    except (ValueError,TypeError,KeyError,OverflowError) as e:
        return jsonify(error=str(e)[:240]),400
    except Exception:
        return jsonify(error='V38.1 research processing failed. Check private Render logs.'),500

@v36_lab.get('/api/research/v38-1/status')
def v38_1_status():
    if os.getenv('TCG_V38_1_VALIDATION_ENABLED')!='1' or os.getenv('TCG_V38_CENTERING_LAB_ENABLED')!='1' or os.getenv('TCG_V36_LAB_ENABLED')!='1':
        return jsonify(error='V38.1 research disabled.'),404
    return jsonify(status='ok',pilot_card_id='swsh11-186',research_only=True,
                   front_calibrated=False,back_calibrated=False,grade_generation=False,
                   feature='two-photo printed-frame repeatability and noncalibrated front-region projection')

# V38.2: the conventional Pokémon-back frame is evaluated INDEPENDENTLY of
# full-art front catalog registration. Research only; grading endpoint unchanged.
@v36_lab.route('/api/research/v38-2/validate', methods=['POST', 'OPTIONS'])
def v38_2_independent_back_measure():
    if request.method == 'OPTIONS':
        return ('', 204)
    required_flags = ('TCG_V36_LAB_ENABLED', 'TCG_V38_CENTERING_LAB_ENABLED',
                      'TCG_V38_2_INDEPENDENT_BACK_ENABLED')
    if any(os.getenv(flag) != '1' for flag in required_flags):
        return jsonify(error='V38.2 research endpoint disabled.'), 404
    origin = request.headers.get('Origin', '').rstrip('/')
    allowed = {s.strip().rstrip('/') for s in os.getenv('ALLOWED_ORIGINS', '').split(',') if s.strip()}
    if origin and origin not in allowed:
        return jsonify(error='Origin not permitted.'), 403
    ip = request.remote_addr or 'unknown'
    q = _history[ip]
    now = time.monotonic()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= 8:
        return jsonify(error='Research request limit reached for this instance.'), 429
    if request.form.get('review_confirmed') != 'yes' or request.form.get('same_card_confirmed') != 'yes':
        return jsonify(error='Confirm physical corner reviews and that the photos show the same physical card.'), 400
    front = request.files.get('photo_front')
    back = request.files.get('photo_back_a')
    optional = request.files.get('photo_back_b')
    if front is None or back is None:
        return jsonify(error='Upload a front photo and one physical back photograph.'), 400
    limit = 8 * 1024 * 1024
    rawa = front.stream.read(limit + 1)
    rawb = back.stream.read(limit + 1)
    rawc = optional.stream.read(limit + 1) if optional is not None else None
    if not rawa or not rawb or any(len(raw) > limit or not raw for raw in (rawa, rawb, *( [rawc] if rawc is not None else [] ))):
        return jsonify(error='Each photo must be a nonempty image no larger than 8 MB.'), 413
    try:
        import json
        coords_a = json.loads(request.form.get('front_corners', 'null'))
        coords_b = json.loads(request.form.get('back_a_corners', 'null'))
        coords_c = json.loads(request.form.get('back_b_corners', 'null')) if rawc is not None else None
        from v38_2_independent_back import analyze_v38_2
        result = analyze_v38_2(rawa, rawb, coords_a, coords_b, rawc, coords_c,
                               card_id=request.form.get('card_id', ''), same_card_confirmed=True)
        q.append(now)
        return jsonify(result)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return jsonify(error=str(exc)[:240]), 400
    except Exception:
        return jsonify(error='V38.2 research processing failed. See private Render logs.'), 500


@v36_lab.get('/api/research/v38-2/status')
def v38_2_research_status():
    flags = ('TCG_V36_LAB_ENABLED', 'TCG_V38_CENTERING_LAB_ENABLED',
             'TCG_V38_2_INDEPENDENT_BACK_ENABLED')
    if any(os.getenv(flag) != '1' for flag in flags):
        return jsonify(error='V38.2 lab disabled.'), 404
    return jsonify(status='ok', version='v38.2', research_only=True,
                   front_identity_required_for_back=False, front_calibrated=False,
                   back_calibrated=False, grade_generation=False,
                   supports_single_back=True, optional_second_back=True,
                   notes='Independent Pokémon-back printed-frame analysis with separate front-match diagnostics.')
