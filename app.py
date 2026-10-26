import os, io, base64, time, json
from collections import defaultdict, deque
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
import requests

app=Flask(__name__, static_folder='assets')
# Both the front and back photos can be 8 MB; allow the pair plus multipart overhead.
app.config['MAX_CONTENT_LENGTH']=16*1024*1024+65536

# V27: optional re-assessment endpoint for saved front/back card photos.
# This runs on the same private AI backend as /api/cards/grade.
from grade_scores_blueprint import score_blueprint
app.register_blueprint(score_blueprint)

# V34: Strictly opt-in staging photo-template alignment lab. It never grades a card.
# Enable ONLY on staging via TCG_V34_TEMPLATE_LAB_ENABLED=1.
if os.getenv('TCG_V34_TEMPLATE_LAB_ENABLED') == '1':
    from template_lab_route_v34 import v34_template_lab
    app.register_blueprint(v34_template_lab)


ALLOWED_ORIGINS={x.strip() for x in os.getenv('ALLOWED_ORIGINS','https://tcg-quest-v4.onrender.com,https://tcg-quest-v4-ai.onrender.com,https://tcg-quest-v8-test.onrender.com,https://jadelycan.github.io').split(',') if x.strip()}
requests_by_ip=defaultdict(deque)

@app.after_request
def cors(response):
    origin=request.headers.get('Origin','').rstrip('/')
    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin']=origin
        response.headers['Vary']='Origin'
        response.headers['Access-Control-Allow-Methods']='POST, OPTIONS, GET'
        response.headers['Access-Control-Allow-Headers']='Content-Type'
    response.headers['Cache-Control']='no-store'
    return response

@app.get('/')
def home(): return send_from_directory('.', 'index.html')

@app.get('/assets/<path:filename>')
def assets(filename): return send_from_directory('assets',filename)

@app.get('/health')
def health(): return jsonify(status='ok', ai_configured=bool(os.getenv('OPENAI_API_KEY')))

@app.route('/api/anime/generate', methods=['POST','OPTIONS'])
def generate():
    if request.method=='OPTIONS':return ('',204)
    origin=request.headers.get('Origin','').rstrip('/')
    if origin and origin not in ALLOWED_ORIGINS:return jsonify(error='Origin not allowed'),403
    key=os.getenv('OPENAI_API_KEY')
    if not key:return jsonify(error='AI service not configured: add OPENAI_API_KEY in Render environment settings.'),503
    # Basic per-instance throttling; for production use shared rate limiting, authentication and quotas.
    ip=request.remote_addr or 'unknown'; now=time.monotonic();q=requests_by_ip[ip]
    while q and now-q[0]>3600:q.popleft()
    if len(q)>=5:return jsonify(error='Hourly demo limit reached. Try later.'),429
    file=request.files.get('photo')
    if not file:return jsonify(error='Upload a photo.'),400
    raw=file.read(8*1024*1024+1)
    if len(raw)>8*1024*1024:return jsonify(error='Image must be 8 MB or smaller.'),413
    try:
        im=Image.open(io.BytesIO(raw));im.verify()
        im=Image.open(io.BytesIO(raw));im.load()
        if im.format not in ('JPEG','PNG','WEBP'):raise ValueError('Unsupported format')
        if im.width*im.height>20_000_000:raise ValueError('Image dimensions too large')
        im.thumbnail((1536,1536))
        im=im.convert('RGB');buf=io.BytesIO();im.save(buf,format='PNG');image_bytes=buf.getvalue()
    except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError):return jsonify(error='Invalid or oversized image.'),400
    style=request.form.get('style','Anime Adventurer')
    styles={'Anime Adventurer':'electric blue adventure anime, green cap and sporty hoodie','Trading Card Champion':'high detail collectible card champion anime, energetic colorful card aura','Sports Anime Hero':'sports anime illustration, athletic jersey and stadium lighting'}
    if style not in styles:return jsonify(error='Invalid style.'),400
    prompt=('Transform the person in the input photo into a polished, original anime-style trading-card collector portrait. '
            'Preserve the person’s recognizable facial features, complexion and general likeness. '
            'Show head and upper torso, clear face, expressive eyes, clean ink lines, rich cel shading, dramatic neon lighting. '
            'Style: '+styles[style]+'. No text, logos, watermarks, real trading card brands, or copyrighted characters. Square portrait.')
    q.append(now)
    try:
        r=requests.post('https://api.openai.com/v1/images/edits',headers={'Authorization':'Bearer '+key},
           files={'image':('selfie.png',image_bytes,'image/png')},
           data={'model':os.getenv('OPENAI_IMAGE_MODEL','gpt-image-1.5'),'prompt':prompt,'size':'1024x1024'},timeout=180)
        if not r.ok:
            if r.status_code==429:return jsonify(error='AI provider rate limit or insufficient API quota. Check API billing.'),429
            return jsonify(error='AI provider rejected request (HTTP '+str(r.status_code)+'). Check model access and Render logs.'),502
        payload=r.json();entry=payload['data'][0]
        if entry.get('b64_json'): encoded=entry['b64_json']
        else:return jsonify(error='AI provider returned an unexpected image format.'),502
        return jsonify(image='data:image/png;base64,'+encoded)
    except (requests.RequestException,KeyError,IndexError,ValueError) as exc:
        app.logger.warning('AI generation failed: %s',type(exc).__name__)
        return jsonify(error='AI provider unavailable or returned invalid data.'),502


# Development scanner: visual identification is an estimate, not a verified catalog match.
scanner_requests=defaultdict(deque)

@app.route('/api/cards/identify',methods=['POST','OPTIONS'])
def identify_card():
    if request.method=='OPTIONS':return ('',204)
    origin=request.headers.get('Origin','').rstrip('/')
    if origin and origin not in ALLOWED_ORIGINS:return jsonify(error='Origin not allowed'),403
    key=os.getenv('OPENAI_API_KEY')
    if not key:return jsonify(error='AI service not configured.'),503
    ip=request.remote_addr or 'unknown'; now=time.monotonic(); q=scanner_requests[ip]
    while q and now-q[0]>3600:q.popleft()
    if len(q)>=5:return jsonify(error='Scanner demo limit reached (5/hour). Try later.'),429
    upload=request.files.get('card')
    if not upload:return jsonify(error='Choose a card photo.'),400
    raw=upload.read(8*1024*1024+1)
    if len(raw)>8*1024*1024:return jsonify(error='Card image must be 8 MB or smaller.'),413
    try:
        with Image.open(io.BytesIO(raw)) as source:
            image_format=source.format
            width,height=source.size
            if image_format not in ('JPEG','PNG','WEBP'):
                return jsonify(error='Unsupported card image format. Please upload a JPG, PNG, or WebP file.'),400
            if width<100 or height<100:
                return jsonify(error='Card photo is too small. Use an image at least 100 pixels wide and tall.'),400
            if width*height>20_000_000:
                return jsonify(error='Card photo resolution is too large (over 20 megapixels). Please resize it.'),400
            source.load()
            im=ImageOps.exif_transpose(source)
            im.thumbnail((1200,1200))
            im=im.convert('RGB')
            buf=io.BytesIO()
            im.save(buf,format='JPEG',quality=82)
            encoded=base64.b64encode(buf.getvalue()).decode('ascii')
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError) as exc:
        app.logger.warning('Scanner image decoding failed: %s',type(exc).__name__)
        return jsonify(error='Could not read card image. Please try exporting it as a new JPG or PNG.'),400
    q.append(now)
    instructions=('You inspect photos of collectible trading cards. Identify ONLY text and details actually visible. '
      'Never invent a set number, year, rarity, brand, or athlete. Return a JSON object with exactly these string keys: '
      'name, category, set, number, year, rarity, confidence, notes. '
      'category must be one of Pokémon, One Piece, Sports, Other TCG, Unknown. '
      'confidence must be high, medium, or low. Use empty strings for unreadable values. '
      'If no collectible card is visible, set category Unknown and explain in notes. '
      'This is an image-only estimate, not a catalog-verified identification.')
    try:
        r=requests.post('https://api.openai.com/v1/chat/completions',
          headers={'Authorization':'Bearer '+key},json={
          'model':os.getenv('OPENAI_VISION_MODEL','gpt-4.1-mini'),
          'messages':[{'role':'system','content':instructions},{'role':'user','content':[
          {'type':'text','text':'Identify this trading card from the photo. Respond only in JSON.'},
          {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+encoded}}]}],
          'response_format':{'type':'json_object'},'max_tokens':350},timeout=90)
        if not r.ok:
            if r.status_code==429:return jsonify(error='AI rate limit or insufficient API quota.'),429
            app.logger.warning('Scanner provider status %s',r.status_code)
            return jsonify(error='Scanner AI provider error (HTTP '+str(r.status_code)+').'),502
        result=json.loads(r.json()['choices'][0]['message']['content'])
        fields=('name','category','set','number','year','rarity','confidence','notes')
        clean={k:str(result.get(k,'') or '')[:300] for k in fields}
        if clean['category'] not in ('Pokémon','One Piece','Sports','Other TCG','Unknown'):clean['category']='Unknown'
        if clean['confidence'] not in ('high','medium','low'):clean['confidence']='low'
        return jsonify(card=clean,verified=False)
    except (requests.RequestException,KeyError,IndexError,ValueError,TypeError):
        app.logger.warning('Scanner request failed')
        return jsonify(error='Scanner AI unavailable or returned invalid data.'),502

# Phase 4 — AI Card Grading
grading_requests = defaultdict(deque)

@app.route('/api/cards/grade', methods=['POST', 'OPTIONS'])
def grade_card():
    if request.method == 'OPTIONS':
        return ('', 204)

    origin = request.headers.get('Origin', '').rstrip('/')
    if origin and origin not in ALLOWED_ORIGINS:
        return jsonify(error='Origin not allowed'), 403

    key = os.getenv('OPENAI_API_KEY')
    if not key:
        return jsonify(error='AI grading service not configured.'), 503

    front = request.files.get('front')
    back = request.files.get('back')

    if not front or not back:
        return jsonify(error='Upload both front and back card photos.'), 400
    def prepare_card_photo(upload):
        raw = upload.read(8 * 1024 * 1024 + 1)

        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('Each photo must be 8 MB or smaller.')

        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Use JPG, PNG, or WebP photos.')

            width, height = source.size
            if width < 100 or height < 100:
                raise ValueError('Card photos must be at least 100 pixels wide and tall.')

            if width * height > 20_000_000:
                raise ValueError('Photo resolution is too large.')

            source.load()
            image = ImageOps.exif_transpose(source)
            image.thumbnail((1536, 1536))
            image = image.convert('RGB')

            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=88)

            return base64.b64encode(buffer.getvalue()).decode('ascii')
    try:
        front_image = prepare_card_photo(front)
        back_image = prepare_card_photo(back)
    except (ValueError, OSError, UnidentifiedImageError,
            Image.DecompressionBombError) as exc:
        return jsonify(error='Invalid card photo. Check the format, size, and resolution.'), 400  
                
    ip = request.remote_addr or 'unknown'
    now = time.monotonic()
    q = grading_requests[ip]

    while q and now - q[0] > 3600:
        q.popleft()

    if len(q) >= 5:
        return jsonify(error='AI grading demo limit reached (5/hour). Try later.'), 429
    grading_instructions = (
        "You are a collectible trading card condition inspector. "
        "Examine the FRONT and BACK photographs of the same card. "
        "Assess visible corner wear, corner whitening, edge wear, "
        "edge chipping, scratches, print lines, stains, and surface defects. "
        "Do not invent defects that cannot be seen. "
        "Distinguish possible reflections, glare, and shadows from actual damage. "
        "If image quality prevents reliable inspection, state that clearly. "
        "Do not claim to detect microscopic defects or authenticate the card. "
        "Return a JSON object with exactly these keys: "
        "corners, edges, surface, photo_quality, overall_notes, estimated_grade, grade_low, grade_high, grade_reason, detailed_inspection, category_scores, category_score_notes, side_scores, side_score_notes, evidence_candidates. "
        "category_scores must be an object with centering, corners, edges, surface keys. "
        "Grade each category independently from the uploaded FRONT and BACK physical photos only. "
        "Use numeric scores from 1 to 10 in increments of 0.5 (e.g. 8, 8.5), or null "
        "when photo quality, card design, glare, sleeves, or visibility prevent a reliable evaluation. "
        "Do not derive category scores from the overall estimated grade, and do not assume "
        "perfect condition when defects are not visible. Do not use any catalog image. "
        "Centering concerns visible print borders on both sides; where borders cannot "
        "be assessed, use null. "
        "category_score_notes must be an object with the same four keys and brief "
        "evidence-based explanations describing WHY each score was given, using the FRONT and BACK photos and the visible defect locations. "
        "Additionally return side_scores with front and back fields, each a numeric 1–10 value in 0.5 increments or null if that side cannot be evaluated. "
        "Return side_score_notes with front and back keys; each must explain that side’s independent score by referencing visible corner, edge or surface evidence. "
        "Front/back estimates are independent assessments, not mathematically inferred from the overall score. "
        "A null side score must be explained; do not assert a number when evidence is unclear. "
        "Evidence candidate locations: return evidence_candidates as an ARRAY of 0–10 entries. "
        "Add an entry only for a specific visible possible defect (not a clean inspected region). "
        "Use {side: front or back, type: corner_whitening/corner_wear/edge_chipping/edge_wear/surface_scratch/print_line/crease/dent/stain/other_visible_flaw, bbox: [left,top,right,bottom], observation: short visible evidence, confidence: high or medium}. "
        "bbox must tightly bound the suspected defect area in FULL PHOTO coordinates scaled from 0 to 1000 (left=0, right=1000, top=0, bottom=1000); not card-relative coordinates. "
        "Each bbox must span 5–280 units per dimension. Do not mark ordinary design details, glare, sleeves, shadows, or uncertain flaws. "
        "Exclude evidence that cannot be approximately located; never add placeholders for the 8 standard locations. An empty array is valid. "
        "detailed_inspection must be an object with front and back keys; each side has corners and edges objects. "
        "Each corners object must have top_left, top_right, bottom_left, bottom_right keys. "
        "Each edges object must have top, right, bottom, left keys. "
        "Each of these 16 entries must be a concise observation based only on visible evidence; "
        "use Cannot assess reliably if a specific area is obscured or unclear. "
        "Describe possible wear and its location without inventing defects. "
        "The five condition fields and grade_reason must be short descriptive strings. "
        "estimated_grade, grade_low, grade_high must be integers from 1 to 10. "
        "Base the estimate on visible evidence only; grade_low and grade_high must express meaningful uncertainty. "
        "Do not assume perfect condition from unclear photos. If quality is poor, widen the range. "
        "This is an experimental visual condition assessment, "
        "not an official PSA grade."
    ) 
    # Give the model the actually measured ratios as supporting context for
    # its CENTERING condition explanation; not as a forced card grade.
    # These are image-derived experimental values, not calibrated PSA ratios.
    # Image analysis runs only AFTER rate-limit checks, to limit server CPU use.
    from precision_centering import measure_pair
    independent_geometry = measure_pair(front_image, back_image)
    supported = []
    for inspected_side in ('front', 'back'):
        measured = independent_geometry.get(inspected_side) or {}
        if measured.get('status') in ('measured', 'partial'):
            values = []
            if measured.get('lr') is not None:
                values.append('left/right ' + str(measured['lr']) + '/' + str(round(100 - measured['lr'], 1)))
            if measured.get('tb') is not None:
                values.append('top/bottom ' + str(measured['tb']) + '/' + str(round(100 - measured['tb'], 1)))
            supported.append(inspected_side.upper() + ': ' + ', '.join(values))
    if supported:
        grading_instructions += (' Independent repeatable print-frame measurements from photo geometry: '
            + '; '.join(supported) + '. These are experimental measurements, not official grading criteria. '
            'Use them as supporting evidence when explaining the centering category, but independently review '
            'image quality and card design. Do not pretend unmeasured borders were verified.')
    else:
        grading_instructions += (' Automatic photo centering did not find a repeatable printed frame on either side. '
                                 'Do not invent numerical border splits from preset overlays or catalog art.')
    q.append(now)

    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': 'Bearer ' + key
            },
            json={
                'model': os.getenv('OPENAI_VISION_MODEL', 'gpt-4.1-mini'),
                'messages': [
                    {
                        'role': 'system',
                        'content': grading_instructions
                    },
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Inspect these two photos of the same trading card. The first photo is the FRONT and the second is the BACK. Return only JSON.'
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': 'data:image/jpeg;base64,' + front_image
                                }
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': 'data:image/jpeg;base64,' + back_image
                                }
                            }
                        ]
                    }
                ],
                'response_format': {'type': 'json_object'},
                'max_tokens': 4200
            },
            timeout=90
        )
        if not response.ok:
            if response.status_code == 429:
                return jsonify(
                    error='AI rate limit or insufficient API quota.'
                ), 429

            app.logger.warning(
                'Grading provider status %s',
                response.status_code
            )
            return jsonify(
                error='AI grading provider error.'
            ), 502

        result = json.loads(
            response.json()['choices'][0]['message']['content']
        )

        fields = (
            'corners',
            'edges',
            'surface',
            'photo_quality',
            'overall_notes',
            'grade_reason'
        )

        clean = {
            field: str(result.get(field, '') or '')[:500]
            for field in fields
        }

        try:
            estimate = int(result['estimated_grade'])
            low = int(result['grade_low'])
            high = int(result['grade_high'])
            if not (1 <= low <= estimate <= high <= 10):
                raise ValueError('Invalid grade range')
        except (KeyError, TypeError, ValueError):
            estimate = low = high = None

        # Normalize all 16 location-specific observations without trusting arbitrary model structure.
        detail = result.get('detailed_inspection')
        detail = detail if isinstance(detail, dict) else {}
        normalized = {}
        for side in ('front', 'back'):
            side_data = detail.get(side)
            side_data = side_data if isinstance(side_data, dict) else {}
            normalized[side] = {}
            for area, locations in (
                ('corners', ('top_left', 'top_right', 'bottom_left', 'bottom_right')),
                ('edges', ('top', 'right', 'bottom', 'left')),
            ):
                values = side_data.get(area)
                values = values if isinstance(values, dict) else {}
                normalized[side][area] = {
                    location: str(values.get(location) or 'Cannot assess reliably')[:220]
                    for location in locations
                }
        clean['detailed_inspection'] = normalized

        # Return evidence-based category scores in the SAME AI request as the
        # condition report. Missing or malformed scores remain null (never
        # inferred from the overall grade). The dashboard understands this key.
        from grade_scores_blueprint import _validated_score
        category_scores = result.get('category_scores')
        if not isinstance(category_scores, dict):
            category_scores = {}
        clean['category_scores'] = {
            category: _validated_score(category_scores.get(category))
            for category in ('centering', 'corners', 'edges', 'surface')
        }
        category_notes = result.get('category_score_notes')
        category_notes = category_notes if isinstance(category_notes, dict) else {}
        clean['category_score_notes'] = {
            category: str(category_notes.get(category) or '')[:350]
            for category in ('centering', 'corners', 'edges', 'surface')
        }
        # Independent front/back assessments for evidence-based per-side reports.
        # A missing or invalid AI score remains null, never copied from the overall grade.
        side_scores = result.get('side_scores')
        side_scores = side_scores if isinstance(side_scores, dict) else {}
        clean['side_scores'] = {
            side: _validated_score(side_scores.get(side))
            for side in ('front', 'back')
        }
        side_notes = result.get('side_score_notes')
        side_notes = side_notes if isinstance(side_notes, dict) else {}
        clean['side_score_notes'] = {
            side: str(side_notes.get(side) or '')[:500]
            for side in ('front', 'back')
        }
        clean['category_score_status'] = (
            'Scores supplied by AI'
            if all(score is not None for score in clean['category_scores'].values())
            else 'Some categories cannot be assessed from the photographs'
        )
        # V29: second-pass crop verification. If unsupported or unavailable, no map markers.
        from evidence import verify_evidence
        proposed_evidence = result.get("evidence_candidates")
        clean["evidence_findings"] = verify_evidence(
            proposed_evidence, {"front":front_image,"back":back_image}, key)
        clean["evidence_version"] = 1
        clean["evidence_explanation"] = (
            "Markers are approximate AI-corroborated photo regions, not pixel-accurate "
            "damage detection or a laboratory measurement. No marker does not prove no flaw.")
        # V32: Ensemble outer-edge detection with conservative border measurement.
        clean["centering_geometry"] = independent_geometry
        clean.update(estimated_grade=estimate, grade_low=low, grade_high=high)

        return jsonify(
            grading=clean,
            verified=False
        )

    except (
        requests.RequestException,
        KeyError,
        IndexError,
        ValueError,
        TypeError
    ):
        app.logger.warning('AI grading request failed')
        return jsonify(
            error='AI grading unavailable or returned invalid data.'
        ), 502        
   
                


# V8 Phase 7 — eBay active listing research (NOT completed-sale values).
# Production eBay OAuth credentials must be stored in Render environment variables.
import threading
import statistics

_ebay_lock = threading.Lock()
_ebay_token = {'value': None, 'expires': 0}
_ebay_requests = defaultdict(deque)


def _ebay_access_token():
    now = time.time()
    with _ebay_lock:
        if _ebay_token['value'] and now < _ebay_token['expires']:
            return _ebay_token['value']
        client_id = os.getenv('EBAY_CLIENT_ID')
        client_secret = os.getenv('EBAY_CLIENT_SECRET')
        if not client_id or not client_secret:
            raise RuntimeError('eBay credentials are not configured')
        response = requests.post(
            'https://api.ebay.com/identity/v1/oauth2/token',
            auth=(client_id, client_secret),
            data={'grant_type': 'client_credentials',
                  'scope': 'https://api.ebay.com/oauth/api_scope'},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload['access_token']
        _ebay_token.update(value=token, expires=now + max(60, int(payload.get('expires_in', 7200)) - 120))
        return token


def _ebay_listings(query, token):
    response = requests.get(
        'https://api.ebay.com/buy/browse/v1/item_summary/search',
        headers={'Authorization': 'Bearer ' + token,
                 'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US',
                 'Accept': 'application/json'},
        params={'q': query, 'limit': '35', 'filter': 'buyingOptions:{FIXED_PRICE}'},
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get('itemSummaries', [])


def _ebay_grade_match(title, grade):
    import re
    title = title.upper()
    # Only include explicit PSA grade listings in graded buckets.
    has_psa = bool(re.search(r'\bPSA\s*(?:GEM\s*MT\s*)?(?:10|9|8|7|6|5|4|3|2|1)\b', title))
    if grade == 'raw':
        # Avoid mistaking other slabbed cards for raw cards.
        return not (has_psa or re.search(r'\b(BGS|CGC|SGC|GRADED|SLABBED)\b', title))
    return bool(re.search(r'\bPSA\s*(?:GEM\s*MT\s*)?' + grade + r'\b', title))


def _ebay_bucket(query, grade, token):
    import re
    suffix = {'raw': ' -PSA -BGS -CGC -SGC', '9': ' PSA 9', '10': ' PSA 10'}[grade]
    listings = _ebay_listings(query + suffix, token)
    matches = []
    for item in listings:
        title = str(item.get('title', ''))
        if not _ebay_grade_match(title, grade):
            continue
        price = item.get('price') or {}
        if price.get('currency') != 'USD':
            continue
        try:
            amount = float(price['value'])
        except (KeyError, ValueError, TypeError):
            continue
        if not 0 < amount < 1000000:
            continue
        url = item.get('itemWebUrl', '')
        if not url.startswith('https://www.ebay.com/'):
            continue
        matches.append({'title': title[:180], 'price_usd': round(amount, 2),
                        'url': url, 'item_id': str(item.get('itemId', ''))[:150]})
    values = [x['price_usd'] for x in matches]
    return {'median_asking_price_usd': round(statistics.median(values), 2) if values else None,
            'listing_count': len(matches), 'listings': matches[:12]}


@app.route('/api/ebay/prices', methods=['GET', 'OPTIONS'])
def ebay_prices():
    if request.method == 'OPTIONS':
        return ('', 204)
    origin = request.headers.get('Origin', '').rstrip('/')
    if origin and origin not in ALLOWED_ORIGINS:
        return jsonify(error='Origin not allowed'), 403
    query = (request.args.get('q') or '').strip()
    if len(query) < 3 or len(query) > 120 or any(ord(c) < 32 for c in query):
        return jsonify(error='Enter a card name (3–120 characters). Include set, number and variant for accuracy.'), 400
    ip = request.remote_addr or 'unknown'
    now = time.monotonic()
    rate = _ebay_requests[ip]
    while rate and now - rate[0] > 60:
        rate.popleft()
    if len(rate) >= 12:
        return jsonify(error='Too many price lookups. Try again in a minute.'), 429
    rate.append(now)
    try:
        token = _ebay_access_token()
        result = {grade: _ebay_bucket(query, grade, token) for grade in ('raw', '9', '10')}
    except RuntimeError:
        return jsonify(error='eBay Production credentials are not configured on the backend.'), 503
    except (requests.RequestException, ValueError, KeyError):
        app.logger.warning('eBay pricing request failed')
        return jsonify(error='eBay listing data is temporarily unavailable. Check API access and try again.'), 502
    payload = {'query': query, 'source': 'eBay active fixed-price listings',
               'price_type': 'asking_price_not_sold_value', 'currency': 'USD',
               'retrieved_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
               'grades': result,
               'disclaimer': 'Active listing asking prices are not completed-sale market values. Search results may include mismatched variants; verify card, set, number and condition.'}
    return jsonify(payload)

@app.errorhandler(413)
def too_large(_):return jsonify(error='Photo exceeds 8 MB.'),413

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','10000')))