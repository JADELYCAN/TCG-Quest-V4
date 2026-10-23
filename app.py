import os, io, base64, time, json
from collections import defaultdict, deque
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
import requests

app=Flask(__name__, static_folder='assets')
app.config['MAX_CONTENT_LENGTH']=8*1024*1024+4096

ALLOWED_ORIGINS={x.strip() for x in os.getenv('ALLOWED_ORIGINS','https://tcg-quest-v4.onrender.com,https://tcg-quest-v4-ai.onrender.com,https://jadelycan.github.io').split(',') if x.strip()}
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
        "corners, edges, surface, photo_quality, overall_notes, estimated_grade, grade_low, grade_high, grade_reason. "
        "The five condition fields and grade_reason must be short descriptive strings. "
        "estimated_grade, grade_low, grade_high must be integers from 1 to 10. "
        "Base the estimate on visible evidence only; grade_low and grade_high must express meaningful uncertainty. "
        "Do not assume perfect condition from unclear photos. If quality is poor, widen the range. "
        "This is an experimental visual condition assessment, "
        "not an official PSA grade."
    ) 
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
                'max_tokens': 600
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
   
                
@app.errorhandler(413)
def too_large(_):return jsonify(error='Photo exceeds 8 MB.'),413

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','10000')))
