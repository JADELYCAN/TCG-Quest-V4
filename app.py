import os, io, base64, time
from collections import defaultdict, deque
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, UnidentifiedImageError
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

@app.errorhandler(413)
def too_large(_):return jsonify(error='Photo exceeds 8 MB.'),413

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','10000')))
