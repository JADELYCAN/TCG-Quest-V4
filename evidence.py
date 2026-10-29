"""V29: conservative, optional AI photo evidence validation.

Candidate coordinates are relative to the original *submitted image*, on a 0-1000 grid.
A second vision pass reviews close-up image crops and suppresses unsupported
candidates. Validated = corroborated by a model, NOT laboratory proof.
"""
from __future__ import annotations
import base64
import io
import json
import math
import os
import requests
from PIL import Image, ImageDraw

LIMIT_PER_SIDE = 6
LIMIT_TOTAL = 10
TYPES = {'corner_whitening', 'corner_wear', 'edge_chipping', 'edge_wear',
         'surface_scratch', 'print_line', 'crease', 'dent', 'stain', 'other_visible_flaw'}


def normalize_candidates(raw):
    if not isinstance(raw, list):
        return []
    out = []
    counts = {'front': 0, 'back': 0}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        side = entry.get('side')
        kind = entry.get('type')
        box = entry.get('bbox')
        if side not in counts or kind not in TYPES or not isinstance(box, list) or len(box) != 4:
            continue
        if counts[side] >= LIMIT_PER_SIDE or len(out) >= LIMIT_TOTAL:
            continue
        if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in box):
            continue
        x1, y1, x2, y2 = [float(n) for n in box]
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            continue
        if x2 - x1 > 280 or y2 - y1 > 280 or min(x2 - x1, y2 - y1) < 5:
            continue
        note = str(entry.get('observation') or '').strip()[:240]
        if len(note) < 12 or entry.get('confidence') not in ('medium', 'high'):
            continue
        counts[side] += 1
        out.append({'id': 'F' + str(counts['front']) if side == 'front' else 'B' + str(counts['back']),
                    'side': side, 'type': kind, 'bbox': [round(n, 2) for n in (x1, y1, x2, y2)],
                    'observation': note, 'confidence': entry['confidence']})
    return out


def on_card(box, quad):
    """Require a valid convex outline and a marker center inside it; fail closed."""
    if not isinstance(quad, (list, tuple)) or len(quad) != 4:
        return False
    try:
        if any(len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in quad):
            return False
        def cross(a,b,c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        turns=[cross(quad[i],quad[(i+1)%4],quad[(i+2)%4]) for i in range(4)]
        if not (min(turns)>1e-6 or max(turns)<-1e-6):
            return False
        center=((box[0]+box[2])/2000,(box[1]+box[3])/2000)
        edges=[cross(quad[i],quad[(i+1)%4],center) for i in range(4)]
        return min(edges)>=-1e-9 or max(edges)<=1e-9
    except (TypeError,ValueError):
        return False


def _crop_data_url(encoded_jpg, box):
    raw = base64.b64decode(encoded_jpg)
    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
        x1, y1, x2, y2 = box
        cx = (x1+x2)/2000 * w
        cy = (y1+y2)/2000 * h
        # Add enough context to see the corner/edge, not an invented single-pixel finding.
        radius = max(75, (x2-x1)/1000*w*1.6, (y2-y1)/1000*h*1.6)
        xlo, ylo = max(0, round(cx-radius)), max(0, round(cy-radius))
        xhi, yhi = min(w, round(cx+radius)), min(h, round(cy+radius))
        crop = im.crop((xlo, ylo, xhi, yhi)).convert('RGB')
        # Highlight the exact proposed full-photo box, not the wider context.
        ImageDraw.Draw(crop).rectangle((x1/1000*w-xlo, y1/1000*h-ylo,
                                       x2/1000*w-xlo, y2/1000*h-ylo), outline=(255,0,255), width=3)
        crop.thumbnail((900, 900))
        buf = io.BytesIO()
        crop.convert('RGB').save(buf, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def verify_evidence(proposed, photos, api_key, model=None, card_quads=None):
    """Return only defects the second vision pass can corroborate.

    On provider failure return [], not invented or unverified map markers.
    """
    candidates = [c for c in normalize_candidates(proposed) if on_card(c['bbox'], (card_quads or {}).get(c['side']))]
    if not candidates or not api_key:
        return []
    content = [{'type': 'text', 'text': (
        'You independently assess candidate visible card flaws in the labeled photo crops below. '
        'Each image is a crop from the collector-uploaded photo and may include glare, print artwork, '
        'reflections, or lens blur. For each id: confirmed=true ONLY if there is credible VISIBLE '
        'evidence for the described flaw INSIDE the magenta rectangle; otherwise false. The rectangle is a synthetic annotation, not damage. Surrounding pixels provide context only; never confirm a nearby flaw outside the rectangle. Be conservative. '
        'Return ONLY JSON {"results":[{"id":"F1","confirmed":false,"reason":"..."}]}. '
        'Do not confirm something based only on the description; inspect the pixels.'
    )}]
    sent = []
    for item in candidates:
        try:
            cropped = _crop_data_url(photos[item['side']], item['bbox'])
        except (ValueError, OSError):
            continue
        sent.append(item)
        content.append({'type':'text','text':item['id']+' / '+item['side']+' / '+item['type']+' / candidate: '+item['observation']})
        content.append({'type':'image_url','image_url':{'url':cropped,'detail':'high'}})
    if len(content) < 3:
        return []
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions',
            headers={'Authorization':'Bearer '+api_key}, json={
                'model': model or os.getenv('OPENAI_VISION_MODEL','gpt-4.1-mini'),
                'messages':[{'role':'user','content':content}],
                'response_format':{'type':'json_object'}, 'max_tokens':950}, timeout=65)
        if not response.ok:
            return []
        data = json.loads(response.json()['choices'][0]['message']['content'])
        verdicts = {r.get('id'):r for r in data.get('results', []) if isinstance(r, dict)}
        result = []
        for item in sent:
            verdict = verdicts.get(item['id'])
            if isinstance(verdict, dict) and verdict.get('confirmed') is True:
                result.append({**item, 'status':'AI corroborated',
                               'verification_note':str(verdict.get('reason') or '')[:240]})
        return result
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return []
