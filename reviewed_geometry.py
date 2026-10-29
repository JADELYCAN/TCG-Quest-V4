"""Validate V2 reviewed guides and calculate ratios server-side; no grading rubric.

Prepared for the existing Flask AI backend. Not active until installed there.
Normalized coordinates refer to the submitted, EXIF-oriented full photograph.
"""
import json
import math
import re


def _quad(raw):
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError('Four ordered points required')
    points = []
    for p in raw:
        if not isinstance(p, dict):
            raise ValueError('Invalid point')
        xy = [p.get('x'), p.get('y')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in xy):
            raise ValueError('Invalid normalized point')
        points.append(xy)
    turns = []
    for i in range(4):
        a, b, c = points[i], points[(i+1)%4], points[(i+2)%4]
        turns.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    if min(turns) <= 1e-6:
        raise ValueError('Clockwise convex TL/TR/BR/BL guides required')
    area = sum(points[i][0]*points[(i+1)%4][1]-points[(i+1)%4][0]*points[i][1] for i in range(4))/2
    if area < .0001:
        raise ValueError('Degenerate outline')
    return points


def _measure(side):
    import numpy as np
    outer, inner = _quad(side.get('outer')), _quad(side.get('inner'))
    matrix, rhs = [], []
    for (x,y),(u,v) in zip(outer, ((0,0),(1,0),(1,1),(0,1))):
        matrix.extend(([x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]))
        rhs.extend((u,v))
    try:
        h = np.linalg.solve(np.array(matrix), np.array(rhs))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Degenerate perspective transform') from exc
    q = []
    for x,y in inner:
        den = h[6]*x+h[7]*y+1
        if abs(den) < 1e-9:
            raise ValueError('Invalid projection')
        u,v = (h[0]*x+h[1]*y+h[2])/den, (h[3]*x+h[4]*y+h[5])/den
        if not (0 < u < 1 and 0 < v < 1):
            raise ValueError('Printed border must be inside card')
        q.append((u,v))
    left, right = (q[0][0]+q[3][0])/2, 1-(q[1][0]+q[2][0])/2
    top, bottom = (q[0][1]+q[1][1])/2, 1-(q[2][1]+q[3][1])/2
    if min(left,right,top,bottom) <= .001:
        raise ValueError('Printed border too close to edge')
    return dict(status='measured', method='collector_reviewed_v1', lr=round(100*left/(left+right),1), tb=round(100*top/(top+bottom),1), confidence='collector-reviewed, not independently verified')


def parse_reviewed_geometry(raw):
    if not raw:
        return None
    if not isinstance(raw,str) or len(raw) > 6000:
        raise ValueError('Invalid guide payload size')
    try:
        data=json.loads(raw)
    except (TypeError,ValueError) as exc:
        raise ValueError('Invalid guide JSON') from exc
    if not isinstance(data,dict) or data.get('version') != 1 or data.get('confirmed') is not True or not isinstance(data.get('revision'),str) or not re.fullmatch(r'[A-Za-z0-9-]{1,80}',data['revision']):
        raise ValueError('Invalid guide revision')
    if not all(isinstance(data.get(s),dict) for s in ('front','back')):
        raise ValueError('Both sides required')
    return dict(revision=data['revision'],front=_measure(data['front']),back=_measure(data['back']))


def prompt_context(data):
    if not data:
        return ''
    values = '; '.join(s.upper()+': L/R '+str(data[s]['lr'])+'/'+str(round(100-data[s]['lr'],1))+', T/B '+str(data[s]['tb'])+'/'+str(round(100-data[s]['tb'],1)) for s in ('front','back'))
    return (' Collector-reviewed V2 guide geometry calculated server-side: '+values+'. '
            'Consider these ratios when estimating centering and the overall grade, together with visible corners, edges and surface. '
            'They are user-positioned guides, NOT independently verified measurements. Compare them against the photographs and automatic geometry. '
            'If guides do not represent a genuine printed frame, or measurements conflict materially, explain the conflict and return null for centering and all overall grade fields. '
            'Do not replace absent condition evidence with a perfect score. Explain how supported centering affected the estimate; do not claim a calibrated grading formula.')
