"""Research-only Pokemon back cut-edge detector using the uniform blue stock design.

This detects the *physical* cut boundary; it does not infer printed centering.
Returns no grade and withholds ambiguous candidates. CPU OpenCV only.
"""
from __future__ import annotations
import cv2
import numpy as np


def detect_pokemon_back_cut_edge(bgr):
    h, w = bgr.shape[:2]
    if min(h, w) < 350:
        return {'status':'unavailable', 'reason':'Image too small for back cut-edge detection.'}
    b,g,r=[x.astype(np.float32) for x in cv2.split(bgr)]
    # Blue printed stock dominates Pokemon backs, but not their neutral backgrounds.
    # Deliberately avoid a generic dark-pixel threshold: dark shadows can merge with cards.
    mask=((b>r*1.14)&(b>g*1.07)&(b>35)&(r<195)).astype(np.uint8)*255
    k=max(5,(round(min(h,w)*.011)//2)*2+1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return {'status':'unavailable','reason':'Blue card region not found.'}
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    c=contours[0];x,y,bw,bh=cv2.boundingRect(c)
    area=float(cv2.contourArea(c));aspect=bw/max(bh,1)
    if area<.11*w*h or not (.55<aspect<.91) or bw<.26*w or bh<.30*h:
        return {'status':'unavailable','reason':'No plausible Pokémon back silhouette found.'}
    # Check that dominant blue region itself is coherent, not a distant patch.
    fill=float(np.count_nonzero(mask[y:y+bh,x:x+bw]))/(bw*bh)
    if fill<.25:return {'status':'unavailable','reason':'Blue backing region is fragmented.'}
    # Fit side borders using contour extrema sampled away from rounded corners.
    def line_fits(which):
        samples=[]
        if which in ('left','right'):
            for yy in np.linspace(y+.16*bh,y+.84*bh,31):
                row=mask[int(yy),max(0,x-18):min(w,x+bw+18)]
                xs=np.flatnonzero(row)
                if len(xs)<.35*bw: continue
                val=(xs[0] if which=='left' else xs[-1])+max(0,x-18)
                samples.append((yy,float(val)))
        else:
            for xx in np.linspace(x+.16*bw,x+.84*bw,31):
                col=mask[max(0,y-18):min(h,y+bh+18),int(xx)]
                ys=np.flatnonzero(col)
                if len(ys)<.35*bh:continue
                val=(ys[0] if which=='top' else ys[-1])+max(0,y-18)
                samples.append((xx,float(val)))
        if len(samples)<15:return None
        samples=np.array(samples,dtype=np.float32)
        a,beta=np.polyfit(samples[:,0],samples[:,1],1)
        residual=np.abs(samples[:,1]-(a*samples[:,0]+beta))
        good=residual<max(3,0.006*min(bw,bh))
        if good.sum()<.7*len(samples):return None
        a,beta=np.polyfit(samples[good,0],samples[good,1],1)
        error=float(np.median(np.abs(samples[good,1]-(a*samples[good,0]+beta))))
        return (float(a),float(beta),error)
    left,right,top,bottom=[line_fits(key) for key in ('left','right','top','bottom')]
    if any(z is None for z in (left,right,top,bottom)):
        return {'status':'unavailable','reason':'Back cut edges disagree along their lengths.'}
    # left/right: x=a*y+b; top/bottom: y=c*x+d
    def cross(s,t):
        a,b,_=s;c,d,_=t;den=1-c*a
        if abs(den)<.1:raise ValueError('nearly parallel')
        xx=(a*d+b)/den;yy=c*xx+d
        return [float(xx),float(yy)]
    try:
        q=np.float32([cross(left,top),cross(right,top),cross(right,bottom),cross(left,bottom)])
    except Exception:return {'status':'unavailable','reason':'Could not intersect the four cut edges.'}
    if np.any(q[:,0]<0) or np.any(q[:,0]>=w) or np.any(q[:,1]<0) or np.any(q[:,1]>=h):
        return {'status':'unavailable','reason':'Back border falls outside the photograph.'}
    if not cv2.isContourConvex(q.astype(np.int32)) or cv2.contourArea(q)<.1*w*h:
        return {'status':'unavailable','reason':'Back cut boundary failed geometry checks.'}
    # Cross-check against mask extent (prevents art-only interior rectangles).
    qleft=float(min(q[0,0],q[3,0]));qright=float(max(q[1,0],q[2,0]))
    if abs(qleft-x)>.03*bw or abs(qright-(x+bw-1))>.03*bw:
        return {'status':'unavailable','reason':'Blue-frame edge fit is not consistent with full silhouette.'}
    return {'status':'candidate_found','method':'v36_1_pokemon_back_blue_stock_boundary',
            'physical_quad':[[round(float(u),2),round(float(v),2)] for u,v in q],
            'confidence':'research_candidate_only','verified_card_edges':False,
            'edge_fit_median_px':{k:round(v[2],2) for k,v in zip(('left','right','top','bottom'),(left,right,top,bottom))},
            'note':'Blue-stock physical cut-edge candidate; verify all four corners before locking. This is not a centering measurement.'}
