"""V36 candidate outer-card-edge refinement, research only.

A *confirmed* V34 SIFT registration gives an approximate card location, but
its catalog crop is NOT a calibrated card perimeter. This module samples image
color transitions near each expected physical edge, fits a slightly shifted
quadrilateral, and returns per-side evidence. It never grades or measures
print centering. No OpenAI or PyTorch calls; CPU OpenCV/NumPy only.
"""
from __future__ import annotations
import math
from typing import Any
import cv2
import numpy as np

SIDES = ('top', 'right', 'bottom', 'left')


def _signed_area(q):
    return float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])) / 2


def _check_quad(q, w, h):
    q = np.asarray(q, np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return False
    if cv2.contourArea(q) < 0.09 * w * h:
        return False
    if np.any(q[:, 0] < -0.03*w) or np.any(q[:, 0] > 1.03*w) or np.any(q[:, 1] < -0.03*h) or np.any(q[:, 1] > 1.03*h):
        return False
    # Four non-self-crossing corners in TL,TR,BR,BL order, convex card.
    cross = [float((q[(i+1)%4,0]-q[i,0])*(q[(i+2)%4,1]-q[(i+1)%4,1])-(q[(i+1)%4,1]-q[i,1])*(q[(i+2)%4,0]-q[(i+1)%4,0])) for i in range(4)]
    if min(cross) <= 0: return False
    return True


def _line_segment_measure(lab, a, b, inward, d, eps, t_values):
    """Delta-LAB across a candidate edge, median over interior scan points."""
    h,w = lab.shape[:2]
    v = b-a
    points = a[None,:] + t_values[:,None]*v[None,:] + d * inward[None,:]
    inside = points + eps*inward
    outside = points - eps*inward
    ok = ((inside[:,0]>=1)&(inside[:,0]<w-2)&(outside[:,0]>=1)&(outside[:,0]<w-2)&
          (inside[:,1]>=1)&(inside[:,1]<h-2)&(outside[:,1]>=1)&(outside[:,1]<h-2))
    if ok.sum() < int(.8*len(t_values)):
        return None
    def interp(ps):
        x=ps[:,0].astype(np.float32).reshape(1,-1);y=ps[:,1].astype(np.float32).reshape(1,-1)
        return cv2.remap(lab, x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).reshape(-1,3)
    inn=interp(inside[ok]);out=interp(outside[ok]);diff=np.linalg.norm(inn-out,axis=1)
    return float(np.median(diff)),float(np.mean(diff>16)),float(np.percentile(diff,25))


def refine_photo_edges(photo_bgr, template_photo_quad, *, include_overlay=True):
    """Refine physical cut edges from a projected exact-card reference.

    template_photo_quad: TL,TR,BR,BL in original-photo pixel coordinates;
    results in same units. Declines if the four side transitions are weak or
    unstable. Invalid/missing references are a hard failure, not an invitation
    to borrow the reference as a cut-line measurement.
    """
    h,w=photo_bgr.shape[:2]
    q=np.asarray(template_photo_quad,np.float32)
    if not _check_quad(q,w,h):
        return {'status':'unavailable','reason':'Reference projection is not a plausible four-corner card quadrilateral.'}
    card_width=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2
    card_height=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
    if card_width<150 or card_height<230:
        return {'status':'unavailable','reason':'Card too small for edge refinement.'}
    # More than 4% outward/inward would be suspicious for an exact-card template match.
    search_px = max(9.,min(28.,float(card_width)*.043))
    step=1.0 if card_width < 650 else 1.5
    shifts=np.arange(-search_px,search_px+step/2,step,dtype=np.float32)
    lab=cv2.cvtColor(photo_bgr,cv2.COLOR_BGR2LAB).astype(np.float32)
    center=q.mean(axis=0)
    measured=[];evidence={}
    for i,name in enumerate(SIDES):
        a=q[i];b=q[(i+1)%4]
        v=b-a;ln=np.linalg.norm(v)
        inward=center-(a+b)/2
        inward=inward/(np.linalg.norm(inward)+1e-9)
        # The color jump is sampled a few px inward/outward of the candidate.
        eps=max(2.4,min(5.0,card_width*.009))
        parts=[np.linspace(.13,.47,21,dtype=np.float32),np.linspace(.53,.87,21,dtype=np.float32)]
        scores=[]
        for d in shifts:
            pair=[_line_segment_measure(lab,a,b,inward,float(d),eps,ts) for ts in parts]
            if any(p is None for p in pair):scores.append((-1.,None));continue
            median=min(p[0] for p in pair)
            consistency=min(p[1] for p in pair)
            # Favor an actual two-sided color transition close to template.
            merit=median*(.65+.35*consistency) - .11*abs(float(d))
            scores.append((merit,pair))
        best_ix=int(np.argmax([x[0] for x in scores]))
        peak,segments=scores[best_ix]
        best_d=float(shifts[best_ix])
        side_delta=[float(s[0]) for s in segments] if segments else []
        if not segments or peak < 11.5 or min(side_delta)<11.5:
            return {'status':'unavailable','reason':f'{name.title()} physical boundary not clear enough for refinement.',
                    'side_evidence':evidence}
        # Diagnose whether the two halves imply different lines (e.g. shadow/wrinkle).
        half_picks=[]
        for j in range(2):
            options=[(-1 if s[1] is None else s[1][j][0]-.11*abs(float(shifts[k])),k) for k,s in enumerate(scores)]
            half_picks.append(float(shifts[max(options)[1]]))
        half_disagreement=abs(half_picks[0]-half_picks[1])
        if half_disagreement > max(7.0,card_width*.013):
            return {'status':'unavailable','reason':f'{name.title()} boundary disagrees along its length; possible shadow or background fold.',
                    'side_evidence':evidence}
        # Reject distant bright/shadow transitions rather than snap to a fold
        # or a white background edge just because it scores strongly.
        if abs(best_d) > min(18.0, card_width*.024):
            return {'status':'unavailable','reason':f'{name.title()} transition too far from the verified artwork; retake or independently review.',
                    'side_evidence':evidence}
        if abs(best_d) > search_px - step*1.2:
            return {'status':'unavailable','reason':f'{name.title()} boundary is outside the safe template-guided search window.',
                    'side_evidence':evidence}
        evidence[name]={
            'offset_from_uncalibrated_catalog_projection_px':round(best_d,2),
            'color_transition_strength':round(float(min(side_delta)),1),
            'two_half_disagreement_px':round(half_disagreement,2),
            'classification':'candidate_physical_cut_edge',
        }
        measured.append((a,b,inward,best_d))
    lines=[]
    for a,b,inward,d in measured:
        lines.append((a+d*inward,b+d*inward))
    corners=[]
    for i in range(4):
        # Intersect previous side and current side; avoid degenerate parallel edges.
        p,p2=lines[(i-1)%4];r,r2=lines[i];u=p2-p;v=r2-r
        M=np.stack([u,-v],axis=1)
        if abs(np.linalg.det(M))<1e-4:
            return {'status':'unavailable','reason':'Refined boundary lines do not intersect reliably.','side_evidence':evidence}
        t=np.linalg.solve(M,r-p)[0]
        corners.append(p+t*u)
    refined=np.asarray(corners,np.float32)
    if not _check_quad(refined,w,h):
        return {'status':'unavailable','reason':'Resulting boundary polygon failed geometry validation.','side_evidence':evidence}
    # Reject candidate if it differs too far from expected whole-card shape.
    if float(np.mean(np.linalg.norm(q-refined,axis=1))) > search_px*1.42:
        return {'status':'unavailable','reason':'Refinement exceeded safe template proximity.','side_evidence':evidence}
    result={
        'status':'candidate_found','method':'v36_template_guided_color_boundary',
        'physical_quad':[[round(float(x),2),round(float(y),2)] for x,y in refined],
        'side_evidence':evidence,'confidence':'research_only_uncalibrated',
        'verified_card_edges':False,'verified_centering':False,'centering_ratio':None,
        'note':'Four plausible color transitions were found near the projected template. Verify against labeled physical corners; these are not proof of accurate centering.'
    }
    if include_overlay:
        canvas=photo_bgr.copy()
        cv2.polylines(canvas,[np.int32(np.round(q))],True,(0,190,255),2,cv2.LINE_AA)
        cv2.polylines(canvas,[np.int32(np.round(refined))],True,(255,255,0),3,cv2.LINE_AA)
        for i,p in enumerate(refined):
            xy=tuple(np.round(p).astype(int))
            cv2.circle(canvas,xy,7,(45,250,160),-1,cv2.LINE_AA)
            cv2.putText(canvas,str(i+1),(xy[0]+9,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,255,240),2,cv2.LINE_AA)
        result['overlay_bgr']=canvas
    return result
