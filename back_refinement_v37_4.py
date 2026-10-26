"""Conservative, research-only physical edge micro-refinement for Pokémon backs.

Starts from the trusted V36 blue-stock silhouette. Samples the light-to-dark
transition crossing each edge, and accepts only independent, consistent samples.
Does NOT measure printed centering, grade a card, or change the V36 baseline when
edge evidence is insufficient. All results require user review and lock.
"""
from __future__ import annotations
import cv2
import numpy as np


SIDES=('top','right','bottom','left')


def _side_measure(lum, q, side):
    p0=np.asarray(q[side],np.float32); p1=np.asarray(q[(side+1)%4],np.float32)
    v=p1-p0; normlen=np.linalg.norm(v)
    if normlen<50:return {'usable':False,'reason':'Edge too short.'}
    v=v/normlen
    inward=np.float32([-v[1],v[0]]) # TL,TR,BR,BL clockwise
    offsets=[]; strengths=[]
    offs=np.linspace(-8,8,81,dtype=np.float32)
    for t in np.linspace(.13,.87,29,dtype=np.float32):
        origin=p0*(1-t)+p1*t
        points=origin+offs[:,None]*inward
        image_line=cv2.remap(lum,points[:,0].reshape(1,-1),points[:,1].reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)[0]
        image_line=cv2.GaussianBlur(image_line.reshape(1,-1),(1,11),1.6)[0]
        slope=-np.gradient(image_line)
        ix=int(np.argmax(slope));offset=float(offs[ix]);strength=float(slope[ix])
        if strength>=3.5:offsets.append(offset);strengths.append(strength)
    if len(offsets)<23:return {'usable':False,'reason':'Weak light-to-dark border contrast.','samples':len(offsets)}
    med=float(np.median(offsets));mad=float(np.median(np.abs(np.asarray(offsets)-med)))
    # Reject printed interior edges and highly variable shadow boundaries.
    if abs(med)>2.25:return {'usable':False,'reason':'Potential edge is too far from V36 baseline.','offset_px':round(med,2),'samples':len(offsets)}
    if mad>.65:return {'usable':False,'reason':'Edge transitions vary along side.','offset_px':round(med,2),'median_abs_deviation_px':round(mad,2),'samples':len(offsets)}
    if np.percentile(offsets,90)-np.percentile(offsets,10)>2.2:
        return {'usable':False,'reason':'Edge transition spread is too wide.','offset_px':round(med,2),'samples':len(offsets)}
    return {'usable':True,'offset_px':round(med,2),'median_abs_deviation_px':round(mad,2),
            'samples':len(offsets),'median_contrast':round(float(np.median(strengths)),2)}


def _intersect(pa,pb,qa,qb):
    r=pb-pa;s=qb-qa
    cross=lambda u,v:float(u[0]*v[1]-u[1]*v[0])
    den=cross(r,s)
    if abs(den)<1e-3:raise ValueError('Parallel candidate lines.')
    return pa+cross(qa-pa,s)/den*r


def analyze_pokemon_back_edges(bgr, baseline_quad):
    """Return diagnostic results plus a guarded candidate; never a grade."""
    if bgr is None or not hasattr(bgr,'shape') or len(bgr.shape)!=3 or bgr.shape[2]!=3:
        return {'status':'unavailable','reason':'A valid color photo is required.'}
    h,w=bgr.shape[:2]
    q=np.asarray(baseline_quad,np.float32)
    if q.shape!=(4,2) or not np.isfinite(q).all() or not cv2.isContourConvex(q.astype(np.int32)) or cv2.contourArea(q)<.08*w*h:
        return {'status':'unavailable','reason':'V36 baseline has invalid geometry.'}
    lum=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    measurements=[_side_measure(lum,q,i) for i in range(4)]
    detail={name:val for name,val in zip(SIDES,measurements)}
    usable=sum(bool(v.get('usable')) for v in measurements)
    if usable<3:
        return {'status':'baseline_preserved','reason':'Fewer than three sides have independently consistent border transitions.',
                'baseline_quad':q.round(2).tolist(),'refined_quad':None,'side_evidence':detail,'review_required':True}
    displaced=[]
    for i in range(4):
        p0,p1=q[i],q[(i+1)%4];v=p1-p0;v=v/max(float(np.linalg.norm(v)),1e-6);inner=np.array([-v[1],v[0]],np.float32)
        displacement=(float(measurements[i]['offset_px']) if measurements[i].get('usable') else 0.)
        displaced.append((p0+inner*displacement,p1+inner*displacement))
    try:
        new=np.float32([_intersect(*displaced[3],*displaced[0]),_intersect(*displaced[0],*displaced[1]),
                        _intersect(*displaced[1],*displaced[2]),_intersect(*displaced[2],*displaced[3])])
    except ValueError:
        return {'status':'baseline_preserved','reason':'Refined edge intersections were unstable.','baseline_quad':q.round(2).tolist(),'refined_quad':None,'side_evidence':detail,'review_required':True}
    move=np.linalg.norm(new-q,axis=1)
    if not cv2.isContourConvex(new.astype(np.int32)) or (new[:,0]<0).any() or (new[:,0]>=w).any() or (new[:,1]<0).any() or (new[:,1]>=h).any() or max(move)>3.2:
        return {'status':'baseline_preserved','reason':'Refined corner motion failed safety geometry checks.',
                'baseline_quad':q.round(2).tolist(),'refined_quad':None,'side_evidence':detail,'max_corner_shift_px':round(float(max(move)),2),'review_required':True}
    return {'status':'micro_refinement_candidate','method':'v37_4_guarded_back_gradient_crosscheck',
            'baseline_quad':q.round(2).tolist(),'refined_quad':new.round(2).tolist(),
            'max_corner_shift_px':round(float(max(move)),2),'median_corner_shift_px':round(float(np.median(move)),2),
            'side_evidence':detail,'review_required':True,'verified_card_edges':False,
            'note':'Candidate derived from physical-card border gradients, not measured printed centering. Requires independent review.'}
