"""TCG Quest V37.1: conservative physical cut-edge refinement (research only).

Seed is the V36/V36.1 candidate, NEVER a trusted calibration. Searches locally
around each straight physical edge, estimates supporting photo transitions, and
rejects insufficient/discordant samples. Cannot grade printed centering.
"""
from __future__ import annotations
import cv2
import numpy as np

NAMES=('top','right','bottom','left')

def _valid(q,w,h):
    a=np.asarray(q,dtype=np.float64)
    if a.shape!=(4,2) or not np.isfinite(a).all():return False
    if np.min(a[:,0])<1 or np.max(a[:,0])>=w-1 or np.min(a[:,1])<1 or np.max(a[:,1])>=h-1:return False
    if cv2.contourArea(a.astype(np.float32))<0.08*w*h:return False
    cross=[float((a[(i+1)%4,0]-a[i,0])*(a[(i+2)%4,1]-a[(i+1)%4,1])-(a[(i+1)%4,1]-a[i,1])*(a[(i+2)%4,0]-a[(i+1)%4,0])) for i in range(4)]
    return min(cross)>0

def _bilinear(im,xy):
    pts=np.asarray(xy,dtype=np.float32)
    return cv2.remap(im,pts[...,0].reshape(1,-1),pts[...,1].reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101).reshape(-1,im.shape[2] if im.ndim==3 else 1)

def _side_samples(lab,base,a,b,center,card_width,side):
    h,w=lab.shape[:2];v=b-a;unit=v/(np.linalg.norm(v)+1e-8)
    normal=center-.5*(a+b);normal=normal/(np.linalg.norm(normal)+1e-8)
    # Search around the *existing cut edge*, never drift to an arbitrary line.
    radius=float(max(4.0,min(13.0,card_width*.018)))
    offsets=np.arange(-radius,radius+0.5,0.5,dtype=np.float32)
    eps=float(max(2.2,min(4.6,card_width*.007)))
    points=[];evidence=[]
    # Avoid naturally rounded corners and the central fold/shadow if inconsistent.
    for t in np.linspace(.13,.87,53):
        basepoint=a*(1-t)+b*t
        candidate=basepoint+offsets[:,None]*normal
        inside=candidate+eps*normal;outside=candidate-eps*normal
        if not (np.all(inside>=2) and np.all(outside>=2) and np.all(inside[:,0]<w-3) and np.all(outside[:,0]<w-3) and np.all(inside[:,1]<h-3) and np.all(outside[:,1]<h-3)):
            continue
        inn=_bilinear(lab,inside).astype(np.float32);out=_bilinear(lab,outside).astype(np.float32)
        # Boundary should transition away from low-color interior to relatively
        # light background. Color distance detects silver/blue edges.
        chroma=np.linalg.norm(out-inn,axis=1)
        light=out[:,0]-inn[:,0]
        scores=chroma+.40*np.maximum(light,0)-1.1*np.abs(offsets)
        scores[light<5]-=20
        index=int(np.argmax(scores))
        if chroma[index]<13 or light[index]<8 or scores[index]<14:continue
        selected=float(offsets[index])
        # Subpixel peak only when peak is safely interior and locally concave.
        if 0<index<len(offsets)-1:
            y0,y1,y2=[float(scores[k]) for k in (index-1,index,index+1)]
            den=y0-2*y1+y2
            if den<-.5: selected+=float(np.clip(.25*(y0-y2)/den,-.5,.5))
        points.append((float(t),selected))
        evidence.append((float(chroma[index]),float(light[index])))
    return points,evidence,radius

def _fit_side(points, strengths, longside):
    if len(points)<28:return None
    vals=np.asarray(points,float);t=vals[:,0];d=vals[:,1]
    # Robust line fit in t instead of selecting one offset for entire side.
    # Base fit from median; exclude large gaps and inconsistent background lines.
    a,b=np.polyfit(t,d,1)
    for _ in range(5):
        residual=np.abs(d-(a*t+b));med=np.median(residual)
        keep=residual<=max(1.25,2.7*med)
        if keep.sum()<25:return None
        a,b=np.polyfit(t[keep],d[keep],1)
    residual=np.abs(d-(a*t+b));keep=residual<max(2.,.003*longside)
    coverage=float((t[keep].max()-t[keep].min()) if keep.any() else 0)
    if keep.sum()<28 or coverage<.56:return None
    mad=float(np.median(residual[keep]));fraction=float(keep.mean())
    if mad>2.1 or fraction<.73:return None
    # value at endpoint t=0 and t=1
    return float(b),float(a+b),{'support':len(points),'inlier_fraction':round(fraction,3),'median_line_residual_px':round(mad,3),'side_span':round(coverage,3),'offset_t0_px':round(float(b),2),'offset_t1_px':round(float(a+b),2)}

def refine_physical_edges(bgr, baseline_quad, *, side='front'):
    """Returns research-only candidate or withheld; requires a V36 baseline quad.

    Changes are capped to 1% of card width and must be supported by >=28
    consistent local samples per edge. Preserves side-specific diagnostics.
    """
    h,w=bgr.shape[:2];q=np.asarray(baseline_quad,np.float32)
    if side not in ('front','back') or not _valid(q,w,h):
        return {'status':'unavailable','reason':'Missing or implausible V36 cut-edge seed.','research_only':True}
    width=float((np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))*.5)
    lab=cv2.cvtColor(cv2.GaussianBlur(bgr,(3,3),.6),cv2.COLOR_BGR2LAB).astype(np.float32)
    center=q.mean(axis=0);line_pts=[];details={}
    allowed=max(3.0,min(8.0,width*.010))
    for i,name in enumerate(NAMES):
        a=q[i];b=q[(i+1)%4]
        p,ev,radius=_side_samples(lab,q,a,b,center,width,name)
        fit=_fit_side(p,ev,float(np.linalg.norm(b-a)))
        if fit is None:return {'status':'unavailable','reason':f'{name} side lacks stable physical-boundary samples.','side_evidence':details,'research_only':True}
        offset_a,offset_b,info=fit
        if max(abs(offset_a),abs(offset_b))>allowed:
            return {'status':'unavailable','reason':f'{name} requires a shift beyond the conservative safety limit; V36 kept.','side_evidence':details,'research_only':True}
        normal=(center-(a+b)/2);normal=normal/(np.linalg.norm(normal)+1e-9)
        line_pts.append((a+offset_a*normal,b+offset_b*normal))
        details[name]=info
    # Intersect adjacent refined lines (handles perspective slope).
    def intersect(p1,p2,r1,r2):
        u=p2-p1;v=r2-r1;mat=np.stack((u,-v),axis=1)
        if abs(float(np.linalg.det(mat)))<1e-4:raise ValueError('parallel')
        return p1+np.linalg.solve(mat,r1-p1)[0]*u
    try:
        refined=np.asarray([intersect(*line_pts[(i-1)%4],*line_pts[i]) for i in range(4)],np.float32)
    except Exception:return {'status':'unavailable','reason':'Corner intersections unstable.','side_evidence':details,'research_only':True}
    if not _valid(refined,w,h):return {'status':'unavailable','reason':'Refined quadrilateral failed geometry validation.','side_evidence':details,'research_only':True}
    displ=np.linalg.norm(refined-q,axis=1)
    # Important: published V37 back benchmark favors V36 by a clear margin.
    # Any substantial drift must be withheld, even when it has sharp contrast.
    mean_limit, max_limit = (2.25,3.75) if side=='back' else (4.0,7.5)
    if float(displ.mean())>mean_limit or float(displ.max())>max_limit:
        return {'status':'unavailable','reason':'Refinement shift exceeds validated-conservative move budget; retain V36 baseline.',
                'side_evidence':details,'baseline_preserved':True,'research_only':True,
                'proposed_mean_move_px':round(float(displ.mean()),2),
                'proposed_max_move_px':round(float(displ.max()),2)}
    return {'status':'candidate_found','method':'v37_1_conservative_multisample_from_v36',
            'physical_quad':np.round(refined,2).tolist(),
            'baseline_method':'V36 candidate (not ground truth)',
            'side_evidence':details,'mean_move_px':round(float(displ.mean()),2),
            'max_move_px':round(float(displ.max()),2),'research_only':True,
            'verified_card_edges':False,'centering_ratio':None,'grade':None,
            'note':'Only a research candidate; human-reviewed independent photos required before promotion. Not a calibrated centering ratio.'}
