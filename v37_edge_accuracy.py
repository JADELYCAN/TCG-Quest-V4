"""TCG Quest V37: shadow-aware, template-seeded physical-edge candidate finder.

Research diagnostics only. NOT a printed-border or centering estimator, nor a
replacement for human-verified card-corner labels. No AI API calls or grading.
"""
from __future__ import annotations
import cv2
import numpy as np

S = ('top','right','bottom','left')

def _plausible(q,w,h):
    q=np.asarray(q,dtype=np.float32)
    if q.shape!=(4,2) or not np.isfinite(q).all() or cv2.contourArea(q)<.09*w*h:
        return False
    if np.min(q[:,0]) < -2 or np.min(q[:,1]) < -2 or np.max(q[:,0])>=w+2 or np.max(q[:,1])>=h+2:
        return False
    cross=[float((q[(i+1)%4,0]-q[i,0])*(q[(i+2)%4,1]-q[(i+1)%4,1])-(q[(i+1)%4,1]-q[i,1])*(q[(i+2)%4,0]-q[(i+1)%4,0])) for i in range(4)]
    return min(cross)>0

def _robust_line(x,y,weights=None):
    """Deterministic RANSAC line followed by weighted least squares."""
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    if len(x)<9:return None
    # Each point's y is the orthogonal search coordinate; x is travel along edge.
    weights=np.ones_like(x) if weights is None else np.asarray(weights,np.float64)
    rng=np.random.default_rng(20261026)
    best=None
    for _ in range(175):
        ii=rng.choice(len(x),2,replace=False)
        if abs(x[ii[1]]-x[ii[0]]) < .18*np.ptp(x):continue
        a=(y[ii[1]]-y[ii[0]])/(x[ii[1]]-x[ii[0]]); b=y[ii[0]]-a*x[ii[0]]
        res=np.abs(y-(a*x+b));good=res<4.8
        if good.sum()<9:continue
        val=float(np.sum(weights[good])-1.8*np.median(res[good]))
        if best is None or val>best[0]:best=(val,good)
    if best is None:return None
    good=best[1];a,b=np.polyfit(x[good],y[good],1,w=np.sqrt(weights[good]));res=np.abs(y-(a*x+b));good=res<5
    if good.sum()<9:return None
    a,b=np.polyfit(x[good],y[good],1,w=np.sqrt(weights[good]))
    return (float(a),float(b),float(np.median(np.abs(y[good]-(a*x[good]+b)))),
            float(good.mean()),int(good.sum()))

def _sample_sides(bgr,reference_quad,step_px=None,include_samples=False,require_background=True):
    h,w=bgr.shape[:2];q=np.asarray(reference_quad,np.float32)
    if not _plausible(q,w,h):return {'status':'unavailable','reason':'Template seed is not a plausible card quad.'}
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(5,5),1.0)
    sample_data={};fits={}
    width=np.linalg.norm(q[1]-q[0]); height=np.linalg.norm(q[3]-q[0])
    max_search=max(17,min(57,.075*width));half=max(4.5,min(8.5,.010*width))
    for s,side in enumerate(S):
        # x=ax+b for left/right, y=ax+b for top/bottom
        a=q[s];b=q[(s+1)%4];long_side=np.linalg.norm(b-a)
        n=48 if long_side>680 else 36
        param=np.linspace(.1,.9,n)
        positions=[];crossings=[];strengths=[];snapshots=[]
        # Side's travel is top -> right -> bottom -> left, cross-edge axis accordingly.
        is_x=side in ('left','right')
        inside_sign=(1 if side=='left' else -1) if is_x else (1 if side=='top' else -1)
        for t in param:
            pt=a+(b-a)*t
            longitudinal=float(pt[1] if is_x else pt[0]); predicted=float(pt[0] if is_x else pt[1])
            candidates=np.arange(round(predicted-max_search),round(predicted+max_search)+1,1.0)
            # Photo outside must be brighter than dark printed-stock inside.
            # Opposing 5-9-pixel sample windows avoid one-pixel white foil noise.
            ii=candidates+inside_sign*half;oo=candidates-inside_sign*half
            xi=ii if is_x else np.full_like(ii,longitudinal); yi=np.full_like(ii,longitudinal) if is_x else ii
            xo=oo if is_x else np.full_like(oo,longitudinal); yo=np.full_like(oo,longitudinal) if is_x else oo
            valid=(xi>=2)&(xi<w-3)&(xo>=2)&(xo<w-3)&(yi>=2)&(yi<h-3)&(yo>=2)&(yo<h-3)
            if not valid.any():continue
            ixs=np.where(valid)[0];xi=xi[valid].astype(np.float32);yi=yi[valid].astype(np.float32);xo=xo[valid].astype(np.float32);yo=yo[valid].astype(np.float32)
            # remap needs 2D coordinates.
            inn=cv2.remap(gray,xi.reshape(1,-1),yi.reshape(1,-1),cv2.INTER_LINEAR)[0]
            out=cv2.remap(gray,xo.reshape(1,-1),yo.reshape(1,-1),cv2.INTER_LINEAR)[0]
            jump=out-inn
            # Independently sample the expected background well OUTSIDE the projected card.
            # This prevents a dark/bright printed artwork stripe from becoming a cut edge.
            bg_coord=predicted-inside_sign*(max_search+12)
            bgx=bg_coord if is_x else longitudinal; bgy=longitudinal if is_x else bg_coord
            if require_background and (bgx<2 or bgx>=w-3 or bgy<2 or bgy>=h-3):continue
            if require_background:
                bg=float(cv2.remap(gray,np.float32([[bgx]]),np.float32([[bgy]]),cv2.INTER_LINEAR)[0,0])
                background_similarity=np.abs(out-bg)
            else:
                bg=0.0
                background_similarity=np.zeros_like(out)
            if require_background:
                merit=jump - .16*np.abs(candidates[valid]-predicted) - .65*background_similarity
                merit=np.where((jump>16)&(out>155)&(inn<195)&(background_similarity<25)&(bg>180),merit,-100)
            else:
                merit=jump - .16*np.abs(candidates[valid]-predicted)
                merit=np.where((jump>16)&(out>125)&(inn<195),merit,-100)
            bi=int(np.argmax(merit))
            if merit[bi]<14:continue
            px=float(candidates[ixs[bi]])
            positions.append(longitudinal);crossings.append(px);strengths.append(float(jump[bi]));snapshots.append((longitudinal,px))
        if len(positions)<12:
            return {'status':'unavailable','reason':f'{side} cut edge has too few independent transitions.', 'side_evidence':sample_data}
        fit=_robust_line(positions,crossings,np.clip(strengths,15,130))
        if fit is None:return {'status':'unavailable','reason':f'{side} boundary does not form a consistent line.','side_evidence':sample_data}
        slope,intercept,mad,inlier_frac,inlier_n=fit
        # A blurred or occluded area can create a wrong parallel transition.
        if inlier_frac<.62 or mad>3.5 or inlier_n<18:
            return {'status':'unavailable','reason':f'{side} edge transitions disagree. Retake or manually review.', 'side_evidence':sample_data}
        # Predicted physical edge must stay locally near the artwork reference.
        testpts=[a+(b-a)*t for t in (.16,.50,.84)]
        shifts=[abs((slope*(pt[1] if is_x else pt[0])+intercept)-(pt[0] if is_x else pt[1])) for pt in testpts]
        if max(shifts) > max_search*.92:
            return {'status':'unavailable','reason':f'{side} boundary lies too far from the catalog projection.','side_evidence':sample_data}
        fits[side]=(slope,intercept)
        sample_data[side]={'median_line_error_px':round(mad,2),'usable_samples':len(positions),
                           'inliers':inlier_n,'inlier_fraction':round(inlier_frac,3),
                           'median_transition_strength':round(float(np.median(strengths)),2),
                           'max_seed_offset_px':round(float(max(shifts)),2)}
        if include_samples:sample_data[side]['support_points']=[[round(x,1),round(y,1)] for x,y in snapshots]
    def cross(hor,vert):
        # hor: y=c*x+d; vert: x=a*y+b
        c,d=fits[hor];a,b=fits[vert];den=1-a*c
        if abs(den)<.15:raise ValueError('nearly parallel')
        x=(a*d+b)/den;return [float(x),float(c*x+d)]
    try:
        corners=np.float32([cross('top','left'),cross('top','right'),cross('bottom','right'),cross('bottom','left')])
    except Exception:return {'status':'unavailable','reason':'Line intersections are unstable.','side_evidence':sample_data}
    if not _plausible(corners,w,h):return {'status':'unavailable','reason':'Candidate corners failed geometry checks.','side_evidence':sample_data}
    # Avoid aspect/collapse errors. Perspective can tilt edges; gross card ratio is bounded.
    top=np.linalg.norm(corners[1]-corners[0]);bot=np.linalg.norm(corners[2]-corners[3]);left=np.linalg.norm(corners[3]-corners[0]);right=np.linalg.norm(corners[2]-corners[1]);ratio=(top+bot)/(left+right)
    if not (.60<=ratio<=.91):return {'status':'unavailable','reason':'Candidate aspect ratio is implausible.','side_evidence':sample_data}
    return {'status':'candidate_found','method':'v37_multisample_photometric_line_fit','physical_quad':corners.round(2).tolist(),
            'side_evidence':sample_data,'research_only':True,'verified_card_edges':False,'centering_ratios':None,'grade':None,
            'note':'Four candidate physical cut-edge lines fitted from independent photo scans. Requires manual reference verification; no centering or grading claim.'}

def detect_front(bgr, template_photo_quad):
    return _sample_sides(bgr,template_photo_quad)

def refine_back(bgr,baseline_quad):
    # Seeded from the independent blue-stock silhouette, then search local background transitions.
    return _sample_sides(bgr,baseline_quad,require_background=False)

def overlay(bgr,old_quad=None,new_quad=None,reference_quad=None):
    im=bgr.copy()
    for quad,bgrcolor,thickness in ((reference_quad,(0,185,255),2),(old_quad,(255,205,0),3),(new_quad,(70,255,85),3)):
        if quad is None:continue
        pts=np.asarray(quad,dtype=np.int32).reshape(4,2)
        cv2.polylines(im,[pts],True,bgrcolor,thickness,cv2.LINE_AA)
        for k,(x,y) in enumerate(pts):cv2.circle(im,(int(x),int(y)),5,bgrcolor,-1,cv2.LINE_AA)
    return im
