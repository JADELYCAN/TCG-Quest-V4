"""TCG Quest V37.2 — research-only physical-edge fallback.

An independently verified *exact-card match* supplies a location seed only.
The catalog boundary is NOT itself a certified cut-edge or perfect-center master.

Where V36 or V37 fails, fit coherent photo color transitions on all four sides
with trimmed aggregation and constrained offset/slope; decline implausible results.
Returns an UNVERIFIED, optionally low-confidence quad for collector comparison.
Never computes printed centering, condition scores, or grades.

Dependencies: numpy, opencv-python-headless. No OpenAI requests or PyTorch.
"""
from __future__ import annotations
import cv2
import numpy as np

SIDES = ('top', 'right', 'bottom', 'left')


def _valid_quad(q, w, h):
    a = np.asarray(q, np.float64)
    if a.shape != (4, 2) or not np.isfinite(a).all():
        return False
    if min(a[:, 0]) < 2 or min(a[:, 1]) < 2 or max(a[:, 0]) > w - 3 or max(a[:, 1]) > h - 3:
        return False
    if cv2.contourArea(a.astype(np.float32)) < .075 * w * h:
        return False
    cross = [float((a[(i+1)%4,0]-a[i,0])*(a[(i+2)%4,1]-a[(i+1)%4,1])
                   -(a[(i+1)%4,1]-a[i,1])*(a[(i+2)%4,0]-a[(i+1)%4,0])) for i in range(4)]
    if min(cross) <= 0:
        return False
    top = np.linalg.norm(a[1]-a[0]); bot = np.linalg.norm(a[2]-a[3])
    left = np.linalg.norm(a[3]-a[0]); right = np.linalg.norm(a[2]-a[1])
    aspect = (top+bot)/(left+right)
    return .59 <= aspect <= .91


def _line_intersection(pa, pb, pc, pd):
    v = pb-pa; u = pd-pc
    mat = np.stack([v, -u], axis=1)
    if abs(float(np.linalg.det(mat))) < .08 * np.linalg.norm(v)*np.linalg.norm(u):
        raise ValueError('Neighboring cut-edge lines intersect unreliably.')
    t = np.linalg.solve(mat, pc-pa)[0]
    return pa+t*v


def _side_fit(lab, q, side_i, max_shift):
    """Score a coherent physical edge, not whichever single artwork stripe is strongest.

    Each side is tested at 53 distributed locations. For each offset the color
    discontinuity across a 7px band is scored and penalized if its *exterior*
    doesn't resemble the more distant exterior background. Trimmed aggregate
    resists foil reflections and folds. A modest slope is fitted jointly.
    """
    a = q[side_i]; b = q[(side_i+1)%4]
    center=q.mean(axis=0)
    inward=center-(a+b)/2
    inward=inward/(np.linalg.norm(inward)+1e-9)
    t=np.linspace(.12,.88,53,dtype=np.float32)
    d=np.arange(-max_shift,max_shift+1,dtype=np.float32)
    points=a[None,:]+(b-a)[None,:]*t[:,None]
    cand=points[:,None,:]+d[None,:,None]*inward[None,None,:]
    def sample(xy):
        return cv2.remap(lab, np.float32(xy[...,0]), np.float32(xy[...,1]),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inside=sample(cand+3.5*inward)
    outside=sample(cand-3.5*inward)
    bg=sample((points-42*inward)[:,None,:])
    discontinuity=np.linalg.norm(inside-outside,axis=2)
    exterior_difference=np.linalg.norm(outside-bg,axis=2)
    contrast=discontinuity-.90*exterior_difference

    # Efficient constrained slanted-line search. Offsets are inside-directed,
    # so a negative value moves the detected edge away from the center.
    best=(-np.inf,None,None)
    for d0 in range(-max_shift+4,max_shift-3):
        for slope in range(-12,13):
            shifted=d0+slope*(t-.5)
            if min(shifted)<-max_shift+1 or max(shifted)>max_shift-1:
                continue
            # Vectorized interpolated contrast for all 53 longitudinal samples.
            indices=shifted+max_shift
            low=np.floor(indices).astype(np.int32)
            frac=indices-low
            v=contrast[np.arange(len(t)),low]*(1-frac)+contrast[np.arange(len(t)),low+1]*frac
            trimmed=float(np.mean(np.sort(v)[9:-9]))
            # Penalize long jumps and unwarranted tilt relative to reference.
            quality=trimmed-.30*abs(d0)-.80*abs(slope)
            if quality>best[0]:best=(quality,float(d0),float(slope))
    quality,d0,slope=best
    if d0 is None:
        return None,'No valid constrained line fit.'
    offsets=d0+slope*(t-.5)
    indices=offsets+max_shift
    lo=np.floor(indices).astype(np.int32); frac=indices-lo
    contrast_curve=contrast[np.arange(len(t)),lo]*(1-frac)+contrast[np.arange(len(t)),lo+1]*frac
    jump_curve=discontinuity[np.arange(len(t)),lo]*(1-frac)+discontinuity[np.arange(len(t)),lo+1]*frac
    bg_curve=exterior_difference[np.arange(len(t)),lo]*(1-frac)+exterior_difference[np.arange(len(t)),lo+1]*frac
    median_jump=float(np.median(jump_curve))
    support=float(np.mean((jump_curve>16)&(bg_curve<65)))
    if quality<5 or median_jump<13 or support<.35:
        return None,('Photo transition too weak or inconsistent to justify a '
                     f'{SIDES[side_i]} cut-edge candidate.')
    # Edge endpoints from the fitted (constant+slope) offset line.
    p1=a+(d0-.5*slope)*inward
    p2=b+(d0+.5*slope)*inward
    evidence={
        'method':'photo_lab_multi_sample_consensus',
        'median_color_jump':round(median_jump,1),
        'supported_sample_fraction':round(support,3),
        'trimmed_evidence_score':round(float(quality),2),
        'offset_at_center_px':round(d0,2),
        'offset_variation_end_to_end_px':round(slope,2),
        'weak_boundary':bool(quality<22 or support<.60),
    }
    return (p1,p2,evidence),None


def recover_front_from_template(bgr, projected_template_quad):
    """Return unverified candidate independent of the V36 baseline.

    Requires exact card identity to have been confirmed UPSTREAM; this function
    is not an identity validator. Caller must gate it after V34 succeeds.
    """
    if bgr is None or not isinstance(bgr,np.ndarray) or bgr.ndim!=3 or bgr.shape[2]!=3:
        return {'status':'unavailable','reason':'A BGR card photograph is required.'}
    h,w=bgr.shape[:2]
    q=np.asarray(projected_template_quad,dtype=np.float32)
    if not _valid_quad(q,w,h):
        return {'status':'unavailable','reason':'Matched artwork projection failed basic card-geometry checks.'}
    width=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2
    if width < 230:
        return {'status':'unavailable','reason':'Photographed card too small for physical-edge recovery.'}
    max_shift=int(max(17,min(24,width*.04)))
    lab=cv2.cvtColor(cv2.GaussianBlur(bgr,(3,3),.7),cv2.COLOR_BGR2LAB).astype(np.float32)
    fits=[];evidence={}
    for i,side in enumerate(SIDES):
        result,reason=_side_fit(lab,q,i,max_shift)
        if result is None:
            return {'status':'unavailable','reason':f'{side.capitalize()} edge: {reason}',
                    'side_evidence':evidence}
        a,b,ev=result
        fits.append((a,b));evidence[side]=ev
    try:
        corners=np.float32([
            _line_intersection(fits[(i-1)%4][0],fits[(i-1)%4][1],fits[i][0],fits[i][1])
            for i in range(4)
        ])
    except (ValueError,np.linalg.LinAlgError):
        return {'status':'unavailable','reason':'Recovered side intersections are unstable.',
                'side_evidence':evidence}
    if not _valid_quad(corners,w,h):
        return {'status':'unavailable','reason':'Recovered corners fail card geometry constraints.',
                'side_evidence':evidence}
    displacement=np.linalg.norm(corners-q,axis=1)
    if float(np.max(displacement))>max_shift*1.85:
        return {'status':'unavailable','reason':'Recovered outline moved too far from verified artwork.',
                'side_evidence':evidence}
    weak=[s for s,v in evidence.items() if v['weak_boundary']]
    return {
        'status':'candidate_found',
        'method':'v37_2_template_seeded_multiside_consensus',
        'physical_quad':np.round(corners,2).tolist(),
        'side_evidence':evidence,
        'confidence':'low' if weak else 'experimental_moderate',
        'weak_sides':weak,
        'review_required':True,
        'verified':False,
        'calibrated':False,
        'mean_seed_shift_px':round(float(displacement.mean()),2),
        'max_seed_shift_px':round(float(displacement.max()),2),
        'centering_ratio':None,
        'grade':None,
        'note':('Template geometry is a search seed, not a perfect card. '
                'Review all four physical corners before accepting this research candidate. '
                'No grading or centering measurement is calculated.'),
    }
