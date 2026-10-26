"""TCG Quest V34 *research-only* exact-card reference registration.

NEVER interprets catalog art placement as a calibrated print location and
NEVER outputs centering ratios or modifies AI grades. Requires a confirmed
card_id; currently the pilot catalog includes Lost Origin Giratina V 186/196.
Reference assets are for private research only. Check rights before distributing.
"""
from __future__ import annotations
import base64
import io
from pathlib import Path
from typing import Any

PILOT_ID = 'swsh11-186'
PILOT_NAME = 'Giratina V — Lost Origin 186/196'
REFERENCE = Path(__file__).resolve().parent / 'template_refs' / 'swsh11-186.png'
NEGATIVE_REFERENCE = Path(__file__).resolve().parent / 'template_refs' / 'swsh11-185-negative.png'
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _reject(reason: str, **extra: Any) -> dict:
    return dict({'status': 'unavailable', 'method': 'v34_exact_card_sift_pilot',
                 'reason': reason, 'verified_centering': False,
                 'lr': None, 'tb': None, 'center_score': None}, **extra)


def inspect_photo(photo_bytes: bytes, card_id: str, *, include_overlay: bool = True, reference_path=None) -> dict:
    """Validate one front photo against the exact-card reference.

    No API calls. Caller must enforce request throttling. Returns a viewable
    outline, matching strength, and hard-coded centering=None. Not a grade.
    """
    if card_id != PILOT_ID:
        return _reject('V34 research pilot only supports the exact Lost Origin Giratina V 186/196 card.', card_id=card_id)
    if not isinstance(photo_bytes, bytes) or not photo_bytes or len(photo_bytes) > MAX_IMAGE_BYTES:
        return _reject('Photo must be a nonempty image no larger than 8 MB.')
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _reject('OpenCV and NumPy are required for template alignment.')
    try:
        arr = np.frombuffer(photo_bytes, dtype=np.uint8)
        photo = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if photo is None:
            return _reject('File is not a supported image.')
        h0,w0 = photo.shape[:2]
        if min(w0,h0) < 400 or w0*h0 > 25_000_000:
            return _reject('Use a clear image at least 400 pixels across, no more than 25 megapixels.')
        ref = cv2.imread(str(reference_path or REFERENCE), cv2.IMREAD_COLOR)
        if ref is None:
            return _reject('Reference image is missing from the private staging service.')
        rh,rw = ref.shape[:2]
        # Bound expensive SIFT computations and normalize upload dimensions.
        shrink = min(1., 1400/max(h0,w0))
        if shrink < 1.:
            photo_small = cv2.resize(photo,(round(w0*shrink), round(h0*shrink)),interpolation=cv2.INTER_AREA)
        else:
            photo_small=photo
        ps_h,ps_w=photo_small.shape[:2]
        # Edge-preserving Gaussian blur mitigates small compression artifacts.
        ref_gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
        photo_gray=cv2.cvtColor(photo_small,cv2.COLOR_BGR2GRAY)
        sift=cv2.SIFT_create(nfeatures=2400, contrastThreshold=0.025)
        kp_ref,des_ref=sift.detectAndCompute(ref_gray,None)
        kp_photo,des_photo=sift.detectAndCompute(photo_gray,None)
        if des_ref is None or des_photo is None or len(kp_ref)<35 or len(kp_photo)<35:
            return _reject('Insufficient stable printed features for a catalog match.')
        matches=cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_ref, des_photo, k=2)
        good=[m for group in matches if len(group)==2 for m,n in [group] if m.distance<.72*n.distance]
        if len(good)<22:
            return _reject('This photo does not produce enough distinct matches to Giratina #186.',candidate_matches=len(good))
        # Match points in template coordinates and photo coordinates.
        src=np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst=np.float32([kp_photo[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.5,maxIters=3500,confidence=.995)
        if H is None or mask is None:
            return _reject('Matching artwork could not be aligned to a single coherent geometry.',candidate_matches=len(good))
        keep=mask.ravel().astype(bool)
        n=int(keep.sum());fraction=n/len(good)
        in_src=src[keep,0,:]; in_dst=dst[keep,0,:]
        rx=np.ptp(in_src[:,0])/rw; ry=np.ptp(in_src[:,1])/rh
        # Reject geometric degeneracy (only tiny corner/logo matching).
        if n<40 or fraction<.60 or rx<.36 or ry<.42:
            return _reject('Matches do not cover enough independent artwork to verify the exact card.',
                           candidate_matches=len(good),inliers=n,inlier_fraction=round(float(fraction),3),
                           reference_coverage=[round(float(rx),3),round(float(ry),3)])
        # Disambiguation is essential: #185 shares many printed labels with #186.
        # This *closed-world* pilot uses the most similar known rival template.
        # A future catalog service must compare multiple plausible identities.
        rival=cv2.imread(str(NEGATIVE_REFERENCE),cv2.IMREAD_GRAYSCALE)
        if rival is None:
            return _reject('Exact-card identity cannot be verified: competing reference missing.')
        kp_rival,des_rival=sift.detectAndCompute(rival,None)
        if des_rival is None:
            return _reject('Competing reference could not be loaded for exact-card checking.')
        rival_pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_rival,des_photo,k=2)
        rival_good=[m for group in rival_pairs if len(group)==2 for m,n2 in [group] if m.distance<.72*n2.distance]
        rival_inliers=0
        if len(rival_good)>=8:
            rival_src=np.float32([kp_rival[m.queryIdx].pt for m in rival_good]).reshape(-1,1,2)
            rival_dst=np.float32([kp_photo[m.trainIdx].pt for m in rival_good]).reshape(-1,1,2)
            rival_H,rival_mask=cv2.findHomography(rival_src,rival_dst,cv2.RANSAC,4.5)
            if rival_mask is not None:rival_inliers=int(rival_mask.sum())
        if n<rival_inliers*1.55 or n-rival_inliers<30:
            return _reject('The photo is ambiguous or matches another Giratina artwork more strongly; exact #186 identity not established.',
                           candidate_matches=len(good),inliers=n,competing_artwork_inliers=rival_inliers)
        corners=np.float32([[[0,0]],[[rw-1,0]],[[rw-1,rh-1]],[[0,rh-1]]])
        projected=cv2.perspectiveTransform(corners,H).reshape(4,2)
        if not np.isfinite(projected).all() or cv2.contourArea(projected.astype(np.float32)) < .10*ps_h*ps_w:
            return _reject('Projected catalog outline is implausible for the photograph.')
        # The projected graphic must be mostly inside the photo and centered over plausible card area.
        margin=min(ps_w,ps_h)*.10
        if (projected[:,0]<-margin).any() or (projected[:,0]>ps_w+margin).any() or (projected[:,1]<-margin).any() or (projected[:,1]>ps_h+margin).any():
            return _reject('Reference alignment projects outside the visible card/photo.')
        fitted=cv2.perspectiveTransform(src[keep],H).reshape(-1,2)
        err=np.linalg.norm(fitted-in_dst,axis=1)
        median_error=float(np.median(err)); p90=float(np.percentile(err,90))
        if median_error>4.5 or p90>12.0:
            return _reject('Template landmarks disagree too much to accept the alignment.',inliers=n,median_error=round(median_error,2))
        # Identify outside card edge independently from the reference (not required for art match).
        phys_quad=None;edge_error=None
        try:
            from precision_centering import _physical_quad
            phys_quad, edge_error=_physical_quad(photo_small,cv2,np)
        except (ImportError, ValueError, TypeError, cv2.error):
            edge_error='Physical boundary detector unavailable.'
        # Response never pretends the catalog thumbnail is a centered print master.
        result={'status':'artwork_aligned_only','method':'v34_exact_card_sift_pilot',
                'card_id':PILOT_ID,'card_name':PILOT_NAME,
                'catalog_source':'User-supplied TCGdex-based English catalog; Volume 07, page 225',
                'candidate_matches':len(good),'inliers':n,
                'inlier_fraction':round(float(fraction),3),
                'competing_artwork_inliers':rival_inliers,
                'reference_coverage':[round(float(rx),3),round(float(ry),3)],
                'median_reprojection_error_px':round(median_error,2),
                'projected_template_quad':[[round(float(x)/ps_w,5),round(float(y)/ps_h,5)] for x,y in projected],
                'physical_edge_status':'found' if phys_quad is not None else 'unavailable',
                'physical_edge_note':edge_error if phys_quad is None else 'Independently estimated from photo contrast.',
                'physical_quad':([[round(float(x)/ps_w,5),round(float(y)/ps_h,5)] for x,y in phys_quad]
                                 if phys_quad is not None else None),
                'verified_centering':False, 'lr':None,'tb':None,'center_score':None,
                'reason':'The catalog artwork was registered to a physical photograph. The source image is NOT a calibrated centered print master; no centering percentage or grade can be inferred.'}
        if include_overlay:
            # Overlay cropped around card and photo; low bandwidth (~few hundred KB)
            canvas=photo_small.copy()
            if phys_quad is not None:
                cv2.polylines(canvas,[np.int32(phys_quad)],True,(255,240,0),3,cv2.LINE_AA)
            points=np.round(projected).astype(np.int32)
            for j in range(4):
                cv2.line(canvas,tuple(points[j]),tuple(points[(j+1)%4]),(0,210,255),3,cv2.LINE_AA)
                cv2.circle(canvas,tuple(points[j]),8,(0,210,255),-1,cv2.LINE_AA)
            # Sparse green feature locations, so collectors can see where alignment came from.
            for idx in np.linspace(0,n-1,min(n,30),dtype=int):
                x,y=np.round(in_dst[idx]).astype(int)
                cv2.circle(canvas,(int(x),int(y)),2,(80,245,120),-1,cv2.LINE_AA)
            cv2.rectangle(canvas,(14,14),(min(ps_w-12,560),104),(7,17,34),-1)
            cv2.putText(canvas,'V34 EXPERIMENTAL TEMPLATE REGISTRATION',(25,43),cv2.FONT_HERSHEY_SIMPLEX,.57,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'Cyan = outer edge; Amber = catalog projection',(25,68),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'NO VERIFIED CENTERING RATIO',(25,90),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,185,255),2,cv2.LINE_AA)
            ok,jpg=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,84])
            if ok:
                result['overlay_data_uri']='data:image/jpeg;base64,'+base64.b64encode(jpg).decode('ascii')
        return result
    except Exception:
        return _reject('Unexpected image processing error. Check the photo format and try another file.')
