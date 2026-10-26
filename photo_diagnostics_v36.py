"""V36 photograph quality and non-grading inspection previews.

All detections here are PHOTO diagnostic candidates, not claims about actual
card damage. We never infer a numeric PSA/BGS/CGC category grade.
"""
from __future__ import annotations
import base64
import cv2
import numpy as np


def _encode(bgr, quality=82):
    ok,buf=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,quality])
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.tobytes()).decode('ascii') if ok else None


def _warp(bgr, quad, width=560, height=784):
    q=np.array(quad,dtype=np.float32)
    dest=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    H=cv2.getPerspectiveTransform(q,dest)
    return cv2.warpPerspective(bgr,H,(width,height),flags=cv2.INTER_CUBIC)


def photo_quality(bgr, quad=None):
    h,w=bgr.shape[:2]
    # Blur/glare thresholds are warnings only, not grades or image rejections.
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    blur=float(cv2.Laplacian(cv2.resize(gray,(min(w,900),round(h*min(w,900)/w)),interpolation=cv2.INTER_AREA),cv2.CV_64F).var())
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    near_white=(hsv[:,:,2]>247)&(hsv[:,:,1]<30)
    # Limit photo-quality evaluation to a central region/card quad where provided.
    if quad is not None:
        mask=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(mask,np.int32(quad),255)
        selected=mask>0
    else:
        selected=np.ones((h,w),bool)
    count=max(1,int(selected.sum()))
    glare=round(float((near_white & selected).sum())/count*100,2)
    notes=[]
    if blur<70:notes.append('Fine detail may be blurred. Try a steady, well-focused photo.')
    if glare>2.0:notes.append('Bright, low-saturation areas may hide scratches or border transitions; use diffuse lighting.')
    if min(w,h)<600:notes.append('Higher-resolution photos are better for close-up inspection.')
    return {'blur_proxy_variance':round(blur,1), 'bright_low_saturation_area_pct':glare,
            'width_px':w,'height_px':h,'photo_guidance':notes,
            'note':'These are image-quality indicators only; bright areas may be normal artwork.'}


def inspection_crops(bgr, quad, *, side='front'):
    """Eight fixed review zones, not machine-confirmed damage markers.

    Physical-card warp means crops contain actual rounded corners and edges,
    rather than rectangular markers covering arbitrary background pixels.
    """
    warped=_warp(bgr,quad)
    h,w=warped.shape[:2]
    def crop(x0,y0,x1,y1, key, label):
        patch=warped[max(0,int(y0*h)):min(h,int(y1*h)),max(0,int(x0*w)):min(w,int(x1*w))]
        if patch.size==0:return None
        return {'id':key,'label':label,'side':side,'kind':'review_zone_not_confirmed_flaw',
                'photo_data_uri':_encode(patch),
                'notice':'Review crop only. The system has not localized or verified a defect here.'}
    specs=[(.00,.00,.24,.20,'tl','Top-left corner'),(.76,.00,1,.20,'tr','Top-right corner'),
           (.76,.80,1,1,'br','Bottom-right corner'),(.00,.80,.24,1,'bl','Bottom-left corner'),
           (.20,.00,.80,.105,'top','Top edge'),(.895,.20,1,.80,'right','Right edge'),
           (.20,.895,.80,1,'bottom','Bottom edge'),(.00,.20,.105,.80,'left','Left edge')]
    return [v for args in specs if (v:=crop(*args)) is not None]
