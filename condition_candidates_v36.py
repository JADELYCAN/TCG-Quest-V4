"""V36 exploratory condition-image signals. Not a validated defect detector.

Laplacian signals highlight artwork, halftones, foil, AND defects, and bright
edge pixels highlight silver borders, glare, AND whitening. Therefore the
outputs are titled image anomalies and NEVER translated into wear labels or
grades without a separately trained/validated classifier.
"""
from __future__ import annotations
import cv2
import numpy as np
from photo_diagnostics_v36 import _warp,_encode


def examine_image_signals(bgr,quad):
    rect=_warp(bgr,quad,width=480,height=672)
    gray=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY)
    # High-frequency structure: intentionally not called a scratch detector.
    lap=cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(gray,(3,3),.65),cv2.CV_16S))
    signal=cv2.GaussianBlur(lap,(3,3),0)
    high=np.percentile(signal,93)
    structure=(signal>max(20,high)).astype(np.uint8)*255
    # Edge bright pixels are generic image evidence, NOT real paper wear.
    hsv=cv2.cvtColor(rect,cv2.COLOR_BGR2HSV)
    h,w=gray.shape
    border=np.zeros((h,w),np.uint8)
    thickness=max(4,round(w*.018))
    border[:thickness]=1;border[-thickness:]=1;border[:,:thickness]=1;border[:,-thickness:]=1
    bright=((hsv[:,:,2]>212)&(hsv[:,:,1]<90)&(border>0)).astype(np.uint8)*255
    structure_color=cv2.applyColorMap(structure,cv2.COLORMAP_TURBO)
    bright_color=cv2.applyColorMap(bright,cv2.COLORMAP_TURBO)
    return {
        'research_only':True,
        'labels':{'surface':'Texture and high-frequency image structure, NOT detected scratches',
                  'edge':'Bright edge pixels, NOT confirmed whitening'},
        'surface_structure_map_data_uri':_encode(structure_color,68),
        'bright_edge_map_data_uri':_encode(bright_color,68),
        'surface_signal_coverage_pct':round(float(np.mean(structure>0))*100,2),
        'bright_edge_signal_coverage_pct_of_card':round(float(np.mean(bright>0))*100,2),
        'warning':'Both maps contain many false positives from normal card printing, holo artwork, shadows, and glare. They are NOT defects or grades.'
    }
