"""TCG Quest V35 -- research-only template-to-cut registration repeatability.

This is NOT centering, a card grade, or proof of accurate print registration.
The catalog raster is an uncalibrated crop, not a centered print master.
It estimates its mapped *frame* in cut-card coordinates to study repeatability
across two independent photos of one physical card.
"""
from __future__ import annotations
import hashlib
from typing import Any
from template_alignment_v34 import inspect_photo, PILOT_ID

METHOD = 'v35_uncalibrated_reference_frame_repeatability'


def _failure(reason: str, first: dict | None = None, second: dict | None = None) -> dict[str, Any]:
    return {
        'status': 'unavailable', 'method': METHOD,
        'reason': str(reason)[:320], 'photo_a': first,
        'photo_b': second,
        'comparison': None, 'validated_centering': False,
        'verified_centering': False, 'lr': None, 'tb': None,
        'center_score': None, 'confidence': 'not_calibrated',
    }


def calculate_frame_geometry(registered: dict) -> dict | None:
    """Convert projected *catalog image corners* to cut-card local coordinates.

    Both input quads use normalized PHOTO coordinates in TL, TR, BR, BL order.
    The reference coordinates measure CATALOG CROP geometry, NOT margins of
    a physical printed design. All output numbers are in % of cut width/height.
    """
    import cv2
    import numpy as np
    if registered.get('status') != 'artwork_aligned_only' or registered.get('physical_edge_status') != 'found':
        return None
    try:
        physical = np.array(registered['physical_quad'], dtype=np.float32)
        projected = np.array(registered['projected_template_quad'], dtype=np.float32)
        if physical.shape != (4, 2) or projected.shape != (4, 2) or not np.isfinite(physical).all() or not np.isfinite(projected).all():
            return None
        # Mapping the physical cut corners to a normalized 0..1 planar frame
        # removes most rotation and perspective effects from phone photos.
        unit = np.array([[0,0],[1,0],[1,1],[0,1]], np.float32)
        H = cv2.getPerspectiveTransform(physical, unit)
        local = cv2.perspectiveTransform(projected.reshape(-1,1,2), H).reshape(4,2)
        if not np.isfinite(local).all() or np.any(local < -0.18) or np.any(local > 1.18):
            return None
        # Geometry sanity: projected frame must remain four ordered sides and
        # near-card-sized; otherwise SIFT/physical outline disagreement wins.
        horiz_top = float(np.linalg.norm(local[1]-local[0]))
        horiz_bottom = float(np.linalg.norm(local[2]-local[3]))
        vert_left = float(np.linalg.norm(local[3]-local[0]))
        vert_right = float(np.linalg.norm(local[2]-local[1]))
        if not all(.78 <= v <= 1.22 for v in (horiz_top,horiz_bottom,vert_left,vert_right)):
            return None
        # Negative gap means projected catalog crop falls outside physical cut.
        gaps = {
            'left': float((local[0,0] + local[3,0])/2)*100,
            'right': float((2-local[1,0]-local[2,0])/2)*100,
            'top': float((local[0,1]+local[1,1])/2)*100,
            'bottom': float((2-local[2,1]-local[3,1])/2)*100,
        }
        midpoint = local.mean(axis=0)
        result = {
            'reference_center_x_pct': round(float((midpoint[0]-.5)*100),2),
            'reference_center_y_pct': round(float((midpoint[1]-.5)*100),2),
            'projected_frame_width_pct': round(100*(horiz_top+horiz_bottom)/2,2),
            'projected_frame_height_pct': round(100*(vert_left+vert_right)/2,2),
            'catalog_crop_edge_gaps_pct': {k:round(v,2) for k,v in gaps.items()},
            'reference_corner_coordinates': [[round(float(x),5),round(float(y),5)] for x,y in local],
            'reference_not_calibrated': True,
        }
        return result
    except (ValueError, TypeError, KeyError, cv2.error):
        return None


def _annotated_response(one: dict, include_overlay: bool) -> dict:
    item={k:one.get(k) for k in ('status','reason','inliers','inlier_fraction','physical_edge_status',
                                    'median_reprojection_error_px','candidate_matches','competing_artwork_inliers')}
    if include_overlay and one.get('overlay_data_uri'):
        item['overlay_data_uri']=one['overlay_data_uri']
    item['frame']=calculate_frame_geometry(one)
    item['reference_not_calibrated']=True
    return item


def inspect_repeatability(photo_a_bytes: bytes, card_id: str,
                          photo_b_bytes: bytes | None = None,
                          *, include_overlay: bool = True,
                          reference_path=None) -> dict:
    """Compare two photos of the SAME physical card; never infer grade/ratio.

    Image inputs are processed in memory; no provider calls and no photo saving.
    Without a second photo, output is a frame-location observation only.
    """
    if card_id != PILOT_ID:
        return _failure('Research pilot supports only confirmed Lost Origin Giratina V #186/196.')
    a=inspect_photo(photo_a_bytes,card_id,include_overlay=include_overlay,reference_path=reference_path)
    display_a=_annotated_response(a,include_overlay)
    if a.get('status')!='artwork_aligned_only':
        return _failure('Photo A: '+str(a.get('reason','exact-card matching failed')),display_a)
    if display_a['frame'] is None:
        return _failure('Photo A: artwork matched, but a valid physical-card frame was not independently detected.',display_a)
    display_b=None;compare=None
    duplicate_inputs=photo_b_bytes is not None and hashlib.sha256(photo_a_bytes).digest()==hashlib.sha256(photo_b_bytes).digest()
    if photo_b_bytes is not None:
        b=inspect_photo(photo_b_bytes,card_id,include_overlay=include_overlay,reference_path=reference_path)
        display_b=_annotated_response(b,include_overlay)
        if b.get('status')!='artwork_aligned_only':
            return _failure('Photo B: '+str(b.get('reason','exact-card matching failed')),display_a,display_b)
        if display_b['frame'] is None:
            return _failure('Photo B: artwork matched, but the physical-card frame was not independently detected.',display_a,display_b)
        fa=display_a['frame'];fb=display_b['frame']
        dx=round(fb['reference_center_x_pct']-fa['reference_center_x_pct'],2)
        dy=round(fb['reference_center_y_pct']-fa['reference_center_y_pct'],2)
        dw=round(fb['projected_frame_width_pct']-fa['projected_frame_width_pct'],2)
        dh=round(fb['projected_frame_height_pct']-fa['projected_frame_height_pct'],2)
        gap_deltas={k:round(fb['catalog_crop_edge_gaps_pct'][k]-fa['catalog_crop_edge_gaps_pct'][k],2)
                    for k in ('left','right','top','bottom')}
        largest=max([abs(dx),abs(dy),abs(dw),abs(dh),*(abs(v) for v in gap_deltas.values())])
        if duplicate_inputs:
            status='identical_files_no_independent_test'
            note='The two uploaded files contain identical image bytes. This is NOT a repeatability test; take a second real photograph of the same card.'
        elif largest<=1.5:
            status='repeatable_in_this_photo_pair'
            note='The uncalibrated catalog-frame geometry is reasonably repeatable in these two photographs. This does not prove accuracy.'
        elif largest<=3.0:
            status='variable_between_photos'
            note='The reference-frame geometry moved noticeably between photos. Do not infer centering or grade.'
        else:
            status='unstable_between_photos'
            note='The frame changed too much between photographs for reliable geometric interpretation.'
        compare={'status':status,'max_observed_change_pct_of_card':round(largest,2),
                 'center_x_change_pct_of_card':dx,'center_y_change_pct_of_card':dy,
                 'frame_width_change_pct_of_card':dw,'frame_height_change_pct_of_card':dh,
                 'crop_gap_changes_pct_of_card':gap_deltas,
                 'reference_not_calibrated':True,'interpretation':note}
    return {'status':'uncalibrated_template_frame_observation','method':METHOD,
            'card_id':PILOT_ID,'photo_a':display_a,'photo_b':display_b,
            'comparison':compare,
            'calibration_status':'missing_validated_centered_print_master',
            'verified_centering':False,'validated_centering':False,
            'confidence':'not_calibrated','lr':None,'tb':None,'center_score':None,
            'reason':'Reference frame position is relative to an uncalibrated catalog-image crop, not a certified printed template. The offsets CANNOT be interpreted as actual centering.'}
