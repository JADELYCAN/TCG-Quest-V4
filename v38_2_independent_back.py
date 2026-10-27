"""TCG Quest V38.2 experimental centering research.

A failure to identify a holographic full-art front MUST NOT suppress an
independently evaluated standard Pokémon-back printed-frame measurement.
Back artwork verification establishes *card-back type*, NOT exact physical-card
identity. Images are processed in memory and no grades are assigned.
"""
from __future__ import annotations

import hashlib
from typing import Any

from v38_centering_lab import (PILOT_ID, _read_photo, _reviewed_points,
                               _back_measure, _encode_overlay)
from v38_1_validation import _first, _front_landmark_preview


def _front_diagnostic(raw: bytes, corners: Any) -> dict:
    import cv2
    import numpy as np
    from template_alignment_v34 import inspect_photo

    img = _read_photo(raw)
    h, w = img.shape[:2]
    physical = _reviewed_points(corners, w, h, 'front')
    # Run the existing exact-card verifier; NEVER weaken its thresholds.
    match = inspect_photo(raw, PILOT_ID, include_overlay=False)
    verified = match.get('status') == 'artwork_aligned_only'
    reason = str(match.get('reason') or '')
    overlay = img.copy()
    cv2.polylines(overlay, [np.rint(physical).astype(np.int32)], True,
                  (255, 255, 0), 3, cv2.LINE_AA)
    regions = []
    if verified:
        try:
            preview = _front_landmark_preview(raw, corners, match['projected_template_quad'])
            result_overlay = preview['overlay_data_uri']
            regions = preview['projected_regions']
        except (ValueError, TypeError, KeyError, cv2.error):
            result_overlay = _encode_overlay(overlay)
            reason = 'Artwork matched, but catalog-landmark preview was unavailable.'
    else:
        result_overlay = _encode_overlay(overlay)

    diagnostic = {
        'stage': ('registered_artwork' if verified else 'exact_card_registration'),
        'reason': reason or ('Exact-card feature registration succeeded.' if verified else 'Catalog matching did not meet verification requirements.'),
        'candidate_matches': match.get('candidate_matches'),
        'consistent_matches': match.get('inliers'),
        'matching_fraction': match.get('inlier_fraction'),
        'reference_coverage': match.get('reference_coverage'),
        'competing_artwork_inliers': match.get('competing_artwork_inliers'),
        'median_reprojection_error_px': match.get('median_reprojection_error_px'),
        'possible_causes_unverified': ([] if verified else [
            'Holographic reflections or uneven illumination',
            'Motion blur, loss of fine detail, or camera perspective',
            'Different crop or a mismatch with the exact reference artwork',
        ]),
        'what_to_try': ([] if verified else [
            'Use diffuse lighting and tilt the light instead of the camera.',
            'Keep the entire physical card visible and in sharp focus.',
            'Check the card set and number before matching again.',
        ]),
    }
    # The old module can return status 'unavailable' for actual identification
    # failure, but this is separate from whether a conventional back is measurable.
    return {
        'status': 'catalog_artwork_aligned' if verified else 'template_match_withheld',
        'exact_card_match': verified,
        'card_id': PILOT_ID if verified else None,
        'calibrated': False, 'lr': None, 'tb': None, 'centering_grade': None,
        'explanation': ('Catalog artwork recognized; landmark positions remain uncalibrated.'
                        if verified else 'Front artwork could not be independently verified. Its centering is withheld; independently supported back-frame measurements can still be viewed.'),
        'projected_regions': regions,
        'diagnostics': diagnostic,
        'overlay_data_uri': result_overlay,
    }


def _back_independent(raw: bytes, corners: Any, name: str) -> dict:
    im = _read_photo(raw)
    h, w = im.shape[:2]
    quad = _reviewed_points(corners, w, h, name)
    return _back_measure(im, quad)


def analyze_v38_2(front_raw: bytes, back_a_raw: bytes,
                  front_corners: Any, back_a_corners: Any,
                  back_b_raw: bytes | None = None, back_b_corners: Any = None,
                  card_id: str = PILOT_ID, same_card_confirmed: bool = False) -> dict:
    """Independent back measurements and noncalibrated front match diagnostics.

    A back can be measured even if front catalog registration is withheld.
    Front/back association is merely self-attested by a user, never verified by
    our conventional card-back recognizer. No measured value changes grades.
    """
    if card_id != PILOT_ID:
        raise ValueError('The V38.2 pilot currently accepts the declared Giratina V #186 only.')
    if not same_card_confirmed:
        raise ValueError('Confirm that the supplied photographs show the same physical card before comparing them.')
    if back_b_raw is None and back_b_corners is not None:
        raise ValueError('Back photo B corners were supplied without back photo B.')
    if back_b_raw is not None and back_b_corners is None:
        raise ValueError('Mark the four physical corners of optional back photo B.')
    if back_b_raw is not None and hashlib.sha256(back_b_raw).digest() == hashlib.sha256(back_a_raw).digest():
        raise ValueError('Back photos A and B must be different original captures, not the exact same uploaded file.')

    # These tasks must remain independent. A match rejection on the front must
    # never replace the back measurements with 'unavailable'.
    back_a = _back_independent(back_a_raw, back_a_corners, 'back A')
    back_b = _back_independent(back_b_raw, back_b_corners, 'back B') if back_b_raw is not None else None
    front = _front_diagnostic(front_raw, front_corners)

    comparison = {
        'status': 'one_back_only' if back_b is None else 'not_comparable',
        'lr_difference_pp': None, 'tb_difference_pp': None,
        'calibrated': False,
        'explanation': ('Upload another independent back photo to compare repeatability.'
                        if back_b is None else 'At least one back measurement could not be established independently.'),
    }
    if back_b is not None and back_a.get('status') == back_b.get('status') == 'experimental_measurement':
        comparison = {
            'status': 'compared',
            'lr_difference_pp': round(abs(_first(back_a['lr']) - _first(back_b['lr'])), 2),
            'tb_difference_pp': round(abs(_first(back_a['tb']) - _first(back_b['tb'])), 2),
            'calibrated': False,
            'explanation': 'Difference between two independent back-photo measurements. Agreement does not prove absolute centering accuracy.',
        }

    complete = back_a.get('status') == 'experimental_measurement'
    return {
        'status': 'research_only', 'version': 'v38.2',
        'pilot_card_id_declared': card_id,
        'front_identity_status': 'verified_from_artwork' if front['exact_card_match'] else 'unverified',
        'same_card_relation': 'collector_confirmed_only_not_independently_verified',
        'back_measurement_association': ('collector_attested_same_physical_card' if complete else 'not_measured'),
        'back_measurement_type': 'conventional_pokemon_back_blue_frame',
        'front': front, 'back_a': back_a, 'back_b': back_b, 'repeatability': comparison,
        'calibrated': False, 'grade': None, 'grade_updated': False,
        'centering_category_score': None, 'verified_front_lr': None, 'verified_front_tb': None,
        'limitations': 'Back ratios are pixel-based estimates on an independently recognized Pokémon blue frame. They do not independently prove the front is Giratina or that both photos show the same physical card. Collector-confirmed provenance is not proof. No full-art printed-centering standard is calibrated.',
        'privacy_note': 'Research endpoint processes photos in memory and does not intentionally store them.',
    }
