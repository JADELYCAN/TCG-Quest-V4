"""Research gate for later category-score integration. Not called by live AI grading.

Do not derive a precise grade from uncertified geometry or unvalidated flaw
signals. No imitation of PSA/BGS/CGC rules or invented subgrade caps.
"""
from __future__ import annotations


def research_grading_gate(measurement:dict|None,category_scores:dict|None):
    """Allow comparison only after independent evidence has been established.

    This module NEVER computes an official grade; it provides transparent
    eligibility and the reasons that information should be withheld.
    """
    reasons=[]
    if not measurement or measurement.get('verified_cut_edges') is not True:
        reasons.append('The physical cut edges have not been independently verified.')
    if not measurement or measurement.get('calibrated_reference') is not True:
        reasons.append('The template is not a calibrated ideal printing reference.')
    if not measurement or measurement.get('independently_validated_ratio') is not True:
        reasons.append('The ratio lacks independent validation.')
    if not category_scores or any(category_scores.get(k) is None for k in ('centering','corners','edges','surface')):
        reasons.append('One or more condition categories lack a separately supported score.')
    return {'grade':None,'official_grading_standard':None,
            'can_show_validated_measurements':len(reasons)==0,
            'blockers':reasons,
            'note':'Even a measurement-ready card needs a separately validated, documented TCG Quest grade-calculation policy.'}
