"""Seam 4: everything meets here. All comparison happens in MusicXML space.

    evaluate(predicted: MusicXMLStr, truth: MusicXMLStr) -> Metrics

Project 2 scope is symbol error rate only; OMR-NED and TEDn arrive in Project 3.
"""

from omrt.eval.editdistance import EditOps, levenshtein

__all__ = ["EditOps", "levenshtein"]
