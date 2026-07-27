"""Seam 4: everything meets here. All comparison happens in MusicXML space.

    evaluate(predicted: MusicXMLStr, truth: MusicXMLStr) -> Metrics

Project 2 scope is symbol error rate only; OMR-NED and TEDn arrive in Project 3.
"""

from omrt.eval.editdistance import EditOps, levenshtein
from omrt.eval.symbols import to_symbols

__all__ = ["EditOps", "levenshtein", "to_symbols"]
