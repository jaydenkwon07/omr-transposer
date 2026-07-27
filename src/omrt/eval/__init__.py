"""Seam 4: everything meets here. All comparison happens in MusicXML space.

    evaluate(predicted: MusicXMLStr, truth: MusicXMLStr) -> Metrics

Project 2 scope is symbol error rate only; OMR-NED and TEDn arrive in Project 3.
"""

from omrt.eval.editdistance import EditOps, levenshtein
from omrt.eval.metrics import Metrics, evaluate
from omrt.eval.symbols import to_symbols

__all__ = ["EditOps", "Metrics", "evaluate", "levenshtein", "to_symbols"]
