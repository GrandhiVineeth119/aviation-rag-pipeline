"""
hallucination_guard.py — Multi-layer verification between retrieval and output.

  Layer 1 — Retrieval Gate:     enough high-scoring chunks?
  Layer 2 — Coverage Check:     do chunks contain the query's key terms?
  Layer 3 — Self-Verification:  ask the LLM to fact-check its own answer

Any critical-layer failure → pipeline returns the refusal message.
"""

import re
from dataclasses import dataclass, field
import numpy as np

from app.config import REFUSAL_MESSAGE

_STOP = frozenset(
    "what is the a an in on at to for of and or how does do can could would "
    "should will are was were been be have has had this that these those it "
    "its with from by about which who whom whose where when why tell me "
    "explain describe please i my your we they there here not".split()
)


@dataclass
class GuardResult:
    passed: bool
    answer: str
    confidence: float
    checks: dict = field(default_factory=dict)


class HallucinationGuard:
    def __init__(self, min_chunks=1, min_avg_score=0.20, coverage_floor=0.30):
        self.min_chunks = min_chunks
        self.min_avg_score = min_avg_score
        self.coverage_floor = coverage_floor

    def run(self, query: str, answer: str, retrieved: list[dict],
            self_verify_fn=None) -> GuardResult:
        checks = {}

        # Layer 1 — retrieval gate
        g_ok, g = self._gate(retrieved)
        checks["retrieval_gate"] = g
        if not g_ok:
            return GuardResult(False, REFUSAL_MESSAGE, 0.0, checks)

        # Layer 2 — coverage
        c_ok, c = self._coverage(query, retrieved)
        checks["coverage"] = c

        # Layer 3 — self-verification
        if self_verify_fn:
            sv = self_verify_fn(query, answer, retrieved)
            checks["self_verify"] = sv
            if not sv.get("supported", True):
                return GuardResult(False, REFUSAL_MESSAGE, 0.2, checks)

        conf = float(np.mean([g["score"], c["score"]]))
        return GuardResult(g_ok, answer if g_ok else REFUSAL_MESSAGE, conf, checks)

    def _gate(self, ret):
        n = len(ret)
        if n < self.min_chunks:
            return False, {"passed": False, "score": 0.0, "n": n}
        avg = float(np.mean([r["score"] for r in ret]))
        ok = avg >= self.min_avg_score
        return ok, {"passed": ok, "score": round(avg, 4), "n": n}

    def _coverage(self, query, ret):
        terms = {w for w in re.findall(r"\b[a-zA-Z]{2,}\b", query.lower()) if w not in _STOP}
        if not terms:
            return True, {"passed": True, "score": 1.0}
        ctx = " ".join(r["chunk"].text.lower() for r in ret)
        found = {t for t in terms if t in ctx}
        score = len(found) / len(terms)
        ok = score >= self.coverage_floor
        return ok, {"passed": ok, "score": round(score, 3),
                     "terms": sorted(terms), "found": sorted(found),
                     "missing": sorted(terms - found)}
