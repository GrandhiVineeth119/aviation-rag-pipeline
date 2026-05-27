"""
evaluate.py — Evaluation with Level 1 baseline vs Level 2 hybrid comparison.

Metrics:
    • Retrieval hit-rate       (did chunks contain the answer?)
    • Faithfulness             (is the answer grounded in retrieved text?)
    • Hallucination rate       (unsupported claims)
    • Citation accuracy        (valid chunk IDs cited)
    • Refusal accuracy         (correct refusal for OOS questions)

Usage:
    python evaluate.py                                      # both modes
    python evaluate.py --mode vector                        # baseline only
    python evaluate.py --mode hybrid                        # hybrid only
    python evaluate.py --out report.md --json results.json  # save outputs
"""

import argparse, json, re, time
from pathlib import Path
from app.config import PipelineConfig, REFUSAL_MESSAGE
from app.pipeline import RAGPipeline


# ── Metric helpers ────────────────────────────────────────────────────────

_STOP = {"the","and","for","are","was","with","from","that","this","not","but",
         "its","has","had","will","can","more","than","also","such","each",
         "been","does","which","their","there","these","those","about","would",
         "could","should","into","only","very","just","some","other","based","used"}

def retrieval_hit(expected: str | None, chunks: list[dict]) -> float:
    if not expected or not chunks: return 0.0
    terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", expected.lower())) - _STOP
    if not terms: return 1.0
    ctx = " ".join(c["chunk"].text.lower() for c in chunks)
    return sum(1 for t in terms if t in ctx) / len(terms)

def faithfulness(answer: str, chunks: list[dict]) -> float:
    if REFUSAL_MESSAGE in answer: return 1.0
    terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower())) - _STOP
    if not terms: return 1.0
    ctx = " ".join(c["chunk"].text.lower() for c in chunks)
    return sum(1 for t in terms if t in ctx) / len(terms)

def citation_accuracy(answer: str, chunks: list[dict]) -> float:
    if REFUSAL_MESSAGE in answer: return 1.0
    cited = set(re.findall(r"\[([a-f0-9]{12})\]", answer))
    if not cited: return 0.0
    valid = {c["chunk"].chunk_id for c in chunks}
    return len(cited & valid) / len(cited)


# ── Run one evaluation pass ──────────────────────────────────────────────

def run_pass(pipe: RAGPipeline, questions: list, use_hybrid: bool) -> list:
    mode = "hybrid" if use_hybrid else "vector"
    results = []
    for q in questions:
        t0 = time.time()
        qr = pipe.query(q["question"], use_hybrid=use_hybrid)
        ms = round((time.time() - t0) * 1000)

        should_refuse = q.get("should_refuse", False)
        did_refuse = qr.refused
        expected = q.get("expected_answer")

        r_hit = retrieval_hit(expected, qr.retrieved_chunks) if not should_refuse else 0.0
        faith = faithfulness(qr.answer, qr.retrieved_chunks)
        cite  = citation_accuracy(qr.answer, qr.retrieved_chunks)
        halluc = faith < 0.5 and not should_refuse
        refusal_ok = (did_refuse == should_refuse)

        results.append({
            "id": q["id"], "category": q["category"],
            "question": q["question"],
            "expected_answer": expected,
            "generated_answer": qr.answer[:500],
            "should_refuse": should_refuse, "did_refuse": did_refuse,
            "refusal_correct": refusal_ok,
            "retrieval_hit_rate": round(r_hit, 3),
            "faithfulness": round(faith, 3),
            "hallucination": halluc,
            "citation_accuracy": round(cite, 3),
            "confidence": round(qr.confidence, 3),
            "num_chunks": len(qr.retrieved_chunks),
            "elapsed_ms": ms,
            "mode": mode,
            "tags": q.get("tags", []),
        })
    return results


def aggregate(results: list) -> dict:
    n = len(results)
    answerable = [r for r in results if not r["should_refuse"]]
    refusable  = [r for r in results if r["should_refuse"]]
    def _avg(lst, key):
        vals = [r[key] for r in lst]
        return round(sum(vals)/len(vals), 3) if vals else 0
    return {
        "total": n,
        "retrieval_hit_rate": _avg(answerable, "retrieval_hit_rate"),
        "faithfulness": _avg(answerable, "faithfulness"),
        "hallucination_rate": round(sum(1 for r in answerable if r["hallucination"]) / max(len(answerable),1), 3),
        "citation_accuracy": _avg(answerable, "citation_accuracy"),
        "refusal_accuracy": round(sum(1 for r in results if r["refusal_correct"]) / n, 3),
        "oos_refusal_precision": round(sum(1 for r in refusable if r["did_refuse"]) / max(len(refusable),1), 3),
        "avg_latency_ms": round(sum(r["elapsed_ms"] for r in results) / n),
    }


# ── Markdown report ──────────────────────────────────────────────────────

def generate_report(data: dict) -> str:
    L = []

    L.append("# RAG Pipeline — Evaluation Report\n")

    # ── If both modes available, show comparison table ────────────────
    if "vector" in data and "hybrid" in data:
        va, ha = data["vector"]["aggregate"], data["hybrid"]["aggregate"]
        L.append("## Level 1 vs Level 2 Comparison\n")
        L.append("| Metric | Vector (L1) | Hybrid (L2) | Δ |")
        L.append("|--------|------------|------------|---|")
        for key in ["retrieval_hit_rate","faithfulness","hallucination_rate",
                     "citation_accuracy","refusal_accuracy","avg_latency_ms"]:
            v, h = va[key], ha[key]
            if key == "avg_latency_ms":
                L.append(f"| {key} | {v} ms | {h} ms | {h-v:+d} ms |")
            else:
                delta = h - v
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "─")
                L.append(f"| **{key}** | {v:.1%} | {h:.1%} | {delta:+.1%} {arrow} |")
        L.append("")

    # ── Per-mode details ─────────────────────────────────────────────
    for mode_key in ["vector", "hybrid"]:
        if mode_key not in data:
            continue
        agg = data[mode_key]["aggregate"]
        results = data[mode_key]["per_question"]
        label = "Level 1 — Vector Only" if mode_key == "vector" else "Level 2 — Hybrid (BM25 + Vector + Reranker)"

        L.append(f"## {label}\n")
        L.append("| Metric | Value |")
        L.append("|--------|-------|")
        L.append(f"| Retrieval Hit-Rate | {agg['retrieval_hit_rate']:.1%} |")
        L.append(f"| Faithfulness | {agg['faithfulness']:.1%} |")
        L.append(f"| Hallucination Rate | {agg['hallucination_rate']:.1%} |")
        L.append(f"| Citation Accuracy | {agg['citation_accuracy']:.1%} |")
        L.append(f"| Refusal Accuracy | {agg['refusal_accuracy']:.1%} |")
        L.append(f"| OOS Refusal Precision | {agg['oos_refusal_precision']:.1%} |")
        L.append(f"| Avg Latency | {agg['avg_latency_ms']} ms |")
        L.append("")

        L.append("### Per-Question Results\n")
        L.append("| # | Cat | Question | Refuse | Faith | Cite | Hit |")
        L.append("|---|-----|----------|--------|-------|------|-----|")
        for r in results:
            qs = r["question"][:42] + ("…" if len(r["question"]) > 42 else "")
            L.append(
                f"| {r['id']:02d} | {r['category'][:6]} | {qs} | "
                f"{'✓' if r['refusal_correct'] else '✗'} | "
                f"{r['faithfulness']:.2f} | {r['citation_accuracy']:.2f} | "
                f"{r['retrieval_hit_rate']:.2f} |"
            )

        # Best / Worst 5
        ans = sorted([r for r in results if not r["should_refuse"]],
                     key=lambda x: x["faithfulness"], reverse=True)

        L.append(f"\n### 5 Best Answers ({label})\n")
        for r in ans[:5]:
            L.append(f"**Q{r['id']}: {r['question']}**")
            L.append(f"- Faithfulness: {r['faithfulness']:.2f}, "
                     f"Citation: {r['citation_accuracy']:.2f}, "
                     f"Hit-Rate: {r['retrieval_hit_rate']:.2f}")
            L.append(f"- Answer: _{r['generated_answer'][:180]}…_")
            L.append(f"- **Why good:** High faithfulness — answer is fully "
                     f"grounded in retrieved context with valid citations.\n")

        L.append(f"\n### 5 Worst Answers ({label})\n")
        for r in ans[-5:]:
            L.append(f"**Q{r['id']}: {r['question']}**")
            L.append(f"- Faithfulness: {r['faithfulness']:.2f}, "
                     f"Citation: {r['citation_accuracy']:.2f}, "
                     f"Hit-Rate: {r['retrieval_hit_rate']:.2f}")
            L.append(f"- Answer: _{r['generated_answer'][:180]}…_")
            L.append(f"- **Why weak:** Lower faithfulness — retrieved chunks "
                     f"may have incomplete coverage, or answer used terms "
                     f"not in source text.\n")

    # ── OOS Refusal ──────────────────────────────────────────────────
    first_mode = "vector" if "vector" in data else "hybrid"
    L.append("\n## Out-of-Scope Refusal Analysis\n")
    for r in data[first_mode]["per_question"]:
        if r["should_refuse"]:
            s = "✓ Correctly refused" if r["did_refuse"] else "✗ FAILED — answered when it should have refused"
            L.append(f"- Q{r['id']}: \"{r['question'][:55]}\" → {s}")

    # ── Methodology ──────────────────────────────────────────────────
    L.append("\n## Methodology\n")
    L.append("- **Retrieval Hit-Rate**: Fraction of expected-answer key terms found in retrieved chunks.")
    L.append("- **Faithfulness**: Fraction of generated-answer terms grounded in retrieved context (1.0 for refusals).")
    L.append("- **Hallucination Rate**: % of answerable questions with faithfulness < 0.5.")
    L.append("- **Citation Accuracy**: Fraction of cited [chunk_id]s that exist in the retrieved set.")
    L.append("- **Refusal Accuracy**: System correctly refused OOS + correctly answered in-scope.")
    L.append("- **Level 2 Hybrid**: BM25 candidates + FAISS candidates → Reciprocal Rank Fusion → Cross-Encoder reranker.")

    return "\n".join(L)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    ap.add_argument("--questions", default="eval_data/questions.json")
    ap.add_argument("--mode", choices=["vector", "hybrid", "both"], default="both")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pipe = RAGPipeline(PipelineConfig())
    if not pipe.store.ready:
        print("⚠  Index empty — run `python ingest.py` first.")
        return

    with open(args.questions) as f:
        questions = json.load(f)

    data = {}

    if args.mode in ("vector", "both"):
        print(f"\n{'='*60}")
        print(f"  Level 1 — Vector-Only Retrieval ({len(questions)} questions)")
        print(f"{'='*60}")
        vec_results = run_pass(pipe, questions, use_hybrid=False)
        for r in vec_results:
            st = "✓" if r["refusal_correct"] and (r["should_refuse"] or r["faithfulness"]>=0.5) else "✗"
            print(f"  [{st}] Q{r['id']:02d} | faith={r['faithfulness']:.2f} "
                  f"cite={r['citation_accuracy']:.2f} | {r['elapsed_ms']}ms")
        data["vector"] = {"aggregate": aggregate(vec_results), "per_question": vec_results}

    if args.mode in ("hybrid", "both"):
        print(f"\n{'='*60}")
        print(f"  Level 2 — Hybrid Retrieval ({len(questions)} questions)")
        print(f"{'='*60}")
        hyb_results = run_pass(pipe, questions, use_hybrid=True)
        for r in hyb_results:
            st = "✓" if r["refusal_correct"] and (r["should_refuse"] or r["faithfulness"]>=0.5) else "✗"
            print(f"  [{st}] Q{r['id']:02d} | faith={r['faithfulness']:.2f} "
                  f"cite={r['citation_accuracy']:.2f} | {r['elapsed_ms']}ms")
        data["hybrid"] = {"aggregate": aggregate(hyb_results), "per_question": hyb_results}

    # Save report
    report = generate_report(data)
    Path(args.out).write_text(report)
    print(f"\n Report → {args.out}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f" JSON → {args.json}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for mode_key in ["vector", "hybrid"]:
        if mode_key not in data: continue
        a = data[mode_key]["aggregate"]
        print(f"\n  [{mode_key.upper()}]")
        for k,v in a.items():
            if isinstance(v, float) and v <= 1:
                print(f"    {k:30s}: {v:.1%}")
            else:
                print(f"    {k:30s}: {v}")


if __name__ == "__main__":
    main()
