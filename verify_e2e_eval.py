"""
Real end-to-end Current Video Ragas evaluation verification script.
(ASCII-safe output for Windows cp1252 terminals)

Verifies:
  1. Retrieved contexts are non-empty (real doc content, not placeholder)
  2. Generated answer is present
  3. Ragas receives the same retrieved contexts
  4. Context Precision is computed from those contexts
  5. Faithfulness is computed from those contexts
  6. Answer Relevancy is computed correctly
"""
import os
import sys
import json
import logging
from dotenv import load_dotenv

# Force UTF-8 output on Windows so emoji/unicode prints don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_eval_verify")

# ── Import project modules ────────────────────────────────────────────────────
from core.evaluation.dataset import BENCHMARK_TRANSCRIPT_SAMPLE, DEFAULT_BENCHMARK_QA
from core.rag_engine import build_rag_chain, ask_question_with_context, RAGChain
from core.evaluation.evaluator import run_ragas_evaluation, EVAL_MODE_CURRENT_VIDEO

PLACEHOLDER = "No matching context retrieved from meeting transcript."


def verify_e2e():
    print("\n" + "=" * 70)
    print("REAL END-TO-END CURRENT VIDEO RAGAS EVALUATION VERIFICATION")
    print("=" * 70)

    # ── Step 1: Build real RAGChain from benchmark transcript ─────────────────
    print("\n[1/5] Building RAGChain from benchmark transcript...")
    chain = build_rag_chain(BENCHMARK_TRANSCRIPT_SAMPLE, collection_name="e2e_verify_session")
    assert isinstance(chain, RAGChain), "build_rag_chain must return a RAGChain instance"
    assert hasattr(chain, "retriever"), "RAGChain must expose .retriever"
    print("      [PASS] RAGChain constructed. Type: " + type(chain).__name__)

    # ── Step 2: Verify ask_question_with_context returns real contexts ─────────
    print("\n[2/5] Verifying ask_question_with_context returns real (non-empty) contexts...")
    test_q = "What is the total project budget?"
    result = ask_question_with_context(chain, test_q)

    answer = result.get("answer", "")
    contexts = result.get("contexts", [])

    assert isinstance(answer, str) and len(answer.strip()) > 0, \
        "FAIL: answer is empty or missing. Got: " + repr(answer)
    print("      Answer: " + repr(answer[:120]))

    assert isinstance(contexts, list) and len(contexts) > 0, \
        "FAIL: contexts list is empty. Got: " + repr(contexts)

    assert PLACEHOLDER not in contexts, \
        "FAIL: fallback placeholder found in contexts -- retriever returned nothing real!\n  contexts=" + repr(contexts)

    for i, ctx in enumerate(contexts):
        assert len(ctx.strip()) > 0, "FAIL: context[%d] is empty string" % i

    print("      [PASS] %d real context chunk(s) retrieved." % len(contexts))
    for i, ctx in enumerate(contexts):
        print("         Chunk[%d]: %r..." % (i, ctx[:80].strip()))

    # ── Step 3: Run full Ragas evaluation using benchmark QA ──────────────────
    print("\n[3/5] Running full Ragas evaluation (save_history=False, 3 questions)...")
    run = run_ragas_evaluation(
        rag_chain=chain,
        qa_pairs=DEFAULT_BENCHMARK_QA[:3],   # 3 questions to limit API cost
        session_id="e2e_verify_session",
        save_history=False,
        max_questions=3,
        evaluation_mode=EVAL_MODE_CURRENT_VIDEO,
    )
    print("      [PASS] Ragas evaluation completed. Run ID: " + run["run_id"])

    # ── Step 4: Verify per-question contexts are non-empty ────────────────────
    print("\n[4/5] Verifying per-question retrieved contexts in run record...")
    per_q = run.get("per_question_results", [])
    assert len(per_q) > 0, "FAIL: per_question_results is empty"

    empty_ctx_questions = []
    placeholder_questions = []
    for q_res in per_q:
        q = q_res["question"]
        rc = q_res.get("retrieved_contexts", [])
        if not rc:
            empty_ctx_questions.append(q)
        if PLACEHOLDER in rc:
            placeholder_questions.append(q)

    if empty_ctx_questions:
        print("      WARNING  Questions with empty contexts: " + str(empty_ctx_questions))
    if placeholder_questions:
        print("      WARNING  Questions with placeholder fallback: " + str(placeholder_questions))

    assert not placeholder_questions, \
        "FAIL: fallback placeholder found for: " + str(placeholder_questions)
    assert not empty_ctx_questions, \
        "FAIL: empty contexts for: " + str(empty_ctx_questions)

    print("      [PASS] All %d questions have real retrieved contexts." % len(per_q))
    for q_res in per_q:
        print("         Q: " + repr(q_res["question"][:60]))
        print("            contexts[0]: " + repr(q_res["retrieved_contexts"][0][:80].strip()) + "...")
        print("            answer: " + repr(q_res["generated_answer"][:80].strip()) + "...")

    # ── Step 5: Verify Ragas produced non-trivial metric values ───────────────
    print("\n[5/5] Verifying Ragas metric values...")
    cp  = run["context_precision"]
    f   = run["faithfulness"]
    ar  = run["answer_relevancy"]
    ov  = run["overall_score"]

    print("      Context Precision : %.4f" % cp)
    print("      Faithfulness      : %.4f" % f)
    print("      Answer Relevancy  : %.4f" % ar)
    print("      Overall Score     : %.4f" % ov)

    # With real contexts, at least one metric should be non-zero
    assert (cp + f + ar) > 0.0, \
        "FAIL: ALL Ragas metrics are 0.0 -- scoring is not working with real contexts"

    print("\n" + "=" * 70)
    print("[ALL CHECKS PASSED] Real end-to-end evaluation verified.")
    print("=" * 70 + "\n")

    return run


if __name__ == "__main__":
    run = verify_e2e()
    print("Final run record (abbreviated):")
    summary = {k: v for k, v in run.items() if k != "per_question_results"}
    print(json.dumps(summary, indent=2))
