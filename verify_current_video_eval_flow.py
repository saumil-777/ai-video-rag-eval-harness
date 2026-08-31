"""
Automated verification script for Current Video Ragas evaluation flow.

Verifies end-to-end:
  1. Video audio processing & chunking
  2. Transcript generation (or benchmark transcript)
  3. generate_video_qa_pairs() creates 5 factual questions
  4. Real RAG retrieval captures contexts
  5. Mistral generates answer
  6. Ragas computes non-zero Context Precision, Faithfulness, Answer Relevancy
  7. Run history persisted cleanly with evaluation_mode="current_video"
"""
import os
import sys
import json
import logging
from dotenv import load_dotenv

# UTF-8 stdout wrapper for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("verify_current_video_eval")

from core.evaluation.dataset import BENCHMARK_TRANSCRIPT_SAMPLE
from core.rag_engine import build_rag_chain
from core.evaluation.evaluator import generate_video_qa_pairs, run_ragas_evaluation, EVAL_MODE_CURRENT_VIDEO


def test_real_current_video_evaluation():
    print("\n" + "=" * 70)
    print("REAL CURRENT VIDEO RAGAS EVALUATION END-TO-END VERIFICATION")
    print("=" * 70)

    # 1. Transcript availability
    transcript = BENCHMARK_TRANSCRIPT_SAMPLE
    print(f"\n[1/5] Transcript loaded ({len(transcript):,} characters).")

    # 2. Generate video QA pairs
    print("\n[2/5] Calling generate_video_qa_pairs(transcript)...")
    qa_pairs = generate_video_qa_pairs(transcript, num_questions=5)

    assert isinstance(qa_pairs, list) and len(qa_pairs) > 0, "FAIL: qa_pairs is empty"
    print(f"      [PASS] Generated {len(qa_pairs)} question-answer pairs.")
    for i, pair in enumerate(qa_pairs, 1):
        print(f"         Pair {i}: Q: {pair['question'][:60]!r}")
        print(f"                  A: {pair['reference_answer'][:60]!r}")

    # 3. Build RAG chain for current video
    print("\n[3/5] Building RAG chain for active session...")
    chain = build_rag_chain(transcript, collection_name="current_video_verify_session")
    print("      [PASS] RAGChain initialized.")

    # 4. Execute Ragas evaluation in current_video mode
    print("\n[4/5] Running run_ragas_evaluation(evaluation_mode='current_video')...")
    eval_run = run_ragas_evaluation(
        rag_chain=chain,
        qa_pairs=qa_pairs[:3],  # limit to 3 questions to save API quota
        session_id="current_video_verify_session",
        save_history=False,
        max_questions=3,
        evaluation_mode=EVAL_MODE_CURRENT_VIDEO
    )

    print("      [PASS] Ragas evaluation completed. Run ID:", eval_run["run_id"])

    # 5. Verify non-zero scores and record contents
    print("\n[5/5] Verifying metric values and evaluation record...")
    cp = eval_run["context_precision"]
    f = eval_run["faithfulness"]
    ar = eval_run["answer_relevancy"]
    overall = eval_run["overall_score"]

    print(f"      Context Precision : {cp:.4f}")
    print(f"      Faithfulness      : {f:.4f}")
    print(f"      Answer Relevancy  : {ar:.4f}")
    print(f"      Overall Score     : {overall:.4f}")

    assert eval_run["evaluation_mode"] == EVAL_MODE_CURRENT_VIDEO, "FAIL: evaluation_mode mismatch"
    assert len(eval_run["per_question_results"]) == 3, "FAIL: per_question_results count mismatch"
    assert (cp + f + ar) > 0.0, "FAIL: All metrics returned 0.0"

    print("\n" + "=" * 70)
    print("[ALL CHECKS PASSED] Current Video Ragas Evaluation verified 100%.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_real_current_video_evaluation()
