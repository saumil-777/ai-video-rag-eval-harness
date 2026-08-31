import re
import sys
import types
import uuid
import time

import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# 1. Compatibility Shim for LangChain 0.4 / Ragas VertexAI import boundary
try:
    import langchain_community.chat_models.vertexai
except (ImportError, ModuleNotFoundError):
    import langchain_core.language_models.chat_models
    shim = types.ModuleType("langchain_community.chat_models.vertexai")
    shim.ChatVertexAI = langchain_core.language_models.chat_models.BaseChatModel
    sys.modules["langchain_community.chat_models.vertexai"] = shim

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

# Import answer_relevancy with fallback for version compatibility
try:
    from ragas.metrics import answer_relevancy
except ImportError:
    try:
        from ragas.metrics.collections import answer_relevancy
    except ImportError:
        from ragas.metrics import answer_relevance as answer_relevancy

from core.llm import get_llm
from core.vector_store import get_embeddings
from core.rag_engine import ask_question_with_context
from core.evaluation.dataset import get_benchmark_dataset, BENCHMARK_DATASET_VERSION
from core.evaluation.history import save_evaluation_run, load_evaluation_history, compute_regression_deltas

logger = logging.getLogger(__name__)

# Evaluation mode constants
EVAL_MODE_BENCHMARK = "benchmark"
EVAL_MODE_CURRENT_VIDEO = "current_video"


def _parse_qa_pairs_from_text(raw_text: str) -> List[Dict[str, str]]:
    """Parse question-answer pairs from LLM text output in a robust, multi-format manner.

    Supports:
      - Q: ... / A: ...
      - 1. Q: ... / 1. A: ...
      - **Q:** ... / **A:** ...
      - **Q1:** ... / **A1:** ...
      - Question 1: ... / Answer 1: ...
      - Multi-line answers
    """
    if not raw_text or not raw_text.strip():
        return []

    qa_pairs = []
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]

    current_q = None
    current_a_parts = []

    for line in lines:
        q_match = re.match(
            r'^(?:\d+[\.\)]\s*)?[\*\#\s]*\(?question\b\s*\d*[\*\#\s]*[:\.-]?[\*\#\s]*|^(?:\d+[\.\)]\s*)?[\*\#\s]*\(?q\d*[\*\#\s]*[:\.-]\s*[\*\#\s]*',
            line,
            re.IGNORECASE
        )
        a_match = re.match(
            r'^(?:\d+[\.\)]\s*)?[\*\#\s]*\(?answer\b\s*\d*[\*\#\s]*[:\.-]?[\*\#\s]*|^(?:\d+[\.\)]\s*)?[\*\#\s]*\(?a\d*[:\.-]\s*[\*\#\s]*',
            line,
            re.IGNORECASE
        )

        if q_match:
            if current_q and current_a_parts:
                ans_str = " ".join(current_a_parts).strip()
                if current_q and ans_str:
                    qa_pairs.append({"question": current_q, "reference_answer": ans_str})

            q_text = line[q_match.end():].strip().rstrip("*").strip()
            current_q = q_text
            current_a_parts = []

        elif a_match and current_q:
            a_text = line[a_match.end():].strip().rstrip("*").strip()
            current_a_parts = [a_text]

        elif current_q and current_a_parts:
            current_a_parts.append(line)

    if current_q and current_a_parts:
        ans_str = " ".join(current_a_parts).strip()
        if current_q and ans_str:
            qa_pairs.append({"question": current_q, "reference_answer": ans_str})

    return qa_pairs


def generate_video_qa_pairs(transcript: str, num_questions: int = 5) -> List[Dict[str, str]]:
    """
    Generates grounded evaluation QA pairs from a video transcript using the LLM.
    Questions and reference answers are derived solely from transcript content.
    Never uses benchmark questions or invented information.
    """
    if not transcript or not isinstance(transcript, str) or len(transcript.strip()) < 50:
        raise ValueError("Transcript is empty or too short (< 50 characters) to generate evaluation questions.")

    llm = get_llm(temperature=0.1)

    prompt = f"""You are an evaluation question generator for a RAG system.

Given the following transcript, generate exactly {num_questions} factual question-answer pairs.

STRICT RULES:
1. Questions must be answerable using ONLY the transcript below.
2. Reference answers must be grounded in the transcript — do not invent any information.
3. Questions should cover different facts spread across the transcript.
4. Format your output as a numbered list exactly like this:
   Q: <question text>
   A: <reference answer text>

TRANSCRIPT:
{transcript[:4000]}

Generate {num_questions} question-answer pairs now:"""

    raw_text = ""
    last_exception = None

    for attempt in range(1, 4):
        try:
            response = llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            if raw_text and raw_text.strip():
                break
        except Exception as ex:
            last_exception = ex
            if "429" in str(ex) or "rate_limited" in str(ex):
                logger.warning(f"Rate limit hit generating QA pairs (attempt {attempt}/3). Sleeping 3s... ({ex})")
                time.sleep(3.0)
            else:
                logger.error(f"LLM error generating QA pairs (attempt {attempt}/3): {type(ex).__name__}: {ex}")
                raise RuntimeError(
                    f"LLM call failed to generate evaluation questions: {type(ex).__name__}: {ex}"
                ) from ex

    if not raw_text or not raw_text.strip():
        if last_exception:
            raise RuntimeError(
                f"LLM call failed to generate evaluation questions: {type(last_exception).__name__}: {last_exception}"
            ) from last_exception
        else:
            raise RuntimeError("LLM returned an empty response when generating evaluation questions.")

    # Parse Q:/A: pairs from response using robust multi-format parser
    qa_pairs = _parse_qa_pairs_from_text(raw_text)

    if not qa_pairs:
        raise ValueError(
            "Could not parse QA pairs from LLM response. "
            f"Raw response snippet: {raw_text[:300]}"
        )

    logger.info(f"Generated {len(qa_pairs)} evaluation QA pairs from video transcript.")
    return qa_pairs


def run_ragas_evaluation(
    rag_chain,
    qa_pairs: Optional[List[Dict[str, str]]] = None,
    session_id: str = "default_session",
    save_history: bool = True,
    max_questions: int = 5,
    evaluation_mode: str = EVAL_MODE_BENCHMARK
) -> Dict[str, Any]:
    """
    Executes the REAL existing RAG pipeline against evaluation questions,
    runs Ragas metrics (Context Precision, Faithfulness, Answer Relevancy),
    and records persistent history with regression comparison.

    evaluation_mode: "benchmark" (default) or "current_video"
    """
    if qa_pairs is None:
        if evaluation_mode == EVAL_MODE_CURRENT_VIDEO:
            raise ValueError(
                "qa_pairs must be provided for current_video evaluation mode. "
                "Call generate_video_qa_pairs(transcript) first."
            )
        benchmark = get_benchmark_dataset()
        qa_pairs = benchmark["qa_pairs"]

    if not qa_pairs:
        raise ValueError("QA dataset for evaluation cannot be empty.")

    # Limit questions per run to avoid exceeding API rate limits
    if max_questions and len(qa_pairs) > max_questions:
        qa_pairs = qa_pairs[:max_questions]

    run_id = f"eval_run_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat()
    logger.info(
        f"Starting RAG Evaluation Run {run_id} | mode={evaluation_mode} | {len(qa_pairs)} question(s)"
    )

    # Step 1: Execute existing RAG pipeline for all questions with rate pacing
    eval_data = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": []
    }

    per_question_raw = []

    for idx, item in enumerate(qa_pairs):
        q_text = item["question"]
        ref_text = item.get("reference_answer", "")

        # Execute real RAG pipeline with retry for rate limits
        rag_res = None
        for attempt in range(3):
            try:
                rag_res = ask_question_with_context(rag_chain, q_text)
                break
            except Exception as ex:
                if "429" in str(ex) or "rate_limited" in str(ex):
                    logger.warning(f"Rate limit hit on question {idx+1}. Retrying in 3s... ({ex})")
                    time.sleep(3.0)
                else:
                    logger.error(f"Error on question {idx+1}: {ex}")
                    rag_res = {"answer": f"Error: {ex}", "contexts": []}
                    break

        if not rag_res:
            rag_res = {"answer": "No answer produced.", "contexts": []}

        ans_text = rag_res.get("answer", "")
        ctx_list = rag_res.get("contexts", [])

        # Fallback for empty contexts if retriever found no matches
        if not ctx_list:
            ctx_list = ["No matching context retrieved from meeting transcript."]

        eval_data["question"].append(q_text)
        eval_data["contexts"].append(ctx_list)
        eval_data["answer"].append(ans_text)
        eval_data["ground_truth"].append(ref_text)

        per_question_raw.append({
            "question": q_text,
            "reference_answer": ref_text,
            "answer": ans_text,
            "contexts": ctx_list
        })

        # Pace calls to respect LLM rate limits
        time.sleep(1.0)

    # Step 2: Build dataset and run Ragas Evaluation
    # Ragas 0.4.x v1 column names: question, contexts, answer, ground_truth
    # convert_v1_to_v2_dataset() renames them to: user_input, retrieved_contexts, response, reference
    dataset = Dataset.from_dict(eval_data)
    llm = get_llm()
    embeddings = get_embeddings()

    logger.info(
        f"Dataset columns sent to Ragas: {dataset.column_names} | "
        f"rows={len(dataset)} | "
        f"contexts populated: {all(len(c) > 0 for c in eval_data['contexts'])} | "
        f"answers populated: {all(len(a) > 0 for a in eval_data['answer'])}"
    )

    logger.info("Executing Ragas metrics calculation...")
    results = None
    last_ragas_error = None
    for attempt in range(3):
        try:
            # NOTE: is_async is NOT a valid Ragas 0.4.x parameter — removed.
            # raise_exceptions=False makes Ragas return NaN for failed rows instead of crashing.
            results = evaluate(
                dataset=dataset,
                metrics=[context_precision, faithfulness, answer_relevancy],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
            )
            last_ragas_error = None
            break
        except Exception as ex:
            last_ragas_error = ex
            if "429" in str(ex) or "rate_limited" in str(ex):
                logger.warning(
                    f"Rate limit during Ragas evaluate(). Sleeping 5s before retry "
                    f"(attempt {attempt+1}/3): {ex}"
                )
                time.sleep(5.0)
            else:
                # Log the REAL error — do NOT silently swallow it
                logger.error(
                    f"Ragas evaluate() raised an exception (attempt {attempt+1}/3): "
                    f"{type(ex).__name__}: {ex}"
                )
                break

    if last_ragas_error is not None and results is None:
        raise RuntimeError(
            f"Ragas evaluation failed after retries. "
            f"Last error: {type(last_ragas_error).__name__}: {last_ragas_error}"
        )

    # Log what Ragas returned so we can verify scores are real
    if results is not None:
        logger.info(f"Ragas EvaluationResult repr: {results!r}")

    def extract_metric_val(res_obj, key):
        """Extract a scalar float from Ragas EvaluationResult or pandas Series.

        Ragas 0.4.x EvaluationResult.__getitem__(key) returns a LIST of per-row
        floats, not a scalar. We take nanmean across the list.
        """
        if res_obj is None:
            return 0.0

        # Try subscript access (covers EvaluationResult and dict)
        if hasattr(res_obj, "__getitem__"):
            try:
                val = res_obj[key]
                # EvaluationResult returns list[float] — take mean
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    arr = np.array([v for v in val if v is not None], dtype=float)
                    if len(arr) == 0 or np.isnan(arr).all():
                        return 0.0
                    return float(np.nanmean(arr))
                # Scalar float
                if isinstance(val, (int, float)) and not np.isnan(float(val)):
                    return float(val)
            except (KeyError, IndexError, TypeError):
                pass

        # Pandas Series: use .get(key) or attribute
        if hasattr(res_obj, "get"):
            val = res_obj.get(key)
            if val is not None:
                try:
                    f = float(val)
                    return 0.0 if np.isnan(f) else f
                except (TypeError, ValueError):
                    pass

        # Attribute access
        if hasattr(res_obj, key):
            val = getattr(res_obj, key)
            try:
                f = float(val)
                return 0.0 if np.isnan(f) else f
            except (TypeError, ValueError):
                pass

        return 0.0

    cp_score = extract_metric_val(results, "context_precision")
    f_score = extract_metric_val(results, "faithfulness")
    ar_score = extract_metric_val(results, "answer_relevancy")

    # Overall RAG score is weighted average
    overall_score = round((cp_score * 0.4) + (f_score * 0.4) + (ar_score * 0.2), 4)

    # Attach individual question scores
    df_results = None
    if results is not None and hasattr(results, "to_pandas"):
        try:
            df_results = results.to_pandas()
        except Exception:
            df_results = None

    per_question_results = []
    for idx, item in enumerate(per_question_raw):
        q_cp = cp_score
        q_f = f_score
        q_ar = ar_score

        if df_results is not None and idx < len(df_results):
            row = df_results.iloc[idx]
            q_cp = extract_metric_val(row, "context_precision") or cp_score
            q_f = extract_metric_val(row, "faithfulness") or f_score
            q_ar = extract_metric_val(row, "answer_relevancy") or ar_score

        q_overall = round((q_cp * 0.4) + (q_f * 0.4) + (q_ar * 0.2), 4)

        per_question_results.append({
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "generated_answer": item["answer"],
            "retrieved_contexts": item["contexts"],
            "context_precision": round(q_cp, 4),
            "faithfulness": round(q_f, 4),
            "answer_relevancy": round(q_ar, 4),
            "overall_score": q_overall
        })

    # Compare with previous run of the SAME mode for regression deltas
    existing_history = load_evaluation_history()
    # Only compare against same-mode runs for meaningful regression
    same_mode_runs = [r for r in existing_history if r.get("evaluation_mode", EVAL_MODE_BENCHMARK) == evaluation_mode]
    previous_run = same_mode_runs[-1] if same_mode_runs else None

    run_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "evaluation_mode": evaluation_mode,
        "dataset_version": BENCHMARK_DATASET_VERSION,
        "rag_config_version": "1.0.0",
        "session_id": session_id,
        "num_questions": len(qa_pairs),
        "context_precision": round(cp_score, 4),
        "faithfulness": round(f_score, 4),
        "answer_relevancy": round(ar_score, 4),
        "overall_score": overall_score,
        "per_question_results": per_question_results,
    }

    deltas = compute_regression_deltas(run_record, previous_run)
    run_record.update(deltas)

    if save_history:
        save_evaluation_run(run_record)

    return run_record

