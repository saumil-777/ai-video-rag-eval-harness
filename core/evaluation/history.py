"""
RAG Evaluation History Persistence & Regression Benchmarking Module
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

HISTORY_FILE_PATH = os.path.join(os.path.dirname(__file__), "evaluation_history.json")


def load_evaluation_history() -> List[Dict[str, Any]]:
    """Loads all historical evaluation runs from local persistent JSON storage."""
    if not os.path.exists(HISTORY_FILE_PATH):
        return []
    try:
        with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as ex:
        logger.error(f"Error loading evaluation history from {HISTORY_FILE_PATH}: {ex}")
        return []


def save_evaluation_run(run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Appends a new evaluation run record to historical JSON storage."""
    history = load_evaluation_history()
    
    # Clean sensitive or non-serializable fields if any
    clean_record = {
        "run_id": run_data.get("run_id"),
        "timestamp": run_data.get("timestamp"),
        "evaluation_mode": run_data.get("evaluation_mode", "benchmark"),
        "dataset_version": run_data.get("dataset_version", "1.0.0"),
        "rag_config_version": run_data.get("rag_config_version", "1.0.0"),
        "session_id": run_data.get("session_id", "default"),
        "num_questions": run_data.get("num_questions", 0),
        "context_precision": round(float(run_data.get("context_precision", 0.0)), 4),
        "faithfulness": round(float(run_data.get("faithfulness", 0.0)), 4),
        "answer_relevancy": round(float(run_data.get("answer_relevancy", 0.0)), 4),
        "overall_score": round(float(run_data.get("overall_score", 0.0)), 4),
        "per_question_results": run_data.get("per_question_results", []),
    }

    history.append(clean_record)
    
    try:
        with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation run {clean_record['run_id']} to {HISTORY_FILE_PATH}")
    except Exception as ex:
        logger.error(f"Failed saving evaluation run to {HISTORY_FILE_PATH}: {ex}")

    return history


def compute_regression_deltas(current_run: Dict[str, Any], previous_run: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Computes metric score deltas (current - previous) for regression comparison."""
    if not previous_run:
        return {
            "context_precision_delta": 0.0,
            "faithfulness_delta": 0.0,
            "answer_relevancy_delta": 0.0,
            "overall_score_delta": 0.0,
        }

    return {
        "context_precision_delta": round(current_run.get("context_precision", 0.0) - previous_run.get("context_precision", 0.0), 4),
        "faithfulness_delta": round(current_run.get("faithfulness", 0.0) - previous_run.get("faithfulness", 0.0), 4),
        "answer_relevancy_delta": round(current_run.get("answer_relevancy", 0.0) - previous_run.get("answer_relevancy", 0.0), 4),
        "overall_score_delta": round(current_run.get("overall_score", 0.0) - previous_run.get("overall_score", 0.0), 4),
    }
