"""
RAG Evaluation & Observability Package
"""

from core.evaluation.dataset import get_benchmark_dataset, BENCHMARK_DATASET_VERSION
from core.evaluation.evaluator import (
    run_ragas_evaluation,
    generate_video_qa_pairs,
    EVAL_MODE_BENCHMARK,
    EVAL_MODE_CURRENT_VIDEO,
)
from core.evaluation.history import load_evaluation_history, save_evaluation_run

__all__ = [
    "get_benchmark_dataset",
    "BENCHMARK_DATASET_VERSION",
    "run_ragas_evaluation",
    "generate_video_qa_pairs",
    "EVAL_MODE_BENCHMARK",
    "EVAL_MODE_CURRENT_VIDEO",
    "load_evaluation_history",
    "save_evaluation_run",
]
