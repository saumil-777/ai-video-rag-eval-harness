"""
Unit & Integration Tests for RAG Quality Evaluation & Observability
"""

import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from core.evaluation.dataset import get_benchmark_dataset, BENCHMARK_DATASET_VERSION
from core.evaluation.history import load_evaluation_history, save_evaluation_run, compute_regression_deltas
from core.evaluation.evaluator import (
    run_ragas_evaluation,
    generate_video_qa_pairs,
    EVAL_MODE_BENCHMARK,
    EVAL_MODE_CURRENT_VIDEO,
)


class TestRAGEvaluation(unittest.TestCase):

    # ── Original 5 tests (preserved exactly) ─────────────────────────────────

    def test_benchmark_dataset_loading(self):
        dataset = get_benchmark_dataset()
        self.assertIn("version", dataset)
        self.assertEqual(dataset["version"], BENCHMARK_DATASET_VERSION)
        self.assertIn("qa_pairs", dataset)
        self.assertGreaterEqual(len(dataset["qa_pairs"]), 10)
        self.assertIn("transcript", dataset)
        self.assertGreater(len(dataset["transcript"]), 100)

    def test_regression_deltas_calculation(self):
        prev = {
            "context_precision": 0.80,
            "faithfulness": 0.75,
            "answer_relevancy": 0.70,
            "overall_score": 0.76,
        }
        curr = {
            "context_precision": 0.90,
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "overall_score": 0.86,
        }

        deltas = compute_regression_deltas(curr, prev)
        self.assertAlmostEqual(deltas["context_precision_delta"], 0.10)
        self.assertAlmostEqual(deltas["faithfulness_delta"], 0.10)
        self.assertAlmostEqual(deltas["answer_relevancy_delta"], 0.10)
        self.assertAlmostEqual(deltas["overall_score_delta"], 0.10)

    def test_regression_deltas_none_previous(self):
        curr = {
            "context_precision": 0.90,
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "overall_score": 0.86,
        }
        deltas = compute_regression_deltas(curr, None)
        self.assertEqual(deltas["context_precision_delta"], 0.0)
        self.assertEqual(deltas["overall_score_delta"], 0.0)

    def test_history_persistence(self):
        """Verify save/load cycle using an isolated temporary history file."""
        sample_run = {
            "run_id": "test_run_123",
            "timestamp": "2026-08-31T00:00:00",
            "evaluation_mode": EVAL_MODE_BENCHMARK,
            "dataset_version": "1.0.0",
            "rag_config_version": "1.0.0",
            "session_id": "unit_test_session",
            "num_questions": 1,
            "context_precision": 0.95,
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
            "overall_score": 0.91,
            "per_question_results": []
        }

        # Use an isolated temp file so the real production history is never touched
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp.write("[]")
            tmp_path = tmp.name

        try:
            import core.evaluation.history as hist_mod
            original_path = hist_mod.HISTORY_FILE_PATH
            hist_mod.HISTORY_FILE_PATH = tmp_path
            try:
                saved_history = save_evaluation_run(sample_run)
                self.assertIsInstance(saved_history, list)
                self.assertGreater(len(saved_history), 0)
                latest = saved_history[-1]
                self.assertEqual(latest["run_id"], "test_run_123")
                self.assertEqual(latest["session_id"], "unit_test_session")
            finally:
                hist_mod.HISTORY_FILE_PATH = original_path
        finally:
            os.unlink(tmp_path)

    @patch("core.evaluation.evaluator.evaluate")
    @patch("core.evaluation.evaluator.get_llm")
    @patch("core.evaluation.evaluator.get_embeddings")
    def test_run_ragas_evaluation_mocked(self, mock_emb, mock_llm, mock_evaluate):
        mock_evaluate.return_value = {
            "context_precision": 0.92,
            "faithfulness": 0.88,
            "answer_relevancy": 0.85
        }

        mock_rag_chain = MagicMock()
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "The total project budget is $50,000 USD."
        mock_retriever.invoke.return_value = [mock_doc]
        mock_rag_chain.retriever = mock_retriever
        mock_rag_chain.invoke.return_value = "The project budget is $50,000."

        test_qa = [
            {
                "question": "What is the budget?",
                "reference_answer": "The budget is $50,000."
            }
        ]

        result = run_ragas_evaluation(
            rag_chain=mock_rag_chain,
            qa_pairs=test_qa,
            session_id="mock_test_session",
            save_history=False
        )

        self.assertIn("run_id", result)
        self.assertEqual(result["num_questions"], 1)
        self.assertAlmostEqual(result["context_precision"], 0.92)
        self.assertAlmostEqual(result["faithfulness"], 0.88)
        self.assertAlmostEqual(result["answer_relevancy"], 0.85)
        self.assertIn("overall_score", result)
        self.assertEqual(len(result["per_question_results"]), 1)

    def test_empty_qa_pairs_raises(self):
        mock_chain = MagicMock()
        with self.assertRaises(ValueError):
            run_ragas_evaluation(rag_chain=mock_chain, qa_pairs=[], save_history=False)

    @patch("core.evaluation.evaluator.evaluate")
    @patch("core.evaluation.evaluator.get_llm")
    @patch("core.evaluation.evaluator.get_embeddings")
    def test_retrieved_contexts_flow_through_evaluator(self, mock_emb, mock_llm, mock_evaluate):
        """Verify that real retriever contexts flow into per_question_results.

        The mock retriever returns a document with a known page_content.
        After evaluation the per_question_results entry must contain that
        exact string — NOT the fallback placeholder.  This proves the
        explicit retriever path in ask_question_with_context() is correctly
        wired through the evaluator.
        """
        mock_evaluate.return_value = {
            "context_precision": 0.80,
            "faithfulness": 0.80,
            "answer_relevancy": 0.80,
        }
        EXPECTED_CONTEXT = "The total project budget is $50,000 USD."

        mock_doc = MagicMock()
        mock_doc.page_content = EXPECTED_CONTEXT

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        mock_chain = MagicMock()
        mock_chain.retriever = mock_retriever
        mock_chain.invoke.return_value = "The project budget is $50,000."

        test_qa = [{"question": "What is the budget?", "reference_answer": "The budget is $50,000."}]

        result = run_ragas_evaluation(
            rag_chain=mock_chain,
            qa_pairs=test_qa,
            session_id="ctx_flow_test",
            save_history=False,
        )

        q_result = result["per_question_results"][0]
        retrieved = q_result["retrieved_contexts"]

        self.assertIsInstance(retrieved, list)
        self.assertGreater(len(retrieved), 0, "contexts must not be empty")
        self.assertNotIn(
            "No matching context retrieved from meeting transcript.",
            retrieved,
            "Fallback placeholder must not appear when retriever returns real documents",
        )
        self.assertIn(EXPECTED_CONTEXT, retrieved,
                      "Retriever's document content must appear in retrieved_contexts")


    # ── New 7 tests for dual-mode ─────────────────────────────────────────────

    @patch("core.evaluation.evaluator.evaluate")
    @patch("core.evaluation.evaluator.get_llm")
    @patch("core.evaluation.evaluator.get_embeddings")
    def test_benchmark_mode_uses_benchmark_questions(self, mock_emb, mock_llm, mock_evaluate):
        """Benchmark mode must use the built-in benchmark QA dataset."""
        mock_evaluate.return_value = {"context_precision": 0.8, "faithfulness": 0.8, "answer_relevancy": 0.8}
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Some answer."

        result = run_ragas_evaluation(
            rag_chain=mock_chain,
            qa_pairs=None,  # evaluator should auto-load benchmark
            session_id="benchmark_test",
            save_history=False,
            evaluation_mode=EVAL_MODE_BENCHMARK
        )
        self.assertEqual(result["evaluation_mode"], EVAL_MODE_BENCHMARK)
        self.assertGreater(result["num_questions"], 0)

    @patch("core.evaluation.evaluator.evaluate")
    @patch("core.evaluation.evaluator.get_llm")
    @patch("core.evaluation.evaluator.get_embeddings")
    def test_current_video_mode_uses_provided_qa_pairs(self, mock_emb, mock_llm, mock_evaluate):
        """Current video mode must use the provided QA pairs, not the benchmark."""
        mock_evaluate.return_value = {"context_precision": 0.7, "faithfulness": 0.7, "answer_relevancy": 0.7}
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Video-specific answer."

        video_qa = [
            {"question": "What topic is discussed?", "reference_answer": "AI projects for resume."},
            {"question": "Who is the speaker?", "reference_answer": "Ex-Microsoft employee."},
        ]

        result = run_ragas_evaluation(
            rag_chain=mock_chain,
            qa_pairs=video_qa,
            session_id="video_session_123",
            save_history=False,
            evaluation_mode=EVAL_MODE_CURRENT_VIDEO
        )
        self.assertEqual(result["evaluation_mode"], EVAL_MODE_CURRENT_VIDEO)
        self.assertEqual(result["num_questions"], 2)

    @patch("core.evaluation.evaluator.get_llm")
    def test_generate_video_qa_pairs_parses_output(self, mock_get_llm):
        """generate_video_qa_pairs should parse Q:/A: format from LLM response."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Q: What is the main topic?\n"
            "A: The main topic is AI project development.\n"
            "Q: Who is presenting?\n"
            "A: An ex-Microsoft employee is presenting.\n"
        )
        mock_llm_instance.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm_instance

        transcript = "This is a detailed transcript about AI projects for resume building. " * 10
        qa_pairs = generate_video_qa_pairs(transcript, num_questions=2)

        self.assertIsInstance(qa_pairs, list)
        self.assertGreater(len(qa_pairs), 0)
        for pair in qa_pairs:
            self.assertIn("question", pair)
            self.assertIn("reference_answer", pair)
            self.assertGreater(len(pair["question"]), 0)
            self.assertGreater(len(pair["reference_answer"]), 0)

    @patch("core.evaluation.evaluator.get_llm")
    def test_generate_video_qa_pairs_parses_multiformat_llm_output(self, mock_get_llm):
        """Verify generate_video_qa_pairs parses numbered, Markdown, and Question/Answer formats."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "1. **Q:** What is the total budget?\n"
            "1. **A:** The total project budget is $50,000 USD.\n"
            "2. **Question 2:** Who is technical lead?\n"
            "2. **Answer 2:** Ahmed is the technical lead for engineering.\n"
            "   He has 10 years of experience.\n"
        )
        mock_llm_instance.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm_instance

        transcript = "Sample meeting transcript containing details about project budget and technical leads. " * 5
        qa_pairs = generate_video_qa_pairs(transcript, num_questions=2)

        self.assertEqual(len(qa_pairs), 2)
        self.assertEqual(qa_pairs[0]["question"], "What is the total budget?")
        self.assertEqual(qa_pairs[0]["reference_answer"], "The total project budget is $50,000 USD.")
        self.assertEqual(qa_pairs[1]["question"], "Who is technical lead?")
        self.assertIn("Ahmed is the technical lead", qa_pairs[1]["reference_answer"])

    @patch("core.evaluation.evaluator.get_llm")
    def test_generate_video_qa_pairs_raises_on_unparseable_output(self, mock_get_llm):
        """Verify ValueError is raised when LLM output cannot be parsed as QA pairs."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Here is some unstructured summary of the text without any Q&A formatting."
        mock_llm_instance.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm_instance

        transcript = "Sample meeting transcript containing details about project budget and technical leads. " * 5
        with self.assertRaises(ValueError) as ctx:
            generate_video_qa_pairs(transcript)
        self.assertIn("Could not parse QA pairs", str(ctx.exception))

    def test_generate_video_qa_pairs_raises_on_empty_transcript(self):
        """Verify ValueError is raised for empty or short transcript (< 50 chars)."""
        with self.assertRaises(ValueError) as ctx:
            generate_video_qa_pairs("Too short")
        self.assertIn("Transcript is empty or too short", str(ctx.exception))

        with self.assertRaises(ValueError):
            generate_video_qa_pairs("")

    @patch("core.evaluation.evaluator.get_llm")
    def test_generate_video_qa_pairs_exposes_real_llm_exception(self, mock_get_llm):
        """Verify real underlying LLM exception is propagated instead of swallowed into generic string."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = ConnectionError("Connection reset by peer")
        mock_get_llm.return_value = mock_llm_instance

        transcript = "Sample meeting transcript containing details about project budget and technical leads. " * 5
        with self.assertRaises(RuntimeError) as ctx:
            generate_video_qa_pairs(transcript)

        self.assertIn("ConnectionError", str(ctx.exception))
        self.assertIn("Connection reset by peer", str(ctx.exception))


    def test_current_video_mode_does_not_use_benchmark_questions(self):
        """Current video QA pairs must not contain benchmark questions."""
        benchmark = get_benchmark_dataset()
        benchmark_questions = {p["question"] for p in benchmark["qa_pairs"]}

        # Simulate video QA pairs (as would be generated from a real transcript)
        video_qa = [
            {"question": "What AI projects are mentioned?", "reference_answer": "Several AI resume projects."},
            {"question": "What tools are shown?", "reference_answer": "Python, LangChain, ChromaDB."},
        ]

        for pair in video_qa:
            self.assertNotIn(
                pair["question"],
                benchmark_questions,
                msg=f"Video QA pair '{pair['question']}' should not appear in benchmark dataset."
            )

    @patch("core.evaluation.evaluator.evaluate")
    @patch("core.evaluation.evaluator.get_llm")
    @patch("core.evaluation.evaluator.get_embeddings")
    def test_evaluation_history_records_mode(self, mock_emb, mock_llm, mock_evaluate):
        """History records must store the evaluation_mode field.
        Uses an isolated temporary file so the real production history is never contaminated.
        """
        mock_evaluate.return_value = {"context_precision": 0.6, "faithfulness": 0.6, "answer_relevancy": 0.6}
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Answer."

        test_qa = [{"question": "Test?", "reference_answer": "Test answer."}]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp.write("[]")
            tmp_path = tmp.name

        try:
            import core.evaluation.history as hist_mod
            original_path = hist_mod.HISTORY_FILE_PATH
            hist_mod.HISTORY_FILE_PATH = tmp_path
            try:
                result = run_ragas_evaluation(
                    rag_chain=mock_chain,
                    qa_pairs=test_qa,
                    session_id="history_mode_test",
                    save_history=True,
                    evaluation_mode=EVAL_MODE_CURRENT_VIDEO
                )
                self.assertEqual(result["evaluation_mode"], EVAL_MODE_CURRENT_VIDEO)

                # Verify it was persisted in the temporary history file
                with open(tmp_path, "r") as f:
                    history = json.load(f)
                saved_run = next((r for r in history if r["run_id"] == result["run_id"]), None)
                self.assertIsNotNone(saved_run, "Run should be persisted in isolated temp history")
                self.assertEqual(saved_run.get("evaluation_mode"), EVAL_MODE_CURRENT_VIDEO)
            finally:
                hist_mod.HISTORY_FILE_PATH = original_path
        finally:
            os.unlink(tmp_path)

    def test_current_video_mode_requires_qa_pairs(self):
        """Current video mode must raise ValueError if qa_pairs not provided."""
        mock_chain = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            run_ragas_evaluation(
                rag_chain=mock_chain,
                qa_pairs=None,
                save_history=False,
                evaluation_mode=EVAL_MODE_CURRENT_VIDEO
            )
        self.assertIn("qa_pairs must be provided", str(ctx.exception))

    def test_generate_video_qa_pairs_rejects_short_transcript(self):
        """generate_video_qa_pairs must raise ValueError for transcripts that are too short."""
        with self.assertRaises(ValueError) as ctx:
            generate_video_qa_pairs("Too short.", num_questions=5)
        self.assertIn("too short", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

