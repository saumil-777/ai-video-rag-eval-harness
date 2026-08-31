import unittest
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

from core.llm import get_llm
from core.summarizer import summarize, generate_title
from core.rag_engine import build_rag_chain, ask_question
from utils.exporter import generate_txt_report, generate_pdf_report


class TestPhase3(unittest.TestCase):

    def test_01_central_llm_factory(self):
        """Verify centralized LLM factory instantiates ChatMistralAI correctly."""
        llm_low_temp = get_llm(temperature=0.1)
        llm_high_temp = get_llm(temperature=0.8)

        self.assertEqual(llm_low_temp.temperature, 0.1)
        self.assertEqual(llm_high_temp.temperature, 0.8)
        self.assertEqual(llm_low_temp.model, "mistral-small-latest")

    def test_02_parallel_map_reduce_summarize(self):
        """Verify parallel batch summarization works on single and multi-chunk transcripts."""
        short_transcript = "In this meeting, the team decided to migrate our server to AWS by Q3."
        summary_short = summarize(short_transcript)
        self.assertTrue(len(summary_short) > 0)

        # Multi-chunk transcript (>3000 chars)
        chunk1 = "Section 1: The engineering team reviewed database query optimization. " * 50
        chunk2 = "Section 2: The design team presented the new dark mode user interface. " * 50
        multi_chunk_transcript = chunk1 + "\n\n" + chunk2

        start_time = time.time()
        summary_multi = summarize(multi_chunk_transcript)
        elapsed = time.time() - start_time

        self.assertTrue(len(summary_multi) > 0)
        print(f"\n[Parallel Summarization Benchmark] 2 chunks processed via batch() in {elapsed:.2f}s")

    def test_03_txt_export_generation(self):
        """Verify TXT report generation contains all essential sections."""
        txt_output = generate_txt_report(
            title="Q3 Roadmap Sync",
            summary="Key roadmap decisions finalized.",
            action_items="1. Alex to deploy backend.",
            key_decisions="1. Approved AWS migration.",
            open_questions="1. What is the budget limit?",
            transcript="Full transcript text here.",
        )

        self.assertIn("Q3 Roadmap Sync", txt_output)
        self.assertIn("Key roadmap decisions finalized.", txt_output)
        self.assertIn("1. Alex to deploy backend.", txt_output)
        self.assertIn("1. Approved AWS migration.", txt_output)
        self.assertIn("1. What is the budget limit?", txt_output)
        self.assertIn("Full transcript text here.", txt_output)

    def test_04_pdf_export_generation(self):
        """Verify PDF report generation produces valid PDF bytes."""
        pdf_bytes = generate_pdf_report(
            title="Q3 Roadmap Sync",
            summary="Key roadmap decisions finalized.",
            action_items="1. Alex to deploy backend.",
            key_decisions="1. Approved AWS migration.",
            open_questions="1. What is the budget limit?",
            transcript="Full transcript text here.",
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(len(pdf_bytes) > 500)
        # Standard PDF file header check
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_05_phase2_regression_isolation(self):
        """Phase 2 Regression: Verify per-session vector isolation remains 100% functional."""
        session_a = f"session_reg_a_{uuid.uuid4().hex[:8]}"
        session_b = f"session_reg_b_{uuid.uuid4().hex[:8]}"

        chain_a = build_rag_chain("Project Alpha uses Python and Django.", collection_name=session_a)
        chain_b = build_rag_chain("Project Beta uses Rust and WebAssembly.", collection_name=session_b)

        ans_b = ask_question(chain_b, "What language does Project Alpha use?")
        self.assertIn("could not find", ans_b.lower())

        ans_a = ask_question(chain_a, "What language does Project Alpha use?")
        self.assertIn("Python", ans_a)


if __name__ == "__main__":
    unittest.main()
