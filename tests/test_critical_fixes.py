import os
import shutil
import unittest
import uuid
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

from core.vector_store import build_vector_store, load_vector_store, get_retriever, sanitize_collection_name
from core.rag_engine import build_rag_chain, ask_question, ask_question_with_context
from utils.audio_processor import cleanup_audio_files, DOWNLOADS_DIR


class TestCriticalFixes(unittest.TestCase):

    def setUp(self):
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        self.session_a = f"session_{uuid.uuid4().hex[:12]}"
        self.session_b = f"session_{uuid.uuid4().hex[:12]}"

    def test_01_collection_name_sanitization(self):
        """Verify collection names are sanitized according to ChromaDB rules."""
        raw_name = "Meeting #123! @Test/Video (Final).mp4"
        sanitized = sanitize_collection_name(raw_name)
        self.assertTrue(len(sanitized) >= 3 and len(sanitized) <= 63)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertNotIn("!", sanitized)
        self.assertNotIn("@", sanitized)

    def test_02_vector_store_isolation(self):
        """Verify Video A and Video B create separate collections in ChromaDB."""
        doc_a = "Video A discussing Apple company revenue."
        doc_b = "Video B discussing Banana fruit farming."

        vs_a = build_vector_store(doc_a, collection_name=self.session_a)
        vs_b = build_vector_store(doc_b, collection_name=self.session_b)

        self.assertNotEqual(vs_a._collection.name, vs_b._collection.name)
        self.assertEqual(vs_a._collection.name, sanitize_collection_name(self.session_a))
        self.assertEqual(vs_b._collection.name, sanitize_collection_name(self.session_b))

    def test_03_cross_video_retrieval_isolation(self):
        """Verify Video B retriever NEVER returns Video A content."""
        content_a = "APPLE_TEST_UNIQUE_KEYWORD: The secret project code is TimCook123."
        content_b = "BANANA_TEST_UNIQUE_KEYWORD: The agricultural yields increased by 50 percent."

        vs_a = build_vector_store(content_a, collection_name=self.session_a)
        vs_b = build_vector_store(content_b, collection_name=self.session_b)

        retriever_b = get_retriever(vs_b, k=4)
        results_b = retriever_b.invoke("What is the secret project code?")
        page_contents_b = [doc.page_content for doc in results_b]

        # Verify Video B query does NOT return Video A's unique keyword
        for text in page_contents_b:
            self.assertNotIn("APPLE_TEST_UNIQUE_KEYWORD", text)
            self.assertNotIn("TimCook123", text)

        # Verify Video A query DOES return Video A content
        retriever_a = get_retriever(vs_a, k=4)
        results_a = retriever_a.invoke("What is the secret project code?")
        page_contents_a = [doc.page_content for doc in results_a]
        found = any("TimCook123" in text for text in page_contents_a)
        self.assertTrue(found, "Video A retriever failed to find Video A content")

    def test_04_audio_cleanup_normal(self):
        """Verify temporary WAV files exist during processing and are removed by cleanup_audio_files."""
        test_wav = os.path.join(DOWNLOADS_DIR, f"dummy_{uuid.uuid4().hex[:8]}.wav")
        chunk_wav = f"{test_wav}_chunk_0.wav"

        # Create dummy audio files
        audio = AudioSegment.silent(duration=2000)
        audio.export(test_wav, format="wav")
        audio.export(chunk_wav, format="wav")

        self.assertTrue(os.path.exists(test_wav))
        self.assertTrue(os.path.exists(chunk_wav))

        # Perform cleanup
        cleanup_audio_files(file_paths=[chunk_wav], main_wav_path=test_wav)

        # Assert files are deleted
        self.assertFalse(os.path.exists(test_wav))
        self.assertFalse(os.path.exists(chunk_wav))

    def test_05_audio_cleanup_after_failure(self):
        """Verify temporary files are cleaned up even when an exception occurs."""
        test_wav = os.path.join(DOWNLOADS_DIR, f"dummy_fail_{uuid.uuid4().hex[:8]}.wav")
        audio = AudioSegment.silent(duration=2000)
        audio.export(test_wav, format="wav")

        self.assertTrue(os.path.exists(test_wav))

        with self.assertRaises(RuntimeError):
            try:
                raise RuntimeError("Simulated transcription pipeline crash!")
            finally:
                cleanup_audio_files(main_wav_path=test_wav)

        # File must be cleaned up despite the exception
        self.assertFalse(os.path.exists(test_wav))

    def test_06_multiple_sessions_isolation(self):
        """Verify running multiple sessions maintains complete isolated RAG contexts."""
        session_1 = f"session_{uuid.uuid4().hex[:12]}"
        session_2 = f"session_{uuid.uuid4().hex[:12]}"

        chain_1 = build_rag_chain("Session 1: Key topic is Quantum Computing breakthroughs.", collection_name=session_1)
        chain_2 = build_rag_chain("Session 2: Key topic is Deep Sea Marine Biology.", collection_name=session_2)

        ans_1 = ask_question(chain_1, "What is the key topic?")
        ans_2 = ask_question(chain_2, "What is the key topic?")

        self.assertIn("Quantum", ans_1)
        self.assertIn("Marine", ans_2)
        self.assertNotIn("Marine", ans_1)
        self.assertNotIn("Quantum", ans_2)

    def test_07_ask_question_with_context_returns_real_contexts(self):
        """Verify ask_question_with_context returns actual retrieved document contexts.

        This is the critical regression test for the RAGChain refactor:
        - contexts must be a non-empty list[str]
        - each context must be a non-empty string
        - at least one context must contain content from the indexed transcript
        - answer must be a non-empty string
        """
        content = (
            "The project deadline is December 31st and the total budget is $100,000. "
            "The engineering lead is Alice and QA lead is Bob. "
            "All services will be deployed on AWS using Docker containers."
        )
        session_id = f"session_ctx_{uuid.uuid4().hex[:12]}"
        chain = build_rag_chain(content, collection_name=session_id)

        result = ask_question_with_context(chain, "What is the project deadline?")

        # Structure checks
        self.assertIn("answer", result, "Result must contain 'answer' key")
        self.assertIn("contexts", result, "Result must contain 'contexts' key")
        self.assertIsInstance(result["answer"], str)
        self.assertIsInstance(result["contexts"], list)

        # Non-empty answer
        self.assertGreater(len(result["answer"].strip()), 0, "Answer must not be empty")

        # Non-empty contexts — this is the primary regression check
        self.assertGreater(
            len(result["contexts"]), 0,
            "contexts must be non-empty; retriever must return real documents"
        )
        for ctx in result["contexts"]:
            self.assertIsInstance(ctx, str)
            self.assertGreater(len(ctx.strip()), 0, "Each context chunk must be non-empty")

        # At least one chunk must contain transcript content
        combined = " ".join(result["contexts"])
        self.assertIn(
            "December", combined,
            "Retrieved contexts must contain content from the indexed transcript"
        )

    def test_08_chat_mistral_ai_combine_outputs_compatibility(self):
        """Verify ChatMistralAI._combine_llm_outputs safely handles dict token_usage values without TypeError.

        This protects multi-generation calls (like Ragas AnswerRelevancy metric) from crashing.
        """
        from langchain_mistralai import ChatMistralAI
        from core.llm import get_llm

        # Get an instance (or instantiate directly)
        llm = ChatMistralAI(model="mistral-small-latest", api_key="dummy_key")

        o1 = {"token_usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}, "service_tier": "standard"}}
        o2 = {"token_usage": {"prompt_tokens": 15, "prompt_tokens_details": {"cached_tokens": 5}, "service_tier": "standard"}}

        res = llm._combine_llm_outputs([o1, o2])
        self.assertIn("token_usage", res)
        self.assertEqual(res["token_usage"]["prompt_tokens"], 25)
        self.assertEqual(res["token_usage"]["prompt_tokens_details"]["cached_tokens"], 5)


if __name__ == "__main__":
    unittest.main()

