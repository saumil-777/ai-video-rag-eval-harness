import os
import unittest
import uuid
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

from core.vector_store import build_vector_store, get_retriever
from utils.audio_processor import cleanup_audio_files, DOWNLOADS_DIR


class TestRegression(unittest.TestCase):

    def setUp(self):
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    def test_01_vector_store_cross_video_isolation_regression(self):
        """MANDATORY REGRESSION: Verify Meeting B retriever NEVER returns Meeting A content."""
        session_a = f"session_reg_apple_{uuid.uuid4().hex[:8]}"
        session_b = f"session_reg_banana_{uuid.uuid4().hex[:8]}"

        doc_a = "APPLE_SECRET_TOKEN_99: Meeting A discussed secret strategy Alpha."
        doc_b = "BANANA_SECRET_TOKEN_88: Meeting B discussed agricultural quotas."

        vs_a = build_vector_store(doc_a, collection_name=session_a)
        vs_b = build_vector_store(doc_b, collection_name=session_b)

        # Query Meeting B for Meeting A secret token
        retriever_b = get_retriever(vs_b, k=4)
        results_b = retriever_b.invoke("What is APPLE_SECRET_TOKEN_99?")
        for doc in results_b:
            self.assertNotIn("APPLE_SECRET_TOKEN_99", doc.page_content)
            self.assertNotIn("strategy Alpha", doc.page_content)

        # Query Meeting A for Meeting A secret token
        retriever_a = get_retriever(vs_a, k=4)
        results_a = retriever_a.invoke("What is APPLE_SECRET_TOKEN_99?")
        found = any("APPLE_SECRET_TOKEN_99" in doc.page_content for doc in results_a)
        self.assertTrue(found)

    def test_02_temp_file_cleanup_on_success_and_failure_regression(self):
        """MANDATORY REGRESSION: Verify temporary files are deleted under normal and failure conditions."""
        wav_normal = os.path.join(DOWNLOADS_DIR, f"temp_norm_{uuid.uuid4().hex[:8]}.wav")
        wav_fail = os.path.join(DOWNLOADS_DIR, f"temp_fail_{uuid.uuid4().hex[:8]}.wav")

        AudioSegment.silent(duration=1000).export(wav_normal, format="wav")
        AudioSegment.silent(duration=1000).export(wav_fail, format="wav")

        # Normal cleanup
        cleanup_audio_files(main_wav_path=wav_normal)
        self.assertFalse(os.path.exists(wav_normal))

        # Failure cleanup
        with self.assertRaises(RuntimeError):
            try:
                raise RuntimeError("Simulated Pipeline Failure")
            finally:
                cleanup_audio_files(main_wav_path=wav_fail)
        self.assertFalse(os.path.exists(wav_fail))


if __name__ == "__main__":
    unittest.main()
