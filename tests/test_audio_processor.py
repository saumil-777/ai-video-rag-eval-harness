import os
import unittest
import uuid
from pydub import AudioSegment
from utils.audio_processor import chunk_audio, cleanup_audio_files, DOWNLOADS_DIR


class TestAudioProcessor(unittest.TestCase):

    def setUp(self):
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        self.test_wav = os.path.join(DOWNLOADS_DIR, f"unit_test_{uuid.uuid4().hex[:8]}.wav")

    def tearDown(self):
        if os.path.exists(self.test_wav):
            try:
                os.remove(self.test_wav)
            except Exception:
                pass

    def test_01_chunk_audio(self):
        """Verify audio chunking splits a 12-minute audio file into two 10-minute chunks."""
        # 12 minutes silent audio = 12 * 60 * 1000 ms
        duration_ms = 12 * 60 * 1000
        audio = AudioSegment.silent(duration=duration_ms)
        audio.export(self.test_wav, format="wav")

        chunks = chunk_audio(self.test_wav, chunk_minutes=10)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertTrue(os.path.exists(chunk))

        # Cleanup generated chunk files
        cleanup_audio_files(file_paths=chunks, main_wav_path=self.test_wav)
        for chunk in chunks:
            self.assertFalse(os.path.exists(chunk))

    def test_02_chunk_empty_audio_raises_error(self):
        """Verify passing 0-duration audio raises ValueError."""
        empty_audio = AudioSegment.silent(duration=0)
        empty_audio.export(self.test_wav, format="wav")

        with self.assertRaises(ValueError):
            chunk_audio(self.test_wav)


if __name__ == "__main__":
    unittest.main()
