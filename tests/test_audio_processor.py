import os
import unittest
import uuid
from pydub import AudioSegment
from unittest.mock import patch, MagicMock
from utils.audio_processor import chunk_audio, cleanup_audio_files, download_youtube_audio, DOWNLOADS_DIR


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

    @patch("yt_dlp.YoutubeDL")
    def test_03_youtube_download_configuration(self, mock_ytdl_cls):
        """Verify download_youtube_audio configures yt-dlp with cloud-compatible options."""
        mock_ydl = MagicMock()
        mock_ytdl_cls.return_value.__enter__.return_value = mock_ydl

        fake_base = os.path.join(DOWNLOADS_DIR, "mock_vid")
        fake_wav = fake_base + ".wav"
        mock_ydl.prepare_filename.return_value = fake_base + ".mp4"
        mock_ydl.extract_info.return_value = {"id": "mock_vid", "title": "Mock Video"}

        # Create dummy wav file so prepare_filename check succeeds
        with open(fake_wav, "wb") as f:
            f.write(b"RIFF dummy wav data")

        try:
            result = download_youtube_audio("https://www.youtube.com/watch?v=mock_vid", max_retries=1)
            self.assertEqual(result, fake_wav)

            # Check that YoutubeDL was initialized with ydl_opts containing cloud-resilience options
            mock_ytdl_cls.assert_called()
            ydl_opts = mock_ytdl_cls.call_args[0][0]

            self.assertTrue(ydl_opts.get("nopart"))
            self.assertTrue(ydl_opts.get("overwrites"))
            self.assertIn("extractor_args", ydl_opts)
            self.assertIn("youtube", ydl_opts["extractor_args"])
            self.assertIn("player_client", ydl_opts["extractor_args"]["youtube"])
            self.assertIn("mweb", ydl_opts["extractor_args"]["youtube"]["player_client"])
            self.assertIn("http_headers", ydl_opts)
            self.assertIn("User-Agent", ydl_opts["http_headers"])
        finally:
            if os.path.exists(fake_wav):
                try:
                    os.remove(fake_wav)
                except Exception:
                    pass

    @patch("yt_dlp.YoutubeDL")
    def test_04_youtube_download_failure_cleanup(self, mock_ytdl_cls):
        """Verify YouTube download failure raises RuntimeError after retries without leaking files."""
        mock_ydl = MagicMock()
        mock_ytdl_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = Exception("HTTP Error 403: Forbidden")

        with self.assertRaises(RuntimeError) as ctx:
            download_youtube_audio("https://www.youtube.com/watch?v=mock_fail", max_retries=2)

        self.assertIn("Failed to download YouTube audio after 2 attempt(s)", str(ctx.exception))
        self.assertEqual(mock_ydl.extract_info.call_count, 2)


if __name__ == "__main__":
    unittest.main()

