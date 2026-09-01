import os
import unittest
import uuid
from pydub import AudioSegment
from unittest.mock import patch, MagicMock
from utils.audio_processor import (
    chunk_audio,
    cleanup_audio_files,
    download_youtube_audio,
    extract_youtube_video_id,
    fetch_youtube_transcript,
    process_input,
    DOWNLOADS_DIR,
)


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

    def test_05_extract_youtube_video_id(self):
        """Verify extract_youtube_video_id parses 11-char video IDs from various URL formats."""
        self.assertEqual(extract_youtube_video_id("https://www.youtube.com/watch?v=jNQXAC9IVRw"), "jNQXAC9IVRw")
        self.assertEqual(extract_youtube_video_id("https://youtu.be/jNQXAC9IVRw"), "jNQXAC9IVRw")
        self.assertEqual(extract_youtube_video_id("https://www.youtube.com/embed/jNQXAC9IVRw"), "jNQXAC9IVRw")
        self.assertEqual(extract_youtube_video_id("jNQXAC9IVRw"), "jNQXAC9IVRw")
        self.assertEqual(extract_youtube_video_id("invalid_string"), "")

    @patch("youtube_transcript_api.YouTubeTranscriptApi")
    def test_06_fetch_youtube_transcript_api(self, mock_ytt_cls):
        """Verify fetch_youtube_transcript returns transcript text using YouTubeTranscriptApi."""
        mock_ytt_inst = MagicMock()
        mock_ytt_cls.return_value = mock_ytt_inst

        mock_snippet_1 = MagicMock()
        mock_snippet_1.text = "Hello world"
        mock_snippet_2 = MagicMock()
        mock_snippet_2.text = "this is a test transcript"

        mock_ytt_inst.fetch.return_value = [mock_snippet_1, mock_snippet_2]

        transcript = fetch_youtube_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        self.assertIn("Hello world this is a test transcript", transcript)

    @patch("utils.audio_processor.fetch_youtube_transcript")
    @patch("utils.audio_processor.download_youtube_audio")
    def test_07_process_input_preferred_caption_path(self, mock_download, mock_fetch_transcript):
        """Verify process_input uses closed captions as the FIRST preferred path for YouTube URLs without attempting audio download."""
        mock_fetch_transcript.return_value = "Primary transcript content fetched via closed captions."

        chunks, main_wav_path, transcript_override = process_input("https://www.youtube.com/watch?v=mock_caption_video")

        self.assertEqual(chunks, [])
        self.assertIsNone(main_wav_path)
        self.assertEqual(transcript_override, "Primary transcript content fetched via closed captions.")
        mock_fetch_transcript.assert_called_once()
        mock_download.assert_not_called()

    @patch("utils.audio_processor.fetch_youtube_transcript")
    @patch("utils.audio_processor.download_youtube_audio")
    @patch("utils.audio_processor.chunk_audio")
    def test_08_process_input_fallback_to_audio_download(self, mock_chunk, mock_download, mock_fetch_transcript):
        """Verify process_input falls back to audio download when closed captions are unavailable."""
        mock_fetch_transcript.side_effect = RuntimeError("No closed captions found")
        mock_download.return_value = "/path/to/downloaded.wav"
        mock_chunk.return_value = ["/path/to/downloaded.wav_chunk_0.wav"]

        chunks, main_wav_path, transcript_override = process_input("https://www.youtube.com/watch?v=mock_no_caption_video")

        self.assertEqual(chunks, ["/path/to/downloaded.wav_chunk_0.wav"])
        self.assertEqual(main_wav_path, "/path/to/downloaded.wav")
        self.assertIsNone(transcript_override)
        mock_fetch_transcript.assert_called_once()
        mock_download.assert_called_once()

    @patch("utils.audio_processor.fetch_youtube_transcript")
    @patch("utils.audio_processor.download_youtube_audio")
    def test_09_process_input_fails_gracefully_when_both_fail(self, mock_download, mock_fetch_transcript):
        """Verify process_input raises a clean, user-facing RuntimeError when both caption retrieval and audio download fail (HTTP 403 scenario)."""
        mock_fetch_transcript.side_effect = RuntimeError("No closed captions found")
        mock_download.side_effect = RuntimeError("HTTP Error 403: Forbidden")

        with self.assertRaises(RuntimeError) as ctx:
            process_input("https://www.youtube.com/watch?v=mock_403_video")

        self.assertIn("YouTube did not allow this video to be accessed", str(ctx.exception))
        self.assertIn("upload the media file directly", str(ctx.exception))

    @patch("utils.audio_processor.convert_to_wav")
    @patch("utils.audio_processor.chunk_audio")
    def test_10_process_input_local_file(self, mock_chunk, mock_convert):
        """Verify process_input processes local uploaded media files directly without attempting YouTube API calls."""
        # Create a dummy local wav file
        dummy_local = os.path.join(DOWNLOADS_DIR, "dummy_local.mp4")
        with open(dummy_local, "wb") as f:
            f.write(b"dummy mp4 content")

        try:
            mock_convert.return_value = dummy_local + "_converted.wav"
            mock_chunk.return_value = [dummy_local + "_converted.wav_chunk_0.wav"]

            chunks, main_wav_path, transcript_override = process_input(dummy_local)

            self.assertEqual(main_wav_path, dummy_local + "_converted.wav")
            self.assertEqual(chunks, [dummy_local + "_converted.wav_chunk_0.wav"])
            self.assertIsNone(transcript_override)
            mock_convert.assert_called_once_with(dummy_local)
        finally:
            if os.path.exists(dummy_local):
                try:
                    os.remove(dummy_local)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
