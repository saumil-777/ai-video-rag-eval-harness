import unittest
from core.config import (
    validate_source_input,
    validate_api_keys,
    SUMMARIZATION_CHUNK_SIZE,
    RAG_CHUNK_SIZE,
    RETRIEVER_K,
)


class TestConfig(unittest.TestCase):

    def test_01_central_constants(self):
        """Verify central configuration constants are initialized properly."""
        self.assertEqual(SUMMARIZATION_CHUNK_SIZE, 3000)
        self.assertEqual(RAG_CHUNK_SIZE, 500)
        self.assertEqual(RETRIEVER_K, 4)

    def test_02_validate_source_input_youtube(self):
        """Verify valid and invalid YouTube URLs."""
        is_valid, _ = validate_source_input("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(is_valid)

        is_valid_short, _ = validate_source_input("https://youtu.be/dQw4w9WgXcQ")
        self.assertTrue(is_valid_short)

        is_invalid, err = validate_source_input("https://notyoutube.com/video")
        self.assertFalse(is_invalid)
        self.assertIn("Invalid URL format", err)

    def test_03_validate_source_input_local_file(self):
        """Verify local file validation rules."""
        is_invalid_empty, err_empty = validate_source_input("")
        self.assertFalse(is_invalid_empty)
        self.assertIn("cannot be empty", err_empty)

        is_invalid_nonexistent, err_nonexist = validate_source_input("C:/nonexistent_file_path_123.mp4")
        self.assertFalse(is_invalid_nonexistent)
        self.assertIn("not found", err_nonexist)

    def test_04_path_traversal_prevention(self):
        """Verify path traversal attempts are rejected."""
        is_invalid_traversal, err = validate_source_input("../../etc/passwd")
        self.assertFalse(is_invalid_traversal)
        self.assertIn("path traversal", err)


if __name__ == "__main__":
    unittest.main()
