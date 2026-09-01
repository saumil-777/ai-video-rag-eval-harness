"""
tests/test_retry.py
===================
Tests for the centralized Mistral retry utility (core/retry.py).

Covers:
  1. is_rate_limit_error — detection accuracy
  2. call_with_retry — success on first attempt
  3. call_with_retry — success after one 429 then success
  4. call_with_retry — exhaustion raises MistralRateLimitError
  5. Non-rate-limit error not retried — re-raised immediately
  6. MistralRateLimitError message contains no API URL or key
  7. Fallback helpers produce clearly-labelled output
  8. Summarizer rate-limit fallback (_extractive_summary_fallback)
  9. Title fallback (_fallback_title)
 10. Extractor fallback (_fallback)
"""

import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# 1–5: core.retry unit tests
# ---------------------------------------------------------------------------

class TestIsRateLimitError(unittest.TestCase):

    def _exc(self, msg: str, cls=Exception) -> Exception:
        return cls(msg)

    def test_detects_429_in_message(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("HTTP status 429 Too Many Requests")))

    def test_detects_rate_limit_keyword(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("rate_limit exceeded")))

    def test_detects_rate_limit_mixed_case(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("Rate Limit error code 1300")))

    def test_detects_mistral_error_code_1300(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc('{"message":"Rate limit","code":1300}')))

    def test_detects_via_exception_chain(self):
        from core.retry import is_rate_limit_error
        cause = Exception("429 from upstream")
        outer = RuntimeError("Wrapped error")
        outer.__cause__ = cause
        self.assertTrue(is_rate_limit_error(outer))

    def test_not_rate_limit_for_generic_error(self):
        from core.retry import is_rate_limit_error
        self.assertFalse(is_rate_limit_error(ValueError("Invalid input")))

    def test_not_rate_limit_for_connection_error(self):
        from core.retry import is_rate_limit_error
        self.assertFalse(is_rate_limit_error(ConnectionError("Connection reset by peer")))


class TestCallWithRetry(unittest.TestCase):

    def setUp(self):
        # Patch time.sleep to avoid actual delays in tests
        patcher = patch("core.retry.time.sleep")
        self.mock_sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_on_first_attempt(self):
        from core.retry import call_with_retry
        fn = MagicMock(return_value="ok")
        result = call_with_retry(fn, "arg1", kwarg="val")
        self.assertEqual(result, "ok")
        fn.assert_called_once_with("arg1", kwarg="val")
        self.mock_sleep.assert_not_called()

    def test_retry_on_rate_limit_then_success(self):
        from core.retry import call_with_retry, MistralRateLimitError

        rate_limit_exc = Exception("HTTP 429 rate_limit")
        fn = MagicMock(side_effect=[rate_limit_exc, "success"])

        result = call_with_retry(fn, max_retries=3, base_delay=1.0)

        self.assertEqual(result, "success")
        self.assertEqual(fn.call_count, 2)
        # Should have slept once (after first failure)
        self.mock_sleep.assert_called_once_with(1.0)  # base_delay * 2^0

    def test_retry_exhaustion_raises_mistral_rate_limit_error(self):
        from core.retry import call_with_retry, MistralRateLimitError

        rate_limit_exc = Exception("429 rate_limit exceeded")
        fn = MagicMock(side_effect=rate_limit_exc)

        with self.assertRaises(MistralRateLimitError):
            call_with_retry(fn, max_retries=3, base_delay=1.0)

        self.assertEqual(fn.call_count, 3)
        # Should sleep twice (after attempt 1 and 2; not after final attempt)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_non_rate_limit_error_not_retried(self):
        from core.retry import call_with_retry

        fn = MagicMock(side_effect=ValueError("Bad input"))

        with self.assertRaises(ValueError):
            call_with_retry(fn, max_retries=3, base_delay=1.0)

        # Must NOT retry — only called once
        self.assertEqual(fn.call_count, 1)
        self.mock_sleep.assert_not_called()

    def test_exponential_backoff_delays(self):
        from core.retry import call_with_retry, MistralRateLimitError

        rate_limit_exc = Exception("429")
        fn = MagicMock(side_effect=rate_limit_exc)

        with self.assertRaises(MistralRateLimitError):
            call_with_retry(fn, max_retries=3, base_delay=2.0)

        # delays should be 2.0 and 4.0 (not after final attempt)
        sleep_calls = self.mock_sleep.call_args_list
        self.assertEqual(len(sleep_calls), 2)
        self.assertAlmostEqual(sleep_calls[0][0][0], 2.0)
        self.assertAlmostEqual(sleep_calls[1][0][0], 4.0)


# ---------------------------------------------------------------------------
# 6: MistralRateLimitError message safety
# ---------------------------------------------------------------------------

class TestMistralRateLimitErrorMessage(unittest.TestCase):

    def test_user_message_contains_no_api_url(self):
        from core.retry import MistralRateLimitError
        exc = MistralRateLimitError(attempts=3)
        msg = str(exc)
        self.assertNotIn("https://", msg)
        self.assertNotIn("mistral.ai", msg)
        self.assertNotIn("api_key", msg.lower())
        self.assertNotIn("authorization", msg.lower())

    def test_user_message_is_friendly(self):
        from core.retry import MistralRateLimitError
        exc = MistralRateLimitError(attempts=3)
        msg = str(exc)
        # Should be human-readable
        self.assertGreater(len(msg), 10)
        # Should mention retry / rate limit in user-friendly terms
        self.assertTrue(
            "rate limit" in msg.lower() or "busy" in msg.lower() or "try again" in msg.lower(),
            f"Message not user-friendly: {msg}"
        )

    def test_attempts_attribute(self):
        from core.retry import MistralRateLimitError
        exc = MistralRateLimitError(attempts=5)
        self.assertEqual(exc.attempts, 5)


# ---------------------------------------------------------------------------
# 7–10: Fallback helpers
# ---------------------------------------------------------------------------

class TestFallbackHelpers(unittest.TestCase):

    def test_extractive_summary_fallback_labels_as_demo(self):
        from core.summarizer import _extractive_summary_fallback
        transcript = "This is a test meeting. We discussed budgets. No decisions yet."
        result = _extractive_summary_fallback(transcript)
        self.assertIn("Demo", result)
        self.assertIn("Fallback", result)

    def test_extractive_summary_fallback_non_empty(self):
        from core.summarizer import _extractive_summary_fallback
        transcript = "Meeting about AI projects. Speaker mentioned Python and LangChain."
        result = _extractive_summary_fallback(transcript)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)

    def test_fallback_title_returns_words_from_transcript(self):
        from core.summarizer import _fallback_title
        transcript = "Welcome to the weekly team sync meeting"
        result = _fallback_title(transcript)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should not be blank
        self.assertTrue(result.strip())

    def test_fallback_title_demo_label(self):
        from core.summarizer import _fallback_title
        transcript = "Kickoff meeting for the AI project quarterly review"
        result = _fallback_title(transcript)
        # Fallback title must be clearly labeled
        self.assertIn("Demo", result)

    def test_extractor_fallback_labels_as_demo(self):
        from core.extractor import _fallback
        result = _fallback("Action item")
        self.assertIn("Demo", result)
        self.assertIn("Fallback", result)
        self.assertIn("Action item", result)

    def test_extractor_fallback_no_api_details(self):
        from core.extractor import _fallback
        result = _fallback("Key decision")
        self.assertNotIn("https://", result)
        self.assertNotIn("api_key", result.lower())
        self.assertNotIn("mistral.ai", result)

    def test_extractor_fallback_non_empty(self):
        from core.extractor import _fallback
        result = _fallback("Open question")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)


if __name__ == "__main__":
    unittest.main()
