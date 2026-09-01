"""
tests/test_retry.py
===================
Tests for the centralized Mistral retry utility (core/retry.py) and fault-tolerant extraction.

Covers:
  1. is_rate_limit_error — 429 and 503/502/504 detection accuracy
  2. call_with_retry — success on first attempt
  3. call_with_retry — success after 429 or 503 retry
  4. call_with_retry — exhaustion raises MistralRateLimitError
  5. Non-transient error not retried — re-raised immediately
  6. MistralRateLimitError message contains no API URL or key
  7. Fallback helpers produce clearly-labelled output
  8. Extractor fault tolerance — 503/429 during extraction returns fallback without aborting pipeline
"""

import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# 1: Detection tests
# ---------------------------------------------------------------------------

class TestIsRateLimitError(unittest.TestCase):

    def _exc(self, msg: str, cls=Exception) -> Exception:
        return cls(msg)

    def test_detects_429_in_message(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("HTTP status 429 Too Many Requests")))

    def test_detects_503_in_message(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("HTTP status 503 Service temporarily unavailable due to high load, please retry.")))

    def test_detects_502_504_in_message(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("HTTP 502 Bad Gateway")))
        self.assertTrue(is_rate_limit_error(self._exc("HTTP 504 Gateway Timeout")))

    def test_detects_rate_limit_keyword(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("rate_limit exceeded")))

    def test_detects_service_unavailable_keyword(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("Service unavailable due to high load")))

    def test_detects_rate_limit_mixed_case(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc("Rate Limit error code 1300")))

    def test_detects_mistral_error_code_1300(self):
        from core.retry import is_rate_limit_error
        self.assertTrue(is_rate_limit_error(self._exc('{"message":"Rate limit","code":1300}')))

    def test_detects_via_exception_chain(self):
        from core.retry import is_rate_limit_error
        cause = Exception("HTTP 503 Service temporarily unavailable")
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

    def test_retry_on_429_then_success(self):
        from core.retry import call_with_retry

        rate_limit_exc = Exception("HTTP 429 rate_limit")
        fn = MagicMock(side_effect=[rate_limit_exc, "success"])

        result = call_with_retry(fn, max_retries=3, base_delay=1.0)

        self.assertEqual(result, "success")
        self.assertEqual(fn.call_count, 2)
        self.mock_sleep.assert_called_once_with(1.0)

    def test_retry_on_503_then_success(self):
        from core.retry import call_with_retry

        service_503_exc = Exception("HTTP 503 Service temporarily unavailable due to high load, please retry.")
        fn = MagicMock(side_effect=[service_503_exc, "success_503"])

        result = call_with_retry(fn, max_retries=3, base_delay=1.0)

        self.assertEqual(result, "success_503")
        self.assertEqual(fn.call_count, 2)
        self.mock_sleep.assert_called_once_with(1.0)

    def test_retry_exhaustion_raises_mistral_rate_limit_error(self):
        from core.retry import call_with_retry, MistralRateLimitError

        service_503_exc = Exception("HTTP 503 Service temporarily unavailable")
        fn = MagicMock(side_effect=service_503_exc)

        with self.assertRaises(MistralRateLimitError):
            call_with_retry(fn, max_retries=3, base_delay=1.0)

        self.assertEqual(fn.call_count, 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_non_rate_limit_error_not_retried(self):
        from core.retry import call_with_retry

        fn = MagicMock(side_effect=ValueError("Bad input"))

        with self.assertRaises(ValueError):
            call_with_retry(fn, max_retries=3, base_delay=1.0)

        self.assertEqual(fn.call_count, 1)
        self.mock_sleep.assert_not_called()

    def test_exponential_backoff_delays(self):
        from core.retry import call_with_retry, MistralRateLimitError

        exc = Exception("HTTP 503 Service unavailable")
        fn = MagicMock(side_effect=exc)

        with self.assertRaises(MistralRateLimitError):
            call_with_retry(fn, max_retries=3, base_delay=2.0)

        sleep_calls = self.mock_sleep.call_args_list
        self.assertEqual(len(sleep_calls), 2)
        self.assertAlmostEqual(sleep_calls[0][0][0], 2.0)
        self.assertAlmostEqual(sleep_calls[1][0][0], 4.0)


# ---------------------------------------------------------------------------
# Message safety
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
        self.assertGreater(len(msg), 10)
        self.assertTrue(
            "rate limit" in msg.lower() or "busy" in msg.lower() or "try again" in msg.lower(),
            f"Message not user-friendly: {msg}"
        )


# ---------------------------------------------------------------------------
# Fallback and Extractor Fault Tolerance
# ---------------------------------------------------------------------------

class TestExtractorFaultTolerance(unittest.TestCase):

    @patch("core.extractor.call_with_retry")
    def test_extract_action_items_returns_fallback_on_503(self, mock_retry):
        from core.extractor import extract_action_items
        mock_retry.side_effect = Exception("HTTP 503 Service temporarily unavailable")

        result = extract_action_items("Sample transcript")
        self.assertIn("Demo", result)
        self.assertIn("Fallback", result)
        self.assertIn("Action item", result)

    @patch("core.extractor.call_with_retry")
    def test_extract_key_decisions_returns_fallback_on_503(self, mock_retry):
        from core.extractor import extract_key_decisions
        mock_retry.side_effect = Exception("HTTP 503 Service temporarily unavailable")

        result = extract_key_decisions("Sample transcript")
        self.assertIn("Demo", result)
        self.assertIn("Fallback", result)
        self.assertIn("Key decision", result)

    @patch("core.extractor.call_with_retry")
    def test_extract_questions_returns_fallback_on_503(self, mock_retry):
        from core.extractor import extract_questions
        mock_retry.side_effect = Exception("HTTP 503 Service temporarily unavailable")

        result = extract_questions("Sample transcript")
        self.assertIn("Demo", result)
        self.assertIn("Fallback", result)
        self.assertIn("Open question", result)


if __name__ == "__main__":
    unittest.main()
