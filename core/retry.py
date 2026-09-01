"""
core/retry.py
=============
Centralized retry utility for Mistral API calls.

Handles HTTP 429 / rate-limit responses with bounded exponential backoff.
Raises MistralRateLimitError (a clean, user-safe exception) when all
retries are exhausted so callers can display a friendly message without
exposing raw API URLs or error payloads.
"""

import time
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception – user-safe, no API details
# ---------------------------------------------------------------------------

class MistralRateLimitError(Exception):
    """
    Raised when the Mistral API is rate-limited and all retry attempts
    are exhausted.

    The message is intentionally vague – safe to display in the UI.
    It never contains raw API URLs, response bodies, or credentials.
    """

    USER_MESSAGE = (
        "⚡ The AI service is temporarily busy (rate limit reached). "
        "Please wait a moment and try again."
    )

    def __init__(self, attempts: int = 3):
        super().__init__(self.USER_MESSAGE)
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_rate_limit_error(exc: BaseException) -> bool:
    """
    Return True if *exc* (or any chained cause) looks like a Mistral
    HTTP 429 / rate-limit error.

    Checks the string representation of the exception chain rather than
    importing Mistral/httpx internals, which keeps this compatible with
    any langchain-mistralai version.
    """
    indicators = ("429", "rate_limit", "rate limit", "1300", "RateLimitError")

    # Walk the exception chain (cause + context)
    current: BaseException | None = exc
    while current is not None:
        text = f"{type(current).__name__} {current!s}"
        if any(ind.lower() in text.lower() for ind in indicators):
            return True
        current = current.__cause__ or current.__context__

    return False


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 2.0, **kwargs):
    """
    Call *fn(*args, **kwargs)* with bounded exponential backoff on 429s.

    Parameters
    ----------
    fn          : callable to invoke
    *args       : positional arguments forwarded to fn
    max_retries : maximum number of attempts (default 3)
    base_delay  : initial sleep in seconds; doubles each attempt (2s, 4s, 8s)
    **kwargs    : keyword arguments forwarded to fn

    Returns
    -------
    The return value of fn on success.

    Raises
    ------
    MistralRateLimitError   if the rate limit persists after all retries.
    <original exception>    for any non-rate-limit error (raised immediately,
                            no retry wasted).
    """
    last_exc: BaseException | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)

        except Exception as exc:
            if is_rate_limit_error(exc):
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))  # 2, 4, 8 …
                    logger.warning(
                        "Mistral rate limit hit (attempt %d/%d). "
                        "Retrying in %.0fs…",
                        attempt, max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Mistral rate limit persists after %d attempts. "
                        "Giving up.",
                        max_retries,
                    )
            else:
                # Not a rate-limit error — re-raise immediately, no retry.
                raise

    # All retries exhausted on rate-limit errors
    raise MistralRateLimitError(attempts=max_retries) from last_exc
