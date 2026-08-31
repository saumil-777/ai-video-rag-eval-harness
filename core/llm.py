import os
import logging
from langchain_mistralai import ChatMistralAI
from core.config import MISTRAL_MODEL, validate_api_keys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility Patch for langchain_mistralai ChatMistralAI._combine_llm_outputs
# ---------------------------------------------------------------------------
# In langchain_mistralai (v0.1.x/v0.2.x), ChatMistralAI._combine_llm_outputs
# attempts `overall_token_usage[k] += v` for all keys in token_usage.
# When Mistral API returns nested dicts (e.g. prompt_tokens_details) or
# non-numeric metadata, combining multi-generation outputs (n > 1, as used by
# Ragas AnswerRelevancy metric) fails with:
#   TypeError: unsupported operand type(s) for +=: 'dict' and 'dict'
#
# Patching _combine_llm_outputs safely aggregates numeric token metrics and
# merges sub-dictionaries without raising TypeError.

_original_combine_llm_outputs = getattr(ChatMistralAI, "_combine_llm_outputs", None)


def _safe_combine_llm_outputs(self, llm_outputs: list) -> dict:
    overall_token_usage: dict = {}
    for output in llm_outputs:
        if output is None or not isinstance(output, dict):
            continue
        token_usage = output.get("token_usage")
        if token_usage is not None and isinstance(token_usage, dict):
            for k, v in token_usage.items():
                if isinstance(v, (int, float)):
                    overall_token_usage[k] = overall_token_usage.get(k, 0) + v
                elif isinstance(v, dict):
                    if k not in overall_token_usage or not isinstance(overall_token_usage[k], dict):
                        overall_token_usage[k] = {}
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            overall_token_usage[k][sub_k] = overall_token_usage[k].get(sub_k, 0) + sub_v
                        else:
                            overall_token_usage[k][sub_k] = sub_v
                else:
                    overall_token_usage[k] = v
    model_name = getattr(self, "model", MISTRAL_MODEL)
    return {"token_usage": overall_token_usage, "model_name": model_name}


# Apply patch to ChatMistralAI class globally
ChatMistralAI._combine_llm_outputs = _safe_combine_llm_outputs


def get_llm(temperature: float = 0.3) -> ChatMistralAI:
    """
    Central factory for ChatMistralAI instances.
    Provides uniform API key validation, model selection, and temperature setting.
    """
    is_valid, err_msg = validate_api_keys(require_sarvam=False)
    if not is_valid:
        logger.error("API Key validation failed in get_llm()")
        raise ValueError(err_msg)

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    logger.debug(f"Instantiating ChatMistralAI (model={MISTRAL_MODEL}, temp={temperature})")
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=api_key,
        temperature=temperature,
        max_retries=5,
        timeout=60,
    )
