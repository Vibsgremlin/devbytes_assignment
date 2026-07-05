"""
Source-grounded article rectification.

The goal is to preserve the AI article's phrasing and structure while
correcting only the facts that contradict the source article(s).
"""

from __future__ import annotations

import os
import re
import time
from difflib import SequenceMatcher

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

SYSTEM_PROMPT = """You are a meticulous news copy editor.

Your task is to rectify an AI-generated news article using the supplied source
article(s). Follow these rules strictly:
1. Preserve the AI article's structure, headings, tone, paragraph order, and
wording whenever they are already correct.
2. Change only spans that are factually unsupported or contradicted by the
source article(s).
3. Remove fabricated add-ons such as "Error Annotations", JSON blocks, notes,
or meta commentary if they appear in the AI article.
4. Do not add new sections, bullet lists, or explanations.
5. Do not mention the source article, your reasoning, or that edits were made.
6. Return only the final rectified article text.

Think like a surgeon, not an author: minimal edits, maximum factuality.
"""

STRICT_FALLBACK_PROMPT = """Revise the article again with an even stricter rule:
keep every original word unless a specific span is contradicted by the source.
Delete any fabricated annotation block. Return only the final article text."""

ANNOTATION_MARKERS = (
    "**Error Annotations:**",
    "Error Annotations:",
    "### Error Annotations",
    "## Error Annotations",
)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _cleanup_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    for marker in ANNOTATION_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
            break

    return text.strip()


def _headline_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith("#"))


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _build_user_prompt(source_article: str, ai_generated_content: str) -> str:
    return (
        "Rectify the AI-generated article using the source article(s).\n\n"
        "SOURCE ARTICLE(S):\n"
        f"{source_article.strip()}\n\n"
        "AI-GENERATED ARTICLE TO RECTIFY:\n"
        f"{ai_generated_content.strip()}\n"
    )


def _call_llm(user_prompt: str, extra_instruction: str | None = None) -> str:
    model = _require_env("LLM_MODEL_NAME")
    api_key = _require_env("LLM_API_KEY")
    api_base = _require_env("LLM_API_BASE")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if extra_instruction:
        messages.append({"role": "system", "content": extra_instruction})
    messages.append({"role": "user", "content": user_prompt})

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = completion(
                model=model,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            cleaned = _cleanup_text(content)
            if cleaned:
                return cleaned
            raise RuntimeError("Model returned empty content")
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"LLM rectification failed after retries: {last_error}")


def _needs_stricter_retry(ai_generated_content: str, rectified_content: str) -> bool:
    if not rectified_content:
        return True

    ai_headlines = _headline_count(ai_generated_content)
    rectified_headlines = _headline_count(rectified_content)
    if ai_headlines and rectified_headlines < max(1, ai_headlines - 2):
        return True

    if len(rectified_content) < max(200, int(len(ai_generated_content) * 0.35)):
        return True

    if _similarity(ai_generated_content, rectified_content) < 0.55:
        return True

    return False


def _safe_fallback(ai_generated_content: str) -> str:
    cleaned = _cleanup_text(ai_generated_content)
    return cleaned or ai_generated_content.strip()


def run(ai_generated_content: str, source_article: str) -> str:
    """
    Rectify an AI-generated article using the source article(s).

    Args:
        ai_generated_content: The AI-generated article text to be corrected.
        source_article: The ground-truth source article(s) for the same story.

    Returns:
        The rectified article text.
    """
    user_prompt = _build_user_prompt(source_article, ai_generated_content)

    try:
        rectified = _call_llm(user_prompt)
        if _needs_stricter_retry(ai_generated_content, rectified):
            rectified = _call_llm(user_prompt, extra_instruction=STRICT_FALLBACK_PROMPT)
        return _cleanup_text(rectified)
    except Exception:
        # Keep the batch moving even if the LLM call fails for a single article.
        return _safe_fallback(ai_generated_content)
