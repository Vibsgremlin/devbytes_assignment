"""
Source-grounded article rectification.

The goal is to preserve the AI article's phrasing and structure while
correcting only the facts that contradict the source article(s).
"""

from __future__ import annotations

import os
import re
import site
import time
from difflib import SequenceMatcher
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if VENDOR_DIR.exists():
    site.addsitedir(str(VENDOR_DIR))

from dotenv.main import load_dotenv
from litellm import completion

load_dotenv()

SYSTEM_PROMPT = """You are a meticulous news copy editor.

Your task is to rectify an AI-generated news article using the supplied source
evidence. Follow these rules strictly:
1. Preserve the AI article's structure, headings, tone, paragraph order, and
wording whenever they are already correct.
2. Change only spans that are factually unsupported or contradicted by the
source evidence.
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

WORD_RE = re.compile(r"[a-z0-9]+")


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


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def _split_source_chunks(source_article: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", source_article) if block.strip()]
    merged: list[str] = []
    current = ""

    for block in blocks:
        if block.startswith("#") and current:
            merged.append(current.strip())
            current = block
            continue

        if not current:
            current = block
        elif len(current) + len(block) < 900:
            current = f"{current}\n\n{block}"
        else:
            merged.append(current.strip())
            current = block

    if current:
        merged.append(current.strip())

    return merged


def _select_relevant_source_context(source_article: str, ai_generated_content: str) -> str:
    if len(source_article) <= 3500:
        return source_article.strip()

    source_chunks = _split_source_chunks(source_article)
    ai_sections = [section.strip() for section in re.split(r"\n\s*\n", ai_generated_content) if section.strip()]
    ai_tokens = _tokenize(ai_generated_content)

    scored_chunks: list[tuple[float, int, str]] = []
    for idx, chunk in enumerate(source_chunks):
        chunk_tokens = _tokenize(chunk)
        if not chunk_tokens:
            continue

        overlap = len(ai_tokens & chunk_tokens) / max(1, len(chunk_tokens))
        best_section_match = 0.0
        for section in ai_sections[:12]:
            section_tokens = _tokenize(section)
            if not section_tokens:
                continue
            section_overlap = len(section_tokens & chunk_tokens) / max(1, len(section_tokens))
            if section_overlap > best_section_match:
                best_section_match = section_overlap

        score = overlap + (best_section_match * 2)
        if chunk.startswith("#"):
            score += 0.05
        scored_chunks.append((score, idx, chunk))

    scored_chunks.sort(reverse=True)

    chosen = sorted(scored_chunks[:8], key=lambda item: item[1])
    context = "\n\n".join(chunk for _, _, chunk in chosen).strip()

    if len(context) < 1800:
        return source_article[:3500].strip()

    return context


def _build_user_prompt(source_article: str, ai_generated_content: str) -> str:
    source_context = _select_relevant_source_context(source_article, ai_generated_content)
    return (
        "Rectify the AI-generated article using the source evidence.\n\n"
        "SOURCE EVIDENCE:\n"
        f"{source_context}\n\n"
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
                timeout=90,
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
        return _safe_fallback(ai_generated_content)
