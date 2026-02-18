"""Use an LLM to generate concise, intuitive labels for MoE experts."""

import json
import logging
import os
import time
import httpx
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

LITELLM_URL = os.getenv("LITELLM_URL", "https://litellm.mlinf.lab.ppops.net")
LITELLM_KEY = os.getenv("LITELLM_KEY", "")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "anthropic.claude-opus-4-6")

SYSTEM_PROMPT = """\
You are an expert in machine learning, specifically Mixture-of-Experts (MoE) transformer models. \
Your job is to analyze routing statistics from a character-level MoE model and produce clear, \
intuitive labels that help non-technical people understand what each expert specializes in.

<context>
In this model, a "router" assigns each character in a sentence to one or more "experts." \
Over many sentences, each expert develops preferences — some handle spaces between words, \
some handle vowels inside words, some handle the first letter of words, etc.

You will receive statistical data showing what characters and patterns each expert was routed. \
Your job is to name each expert with TWO labels:
1. A "title" — a short character-level label describing what patterns/characters it handles
2. A "domain" — a broader functional role describing what PURPOSE this expert serves in language processing
</context>

<rules>
- "title": 1-4 words. Creative, memorable, plain English. Describes the character-level pattern.
- "domain": 1-3 words. The broader linguistic function. Think: what role does this expert play in understanding language?
- "description": 1 short sentence a 10-year-old could understand, combining both the character pattern and the broader purpose.
- Look at ALL the data holistically — character frequencies, types, word positions, example words.
- Each expert in the same layer MUST get DISTINCT titles AND distinct domains. Never repeat.
- Return ONLY a valid JSON array. No markdown fences, no explanation outside the JSON.
</rules>

<examples>
<example>
<input>
Expert 0 (450 activations):
  Top characters: ' ' 42.1%, 't' 8.2%, 'a' 6.1%
  Character types: space: 42.1%, consonant: 31.2%, vowel: 18.5%
  Word positions: first: 28.1%, last: 22.3%
  Example words (CAPS = handled): The LITTLE cat JUMPED
</input>
<ideal_output>
{{"expert_id": 0, "title": "Word Glue", "domain": "Separator", "description": "This expert handles the spaces and boundaries between words, acting as the glue that keeps words apart."}}
</ideal_output>
</example>

<example>
<input>
Expert 1 (380 activations):
  Top characters: 'e' 18.3%, 'a' 14.1%, 'i' 12.5%, 'o' 9.8%
  Character types: vowel: 54.7%, consonant: 28.1%
  Word positions: middle: 48.2%, last: 24.1%
  Example words (CAPS = handled): thE lIttlE cAt jUmpEd
</input>
<ideal_output>
{{"expert_id": 1, "title": "Vowel Heart", "domain": "Sound Core", "description": "This expert focuses on the vowels buried inside words — the sounds that give words their melody."}}
</ideal_output>
</example>

<example type="bad">
<input>Expert 2 with consonant/vowel mix</input>
<bad_output>
{{"expert_id": 2, "title": "The Inner Consonant Expert", "domain": "consonant-heavy", "description": "Handles consonants in the middle of words"}}
</bad_output>
<why_bad>Title is generic and uses "Expert" redundantly. Domain just restates character type. Description is dry and technical. Should be more creative and insightful.</why_bad>
</example>
</examples>"""

USER_PROMPT_TEMPLATE = """\
<task>
Analyze {n_experts} experts from Layer {layer_id} of a character-level MoE model.
Generate a title, domain, and description for each expert.
</task>

<expert_data>
{expert_data}
</expert_data>

<output_format>
Return a JSON array with exactly {n_experts} objects, one per expert, in order:
[
  {{"expert_id": 0, "title": "...", "domain": "...", "description": "..."}},
  {{"expert_id": 1, "title": "...", "domain": "...", "description": "..."}}
]

Remember: each title and domain must be DISTINCT across all experts in this layer. \
Focus on what makes each expert UNIQUE compared to the others.
</output_format>"""


def _format_expert_stats(expert: dict) -> str:
    """Format one expert's stats into a readable block for the LLM."""
    lines = [f"Expert {expert['expert_id']} ({expert['total_activations']} activations):"]

    # Top characters
    chars = ", ".join(
        f"'{c['char']}' {c['pct']}%" for c in expert.get("top_chars", [])
    )
    if chars:
        lines.append(f"  Top characters: {chars}")

    # Character type breakdown
    ct = expert.get("char_type_pcts", {})
    if ct:
        parts = [f"{k}: {v:.1f}%" for k, v in sorted(ct.items(), key=lambda x: -x[1]) if v > 2]
        lines.append(f"  Character types: {', '.join(parts)}")

    # Word position breakdown
    wp = expert.get("word_pos_pcts", {})
    if wp:
        parts = [f"{k}: {v:.1f}%" for k, v in sorted(wp.items(), key=lambda x: -x[1]) if v > 2]
        lines.append(f"  Word positions: {', '.join(parts)}")

    # Example words handled
    words = expert.get("example_words", [])
    if words:
        word_strs = []
        for w in words[:5]:
            highlighted = ""
            for i, ch in enumerate(w["word"]):
                if i < len(w["highlights"]) and w["highlights"][i]:
                    highlighted += ch.upper()
                else:
                    highlighted += ch.lower()
            word_strs.append(highlighted)
        lines.append(f"  Example words (CAPS = handled by this expert): {', '.join(word_strs)}")

    return "\n".join(lines)


def label_experts_with_llm(layer_experts: list[dict], layer_id: int) -> list[dict]:
    """Call the LLM to generate labels for all experts in a layer.

    Returns list of {"expert_id": int, "title": str, "description": str}.
    Raises on failure so caller can fall back to programmatic labels.
    """
    if not LITELLM_KEY:
        raise ValueError("No LITELLM_KEY configured")

    expert_blocks = "\n\n".join(_format_expert_stats(e) for e in layer_experts)

    user_msg = USER_PROMPT_TEMPLATE.format(
        n_experts=len(layer_experts),
        layer_id=layer_id,
        expert_data=expert_blocks,
    )

    max_retries = 3
    base_delay = 2.0
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            response = httpx.post(
                f"{LITELLM_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LITELLM_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LITELLM_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            break  # success
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as exc:
            last_exc = exc
            is_server_error = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500
            is_transient = isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout))
            if (is_server_error or is_transient) and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logging.warning(
                    "LLM request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                raise
    else:
        raise last_exc  # type: ignore[misc]

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Parse JSON from response — handle possible markdown wrapping
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    labels = json.loads(text)

    if len(labels) != len(layer_experts):
        raise ValueError(f"LLM returned {len(labels)} labels, expected {len(layer_experts)}")

    return labels
