import os
import json
from openai import OpenAI


def build_paraphrase_prompt(
    prompt_template: str,
    sentence: str,
    id_term: str,
    concept_term: str,
    syntax_type: str,
    lexical_type: str,
    length_min_words: int,
    length_max_words: int,
) -> str:
    return (
        prompt_template
        .replace("{sentence}", sentence)
        .replace("{id_term}", id_term)
        .replace("{concept_term}", concept_term)
        .replace("{syntax_type}", syntax_type)
        .replace("{lexical_type}", lexical_type)
        .replace("{length_min_words}", str(length_min_words))
        .replace("{length_max_words}", str(length_max_words))
    )


def call_llm_for_paraphrase(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 1.0,
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()

def parse_paraphrase_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = (
            text.replace("```json", "")
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

    data = json.loads(text)

    if "paraphrase" not in data:
        raise ValueError(f"Missing paraphrase key in output: {text}")

    return str(data["paraphrase"]) 
def add_rejection_feedback_to_prompt(
    prompt: str,
    rejected_sentence: str,
    filter_reason: str,
    paraphrase_type: str,
    syntax_type: str,
    lexical_type: str,
) -> str:
    return f"""{prompt}

Additional regeneration constraint:

The previous generated paraphrase was rejected.

Rejected paraphrase:
{rejected_sentence}

Reason for rejection:
{filter_reason}

Required paraphrase type:
{paraphrase_type}

Required syntax type:
{syntax_type}

Required lexical type:
{lexical_type}

Generate a new paraphrase that follows the original instructions and avoids the rejection issue.

Important:
- Preserve the original meaning.
- Preserve the same task label, bias type, id term, and concept term.
- Preserve the required paraphrase type.
- Do not repeat the rejected paraphrase.
- Return only the required JSON format.
"""
