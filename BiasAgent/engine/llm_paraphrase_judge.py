import os
import json
from openai import OpenAI


def build_paraphrase_judge_prompt(
    prompt_template: str,
    original_sentence: str,
    paraphrase_sentence: str,
    task: str,
    label: str,
    bias_type: str,
    id_term: str,
    concept_term: str,
    paraphrase_type: str,
    syntax_type: str,
    lexical_type: str,
) -> str:
    return (
        prompt_template
        .replace("{original_sentence}", original_sentence)
        .replace("{paraphrase_sentence}", paraphrase_sentence)
        .replace("{task}", task)
        .replace("{label}", label)
        .replace("{bias_type}", bias_type)
        .replace("{id_term}", id_term)
        .replace("{concept_term}", concept_term)
        .replace("{paraphrase_type}", paraphrase_type)
        .replace("{syntax_type}", syntax_type)
        .replace("{lexical_type}", lexical_type)
    )


def call_llm_paraphrase_judge(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


def parse_paraphrase_judge_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = (
            text.replace("```json", "")
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

    data = json.loads(text)

    return {
        "valid": bool(data["valid"]),
        "flagged": int(data["flagged"]),
        "reason": str(data["reason"]),
    }
