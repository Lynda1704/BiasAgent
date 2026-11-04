import os
import json
from openai import OpenAI


def build_proxy_judge_prompt(
    prompt_template: str,
    baseline_sentence: str,
    proxy_sentence: str,
    task: str,
    label: str,
    bias_type: str,
    id_term: str,
    concept_term: str,
    proxy_type: str,
    phi: float,
    theta: float,
    k: int,
) -> str:
    return (
        prompt_template
        .replace("{baseline_sentence}", baseline_sentence)
        .replace("{proxy_sentence}", proxy_sentence)
        .replace("{task}", task)
        .replace("{label}", label)
        .replace("{bias_type}", bias_type)
        .replace("{id_term}", id_term)
        .replace("{concept_term}", concept_term)
        .replace("{proxy_type}", proxy_type)
        .replace("{phi}", str(phi))
        .replace("{theta}", str(theta))
        .replace("{k}", str(k))
    )


def call_llm_proxy_judge(
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


def parse_proxy_judge_json(text: str) -> dict:
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
