import os
import json
from openai import OpenAI


def build_proxy_generation_prompt(
    prompt_template: str,
    sentence: str,
    task: str,
    label: str,
    bias_type: str,
    id_term: str,
    concept_term: str,
    proxy_type: str,
    phi: float,
    theta: float,
    prior: float,
    k: int,
) -> str:
    return (
        prompt_template
        .replace("{sentence}", sentence)
        .replace("{task}", task)
        .replace("{label}", label)
        .replace("{bias_type}", bias_type)
        .replace("{id_term}", id_term)
        .replace("{concept_term}", concept_term)
        .replace("{proxy_type}", proxy_type)
        .replace("{phi}", str(phi))
        .replace("{theta}", str(theta))
        .replace("{prior}", str(prior))
        .replace("{k}", str(k))
    )


def call_llm_for_proxy_generation(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


def parse_proxy_generation_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = (
            text.replace("```json", "")
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

    data = json.loads(text)

    if "proxy_sentence" not in data:
        raise ValueError(f"Missing proxy_sentence key in output: {text}")

    return str(data["proxy_sentence"])
def add_proxy_rejection_feedback_to_prompt(
    prompt: str,
    rejected_proxy_sentence: str,
    filter_reason: str,
    proxy_type: str,
    phi: float,
    theta: float,
    k: int,
) -> str:
    return f"""{prompt}

Additional regeneration constraint:

The previous generated proxy sentence was rejected.

Rejected proxy sentence:
{rejected_proxy_sentence}

Reason for rejection:
{filter_reason}

Required proxy type:
{proxy_type}

Required proxy parameters:
- phi = {phi}
- theta = {theta}
- k = {k}

Generate a new proxy sentence that follows the original proxy-generation instructions and avoids the rejection issue.

Important:
- Preserve the same task label, bias type, id term, and concept term.
- Preserve the same proxy type and proxy strength.
- Do not repeat the rejected proxy sentence.
- Return only the required JSON format.
"""
