import os
import json
from typing import Dict, Any
from openai import OpenAI


def build_sample_generation_prompt(
    prompt_template: str,
    id_term: str,
    bias_type: str,
    concept_term: str,
    task_name: str,
    labels: list[str],
) -> str:
    prompt = prompt_template
    prompt = prompt.replace("{id_term}", id_term)
    prompt = prompt.replace("{bias_type}", bias_type)
    prompt = prompt.replace("{concept_term}", concept_term)
    prompt = prompt.replace("{task_name}", task_name)
    prompt = prompt.replace("{labels}", ", ".join(labels))
    return prompt


def call_llm_for_samples(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content.strip()


def parse_sample_generation_json(text: str) -> Dict[str, str]:

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("LLM sample output must be a JSON object.")

    parsed = {}

    for label, sentence in data.items():
        parsed[str(label)] = str(sentence)

    return parsed
