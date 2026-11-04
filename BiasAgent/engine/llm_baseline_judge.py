import os
import json
from typing import Dict, Any
from openai import OpenAI


def build_baseline_judge_prompt(
    prompt_template: str,
    refid: int,
    task_name: str,
    label: str,
    bias_type: str,
    rows: list[dict],
) -> str:
    prompt = prompt_template
    prompt = prompt.replace("{refid}", str(refid))
    prompt = prompt.replace("{task_name}", task_name)
    prompt = prompt.replace("{label}", label)
    prompt = prompt.replace("{bias_type}", bias_type)
    prompt = prompt.replace("{rows}", json.dumps(rows, ensure_ascii=False, indent=2))
    return prompt


def call_llm_baseline_judge(
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


def parse_baseline_judge_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    data = json.loads(text)

    if "valid" not in data:
        raise ValueError("Judge output missing 'valid'.")

    if "flagged" not in data:
        raise ValueError("Judge output missing 'flagged'.")

    if "reason" not in data:
        data["reason"] = ""

    data["valid"] = bool(data["valid"])
    data["flagged"] = int(data["flagged"])

    if data["valid"]:
        data["flagged"] = 0
    else:
        data["flagged"] = 1

    return data
