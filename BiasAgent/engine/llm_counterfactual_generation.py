import os
import ast
from typing import List
from openai import OpenAI


def build_counterfactual_prompt(
    prompt_template: str,
    source_id_term: str,
    target_id_term: str,
    sentences: list[str],
) -> str:
    prompt = prompt_template
    prompt = prompt.replace("{source_id_term}", source_id_term)
    prompt = prompt.replace("{target_id_term}", target_id_term)
    prompt = prompt.replace("{sentences}", repr(sentences))
    return prompt


def call_llm_for_counterfactuals(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


def parse_counterfactual_list(text: str) -> List[str]:
    text = text.strip()

    if text.startswith("```"):
        text = (
            text.replace("```python", "")
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:
        data = ast.literal_eval(text)

        if not isinstance(data, list):
            raise ValueError("Counterfactual output must be a Python list.")

        return [str(x) for x in data]

    except Exception:
        cleaned = text

        cleaned = cleaned.strip("[")
        cleaned = cleaned.strip("]")
        cleaned = cleaned.strip()

        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]

        return [cleaned]
def add_baseline_rejection_feedback_to_prompt(
    prompt: str,
    rejected_group_text: str,
    filter_reason: str,
) -> str:
    return f"""{prompt}

Additional regeneration constraint:

The previous counterfactual group was rejected.

Rejected counterfactual group:
{rejected_group_text}

Reason for rejection:
{filter_reason}

Regenerate a new valid counterfactual group that follows the original baseline/counterfactual generation instructions.

Important:
- Preserve the same task, label, bias type, concept term, and demographic terms.
- Keep the generated sentences comparable across demographic variants.
- Avoid the rejection issue.
- Do not repeat the rejected sentences.
- Return only the required JSON format.
"""
