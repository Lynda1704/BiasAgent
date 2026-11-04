import json
from typing import Dict, Any


def load_bias_types(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "bias_types" not in data:
        raise ValueError("bias_types.json must contain a 'bias_types' field.")

    for item in data["bias_types"]:
        if "name" not in item:
            raise ValueError("Each bias type must contain a 'name' field.")

        if "identity_terms" not in item:
            raise ValueError(f"Bias type '{item['name']}' must contain identity_terms.")

        if not isinstance(item["identity_terms"], list):
            raise ValueError(f"identity_terms for '{item['name']}' must be a list.")

    return data
