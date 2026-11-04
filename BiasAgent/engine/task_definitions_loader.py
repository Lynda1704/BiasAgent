import json
from typing import Dict, Any


def load_task_definitions(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("task_definitions.json must contain a JSON object.")

    for task_name, task_info in data.items():
        if "labels" not in task_info:
            raise ValueError(f"Task '{task_name}' is missing 'labels'.")

        if not isinstance(task_info["labels"], list):
            raise ValueError(f"Task '{task_name}' labels must be a list.")

    return data


def get_task_labels(task_definitions: Dict[str, Any], task_name: str) -> list[str]:
    if task_name not in task_definitions:
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {list(task_definitions.keys())}"
        )

    return task_definitions[task_name]["labels"]
