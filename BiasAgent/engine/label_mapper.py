from typing import Dict, List, Any


def get_dataset_labels_for_task(task_df) -> List[str]:
    return sorted(task_df["label"].dropna().unique().tolist())


def build_dataset_label_to_id(labels: List[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(labels)}


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in mapping.items()}


def extract_model_labels_from_sample_outputs(sample_outputs: List[Any]) -> List[str]:
    labels = set()

    for output in sample_outputs:
        if isinstance(output, dict):
            if "label" in output:
                labels.add(output["label"])

        elif isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and "label" in item:
                    labels.add(item["label"])

    return sorted(labels)


def build_observed_label_alignment(
    gold_labels: List[str],
    predicted_labels: List[str]
) -> Dict[str, Dict[str, int]]:
    alignment: Dict[str, Dict[str, int]] = {}

    for gold, pred in zip(gold_labels, predicted_labels):
        if gold not in alignment:
            alignment[gold] = {}
        if pred not in alignment[gold]:
            alignment[gold][pred] = 0
        alignment[gold][pred] += 1

    return alignment
