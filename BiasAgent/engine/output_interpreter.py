from typing import Any, Dict, List, Optional


def normalize_single_output(output: Any) -> Dict[str, Any]:

    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
        output = output[0]

    if isinstance(output, list):
        scores = {}
        best_label = None
        best_score = -1.0

        for item in output:
            label = str(item["label"])
            score = float(item["score"])
            scores[label] = score

            if score > best_score:
                best_score = score
                best_label = label

        return {
            "predicted_label": best_label,
            "predicted_score": best_score,
            "all_scores": scores
        }

    if isinstance(output, dict):
        label = str(output["label"])
        score = float(output["score"])

        return {
            "predicted_label": label,
            "predicted_score": score,
            "all_scores": {
                label: score
            }
        }

    raise ValueError(f"Unsupported model output format: {output}")


def normalize_outputs(outputs: List[Any]) -> List[Dict[str, Any]]:
    return [normalize_single_output(o) for o in outputs]


def get_score_for_label(
    normalized_output: Dict[str, Any],
    target_label: Optional[str] = None
) -> float:

    if target_label is None:
        return float(normalized_output["predicted_score"])

    all_scores = normalized_output.get("all_scores", {})

    if target_label in all_scores:
        return float(all_scores[target_label])


    if normalized_output["predicted_label"] == target_label:
        return float(normalized_output["predicted_score"])

    raise ValueError(
        f"Target label '{target_label}' not found in model output. "
        f"Available labels: {list(all_scores.keys())}"
    )


def extract_predicted_label(output: Any) -> str:
    return normalize_single_output(output)["predicted_label"]


def extract_predicted_score(output: Any) -> float:
    return normalize_single_output(output)["predicted_score"]
