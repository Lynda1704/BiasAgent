from typing import Dict


def select_metric_configuration(
    output_type: str,
    label_mapping_known: bool,
    dataset_structure: str,
    num_groups: int
) -> Dict[str, str]:

    fairness = "counterfactual" if dataset_structure == "refid_counterfactual" else "group"

    if not label_mapping_known:
        if output_type in ["vector_scores", "single_label_score", "single_score"]:
            scoring = "AverageClassPrediction_Score"
        else:
            scoring = "FailureRate"
    else:
        scoring = "F1_Score"

    distance = "AbsoluteDistance"
    contrasting = "pcm" if num_groups >= 2 else "bcm"

    return {
        "scoring_function": scoring,
        "distance_function": distance,
        "fairness_idiom": fairness,
        "contrasting_method": contrasting,
        "reason": "Choix simple par règles"
    }
