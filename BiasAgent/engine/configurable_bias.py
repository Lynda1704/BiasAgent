from typing import Dict, Any, Optional
import itertools
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from engine.output_interpreter import normalize_outputs, get_score_for_label


SCORING_FUNCTIONS = {
    "Accuracy_Score": {
        "return_type": "singleton",
        "requires_target_label": False,
        "requires_ground_truth": True,
        "uses": ["predicted_label", "label"],
    },
    "F1_Score": {
        "return_type": "singleton",
        "requires_target_label": False,
        "requires_ground_truth": True,
        "uses": ["predicted_label", "label"],
    },
    "Precision_Score": {
        "return_type": "singleton",
        
        
        "requires_target_label": True,
        "requires_ground_truth": True,
        "uses": ["predicted_label", "label"],
    },
    "Recall_Score": {
        "return_type": "singleton",
        "requires_target_label": True,
        "requires_ground_truth": True,
        "uses": ["predicted_label", "label"],
    },
    "AverageClassPrediction_Score": {
        "return_type": "singleton",
        "requires_target_label": True,
        "requires_ground_truth": False,
        "uses": ["target_score"],
    },
    "ClassPrediction_Score": {
        "return_type": "set",
        "requires_target_label": True,
        "requires_ground_truth": False,
        "uses": ["target_score"],
    },
    "PredictedScore_Score": {
        "return_type": "singleton",
        "requires_target_label": False,
        "requires_ground_truth": False,
        "uses": ["predicted_score"],
    },
    "PredictedLabelShift": {
        "return_type": "label",
        "requires_target_label": False,
        "requires_ground_truth": False,
        "uses": ["predicted_label"],
    },
    "FailureRate": {
        "return_type": "failure",
        "requires_target_label": False,
        "requires_ground_truth": False,
        "uses": ["predicted_label", "predicted_score"],
    },
}


DISTANCE_FUNCTIONS = {
    "AbsoluteDistance": {
        "input_type": "singleton",
    },
    "WassersteinDistance": {
        "input_type": "set",
    },
    "LabelMismatchDistance": {
        "input_type": "label",
    },
    "FailureDistance": {
        "input_type": "failure",
    },
    "AbsoluteDivergenceFromExpectedOutcome": {
        "input_type": "singleton",
        "arity": "multigroup",
    },
    "PerGroupAbsoluteDivergenceFromExpectedOutcome": {
        "input_type": "singleton",
        "arity": "multigroup",
    },
}


MULTIGROUP_DISTANCE_FUNCTIONS = [
    "AbsoluteDivergenceFromExpectedOutcome",
    "PerGroupAbsoluteDivergenceFromExpectedOutcome",
]


FAIRNESS_IDIOMS = ["counterfactual", "group"]
CONTRASTING_METHODS = ["pcm", "bcm", "vbcm", "mcm", "vmcm", "ncm"]


ELIGIBLE_MULTIGROUP_SCORING_FUNCTIONS = [
    "AverageClassPrediction_Score",
    "PredictedScore_Score",
    "Accuracy_Score",
    "F1_Score",
    "Precision_Score",
    "Recall_Score",
]


SUPERVISED_SCORING_FUNCTIONS = [
    "Accuracy_Score",
    "F1_Score",
    "Precision_Score",
    "Recall_Score",
]


def absolute_distance(x, y):
    return abs(float(x) - float(y))


def label_mismatch_distance(label_a, label_b):
    return 1.0 if label_a != label_b else 0.0


def failure_distance(item_a, item_b, threshold: float = 0.20):
    label_a = item_a["predicted_label"]
    label_b = item_b["predicted_label"]

    score_a = float(item_a["predicted_score"])
    score_b = float(item_b["predicted_score"])

    if label_a != label_b:
        return 1.0

    if abs(score_a - score_b) > threshold:
        return 1.0

    return 0.0


def wasserstein_distance_simple(x, y):
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(x, y))


def absolute_divergence_from_expected_outcome(group_values: dict) -> Optional[float]:
    n = len(group_values)

    if n == 0:
        return None

    total = sum(group_values.values())

    if total == 0:
        return 0.0

    expected_share = 1.0 / n

    deviations = [abs((value / total) - expected_share) for value in group_values.values()]

    return float(np.mean(deviations))


def per_group_absolute_divergence_from_expected_outcome(group_values: dict) -> Dict[str, float]:
    n = len(group_values)

    if n == 0:
        return {}

    total = sum(group_values.values())
    expected_share = 1.0 / n

    if total == 0:
        return {group: 0.0 for group in group_values}

    return {
        group: abs((value / total) - expected_share)
        for group, value in group_values.items()
    }


def compute_multigroup_distance(distance_function: str, group_values: dict):
    if distance_function == "AbsoluteDivergenceFromExpectedOutcome":
        return absolute_divergence_from_expected_outcome(group_values)

    if distance_function == "PerGroupAbsoluteDivergenceFromExpectedOutcome":
        return per_group_absolute_divergence_from_expected_outcome(group_values)

    raise ValueError(f"Unsupported multigroup distance_function: {distance_function}")


def compute_supervised_score(values, scoring_function: str, target_label: Optional[str] = None):
    gold = [v["gold_label"] for v in values]
    pred = [v["predicted_label"] for v in values]

    if scoring_function == "Accuracy_Score":
        return float(accuracy_score(gold, pred))

    if scoring_function == "F1_Score":
        return float(f1_score(gold, pred, average="macro", zero_division=0))

    
    if scoring_function == "Precision_Score":
        return float(precision_score(gold, pred, labels=[target_label], average="micro", zero_division=0))

    if scoring_function == "Recall_Score":
        return float(recall_score(gold, pred, labels=[target_label], average="micro", zero_division=0))

    raise ValueError(f"Unsupported supervised scoring function: {scoring_function}")


def get_single_example_supervised_value(row, scoring_function: str, target_label: Optional[str] = None):
    return compute_supervised_score(
        [{"gold_label": row["label"], "predicted_label": row["predicted_label"]}],
        scoring_function,
        target_label=target_label,
    )


def get_counterfactual_value(row, scoring_function: str, target_label: Optional[str] = None):
    if scoring_function in SUPERVISED_SCORING_FUNCTIONS:
        return get_single_example_supervised_value(row, scoring_function, target_label)
    return get_value_for_scoring(row, scoring_function)


def validate_metric_parameters(
    scoring_function: str,
    distance_function: str,
    target_label: Optional[str] = None,
    fairness_idiom: Optional[str] = None,
):
    if scoring_function not in SCORING_FUNCTIONS:
        raise ValueError(f"Unsupported scoring_function: {scoring_function}")

    if distance_function not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance_function: {distance_function}")

    scoring_info = SCORING_FUNCTIONS[scoring_function]
    distance_info = DISTANCE_FUNCTIONS[distance_function]

    if scoring_info["requires_target_label"] and not target_label:
        raise ValueError(
            f"{scoring_function} requires target_label, but target_label is missing."
        )

    if scoring_function in ("Accuracy_Score", "F1_Score") and target_label is not None:
        print("[WARNING] target_label ignored for Accuracy_Score/F1_Score")

    if scoring_info["return_type"] != distance_info["input_type"]:
        raise ValueError(
            f"Incompatible scoring/distance: {scoring_function} returns "
            f"{scoring_info['return_type']}, but {distance_function} expects "
            f"{distance_info['input_type']}"
        )


def build_scored_dataframe(task_df, clf, target_label: Optional[str] = None):
    df = task_df.copy()

    outputs = clf(df["sentence"].tolist())
    normalized = normalize_outputs(outputs)

    df["predicted_label"] = [o["predicted_label"] for o in normalized]
    df["predicted_score"] = [o["predicted_score"] for o in normalized]
    df["all_scores"] = [o["all_scores"] for o in normalized]

    if target_label is not None:
        df["target_score"] = [get_score_for_label(o, target_label) for o in normalized]
    else:
        df["target_score"] = None

    return df


def get_value_for_scoring(row, scoring_function: str):
    if scoring_function in SUPERVISED_SCORING_FUNCTIONS:
        return {
            "gold_label": row["label"],
            "predicted_label": row["predicted_label"],
        }

    if scoring_function in ["AverageClassPrediction_Score", "ClassPrediction_Score"]:
        return float(row["target_score"])

    if scoring_function == "PredictedScore_Score":
        return float(row["predicted_score"])

    if scoring_function == "PredictedLabelShift":
        return row["predicted_label"]

    if scoring_function == "FailureRate":
        return {
            "predicted_label": row["predicted_label"],
            "predicted_score": row["predicted_score"],
        }

    raise ValueError(f"Unsupported scoring_function: {scoring_function}")


def compute_distance(distance_function: str, x, y, threshold: float = 0.20):
    if distance_function == "AbsoluteDistance":
        return absolute_distance(x, y)

    if distance_function == "WassersteinDistance":
        return wasserstein_distance_simple(x, y)

    if distance_function == "LabelMismatchDistance":
        return label_mismatch_distance(x, y)

    if distance_function == "FailureDistance":
        return failure_distance(x, y, threshold=threshold)

    raise ValueError(f"Unsupported distance_function: {distance_function}")


def aggregate_group_values(values, scoring_function: str, target_label: Optional[str] = None):
    if scoring_function in SUPERVISED_SCORING_FUNCTIONS:
        return compute_supervised_score(values, scoring_function, target_label=target_label)

    if scoring_function == "AverageClassPrediction_Score":
        return float(np.mean(values))

    if scoring_function == "ClassPrediction_Score":
        return list(values)

    if scoring_function == "PredictedScore_Score":
        return float(np.mean(values))

    if scoring_function in ["PredictedLabelShift", "FailureRate"]:
        return list(values)

    raise ValueError(f"Unsupported scoring_function: {scoring_function}")


def compute_counterfactual_pcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_refid = []

    for refid, group_df in scored_df.groupby("refid"):
        rows = list(group_df.to_dict("records"))

        if len(rows) < 2:
            continue

        distances = []
        pairs = []

        for row_a, row_b in itertools.combinations(rows, 2):
            g1 = row_a["id_term"]
            g2 = row_b["id_term"]

            v1 = get_counterfactual_value(row_a, scoring_function, target_label)
            v2 = get_counterfactual_value(row_b, scoring_function, target_label)

            distance = compute_distance(distance_function, v1, v2, threshold)

            distances.append(distance)
            pairs.append((g1, g2))

        score = float(np.mean(distances))

        per_refid.append(
             {
                "refid": refid,
                "bias_type": rows[0]["bias_type"],
                "score": score,
                "pcm_score": score,
                "pairs": pairs,
             }
        )

    global_score = float(np.mean([x["score"] for x in per_refid])) if per_refid else None

    return {
        "global_score": global_score,
        "top_cases": sorted(per_refid, key=lambda x: x["score"], reverse=True)[:10],
        "all_cases": per_refid,
    }


def compute_counterfactual_bcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_refid = []

    for refid, group_df in scored_df.groupby("refid"):
        rows = list(group_df.to_dict("records"))

        if len(rows) < 2:
            continue

        values = [get_counterfactual_value(row, scoring_function, target_label) for row in rows]

        if scoring_function in [
            "AverageClassPrediction_Score", "PredictedScore_Score",
        ] + SUPERVISED_SCORING_FUNCTIONS:
            background = float(np.mean(values))
        else:
            continue

        group_scores = []
        distances = []

        for row, value in zip(rows, values):
            distance = compute_distance(distance_function, value, background, threshold)
            distances.append(distance)
            group_scores.append(
                {
                    "group": row["id_term"],
                    "distance_from_background": distance,
                }
            )

        bcm_score = float(np.mean(distances))

        per_refid.append(
           {
                "refid": refid,
                "bias_type": rows[0]["bias_type"],
                "score": bcm_score,
                "bcm_score": bcm_score,
                "background_score": background,
                "group_scores": group_scores,
            }
        )

    global_score = float(np.mean([x["score"] for x in per_refid])) if per_refid else None

    return {
        "global_score": global_score,
        "top_cases": sorted(per_refid, key=lambda x: x["score"], reverse=True)[:10],
        "all_cases": per_refid,
    }


def compute_group_pcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_bias_type = []

    for bias_type, bias_df in scored_df.groupby("bias_type"):
        group_values = {}

        for group, group_df in bias_df.groupby("id_term"):
            values = [
                get_value_for_scoring(row, scoring_function)
                for row in group_df.to_dict("records")
            ]
            group_values[group] = aggregate_group_values(values, scoring_function, target_label)

        if len(group_values) < 2:
            continue

        distances = []
        pairs = []

        for (g1, v1), (g2, v2) in itertools.combinations(group_values.items(), 2):
            if scoring_function in ["PredictedLabelShift", "FailureRate"]:
                min_len = min(len(v1), len(v2))
                if min_len == 0:
                    continue

                pair_distances = [
                    compute_distance(distance_function, v1[i], v2[i], threshold)
                    for i in range(min_len)
                ]

                distance = float(np.mean(pair_distances))
            else:
                distance = compute_distance(distance_function, v1, v2, threshold)

            distances.append(distance)
            pairs.append((g1, g2))

        score = float(np.mean(distances)) if distances else None

        if score is not None:
            per_bias_type.append(
                {
                    "bias_type": bias_type,
                    "score": score,
                    "pcm_score": score,
                    "pairs": pairs,
                }
            )

    global_score = (
        float(np.mean([x["score"] for x in per_bias_type])) if per_bias_type else None
    )

    return {
        "global_score": global_score,
        "top_cases": sorted(per_bias_type, key=lambda x: x["score"], reverse=True)[:10],
        "all_cases": per_bias_type,
    }


def compute_group_bcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_bias_type = []

    for bias_type, bias_df in scored_df.groupby("bias_type"):
        group_scores = {}

        for group, group_df in bias_df.groupby("id_term"):
            values = [
                get_value_for_scoring(row, scoring_function)
                for row in group_df.to_dict("records")
            ]

            if scoring_function in [
                "AverageClassPrediction_Score",
                "PredictedScore_Score",
            ] + SUPERVISED_SCORING_FUNCTIONS:
                group_scores[group] = aggregate_group_values(values, scoring_function, target_label)

        if len(group_scores) < 2:
            continue

        background = float(np.mean(list(group_scores.values())))

        distances = []
        group_details = []

        for group, value in group_scores.items():
            distance = compute_distance(distance_function, value, background, threshold)
            distances.append(distance)
            group_details.append(
                {
                    "group": group,
                    "distance_from_background": distance,
                }
            )

        score = float(np.mean(distances))

        per_bias_type.append(
            {
                "bias_type": bias_type,
                "score": score,
                "bcm_score": score,
                "background_score": background,
                "group_scores": group_details,
            }
        )

    global_score = (
        float(np.mean([x["score"] for x in per_bias_type])) if per_bias_type else None
    )

    return {
        "global_score": global_score,
        "top_cases": sorted(per_bias_type, key=lambda x: x["score"], reverse=True)[:10],
        "all_cases": per_bias_type,
    }


def compute_counterfactual_vbcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_refid = []

    for refid, group_df in scored_df.groupby("refid"):
        rows = list(group_df.to_dict("records"))

        if len(rows) < 2:
            continue

        values = [get_counterfactual_value(row, scoring_function, target_label) for row in rows]

        if scoring_function in [
            "AverageClassPrediction_Score", "PredictedScore_Score",
        ] + SUPERVISED_SCORING_FUNCTIONS:
            background = float(np.mean(values))
        else:
            continue

        group_details = []

        for row, value in zip(rows, values):
            distance = compute_distance(distance_function, value, background, threshold)
            group_details.append(
                {
                    "group": row["id_term"],
                    "distance_from_background": distance,
                }
            )

        per_group_score = {g["group"]: g["distance_from_background"] for g in group_details}

        per_refid.append(
            {
                "refid": refid,
                "bias_type": rows[0]["bias_type"],
                "score": per_group_score,
                "vbcm_score": per_group_score,
                "background_score": background,
                "group_scores": group_details,
            }
        )

    all_group_distances = [
        d["distance_from_background"] for case in per_refid for d in case["group_scores"]
    ]
    global_score = float(np.mean(all_group_distances)) if all_group_distances else None

    return {
        "global_score": global_score,
        "top_cases": sorted(
            per_refid, key=lambda x: max(x["vbcm_score"].values(), default=0), reverse=True
        )[:10],
        "all_cases": per_refid,
    }


def compute_group_vbcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_bias_type = []

    for bias_type, bias_df in scored_df.groupby("bias_type"):
        group_scores = {}

        for group, group_df in bias_df.groupby("id_term"):
            values = [
                get_value_for_scoring(row, scoring_function)
                for row in group_df.to_dict("records")
            ]

            if scoring_function in ELIGIBLE_MULTIGROUP_SCORING_FUNCTIONS:
                group_scores[group] = aggregate_group_values(values, scoring_function, target_label)

        if len(group_scores) < 2:
            continue

        background = float(np.mean(list(group_scores.values())))

        group_details = []

        for group, value in group_scores.items():
            distance = compute_distance(distance_function, value, background, threshold)
            group_details.append(
                {
                    "group": group,
                    "distance_from_background": distance,
                }
            )

        per_group_score = {g["group"]: g["distance_from_background"] for g in group_details}

        per_bias_type.append(
            {
                "bias_type": bias_type,
                "score": per_group_score,
                "vbcm_score": per_group_score,
                "background_score": background,
                "group_scores": group_details,
            }
        )

    all_group_distances = [
        d["distance_from_background"] for case in per_bias_type for d in case["group_scores"]
    ]
    global_score = float(np.mean(all_group_distances)) if all_group_distances else None

    return {
        "global_score": global_score,
        "top_cases": sorted(
            per_bias_type, key=lambda x: max(x["vbcm_score"].values(), default=0), reverse=True
        )[:10],
        "all_cases": per_bias_type,
    }


def compute_counterfactual_mcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_refid = []

    for refid, group_df in scored_df.groupby("refid"):
        rows = list(group_df.to_dict("records"))

        if len(rows) < 2:
            continue

        if scoring_function not in ELIGIBLE_MULTIGROUP_SCORING_FUNCTIONS:
            continue

        group_values = {
            row["id_term"]: get_counterfactual_value(row, scoring_function, target_label)
            for row in rows
        }

        score = compute_multigroup_distance(distance_function, group_values)

        per_refid.append(
            {
                "refid": refid,
                "bias_type": rows[0]["bias_type"],
                "score": score,
                "mcm_score": score,
                "group_values": group_values,
            }
        )

    valid_scores = [x["score"] for x in per_refid if x["score"] is not None]
    global_score = float(np.mean(valid_scores)) if valid_scores else None

    return {
        "global_score": global_score,
        "top_cases": sorted(per_refid, key=lambda x: x["score"] or 0, reverse=True)[:10],
        "all_cases": per_refid,
    }


def compute_group_mcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_bias_type = []

    for bias_type, bias_df in scored_df.groupby("bias_type"):
        group_values = {}

        for group, group_df in bias_df.groupby("id_term"):
            values = [
                get_value_for_scoring(row, scoring_function)
                for row in group_df.to_dict("records")
            ]

            if scoring_function in ELIGIBLE_MULTIGROUP_SCORING_FUNCTIONS:
                group_values[group] = aggregate_group_values(values, scoring_function, target_label)

        if len(group_values) < 2:
            continue

        score = compute_multigroup_distance(distance_function, group_values)

        per_bias_type.append(
            {
                "bias_type": bias_type,
                "score": score,
                "mcm_score": score,
                "group_values": group_values,
            }
        )

    valid_scores = [x["score"] for x in per_bias_type if x["score"] is not None]
    global_score = float(np.mean(valid_scores)) if valid_scores else None

    return {
        "global_score": global_score,
        "top_cases": sorted(per_bias_type, key=lambda x: x["score"] or 0, reverse=True)[:10],
        "all_cases": per_bias_type,
    }


def compute_counterfactual_vmcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_refid = []

    for refid, group_df in scored_df.groupby("refid"):
        rows = list(group_df.to_dict("records"))

        if len(rows) < 2:
            continue

        if scoring_function not in ELIGIBLE_MULTIGROUP_SCORING_FUNCTIONS:
            continue

        group_values = {
            row["id_term"]: get_counterfactual_value(row, scoring_function, target_label)
            for row in rows
        }

        per_group_score = compute_multigroup_distance(distance_function, group_values)

        per_refid.append(
            {
                "refid": refid,
                "bias_type": rows[0]["bias_type"],
                "score": per_group_score,
                "vmcm_score": per_group_score,
                "group_values": group_values,
            }
        )

    all_group_distances = [
        v for case in per_refid for v in case["vmcm_score"].values()
    ]
    global_score = float(np.mean(all_group_distances)) if all_group_distances else None

    return {
        "global_score": global_score,
        "top_cases": sorted(
            per_refid, key=lambda x: max(x["vmcm_score"].values(), default=0), reverse=True
        )[:10],
        "all_cases": per_refid,
    }


def compute_group_vmcm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    per_bias_type = []

    for bias_type, bias_df in scored_df.groupby("bias_type"):
        group_values = {}

        for group, group_df in bias_df.groupby("id_term"):
            values = [
                get_value_for_scoring(row, scoring_function)
                for row in group_df.to_dict("records")
            ]

            if scoring_function in ELIGIBLE_MULTIGROUP_SCORING_FUNCTIONS:
                group_values[group] = aggregate_group_values(values, scoring_function, target_label)

        if len(group_values) < 2:
            continue

        per_group_score = compute_multigroup_distance(distance_function, group_values)

        per_bias_type.append(
            {
                "bias_type": bias_type,
                "score": per_group_score,
                "vmcm_score": per_group_score,
                "group_values": group_values,
            }
        )

    all_group_distances = [
        v for case in per_bias_type for v in case["vmcm_score"].values()
    ]
    global_score = float(np.mean(all_group_distances)) if all_group_distances else None

    return {
        "global_score": global_score,
        "top_cases": sorted(
            per_bias_type, key=lambda x: max(x["vmcm_score"].values(), default=0), reverse=True
        )[:10],
        "all_cases": per_bias_type,
    }


def compute_counterfactual_ncm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    return compute_counterfactual_mcm(
        scored_df, scoring_function, distance_function, threshold, target_label
    )


def compute_group_ncm(
    scored_df,
    scoring_function,
    distance_function,
    threshold: float = 0.20,
    target_label: Optional[str] = None,
):
    return compute_group_mcm(
        scored_df, scoring_function, distance_function, threshold, target_label
    )


def compute_bias_with_config(
    task_df,
    clf,
    target_label: Optional[str] = None,
    scoring_function: str = "AverageClassPrediction_Score",
    distance_function: str = "AbsoluteDistance",
    fairness_idiom: str = "counterfactual",
    contrasting_method: str = "pcm",
    threshold: float = 0.20,
) -> Dict[str, Any]:
    validate_metric_parameters(
        scoring_function=scoring_function,
        distance_function=distance_function,
        target_label=target_label,
        fairness_idiom=fairness_idiom,
    )

    scored_df = build_scored_dataframe(
        task_df=task_df,
        clf=clf,
        target_label=target_label,
    )

    if fairness_idiom == "counterfactual" and contrasting_method == "pcm":
        result = compute_counterfactual_pcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "counterfactual" and contrasting_method == "bcm":
        result = compute_counterfactual_bcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "group" and contrasting_method == "pcm":
        result = compute_group_pcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "group" and contrasting_method == "bcm":
        result = compute_group_bcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "counterfactual" and contrasting_method == "vbcm":
        result = compute_counterfactual_vbcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "group" and contrasting_method == "vbcm":
        result = compute_group_vbcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "counterfactual" and contrasting_method == "mcm":
        result = compute_counterfactual_mcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "group" and contrasting_method == "mcm":
        result = compute_group_mcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "counterfactual" and contrasting_method == "vmcm":
        result = compute_counterfactual_vmcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "group" and contrasting_method == "vmcm":
        result = compute_group_vmcm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "counterfactual" and contrasting_method == "ncm":
        result = compute_counterfactual_ncm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    elif fairness_idiom == "group" and contrasting_method == "ncm":
        result = compute_group_ncm(
            scored_df, scoring_function, distance_function, threshold, target_label
        )

    else:
        raise ValueError(
            f"Combination not implemented yet: {fairness_idiom} + {contrasting_method}"
        )

    return {
        "global_score": result["global_score"],
        "top_cases": result["top_cases"],
        "all_cases": result["all_cases"],
        "num_examples": len(task_df),
        "executed_parameters": {
            "target_label": target_label,
            "scoring_function": scoring_function,
            "distance_function": distance_function,
            "fairness_idiom": fairness_idiom,
            "contrasting_method": contrasting_method,
            "threshold": threshold,
        },
    }
