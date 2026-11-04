import itertools
from typing import Any, Dict

import numpy as np
import pandas as pd


def absolute_distance(x: float, y: float) -> float:
    return abs(x - y)


def score_dataset(df: pd.DataFrame, clf, get_target_score_fn, target_label: str) -> pd.DataFrame:
    outputs = clf(df["sentence"].tolist())

    target_scores = []
    raw_outputs = []

    for out in outputs:
        target_scores.append(get_target_score_fn(out, target_label))
        raw_outputs.append(out)

    scored = df.copy()
    scored["target_score"] = target_scores
    scored["raw_model_output"] = raw_outputs
    return scored


def compute_pcm_counterfactual_for_refid(group_df: pd.DataFrame, score_col: str = "target_score") -> Dict[str, Any]:
    groups = group_df["id_term"].tolist()
    scores = group_df[score_col].tolist()

    if len(scores) < 2:
        return {
            "pairs": [],
            "pairwise_distances": [],
            "pcm_score": None
        }

    pairs = []
    distances = []

    for (g1, s1), (g2, s2) in itertools.combinations(zip(groups, scores), 2):
        pairs.append((g1, g2))
        distances.append(absolute_distance(s1, s2))

    pcm_score = float(np.mean(distances))

    return {
        "pairs": pairs,
        "pairwise_distances": distances,
        "pcm_score": pcm_score
    }


def compute_global_counterfactual_pcm(df: pd.DataFrame, score_col: str = "target_score") -> Dict[str, Any]:
    per_refid = []

    for refid, group_df in df.groupby("refid"):
        res = compute_pcm_counterfactual_for_refid(group_df, score_col=score_col)
        if res["pcm_score"] is not None:
            per_refid.append({
                "refid": refid,
                "pcm_score": res["pcm_score"],
                "pairs": res["pairs"]
            })

    if not per_refid:
        return {"global_score": None, "per_refid": []}

    global_score = float(np.mean([x["pcm_score"] for x in per_refid]))

    return {
        "global_score": global_score,
        "per_refid": per_refid
    }
