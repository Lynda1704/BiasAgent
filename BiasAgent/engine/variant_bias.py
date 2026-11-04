import os
from typing import Any, Dict, Optional

import pandas as pd

from engine.configurable_bias import compute_bias_with_config, validate_metric_parameters
from engine.data_loader import filter_dataset_by_task, load_baseline


VARIANT_SPECS = {
    "paraphrase": {
        "text_col": "sentence",
        "orig_id_col": "original_id",
        "orig_refid_col": "original_refid",
        "orig_text_col": "original_sentence",
        "type_col": "paraphrase_type",
        "judge_col": "paraphrase_judge_valid",
    },
    "proxy": {
        "text_col": "proxy_sentence",
        "orig_id_col": "baseline_id",
        "orig_refid_col": "baseline_refid",
        "orig_text_col": "baseline_sentence",
        "type_col": "proxy_type",
        "judge_col": "proxy_judge_valid",
    },
}

CONFIG_KEYS = [
    "scoring_function",
    "distance_function",
    "fairness_idiom",
    "contrasting_method",
    "target_label",
]


class CachedClassifier:

    def __init__(self, clf):
        self.clf = clf
        self.cache: Dict[str, Any] = {}

    def __call__(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            outputs = self.clf(missing)
            for text, output in zip(missing, outputs):
                self.cache[text] = output

        results = [self.cache[t] for t in texts]
        return results[0] if single else results


def _read_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1:  
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().replace('"', "").replace("\ufeff", "") for c in df.columns]
    return df


def load_base_task(baseline_csv_path: str, task_name: str, max_rows: Optional[int]) -> pd.DataFrame:
    base_all = load_baseline(baseline_csv_path)
    base_task = filter_dataset_by_task(base_all, task_name)
    if max_rows is not None:
        base_task = base_task.head(max_rows).copy()
    return base_task


def prepare_variant_frame(
    variants_df: pd.DataFrame,
    base_task_df: pd.DataFrame,
    kind: str,
    task_name: str,
) -> pd.DataFrame:
    spec = VARIANT_SPECS[kind]
    columns = ["sentence", "label", "refid", "id_term", "bias_type",
               "variant_type", "orig_id", "base_refid"]

    v = variants_df.copy()
    v = v[v["task"].astype(str).str.strip().str.lower() == task_name.strip().lower()]

    if "flagged" in v.columns:
        v = v[pd.to_numeric(v["flagged"], errors="coerce") == 0]

    if spec["judge_col"] in v.columns:
        v = v[
            v[spec["judge_col"]]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "1.0"])
        ]

    v = v.copy()
    v[spec["orig_id_col"]] = pd.to_numeric(v[spec["orig_id_col"]], errors="coerce")
    v[spec["orig_refid_col"]] = pd.to_numeric(v[spec["orig_refid_col"]], errors="coerce")
    v = v.dropna(subset=[spec["orig_id_col"], spec["orig_refid_col"]])
    v[spec["orig_id_col"]] = v[spec["orig_id_col"]].astype(int)
    v[spec["orig_refid_col"]] = v[spec["orig_refid_col"]].astype(int)

    base_sentence_by_id = {
        int(i): str(s).strip() for i, s in zip(base_task_df["id"], base_task_df["sentence"])
    }

    v = v[v[spec["orig_id_col"]].isin(base_sentence_by_id)]
    if len(v):
        v = v[
            v.apply(
                lambda r: base_sentence_by_id[r[spec["orig_id_col"]]]
                == str(r[spec["orig_text_col"]]).strip(),
                axis=1,
            )
        ]

    if len(v) == 0:
        return pd.DataFrame(columns=columns)

    if "id" in v.columns:
        v = v.sort_values("id")
    v = v.drop_duplicates(
        subset=[spec["orig_refid_col"], spec["type_col"], "id_term"], keep="last"
    )

    group_keys = list(zip(v[spec["orig_refid_col"]], v[spec["type_col"]].astype(str)))
    codes = {k: i for i, k in enumerate(sorted(set(group_keys)))}

    shaped = pd.DataFrame(
        {
            "sentence": v[spec["text_col"]].astype(str).values,
            "label": v["label"].values,
            "refid": [codes[k] for k in group_keys],
            "id_term": v["id_term"].values,
            "bias_type": v["bias_type"].values,
            "variant_type": v[spec["type_col"]].astype(str).values,
            "orig_id": v[spec["orig_id_col"]].values,
            "base_refid": v[spec["orig_refid_col"]].values,
        }
    )
    return shaped.sort_values(["refid", "orig_id"]).reset_index(drop=True)


def build_matched_baseline(shaped: pd.DataFrame, base_task_df: pd.DataFrame) -> pd.DataFrame:
    base = base_task_df[["id", "sentence", "label", "id_term", "bias_type"]].copy()
    base["orig_id"] = base["id"].astype(int)
    base = base.drop(columns=["id"]).drop_duplicates(subset=["orig_id"])

    matched = shaped[["refid", "orig_id"]].merge(base, on="orig_id", how="left")
    return matched[["sentence", "label", "refid", "id_term", "bias_type", "orig_id"]]


def compute_variant_bias_table(
    kind: str,
    base_task_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    task_name: str,
    clf,
    config: Dict[str, Any],
) -> pd.DataFrame:
    spec = VARIANT_SPECS[kind]
    type_col = spec["type_col"]
    fairness_idiom = config["fairness_idiom"]

    out_columns = [
        "task", "bias_type", type_col, "group_base_refid", "n_members",
        "bias_score", "matched_baseline_score", "delta",
    ]

    shaped = prepare_variant_frame(variants_df, base_task_df, kind, task_name)
    if len(shaped) == 0:
        return pd.DataFrame(columns=out_columns)

    rows = []
    for vtype in sorted(shaped["variant_type"].unique()):
        subset = shaped[shaped["variant_type"] == vtype].reset_index(drop=True)
        matched = build_matched_baseline(subset, base_task_df).reset_index(drop=True)

        variant_cases = compute_bias_with_config(task_df=subset, clf=clf, **config)["all_cases"]
        matched_cases = compute_bias_with_config(task_df=matched, clf=clf, **config)["all_cases"]

        key = "refid" if fairness_idiom == "counterfactual" else "bias_type"
        matched_score = {c[key]: c["score"] for c in matched_cases}

        code_to_base_refid = subset.groupby("refid")["base_refid"].first().to_dict()

        for case in variant_cases:
            if fairness_idiom == "counterfactual":
                n_members = int((subset["refid"] == case["refid"]).sum())
                base_refid = int(code_to_base_refid[case["refid"]])
            else:
                n_members = int((subset["bias_type"] == case["bias_type"]).sum())
                base_refid = None

            m = matched_score.get(case[key])
            rows.append(
                {
                    "task": task_name,
                    "bias_type": case["bias_type"],
                    type_col: vtype,
                    "group_base_refid": base_refid,
                    "n_members": n_members,
                    "bias_score": case["score"],
                    "matched_baseline_score": m,
                    "delta": None if m is None else case["score"] - m,
                }
            )

    return pd.DataFrame(rows, columns=out_columns)


def run_variant_bias_tool(
    kind: str,
    model_name: str,
    task_name: str,
    baseline_csv_path: str,
    variants_path: str,
    output_csv_path: str,
    scoring_function: str,
    distance_function: str,
    fairness_idiom: str,
    contrasting_method: str,
    target_label: Optional[str] = None,
    threshold: float = 0.20,
    max_rows: Optional[int] = 200,
    clf=None,
) -> Dict[str, Any]:
    spec = VARIANT_SPECS[kind]
    type_col = spec["type_col"]

    
    validate_metric_parameters(
        scoring_function=scoring_function,
        distance_function=distance_function,
        target_label=target_label,
        fairness_idiom=fairness_idiom,
    )

    if clf is None:
        from engine.hf_model import build_pipeline

        clf = build_pipeline(model_name)
    clf = CachedClassifier(clf)

    config = {
        "target_label": target_label,
        "scoring_function": scoring_function,
        "distance_function": distance_function,
        "fairness_idiom": fairness_idiom,
        "contrasting_method": contrasting_method,
        "threshold": threshold,
    }

    base_task = load_base_task(baseline_csv_path, task_name, max_rows)
    variants_df = _read_csv_robust(variants_path)

    table = compute_variant_bias_table(kind, base_task, variants_df, task_name, clf, config)

    table.insert(0, "model_name", model_name)
    for k in CONFIG_KEYS:
        table[k] = config[k]
    table["threshold"] = threshold

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    table.to_csv(output_csv_path, index=False)

    summary_output_path = output_csv_path.replace(".csv", "_summary.csv")
    if len(table):
        summary_df = (
            table.groupby(["task", type_col])
            .agg(
                bias_rate=("bias_score", "mean"),
                num_examples=("bias_score", "count"),
                num_biased=("bias_score", lambda s: int((s > 0).sum())),
            )
            .reset_index()
        )
        summary_df.insert(0, "model_name", model_name)
        summary_by_type = table.groupby(type_col)["bias_score"].mean().to_dict()
        overall = float(table["bias_score"].mean())
        matched_overall = (
            float(table["matched_baseline_score"].mean())
            if table["matched_baseline_score"].notna().any()
            else None
        )
    else:
        summary_df = pd.DataFrame(
            columns=["model_name", "task", type_col, "bias_rate", "num_examples", "num_biased"]
        )
        summary_by_type, overall, matched_overall = {}, None, None
    summary_df.to_csv(summary_output_path, index=False)

    return {
        "ok": True,
        "model_name": model_name,
        "task_name": task_name,
        "kind": kind,
        "metric_config_used": {k: config[k] for k in CONFIG_KEYS},
        "threshold": threshold,
        "max_rows": max_rows,
        "output_csv_path": output_csv_path,
        "summary_output_path": summary_output_path,
        "n_groups_scored": int(len(table)),
        "bias_score": overall,
        "matched_baseline_score": matched_overall,
        "summary_by_type": summary_by_type,
        "columns": list(table.columns),
    }
