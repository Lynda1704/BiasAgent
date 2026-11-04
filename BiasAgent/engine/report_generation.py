import os
import json
import pandas as pd
from engine.report.loaders import (
    load_counterfactual_results,
    load_paraphrase_results,
    load_proxy_results,
)
from engine.report.summaries import build_executive_summary

from engine.report.tables import (
    build_counterfactual_by_bias,
    build_paraphrase_family_summary,
    build_paraphrase_type_summary,
    build_proxy_type_summary,
    build_overall_summary,
        build_paraphrase_by_bias_type,
    build_proxy_by_bias_type,
    value_column,
)

from engine.report.figures import (
    create_counterfactual_bias_figure,
    create_paraphrase_family_figure,
    create_proxy_figure,
    create_paraphrase_detail_figure,
    create_overall_comparison_figure,
)
PARAPHRASE_FAMILIES = {
    "Linguistic Paraphrases": [
        "question",
        "reorder",
        "passive_active",
        "conjunction",
        "idiomatic",
    ],
    "Syntax-Aware Patterns": [
        "pattern_by_example",
        "taboo_patterns",
    ],
    "Pivot Translation": [
        "pivot_translation",
    ],
}


def _safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _assign_paraphrase_family(paraphrase_type: str) -> str:
    for family, types in PARAPHRASE_FAMILIES.items():
        if str(paraphrase_type) in types:
            return family
    return "Other"


def generate_model_bias_report(
    model_name: str,
    task_name: str,
    counterfactual_results_path: str,
    paraphrase_results_path: str,
    proxy_results_path: str,
    output_dir: str,
):
    model_dir = os.path.join(output_dir, _safe_name(model_name), _safe_name(task_name))
    figures_dir = os.path.join(model_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    counterfactual_df = load_counterfactual_results(counterfactual_results_path)
    paraphrase_df = load_paraphrase_results(paraphrase_results_path)
    proxy_df = load_proxy_results(proxy_results_path)
    

    figures = []


    cf_by_bias = build_counterfactual_by_bias(counterfactual_df)

    fig_path = create_counterfactual_bias_figure(cf_by_bias, figures_dir)
    if fig_path is not None:
      figures.append(fig_path)


    paraphrase_df["paraphrase_family"] = paraphrase_df["paraphrase_type"].apply(
        _assign_paraphrase_family
    )

    family_df = build_paraphrase_family_summary(paraphrase_df)

    fig_path =create_paraphrase_family_figure(family_df, figures_dir)
    if fig_path is not None:
      figures.append(fig_path)


    proxy_by_type = build_proxy_type_summary(proxy_df)

    fig_path = create_proxy_figure(proxy_by_type, figures_dir)
    if fig_path is not None:
      figures.append(fig_path)


    para_by_type = build_paraphrase_type_summary(paraphrase_df)

    fig_path = create_paraphrase_detail_figure(para_by_type, figures_dir)
    if fig_path is not None:
      figures.append(fig_path)


    counterfactual_global = (
        float(counterfactual_df["global_score"].iloc[0])
        if "global_score" in counterfactual_df.columns and len(counterfactual_df)
        else None
    )

    linguistic_rate = float(
        paraphrase_df[
            paraphrase_df["paraphrase_family"] == "Linguistic Paraphrases"
        ][value_column(paraphrase_df)].mean()
    )

    syntax_rate = float(
        paraphrase_df[
            paraphrase_df["paraphrase_family"] == "Syntax-Aware Patterns"
        ][value_column(paraphrase_df)].mean()
    )

    pivot_rate = float(
        paraphrase_df[
            paraphrase_df["paraphrase_family"] == "Pivot Translation"
        ][value_column(paraphrase_df)].mean()
    )

    proxy_global = float(proxy_df[value_column(proxy_df)].mean())

    overall_df = build_overall_summary(
        counterfactual_global=counterfactual_global,
        linguistic_rate=linguistic_rate,
        syntax_rate=syntax_rate,
        pivot_rate=pivot_rate,
        proxy_global=proxy_global,)

    fig_path = create_overall_comparison_figure(overall_df, figures_dir)
    if fig_path is not None:
      figures.append(fig_path)


    cf_by_bias.to_csv(os.path.join(model_dir, "counterfactual_by_bias_type.csv"), index=False)
    family_df.to_csv(os.path.join(model_dir, "paraphrase_family_summary.csv"), index=False)
    paraphrase_by_bias = build_paraphrase_by_bias_type(paraphrase_df)
    proxy_by_bias = build_proxy_by_bias_type(proxy_df)
    paraphrase_by_bias.to_csv(os.path.join(model_dir, "paraphrase_by_bias_type.csv"), index=False)
    proxy_by_bias.to_csv(os.path.join(model_dir, "proxy_by_bias_type.csv"), index=False)
    para_by_type.to_csv(os.path.join(model_dir, "paraphrase_type_summary.csv"), index=False)
    proxy_by_type.to_csv(os.path.join(model_dir, "proxy_type_summary.csv"), index=False)
    overall_df.to_csv(os.path.join(model_dir, "overall_summary.csv"), index=False)
    executive_summary = build_executive_summary(
      model_name=model_name,
      task_name=task_name,
      counterfactual_global=counterfactual_global,
      linguistic_rate=linguistic_rate,
      syntax_rate=syntax_rate,
      pivot_rate=pivot_rate,
      proxy_global=proxy_global,
      overall_df=overall_df,
      )


    metric_config = {}
    for key in ("scoring_function", "distance_function", "fairness_idiom",
                "contrasting_method", "target_label"):
        value = None
        if key in counterfactual_df.columns and len(counterfactual_df):
            raw_value = counterfactual_df[key].iloc[0]
            value = None if pd.isna(raw_value) else str(raw_value)
        metric_config[key] = value

    summary = {
        "ok": True,
        "model_name": model_name,
        "task_name": task_name,
        "output_dir": model_dir,
        "figures_dir": figures_dir,
        "figures": figures,
        "counterfactual_global": counterfactual_global,
        "linguistic_paraphrases_bias_rate": linguistic_rate,
        "syntax_aware_patterns_bias_rate": syntax_rate,
        "pivot_translation_bias_rate": pivot_rate,
        "proxy_bias_rate": proxy_global,
        **metric_config,
        "num_counterfactual_rows": int(len(counterfactual_df)),
        "num_paraphrase_rows": int(len(paraphrase_df)),
        "num_proxy_rows": int(len(proxy_df)),
        "executive_summary": executive_summary,
    }

    summary_path = os.path.join(model_dir, "model_bias_report_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary_txt_path = os.path.join(model_dir, "executive_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(executive_summary)
    summary["summary_path"] = summary_path
    summary["executive_summary_path"] = summary_txt_path
    
    return summary
