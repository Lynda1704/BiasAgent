import json
import os
import sys
import logging
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv


PROJECT_PATH = os.environ.get(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


_ENV_CANDIDATES = [
    os.path.join(PROJECT_PATH, ".env"),
    os.path.join(os.path.dirname(PROJECT_PATH), "keys", ".env"),
]
for _env_path in _ENV_CANDIDATES:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break
else:
    load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

for logger_name in [
    "mcp",
    "mcp.server",
    "mcp.server.fastmcp"
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

RESOURCES_PATH = os.path.join(PROJECT_PATH, "mcp_server", "resources")

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

mcp = FastMCP("bias-testing-mcp")


def load_json_resource(filename: str) -> Dict[str, Any]:
    path = os.path.join(RESOURCES_PATH, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_first_output(outputs):
    if outputs and len(outputs) > 0:
        return outputs[0]
    return None


TASKS_DATA = load_json_resource("tasks.json")
METRICS_DATA = load_json_resource("metrics.json")
PROMPTS_DATA = load_json_resource("prompts.json")


@mcp.resource("bias://tasks")
def get_tasks_resource() -> str:
    return json.dumps(TASKS_DATA, ensure_ascii=False, indent=2)


@mcp.resource("bias://metrics")
def get_metrics_resource() -> str:
    return json.dumps(METRICS_DATA, ensure_ascii=False, indent=2)


@mcp.resource("bias://prompts")
def get_prompts_resource() -> str:
    return json.dumps(PROMPTS_DATA, ensure_ascii=False, indent=2)


@mcp.tool()
def get_metric_definitions() -> Dict[str, Any]:
    return {
        "ok": True,
        "metrics": METRICS_DATA
    }


@mcp.tool()
def mark_final_dataset_rows_as_regenerated(
    target_csv_path: str,
    regenerated_judged_csv_path: str,
    id_col: str = "id",
) -> Dict[str, Any]:
    import os
    import pandas as pd

    if not os.path.exists(target_csv_path) or not os.path.exists(regenerated_judged_csv_path):
        return {"ok": True, "marked_rows": 0, "reason": "missing file"}

    target_df = pd.read_csv(target_csv_path)
    regenerated_df = pd.read_csv(regenerated_judged_csv_path)

    if len(target_df) == 0 or len(regenerated_df) == 0:
        return {"ok": True, "marked_rows": 0, "reason": "empty file"}

    if "source_rejected_id" not in regenerated_df.columns:
        raise ValueError("regenerated_judged_csv_path must contain source_rejected_id")

    source_ids = set(
        regenerated_df["source_rejected_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    if id_col not in target_df.columns:
        raise ValueError(f"target_csv_path must contain {id_col}")

    mask = target_df[id_col].dropna().astype(int).isin(source_ids)

    if "regeneration_attempt" not in target_df.columns:
        target_df["regeneration_attempt"] = 0

    if "was_regenerated" not in target_df.columns:
        target_df["was_regenerated"] = False

    if "source_rejected_id" not in target_df.columns:
        target_df["source_rejected_id"] = None

    target_df.loc[mask, "regeneration_attempt"] = 1
    target_df.loc[mask, "was_regenerated"] = True
    target_df.loc[mask, "source_rejected_id"] = target_df.loc[mask, id_col]

    target_df.to_csv(target_csv_path, index=False)

    return {
        "ok": True,
        "target_csv_path": target_csv_path,
        "regenerated_judged_csv_path": regenerated_judged_csv_path,
        "marked_rows": int(mask.sum()),
    }
@mcp.tool()
def mark_rejected_as_regenerated(
    rejected_csv_path: str,
    regenerated_csv_path: str,
    id_col: str = "id",
) -> Dict[str, Any]:
    import os
    import pandas as pd

    if not os.path.exists(rejected_csv_path) or not os.path.exists(regenerated_csv_path):
        return {"ok": True, "marked_rows": 0, "reason": "missing file"}

    rejected_df = pd.read_csv(rejected_csv_path)
    regenerated_df = pd.read_csv(regenerated_csv_path)

    if len(rejected_df) == 0 or len(regenerated_df) == 0:
        return {"ok": True, "marked_rows": 0, "reason": "empty file"}

    source_ids = set(regenerated_df["source_rejected_id"].dropna().astype(int).tolist())

    mask = rejected_df[id_col].dropna().astype(int).isin(source_ids)

    rejected_df.loc[mask, "regeneration_attempt"] = 1
    rejected_df.loc[mask, "was_regenerated"] = True

    rejected_df.to_csv(rejected_csv_path, index=False)

    return {
        "ok": True,
        "rejected_csv_path": rejected_csv_path,
        "regenerated_csv_path": regenerated_csv_path,
        "marked_rows": int(mask.sum()),
    }
@mcp.tool()
def regenerate_rejected_baseline_with_llm(
    rejected_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    max_refids: int | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_regeneration_attempts: int = 5,
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import pandas as pd

    from engine.llm_counterfactual_generation import (
        build_counterfactual_prompt,
        call_llm_for_counterfactuals,
        parse_counterfactual_list,
        add_baseline_rejection_feedback_to_prompt,
    )

    print("[TOOL] regenerate_rejected_baseline_with_llm: start", file=sys.stderr)

    df = pd.read_csv(rejected_csv_path)

    if len(df) == 0:
        pd.DataFrame().to_csv(output_csv_path, index=False)
        return {
            "ok": True,
            "new_rows": 0,
            "reason": "no rejected baseline groups",
            "output_csv_path": output_csv_path,
        }

    if "regeneration_attempt" not in df.columns:
        df["regeneration_attempt"] = 0

    df["regeneration_attempt"] = df["regeneration_attempt"].fillna(0).astype(int)
    df = df[df["regeneration_attempt"] < max_regeneration_attempts].copy()

    refids = df["refid"].dropna().unique().tolist()

    if max_refids is not None:
        refids = refids[:max_refids]

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["counterfactual_generation_prompt"]["template"]

    rows = []

    for refid in refids:
        group_df = df[df["refid"] == refid].copy()

        if group_df.empty:
            continue

        group_df = group_df.sort_values("id").copy()

        rejected_group_text = "\n".join(
            [
                f"- {row['id_term']}: {row['sentence']}"
                for _, row in group_df.iterrows()
            ]
        )

        if "filter_reason" in group_df.columns and group_df["filter_reason"].notna().any():
            filter_reason = str(group_df["filter_reason"].dropna().iloc[0])
        elif "baseline_judge_reason" in group_df.columns and group_df["baseline_judge_reason"].notna().any():
            filter_reason = str(group_df["baseline_judge_reason"].dropna().iloc[0])
        else:
            filter_reason = ""

        source_row = group_df.iloc[0]
        source_id_term = str(source_row["id_term"])
        source_sentence = str(source_row["sentence"])

        print(
            f"[TOOL] regenerating baseline refid={refid} source={source_id_term}",
            file=sys.stderr,
        )

        for _, old_row in group_df.iterrows():
            target_id_term = str(old_row["id_term"])

            if target_id_term == source_id_term:
                new_sentence = source_sentence
            else:
                base_prompt = build_counterfactual_prompt(
                    prompt_template=prompt_template,
                    source_id_term=source_id_term,
                    target_id_term=target_id_term,
                    sentences=[source_sentence],
                )

                prompt = add_baseline_rejection_feedback_to_prompt(
                    prompt=base_prompt,
                    rejected_group_text=rejected_group_text,
                    filter_reason=filter_reason,
                )

                print(
                    f"[TOOL] regenerating baseline refid={refid}: "
                    f"{source_id_term} -> {target_id_term}",
                    file=sys.stderr,
                )

                raw_output = call_llm_for_counterfactuals(
                    prompt=prompt,
                    model=llm_model,
                    temperature=temperature,
                )

                generated = parse_counterfactual_list(raw_output)

                if len(generated) == 0:
                    raise ValueError(
                        f"No regenerated sentence returned for refid={refid}, "
                        f"target_id_term={target_id_term}"
                    )

                new_sentence = str(generated[0])

            new_row = old_row.to_dict()

            new_row["sentence"] = new_sentence
            new_row["source_rejected_id"] = int(old_row["id"])
            new_row["was_regenerated"] = True
            new_row["regeneration_attempt"] = int(old_row["regeneration_attempt"]) + 1

            new_row["flagged"] = None
            new_row["filter_status"] = None
            new_row["filter_reason"] = None
            new_row["baseline_judge_valid"] = None
            new_row["baseline_judge_reason"] = None

            rows.append(new_row)

    regenerated_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    regenerated_df.to_csv(output_csv_path, index=False)

    print("[TOOL] regenerate_rejected_baseline_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "rejected_csv_path": rejected_csv_path,
        "output_csv_path": output_csv_path,
        "input_refids": int(len(refids)),
        "new_rows": int(len(regenerated_df)),
        "new_refids": int(regenerated_df["refid"].nunique()) if len(regenerated_df) else 0,
        "write_mode": "overwrite_regenerated_baseline",
    }
@mcp.tool()
def replace_accepted_refid_groups(
    target_csv_path: str,
    replacement_csv_path: str,
    output_csv_path: str | None = None,
    refid_col: str = "refid",
    flagged_col: str = "flagged",
) -> Dict[str, Any]:
    import os
    import pandas as pd

    target_df = pd.read_csv(target_csv_path)
    replacement_df = pd.read_csv(replacement_csv_path)

    if output_csv_path is None:
        output_csv_path = target_csv_path

    if len(replacement_df) == 0:
        target_df.to_csv(output_csv_path, index=False)
        return {
            "ok": True,
            "replaced_refids": 0,
            "replaced_rows": 0,
            "reason": "empty replacement file",
        }

    accepted_refids = []

    for refid, group in replacement_df.groupby(refid_col):
        if group[flagged_col].fillna(1).astype(int).eq(0).all():
            accepted_refids.append(refid)

    accepted_df = replacement_df[
        replacement_df[refid_col].isin(accepted_refids)
    ].copy()

    if len(accepted_df) == 0:
        target_df.to_csv(output_csv_path, index=False)
        return {
            "ok": True,
            "replaced_refids": 0,
            "replaced_rows": 0,
            "rejected_regenerated_refids": int(replacement_df[refid_col].nunique()),
            "reason": "no accepted regenerated refid groups",
        }

    kept_df = target_df[
        ~target_df[refid_col].isin(accepted_refids)
    ].copy()

    final_df = pd.concat([kept_df, accepted_df], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)

    return {
        "ok": True,
        "target_csv_path": target_csv_path,
        "replacement_csv_path": replacement_csv_path,
        "output_csv_path": output_csv_path,
        "replaced_refids": int(len(accepted_refids)),
        "replaced_rows": int(len(accepted_df)),
        "total_rows": int(len(final_df)),
    }
@mcp.tool()
def regenerate_rejected_proxies_with_llm(
    rejected_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    max_rows: int | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_regeneration_attempts: int = 5,
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import pandas as pd

    from engine.llm_proxy_generation import (
    build_proxy_generation_prompt,
    call_llm_for_proxy_generation,
    parse_proxy_generation_json,
    add_proxy_rejection_feedback_to_prompt,
)

    print("[TOOL] regenerate_rejected_proxies_with_llm: start", file=sys.stderr)

    df = pd.read_csv(rejected_csv_path)

    if len(df) == 0:
        pd.DataFrame().to_csv(output_csv_path, index=False)
        return {
            "ok": True,
            "new_rows": 0,
            "reason": "no rejected proxies",
            "output_csv_path": output_csv_path,
        }

    df["regeneration_attempt"] = df["regeneration_attempt"].fillna(0).astype(int)
    df = df[df["regeneration_attempt"] < max_regeneration_attempts].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["proxy_generation_prompt"]["template"]

    rows = []

    for _, row in df.iterrows():
        base_prompt = build_proxy_generation_prompt(
            prompt_template=prompt_template,
            sentence=str(row["baseline_sentence"]),
            task=str(row["task"]),
            label=str(row["label"]),
            bias_type=str(row["bias_type"]),
            id_term=str(row["id_term"]),
            concept_term=str(row["concept_term"]),
            proxy_type=str(row["proxy_type"]),
            phi=float(row["phi"]),
            theta=float(row["theta"]),
            prior=float(row["prior"]),
            k=int(row["k"]),
        )

        prompt = add_proxy_rejection_feedback_to_prompt(
            prompt=base_prompt,
            rejected_proxy_sentence=str(row["proxy_sentence"]),
            filter_reason=str(row["filter_reason"]),
            proxy_type=str(row["proxy_type"]),
            phi=float(row["phi"]),
            theta=float(row["theta"]),
            k=int(row["k"]),
        )

        print(
            f"[TOOL] regenerating proxy id={row['id']} type={row['proxy_type']}",
            file=sys.stderr,
        )

        raw_output = call_llm_for_proxy_generation(
            prompt=prompt,
            model=llm_model,
            temperature=temperature,
        )

        new_proxy_sentence = parse_proxy_generation_json(raw_output)

        new_row = row.to_dict()

        new_row["proxy_sentence"] = new_proxy_sentence
        new_row["source_rejected_id"] = int(row["id"])
        new_row["was_regenerated"] = True
        new_row["regeneration_attempt"] = int(row["regeneration_attempt"]) + 1

        new_row["flagged"] = None
        new_row["filter_status"] = None
        new_row["filter_reason"] = None
        new_row["proxy_judge_valid"] = None
        new_row["proxy_judge_reason"] = None

        rows.append(new_row)

    regenerated_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    regenerated_df.to_csv(output_csv_path, index=False)

    print("[TOOL] regenerate_rejected_proxies_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "rejected_csv_path": rejected_csv_path,
        "output_csv_path": output_csv_path,
        "input_rows": int(len(df)),
        "new_rows": int(len(regenerated_df)),
    }
@mcp.tool()
def regenerate_rejected_paraphrases_with_llm(
    rejected_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    max_rows: int | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_regeneration_attempts: int = 5,
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import pandas as pd

    from engine.llm_paraphrase_generation import (
        build_paraphrase_prompt,
        call_llm_for_paraphrase,
        parse_paraphrase_json,
        add_rejection_feedback_to_prompt,
    )

    print("[TOOL] regenerate_rejected_paraphrases_with_llm: start", file=sys.stderr)

    df = pd.read_csv(rejected_csv_path)

    if len(df) == 0:
        pd.DataFrame().to_csv(output_csv_path, index=False)
        return {
            "ok": True,
            "new_rows": 0,
            "reason": "no rejected paraphrases",
            "output_csv_path": output_csv_path,
        }

    df["regeneration_attempt"] = df["regeneration_attempt"].fillna(0).astype(int)
    df = df[df["regeneration_attempt"] < max_regeneration_attempts].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["paraphrase_generation_prompt"]["template"]

    rows = []
    skipped_pivot = 0

    for _, row in df.iterrows():
        paraphrase_type = str(row["paraphrase_type"])

        if paraphrase_type == "pivot_translation":
            skipped_pivot += 1
            continue

        base_prompt = build_paraphrase_prompt(
            prompt_template=prompt_template,
            sentence=str(row["original_sentence"]),
            id_term=str(row["id_term"]),
            concept_term=str(row["concept_term"]),
            syntax_type=str(row["syntax_type"]),
            lexical_type=str(row["lexical_type"]),
            length_min_words=5,
            length_max_words=30,
        )

        prompt = add_rejection_feedback_to_prompt(
            prompt=base_prompt,
            rejected_sentence=str(row["sentence"]),
            filter_reason=str(row["filter_reason"]),
            paraphrase_type=str(row["paraphrase_type"]),
            syntax_type=str(row["syntax_type"]),
            lexical_type=str(row["lexical_type"]),
        )

        print(
            f"[TOOL] regenerating paraphrase id={row['id']} type={paraphrase_type}",
            file=sys.stderr,
        )
        
        try:
          raw_output = call_llm_for_paraphrase(
                prompt=prompt,
                model=llm_model,
                temperature=temperature,
            )
          new_sentence = parse_paraphrase_json(raw_output)
        except Exception as e:
            print(f"[TOOL] SKIPPED paraphrase id={row['id']} (LLM/parsing error: {e})", file=sys.stderr,)
            continue

        new_row = row.to_dict()
        

        new_row["sentence"] = new_sentence
        new_row["source_rejected_id"] = int(row["id"])
        new_row["was_regenerated"] = True
        new_row["regeneration_attempt"] = int(row["regeneration_attempt"]) + 1

        new_row["flagged"] = None
        new_row["filter_status"] = None
        new_row["filter_reason"] = None
        new_row["paraphrase_judge_valid"] = None
        new_row["paraphrase_judge_reason"] = None

        rows.append(new_row)

    regenerated_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    regenerated_df.to_csv(output_csv_path, index=False)

    print("[TOOL] regenerate_rejected_paraphrases_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "rejected_csv_path": rejected_csv_path,
        "output_csv_path": output_csv_path,
        "input_rows": int(len(df)),
        "new_rows": int(len(regenerated_df)),
        "skipped_pivot_translation": int(skipped_pivot),
    }
@mcp.tool()
def replace_accepted_rows_by_id(
    target_csv_path: str,
    replacement_csv_path: str,
    output_csv_path: str | None = None,
    id_col: str = "id",
    flagged_col: str = "flagged",
) -> Dict[str, Any]:
    import os
    import pandas as pd

    target_df = pd.read_csv(target_csv_path)
    replacement_df = pd.read_csv(replacement_csv_path)

    if output_csv_path is None:
        output_csv_path = target_csv_path

    if len(replacement_df) == 0:
        target_df.to_csv(output_csv_path, index=False)
        return {
            "ok": True,
            "replaced_rows": 0,
            "reason": "empty replacement file",
            "output_csv_path": output_csv_path,
        }

    accepted_df = replacement_df[
        replacement_df[flagged_col].fillna(1).astype(int) == 0
    ].copy()

    if len(accepted_df) == 0:
        target_df.to_csv(output_csv_path, index=False)
        return {
            "ok": True,
            "replaced_rows": 0,
            "rejected_regenerated_rows": int(len(replacement_df)),
            "reason": "no accepted regenerated rows",
            "output_csv_path": output_csv_path,
        }

    replacement_ids = set(accepted_df[id_col].dropna().astype(int).tolist())

    kept_df = target_df[
        ~target_df[id_col].dropna().astype(int).isin(replacement_ids)
    ].copy()

    final_df = pd.concat([kept_df, accepted_df], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)

    return {
        "ok": True,
        "target_csv_path": target_csv_path,
        "replacement_csv_path": replacement_csv_path,
        "output_csv_path": output_csv_path,
        "replaced_rows": int(len(accepted_df)),
        "rejected_regenerated_rows": int(len(replacement_df) - len(accepted_df)),
        "total_rows": int(len(final_df)),
    }
@mcp.tool()
def generate_bias_report(
    model_name: str,
    task_name: str,
    counterfactual_results_path: str,
    paraphrase_results_path: str,
    proxy_results_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    from engine.report_generation import generate_model_bias_report

    return generate_model_bias_report(
        model_name=model_name,
        task_name=task_name,
        counterfactual_results_path=counterfactual_results_path,
        paraphrase_results_path=paraphrase_results_path,
        proxy_results_path=proxy_results_path,
        output_dir=output_dir,
    )

@mcp.tool()
def check_task_rows_in_csv(
    csv_path: str,
    task_name: str,
) -> Dict[str, Any]:
    import os
    import pandas as pd

    if not os.path.exists(csv_path):
        return {
            "ok": True,
            "exists": False,
            "csv_path": csv_path,
            "task_name": task_name,
            "has_task_rows": False,
            "rows_for_task": 0,
        }

    df = pd.read_csv(csv_path)

    if "task" not in df.columns:
        return {
            "ok": True,
            "exists": True,
            "csv_path": csv_path,
            "task_name": task_name,
            "has_task_rows": False,
            "rows_for_task": 0,
            "columns": list(df.columns),
        }

    task_df = df[df["task"].astype(str).str.strip() == str(task_name).strip()]

    return {
        "ok": True,
        "exists": True,
        "csv_path": csv_path,
        "task_name": task_name,
        "has_task_rows": len(task_df) > 0,
        "rows_for_task": int(len(task_df)),
        "columns": list(df.columns),
    }


@mcp.tool()
def compute_proxy_bias_after_counterfactual_filter(
    model_name: str,
    counterfactual_dataset_path: str,
    proxies_judged_path: str,
    output_csv_path: str,
    task_name: str,
    scoring_function: str,
    distance_function: str,
    fairness_idiom: str,
    contrasting_method: str,
    target_label: str | None = None,
    threshold: float = 0.2,
    max_rows: int | None = 200,
) -> Dict[str, Any]:
    import sys
    from engine.variant_bias import run_variant_bias_tool

    print("[TOOL] compute_proxy_bias_after_counterfactual_filter: start", file=sys.stderr)

    result = run_variant_bias_tool(
        kind="proxy",
        model_name=model_name,
        task_name=task_name,
        baseline_csv_path=counterfactual_dataset_path,
        variants_path=proxies_judged_path,
        output_csv_path=output_csv_path,
        scoring_function=scoring_function,
        distance_function=distance_function,
        fairness_idiom=fairness_idiom,
        contrasting_method=contrasting_method,
        target_label=target_label,
        threshold=threshold,
        max_rows=max_rows,
    )

    print("[TOOL] compute_proxy_bias_after_counterfactual_filter: done", file=sys.stderr)

    return {
        "ok": True,
        "model_name": model_name,
        "task_name": task_name,
        "counterfactual_dataset_path": counterfactual_dataset_path,
        "proxies_judged_path": proxies_judged_path,
        "output_csv_path": output_csv_path,
        "summary_output_path": result["summary_output_path"],
        "threshold": threshold,
        "max_rows": max_rows,
        "metric_config_used": result["metric_config_used"],
        "evaluated_proxy_groups": result["n_groups_scored"],
        "proxy_bias_rate": result["bias_score"],
        "matched_baseline_score": result["matched_baseline_score"],
        "summary_by_proxy_type": result["summary_by_type"],
        "columns": result["columns"],
    }


@mcp.tool()
def generate_proxy_sentences_with_llm(
    input_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    task_name: str | None = None,
    max_rows: int | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.8,
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import math
    import pandas as pd

    from engine.llm_proxy_generation import (
        build_proxy_generation_prompt,
        call_llm_for_proxy_generation,
        parse_proxy_generation_json,
    )

    print("[TOOL] generate_proxy_sentences_with_llm: start", file=sys.stderr)

    df = pd.read_csv(input_csv_path)

    if task_name is not None:
        df = df[df["task"].astype(str).str.strip() == str(task_name).strip()].copy()

    if "flagged" in df.columns:
        df = df[df["flagged"].astype(int) == 0].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["proxy_generation_prompt"]["template"]

    configs = [
        {"proxy_type": "weak", "phi": 4.0, "theta": 0.8},
        {"proxy_type": "medium", "phi": 7.0, "theta": 0.95},
        {"proxy_type": "strong", "phi": 9.5, "theta": 0.995},
    ]

    default_prior = 0.1

    columns = [
        "id",
        "proxy_refid",
        "baseline_id",
        "baseline_refid",
        "task",
        "bias_type",
        "label",
        "id_term",
        "concept_term",
        "baseline_sentence",
        "proxy_type",
        "phi",
        "theta",
        "prior",
        "k",
        "proxy_sentence",
        "flagged",
    ]

    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)

        for col in columns:
            if col not in existing_df.columns:
                existing_df[col] = None

        existing_df = existing_df[columns].copy()

        max_id = int(existing_df["id"].max()) if len(existing_df) else 0
        max_proxy_refid = int(existing_df["proxy_refid"].max()) if len(existing_df) else 0

        existing_keys = set(
            zip(
                existing_df["task"].astype(str),
                existing_df["baseline_id"].astype(int),
                existing_df["proxy_type"].astype(str),
            )
        )
    else:
        existing_df = pd.DataFrame(columns=columns)
        max_id = 0
        max_proxy_refid = 0
        existing_keys = set()

    rows = []
    current_id = max_id + 1
    current_proxy_refid = max_proxy_refid + 1
    skipped_existing = 0

    for _, row in df.iterrows():
        baseline_id = int(row["id"])
        baseline_refid = int(row["refid"])
        baseline_sentence = str(row["sentence"])
        task = str(row["task"])
        prior = default_prior

        generated_any_for_baseline = False
        proxy_refid_for_this_baseline = current_proxy_refid

        for cfg in configs:
            key = (task, baseline_id, cfg["proxy_type"])

            if key in existing_keys:
                skipped_existing += 1
                continue

            phi = float(cfg["phi"])
            theta = float(cfg["theta"])

            numerator = theta * (1 - prior)
            denominator = prior * (1 - theta)
            k = math.ceil(math.log(numerator / denominator, phi))

            prompt = build_proxy_generation_prompt(
                prompt_template=prompt_template,
                sentence=baseline_sentence,
                task=str(row["task"]),
                label=str(row["label"]),
                bias_type=str(row["bias_type"]),
                id_term=str(row["id_term"]),
                concept_term=str(row["concept_term"]),
                proxy_type=cfg["proxy_type"],
                phi=phi,
                theta=theta,
                prior=prior,
                k=k,
            )

            print(
                f"[TOOL] proxy baseline_id={baseline_id} type={cfg['proxy_type']} k={k}",
                file=sys.stderr,
            )

            raw_output = call_llm_for_proxy_generation(
                prompt=prompt,
                model=llm_model,
                temperature=temperature,
            )

            proxy_sentence = parse_proxy_generation_json(raw_output)

            rows.append({
                "id": current_id,
                "proxy_refid": proxy_refid_for_this_baseline,
                "baseline_id": baseline_id,
                "baseline_refid": baseline_refid,
                "task": row["task"],
                "bias_type": row["bias_type"],
                "label": row["label"],
                "id_term": row["id_term"],
                "concept_term": row["concept_term"],
                "baseline_sentence": baseline_sentence,
                "proxy_type": cfg["proxy_type"],
                "phi": phi,
                "theta": theta,
                "prior": prior,
                "k": int(k),
                "proxy_sentence": proxy_sentence,
                "flagged": 0,
            })

            current_id += 1
            generated_any_for_baseline = True

        if generated_any_for_baseline:
            current_proxy_refid += 1

    new_df = pd.DataFrame(rows, columns=columns)

    latest_output_path = output_csv_path.replace(".csv", "_latest.csv")
    new_df.to_csv(latest_output_path, index=False)

    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df.to_csv(output_csv_path, index=False)

    print("[TOOL] generate_proxy_sentences_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "input_csv_path": input_csv_path,
        "output_csv_path": output_csv_path,
        "latest_output_path": latest_output_path,
        "task_name": task_name,
        "new_rows": int(len(new_df)),
        "total_rows": int(len(final_df)),
        "num_baseline_rows": int(len(df)),
        "skipped_existing": int(skipped_existing),
        "proxy_variants_per_baseline": len(configs),
        "columns": columns,
        "write_mode": "append_only_new_proxy_sentences",
    }
@mcp.tool()
def judge_proxy_sentences_with_llm(
    input_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    task_name: str | None = None,
    max_rows: int | None = None,
    llm_model: str = "gpt-4o-mini",
    rejected_output_csv_path: str | None = None,
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import pandas as pd

    from engine.llm_proxy_judge import (
        build_proxy_judge_prompt,
        call_llm_proxy_judge,
        parse_proxy_judge_json,
    )

    print("[TOOL] judge_proxy_sentences_with_llm: start", file=sys.stderr)

    df = pd.read_csv(input_csv_path)

    if task_name is not None:
        df = df[df["task"].astype(str).str.strip() == str(task_name).strip()].copy()

    already_judged_ids = set()

    if os.path.exists(output_csv_path):
        old_df = pd.read_csv(output_csv_path)

        if "id" in old_df.columns:
            already_judged_ids = set(
                old_df["id"].dropna().astype(int).tolist()
            )
    else:
        old_df = None

    if "id" in df.columns:
        df = df[~df["id"].astype(int).isin(already_judged_ids)].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["proxy_judge_prompt"]["template"]

    for idx, row in df.iterrows():
        prompt = build_proxy_judge_prompt(
            prompt_template=prompt_template,
            baseline_sentence=str(row["baseline_sentence"]),
            proxy_sentence=str(row["proxy_sentence"]),
            task=str(row["task"]),
            label=str(row["label"]),
            bias_type=str(row["bias_type"]),
            id_term=str(row["id_term"]),
            concept_term=str(row["concept_term"]),
            proxy_type=str(row["proxy_type"]),
            phi=float(row["phi"]),
            theta=float(row["theta"]),
            k=int(row["k"]),
        )

        print(f"[TOOL] judging proxy id={row['id']}", file=sys.stderr)

        raw_output = call_llm_proxy_judge(
            prompt=prompt,
            model=llm_model,
            temperature=0.0,
        )

        parsed = parse_proxy_judge_json(raw_output)

        df.loc[idx, "flagged"] = parsed["flagged"]
        df.loc[idx, "proxy_judge_valid"] = parsed["valid"]
        df.loc[idx, "proxy_judge_reason"] = parsed["reason"]
    metadata_defaults = {
        "filter_status": None,
        "filter_reason": None,
        "regeneration_attempt": 0,
        "source_rejected_id": None,
        "was_regenerated": False,
    }

    for col, default in metadata_defaults.items():
        if col not in df.columns:
            df[col] = default

    if len(df):
        df["filter_status"] = df["flagged"].fillna(0).astype(int).apply(
            lambda x: "rejected" if x == 1 else "accepted"
        )
        df["filter_reason"] = df["proxy_judge_reason"]

    latest_output_path = output_csv_path.replace(".csv", "_latest.csv")
    df.to_csv(latest_output_path, index=False)

    if old_df is not None:
        final_df = pd.concat([old_df, df], ignore_index=True)
    else:
        final_df = df.copy()

    for col, default in metadata_defaults.items():
        if col not in final_df.columns:
            final_df[col] = default

    final_df["filter_status"] = final_df["flagged"].fillna(0).astype(int).apply(
        lambda x: "rejected" if x == 1 else "accepted"
    )

    if "proxy_judge_reason" in final_df.columns:
        final_df["filter_reason"] = final_df["proxy_judge_reason"]
    else:
        final_df["filter_reason"] = None

    final_df.to_csv(output_csv_path, index=False)

    rejected_rows_count = 0

    if rejected_output_csv_path is not None:
        rejected_df = final_df[
            (final_df["flagged"].fillna(0).astype(int) == 1)
            & (final_df["regeneration_attempt"].fillna(0).astype(int) == 0)
        ].copy()

        rejected_rows_count = int(len(rejected_df))
        rejected_df.to_csv(rejected_output_csv_path, index=False)
    print("[TOOL] judge_proxy_sentences_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "input_csv_path": input_csv_path,
        "output_csv_path": output_csv_path,
        "latest_output_path": latest_output_path,
        "task_name": task_name,
        "already_judged_rows": int(len(already_judged_ids)),
        "new_rows_judged": int(len(df)),
        "total_rows": int(len(final_df)),
        "flagged_counts_this_run": df["flagged"].value_counts().to_dict() if len(df) else {},
        "columns": list(final_df.columns),
        "write_mode": "append_only_new_judged_proxy_sentences",
        "rejected_output_csv_path": rejected_output_csv_path,
        "rejected_rows_this_run": rejected_rows_count,
    }
@mcp.tool()
def compute_paraphrase_bias_after_counterfactual_filter(
    model_name: str,
    counterfactual_dataset_path: str,
    paraphrases_judged_path: str,
    output_csv_path: str,
    task_name: str,
    scoring_function: str,
    distance_function: str,
    fairness_idiom: str,
    contrasting_method: str,
    target_label: str | None = None,
    threshold: float = 0.2,
    max_rows: int | None = 200,
) -> Dict[str, Any]:
    import sys
    from engine.variant_bias import run_variant_bias_tool

    print("[TOOL] compute_paraphrase_bias_after_counterfactual_filter: start", file=sys.stderr)

    result = run_variant_bias_tool(
        kind="paraphrase",
        model_name=model_name,
        task_name=task_name,
        baseline_csv_path=counterfactual_dataset_path,
        variants_path=paraphrases_judged_path,
        output_csv_path=output_csv_path,
        scoring_function=scoring_function,
        distance_function=distance_function,
        fairness_idiom=fairness_idiom,
        contrasting_method=contrasting_method,
        target_label=target_label,
        threshold=threshold,
        max_rows=max_rows,
    )

    print("[TOOL] compute_paraphrase_bias_after_counterfactual_filter: done", file=sys.stderr)

    return {
        "ok": True,
        "model_name": model_name,
        "task_name": task_name,
        "counterfactual_dataset_path": counterfactual_dataset_path,
        "paraphrases_judged_path": paraphrases_judged_path,
        "output_csv_path": output_csv_path,
        "summary_output_path": result["summary_output_path"],
        "threshold": threshold,
        "max_rows": max_rows,
        "metric_config_used": result["metric_config_used"],
        "evaluated_paraphrase_groups": result["n_groups_scored"],
        "paraphrase_bias_rate": result["bias_score"],
        "matched_baseline_score": result["matched_baseline_score"],
        "summary_by_paraphrase_type": result["summary_by_type"],
        "columns": result["columns"],
    }


@mcp.tool()
def judge_paraphrases_with_llm(
    input_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    task_name: str | None = None,
    max_rows: int | None = None,
    llm_model: str = "gpt-4o-mini",
    rejected_output_csv_path: str | None = None,
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import pandas as pd

    from engine.llm_paraphrase_judge import (
        build_paraphrase_judge_prompt,
        call_llm_paraphrase_judge,
        parse_paraphrase_judge_json,
    )

    print("[TOOL] judge_paraphrases_with_llm: start", file=sys.stderr)

    df = pd.read_csv(input_csv_path)

    if task_name is not None:
        df = df[df["task"].astype(str).str.strip() == str(task_name).strip()].copy()

    already_judged_ids = set()

    if os.path.exists(output_csv_path):
        old_df = pd.read_csv(output_csv_path)

        if "id" in old_df.columns:
            already_judged_ids = set(
                old_df["id"].dropna().astype(int).tolist()
            )
    else:
        old_df = None

    if "id" in df.columns:
        df = df[~df["id"].astype(int).isin(already_judged_ids)].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["paraphrase_judge_prompt"]["template"]

    for idx, row in df.iterrows():
        prompt = build_paraphrase_judge_prompt(
            prompt_template=prompt_template,
            original_sentence=str(row["original_sentence"]),
            paraphrase_sentence=str(row["sentence"]),
            task=str(row["task"]),
            label=str(row["label"]),
            bias_type=str(row["bias_type"]),
            id_term=str(row["id_term"]),
            concept_term=str(row["concept_term"]),
            paraphrase_type=str(row["paraphrase_type"]),
            syntax_type=str(row["syntax_type"]),
            lexical_type=str(row["lexical_type"]),
        )

        print(f"[TOOL] judging paraphrase id={row['id']}", file=sys.stderr)

        raw_output = call_llm_paraphrase_judge(
            prompt=prompt,
            model=llm_model,
            temperature=0.0,
        )

        parsed = parse_paraphrase_judge_json(raw_output)

        df.loc[idx, "flagged"] = parsed["flagged"]
        df.loc[idx, "paraphrase_judge_valid"] = parsed["valid"]
        df.loc[idx, "paraphrase_judge_reason"] = parsed["reason"]

    metadata_defaults = {
        "filter_status": None,
        "filter_reason": None,
        "regeneration_attempt": 0,
        "source_rejected_id": None,
        "was_regenerated": False,
    }

    for col, default in metadata_defaults.items():
        if col not in df.columns:
            df[col] = default

    if len(df):
        df["filter_status"] = df["flagged"].fillna(0).astype(int).apply(
            lambda x: "rejected" if x == 1 else "accepted"
        )
        df["filter_reason"] = df["paraphrase_judge_reason"]

    latest_output_path = output_csv_path.replace(".csv", "_latest.csv")
    df.to_csv(latest_output_path, index=False)

    if old_df is not None:
        final_df = pd.concat([old_df, df], ignore_index=True)
    else:
        final_df = df.copy()

    for col, default in metadata_defaults.items():
        if col not in final_df.columns:
            final_df[col] = default

    final_df["filter_status"] = final_df["flagged"].fillna(0).astype(int).apply(
        lambda x: "rejected" if x == 1 else "accepted"
    )

    if "paraphrase_judge_reason" in final_df.columns:
        final_df["filter_reason"] = final_df["paraphrase_judge_reason"]
    else:
        final_df["filter_reason"] = None

    final_df.to_csv(output_csv_path, index=False)

    rejected_rows_count = 0

    if rejected_output_csv_path is not None:
        rejected_df = final_df[
            (final_df["flagged"].fillna(0).astype(int) == 1)
            & (final_df["regeneration_attempt"].fillna(0).astype(int) == 0)
        ].copy()

        rejected_rows_count = int(len(rejected_df))
        rejected_df.to_csv(rejected_output_csv_path, index=False)
    print("[TOOL] judge_paraphrases_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "input_csv_path": input_csv_path,
        "output_csv_path": output_csv_path,
        "latest_output_path": latest_output_path,
        "task_name": task_name,
        "already_judged_rows": int(len(already_judged_ids)),
        "new_rows_judged": int(len(df)),
        "total_rows": int(len(final_df)),
        "flagged_counts_this_run": df["flagged"].value_counts().to_dict() if len(df) else {},
        "columns": list(final_df.columns),
        "write_mode": "append_only_new_judged_paraphrases",
        "rejected_output_csv_path": rejected_output_csv_path,
        "rejected_rows_this_run": rejected_rows_count,
    }

@mcp.tool()
def generate_paraphrases_with_llm(
    input_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    task_name: str | None = None,
    max_rows: int | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    pivot_languages: list[str] | None = None,
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import pandas as pd

    from engine.pivot_translation import pivot_translate
    from engine.llm_paraphrase_generation import (
        build_paraphrase_prompt,
        call_llm_for_paraphrase,
        parse_paraphrase_json,
    )
    from engine.syntax_pattern_utils import get_frequent_and_rare_patterns

    print("[TOOL] generate_paraphrases_with_llm: start", file=sys.stderr)

    df = pd.read_csv(input_csv_path)

    if task_name is not None:
        df = df[df["task"].astype(str).str.strip() == str(task_name).strip()].copy()

    if "flagged" in df.columns:
        df = df[df["flagged"].astype(int) == 0].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["paraphrase_generation_prompt"]["template"]
    pattern_prompt_template = prompts_data["pattern_by_example_prompt"]["template"]
    taboo_prompt_template = prompts_data["taboo_pattern_prompt"]["template"]

    configs = [
        {"paraphrase_type": "question", "syntax_type": "interrogative reformulation", "lexical_type": "synonym substitution"},
        {"paraphrase_type": "reorder", "syntax_type": "sentence reordering", "lexical_type": "synonym substitution"},
        {"paraphrase_type": "passive_active", "syntax_type": "passive voice transformation", "lexical_type": "synonym substitution"},
        {"paraphrase_type": "conjunction", "syntax_type": "conjunction or subordinating clause", "lexical_type": "synonym substitution"},
        {"paraphrase_type": "negation", "syntax_type": "negation with meaning preserved", "lexical_type": "antonym replacement with polarity preserved"},
        {"paraphrase_type": "idiomatic", "syntax_type": "sentence reordering", "lexical_type": "figurative or idiomatic reformulation"},
        {"paraphrase_type": "pattern_by_example", "syntax_type": "rare_pattern_guided", "lexical_type": "syntax_guided"},
        {"paraphrase_type": "taboo_patterns", "syntax_type": "avoid_frequent_patterns", "lexical_type": "syntax_guided"},
        {"paraphrase_type": "pivot_translation", "syntax_type": "multi_pivot_machine_translation", "lexical_type": "cross_lingual_reformulation"},
    ]

    columns = [
        "id", "paraphrase_refid", "original_id", "original_refid",
        "task", "bias_type", "label", "id_term", "concept_term",
        "original_sentence", "paraphrase_type", "syntax_type",
        "lexical_type", "sentence", "flagged",
    ]

    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)

        for col in columns:
            if col not in existing_df.columns:
                existing_df[col] = None

        existing_df = existing_df[columns].copy()
        max_id = int(existing_df["id"].max()) if len(existing_df) else 0
        max_para_refid = int(existing_df["paraphrase_refid"].max()) if len(existing_df) else 0

        existing_keys = set(
            zip(
                existing_df["task"].astype(str),
                existing_df["original_id"].astype(int),
                existing_df["paraphrase_type"].astype(str),
            )
        )
    else:
        existing_df = pd.DataFrame(columns=columns)
        max_id = 0
        max_para_refid = 0
        existing_keys = set()

    if os.path.exists(output_csv_path) and len(existing_df):
        pattern_info = get_frequent_and_rare_patterns(
            paraphrases_csv_path=output_csv_path,
            task_name=task_name,
            top_k=3,
        )
    else:
        pattern_info = {
            "frequent_patterns": [],
            "rare_patterns": [],
            "pattern_table": [],
        }

    rows = []
    current_id = max_id + 1
    current_para_refid = max_para_refid + 1
    skipped_existing = 0

    if pivot_languages is None:
        pivot_languages = ["ar", "fr"]

    for _, row in df.iterrows():
        original_id = int(row["id"])
        original_refid = int(row["refid"])
        original_sentence = str(row["sentence"])
        task = str(row["task"])

        generated_any_for_original = False
        para_refid_for_this_original = current_para_refid

        for cfg in configs:
            key = (task, original_id, cfg["paraphrase_type"])

            if key in existing_keys:
                skipped_existing += 1
                continue

            generated_any_for_original = True

            print(
                f"[TOOL] paraphrase original_id={original_id} type={cfg['paraphrase_type']}",
                file=sys.stderr,
            )

            if cfg["paraphrase_type"] == "pivot_translation":
                print(
                    f"[TOOL] pivot translation original_id={original_id} pivots={pivot_languages}",
                    file=sys.stderr,
                )

                paraphrase = pivot_translate(
                    text=original_sentence,
                    pivot_languages=pivot_languages,
                    source_lang="en",
                )

            else:
                if cfg["paraphrase_type"] == "pattern_by_example":
                    prompt = pattern_prompt_template
                    prompt = prompt.replace("{sentence}", original_sentence)
                    prompt = prompt.replace("{task}", str(row["task"]))
                    prompt = prompt.replace("{label}", str(row["label"]))
                    prompt = prompt.replace("{id_term}", str(row["id_term"]))
                    prompt = prompt.replace("{concept_term}", str(row["concept_term"]))
                    prompt = prompt.replace(
                        "{rare_patterns}",
                        str(pattern_info["rare_patterns"]),
                    )

                elif cfg["paraphrase_type"] == "taboo_patterns":
                    prompt = taboo_prompt_template
                    prompt = prompt.replace("{sentence}", original_sentence)
                    prompt = prompt.replace("{task}", str(row["task"]))
                    prompt = prompt.replace("{label}", str(row["label"]))
                    prompt = prompt.replace("{id_term}", str(row["id_term"]))
                    prompt = prompt.replace("{concept_term}", str(row["concept_term"]))
                    prompt = prompt.replace(
                        "{frequent_patterns}",
                        str(pattern_info["frequent_patterns"]),
                    )

                else:
                    prompt = build_paraphrase_prompt(
                        prompt_template=prompt_template,
                        sentence=original_sentence,
                        id_term=str(row["id_term"]),
                        concept_term=str(row["concept_term"]),
                        syntax_type=cfg["syntax_type"],
                        lexical_type=cfg["lexical_type"],
                        length_min_words=5,
                        length_max_words=30,
                    )
                try:
                    raw_output = call_llm_for_paraphrase(
                        prompt=prompt,
                        model=llm_model,
                        temperature=temperature,
                    )
                    paraphrase = parse_paraphrase_json(raw_output)
                except Exception as e:
                    print(
                        f"[TOOL] SKIPPED paraphrase generation for id={row.get('id')} (LLM/parsing error: {e})",
                        file=sys.stderr,
                    )
                    continue

            rows.append({
                "id": current_id,
                "paraphrase_refid": para_refid_for_this_original,
                "original_id": original_id,
                "original_refid": original_refid,
                "task": row["task"],
                "bias_type": row["bias_type"],
                "label": row["label"],
                "id_term": row["id_term"],
                "concept_term": row["concept_term"],
                "original_sentence": original_sentence,
                "paraphrase_type": cfg["paraphrase_type"],
                "syntax_type": cfg["syntax_type"],
                "lexical_type": cfg["lexical_type"],
                "sentence": paraphrase,
                "flagged": 0,
            })

            current_id += 1

        if generated_any_for_original:
            current_para_refid += 1

    new_df = pd.DataFrame(rows, columns=columns)
    latest_output_path = output_csv_path.replace(".csv", "_latest.csv")
    new_df.to_csv(latest_output_path, index=False)

    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df.to_csv(output_csv_path, index=False)

    print("[TOOL] generate_paraphrases_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "input_csv_path": input_csv_path,
        "output_csv_path": output_csv_path,
        "latest_output_path": latest_output_path,
        "task_name": task_name,
        "new_rows": int(len(new_df)),
        "total_rows": int(len(final_df)),
        "num_original_rows": int(len(df)),
        "skipped_existing": int(skipped_existing),
        "paraphrases_per_original": len(configs),
        "pivot_languages": pivot_languages,
        "frequent_patterns": pattern_info["frequent_patterns"],
        "rare_patterns": pattern_info["rare_patterns"],
        "columns": columns,
        "write_mode": "append_only_new_paraphrases",
    }
   
@mcp.tool()
def remove_duplicate_refid_groups_before_append(
    central_dataset_path: str,
    new_judged_baseline_path: str,
    output_path: str
) -> Dict[str, Any]:
    import os
    import sys
    import re
    import pandas as pd

    print("[TOOL] remove_duplicate_refid_groups_before_append: start", file=sys.stderr)

    def norm(x):
        x = str(x).lower().strip()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^\w\s]", "", x)
        return x

    new_df = pd.read_csv(new_judged_baseline_path)

    if new_df.empty:
        new_df.to_csv(output_path, index=False)
        return {
            "ok": True,
            "new_rows": 0,
            "removed_rows": 0,
            "remaining_rows": 0,
        }

    inferred_tasks = set(new_df["task"].astype(str).str.strip().tolist())

    if not os.path.exists(central_dataset_path):
        new_df.to_csv(output_path, index=False)
        return {
            "ok": True,
            "tasks_checked": sorted(inferred_tasks),
            "new_rows": int(len(new_df)),
            "removed_rows": 0,
            "remaining_rows": int(len(new_df)),
            "mode": "no_central_dataset",
        }

    central_df = pd.read_csv(central_dataset_path)

    central_same_task = central_df[
        central_df["task"].astype(str).str.strip().isin(inferred_tasks)
    ].copy()

    central_keys = set(
        zip(
            central_same_task["task"].astype(str).str.strip(),
            central_same_task["sentence"].astype(str).apply(norm),
        )
    )

    new_df["_dedup_key"] = list(
        zip(
            new_df["task"].astype(str).str.strip(),
            new_df["sentence"].astype(str).apply(norm),
        )
    )

    duplicate_refids = set(
        new_df.loc[
            new_df["_dedup_key"].isin(central_keys),
            "refid"
        ].dropna().astype(int).tolist()
    )

    deduped_df = new_df[
        ~new_df["refid"].astype(int).isin(duplicate_refids)
    ].copy()

    deduped_df = deduped_df.drop(columns=["_dedup_key"], errors="ignore")

    deduped_df.to_csv(output_path, index=False)

    print("[TOOL] remove_duplicate_refid_groups_before_append: done", file=sys.stderr)

    return {
        "ok": True,
        "central_dataset_path": central_dataset_path,
        "new_judged_baseline_path": new_judged_baseline_path,
        "output_path": output_path,
        "tasks_checked": sorted(inferred_tasks),
        "central_rows_same_task": int(len(central_same_task)),
        "new_rows": int(len(new_df)),
        "removed_refids": int(len(duplicate_refids)),
        "removed_rows": int(len(new_df) - len(deduped_df)),
        "remaining_rows": int(len(deduped_df)),
        "mode": "task_plus_sentence",
    }

@mcp.tool()
def append_to_central_generated_dataset(
    central_dataset_path: str,
    new_judged_baseline_path: str,
    paraphrases_judged_path: str | None = None,
    proxies_judged_path: str | None = None,
) -> Dict[str, Any]:
    import os
    import sys
    import pandas as pd

    print("[TOOL] append_to_central_generated_dataset: start", file=sys.stderr)

    base_columns = [
        "id",
        "bias_type",
        "label",
        "task",
        "id_term",
        "concept_term",
        "sentence",
        "flagged",
        "refid",
        "baseline_judge_valid",
        "baseline_judge_reason",
    ]

    optional_columns = [
        "filter_status",
        "filter_reason",
        "regeneration_attempt",
        "source_rejected_id",
        "was_regenerated",
    ]

    DEFAULT_VALUES = {
        "regeneration_attempt": 0,
        "was_regenerated": False,
    }

    new_df = pd.read_csv(new_judged_baseline_path)

    for col in base_columns:
        if col not in new_df.columns:
            if col == "baseline_judge_valid":
                new_df[col] = None
            elif col == "baseline_judge_reason":
                new_df[col] = ""
            else:
                raise ValueError(f"Missing column in new dataset: {col}")

    columns = base_columns + [c for c in optional_columns if c in new_df.columns or (
        os.path.exists(central_dataset_path)
        and c in pd.read_csv(central_dataset_path, nrows=0).columns
    )]

    for col in optional_columns:
        if col in columns and col not in new_df.columns:
            new_df[col] = DEFAULT_VALUES.get(col, None)

    for col in ("regeneration_attempt", "was_regenerated"):
        if col in new_df.columns:
            new_df[col] = new_df[col].fillna(DEFAULT_VALUES[col])

    new_df = new_df[columns].copy()

    if os.path.exists(central_dataset_path):
        central_df = pd.read_csv(central_dataset_path)

        for col in columns:
            if col not in central_df.columns:
                if col == "baseline_judge_valid":
                    central_df[col] = None
                elif col == "baseline_judge_reason":
                    central_df[col] = ""
                else:
                    central_df[col] = DEFAULT_VALUES.get(col, None)

        for col in ("regeneration_attempt", "was_regenerated"):
            if col in central_df.columns:
                central_df[col] = central_df[col].fillna(DEFAULT_VALUES[col])

        central_df = central_df[columns].copy()
    else:
        central_df = pd.DataFrame(columns=columns)

    if new_df.empty:
        central_df.to_csv(central_dataset_path, index=False)
        return {
            "ok": True,
            "old_rows": int(len(central_df)),
            "added_rows": 0,
            "final_rows": int(len(central_df)),
            "columns": columns,
        }

    max_id = int(central_df["id"].max()) if len(central_df) else 0
    max_refid = int(central_df["refid"].max()) if len(central_df) else 0

    old_ids = new_df["id"].astype(int).tolist()
    new_df["id"] = range(max_id + 1, max_id + 1 + len(new_df))
    id_map = dict(zip(old_ids, new_df["id"]))

    refid_map = {
        old_refid: new_refid
        for new_refid, old_refid in enumerate(
            sorted(new_df["refid"].astype(int).unique()),
            start=max_refid + 1,
        )
    }
    new_df["refid"] = new_df["refid"].astype(int).map(refid_map)

    final_df = pd.concat([central_df, new_df], ignore_index=True)
    final_df.to_csv(central_dataset_path, index=False)
    task_names_added = new_df["task"].unique().tolist()
    if paraphrases_judged_path and os.path.exists(paraphrases_judged_path):
        para_df = pd.read_csv(paraphrases_judged_path)
        mask = para_df["task"].isin(task_names_added)
        if mask.any():
            para_df.loc[mask, "original_refid"] = (
                para_df.loc[mask, "original_refid"].astype(int).map(refid_map)
                .fillna(para_df.loc[mask, "original_refid"])
            )
            para_df.loc[mask, "original_id"] = (
                para_df.loc[mask, "original_id"].astype(int).map(id_map)
                .fillna(para_df.loc[mask, "original_id"])
            )
            para_df.to_csv(paraphrases_judged_path, index=False)
            print(f"[TOOL] {mask.sum()} lignes de paraphrases remappees pour {task_names_added}", file=sys.stderr)

    if proxies_judged_path and os.path.exists(proxies_judged_path):
        proxy_df = pd.read_csv(proxies_judged_path)
        refid_col = "baseline_refid" if "baseline_refid" in proxy_df.columns else "original_refid"
        id_col = "baseline_id" if "baseline_id" in proxy_df.columns else "original_id"
        mask = proxy_df["task"].isin(task_names_added)
        if mask.any():
            proxy_df.loc[mask, id_col] = (
                proxy_df.loc[mask, id_col].astype(int).map(id_map)
                .fillna(proxy_df.loc[mask, id_col])
            )
            proxy_df.loc[mask, refid_col] = (
                proxy_df.loc[mask, refid_col].astype(int).map(refid_map)
                .fillna(proxy_df.loc[mask, refid_col])
            )
            proxy_df.to_csv(proxies_judged_path, index=False)
            print(f"[TOOL] {mask.sum()} lignes de proxies remappees pour {task_names_added}", file=sys.stderr)

    print("[TOOL] append_to_central_generated_dataset: done", file=sys.stderr)
    return {
        "ok": True,
        "central_dataset_path": central_dataset_path,
        "new_judged_baseline_path": new_judged_baseline_path,
        "old_rows": int(len(central_df)),
        "added_rows": int(len(new_df)),
        "final_rows": int(len(final_df)),
        "columns": list(final_df.columns),
        "tasks": final_df["task"].value_counts().to_dict(),
    }

def append_dataframe_preserve_ids(
    new_df,
    output_path: str,
    columns: list[str],
):
    import os
    import pandas as pd

    new_df = new_df[columns].copy()

    if not os.path.exists(output_path):
        new_df.to_csv(output_path, index=False)
        return new_df

    old_df = pd.read_csv(output_path)
    old_df = old_df[columns].copy()

    max_id = int(old_df["id"].max()) if len(old_df) else 0
    max_refid = int(old_df["refid"].max()) if len(old_df) else 0

    new_df["id"] = range(max_id + 1, max_id + 1 + len(new_df))

    refid_map = {
        old_refid: new_refid
        for new_refid, old_refid in enumerate(
            sorted(new_df["refid"].unique()),
            start=max_refid + 1,
        )
    }

    new_df["refid"] = new_df["refid"].map(refid_map)

    final_df = pd.concat([old_df, new_df], ignore_index=True)
    final_df.to_csv(output_path, index=False)

    return final_df
@mcp.tool()
def judge_baseline_counterfactual_groups_with_llm(
    baseline_csv_path: str,
    prompts_path: str,
    output_csv_path: str,
    task_name: str | None = None,
    max_refids: int | None = None,
    llm_model: str = "gpt-4o-mini",
    rejected_output_csv_path: str | None = None,
) -> Dict[str, Any]:
    import sys
    import json
    import pandas as pd

    from engine.llm_baseline_judge import (
        build_baseline_judge_prompt,
        call_llm_baseline_judge,
        parse_baseline_judge_json,
    )

    print("[TOOL] judge_baseline_counterfactual_groups_with_llm: start", file=sys.stderr)

    df = pd.read_csv(baseline_csv_path)

    if task_name is not None:
        df = df[
            df["task"].astype(str).str.strip() == str(task_name).strip()
        ].copy()

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["baseline_counterfactual_judge_prompt"]["template"]

    refids = [int(r) for r in df["refid"].dropna().unique()]

    if max_refids is not None:
        refids = refids[:max_refids]

    judge_results = []

    for refid in refids:
        group_df = df[df["refid"].astype(int) == int(refid)].copy()

        if group_df.empty:
            continue

        current_task_name = str(group_df["task"].iloc[0])
        label = str(group_df["label"].iloc[0])
        bias_type = str(group_df["bias_type"].iloc[0])

        rows = []

        for _, row in group_df.iterrows():
            rows.append({
                "id": int(row["id"]),
                "id_term": str(row["id_term"]),
                "concept_term": str(row["concept_term"]),
                "sentence": str(row["sentence"]),
            })

        prompt = build_baseline_judge_prompt(
            prompt_template=prompt_template,
            refid=int(refid),
            task_name=current_task_name,
            label=label,
            bias_type=bias_type,
            rows=rows,
        )

        print(f"[TOOL] judging baseline refid={refid}", file=sys.stderr)

        raw_output = call_llm_baseline_judge(
            prompt=prompt,
            model=llm_model,
            temperature=0.0,
        )

        parsed = parse_baseline_judge_json(raw_output)

        mask = df["refid"].astype(int) == int(refid)

        df.loc[mask, "flagged"] = parsed["flagged"]
        df.loc[mask, "baseline_judge_valid"] = parsed["valid"]
        df.loc[mask, "baseline_judge_reason"] = parsed["reason"]

        judge_results.append({
            "refid": int(refid),
            "task": current_task_name,
            "flagged": parsed["flagged"],
            "valid": parsed["valid"],
            "reason": parsed["reason"],
        })

    base_columns = [
        "id",
        "bias_type",
        "label",
        "task",
        "id_term",
        "concept_term",
        "sentence",
        "flagged",
        "refid",
        "baseline_judge_valid",
        "baseline_judge_reason",
    ]

    optional_columns = [
        "filter_status",
        "filter_reason",
        "regeneration_attempt",
        "source_rejected_id",
        "was_regenerated",
    ]

    columns = base_columns + [c for c in optional_columns if c in df.columns]

    judged_df = df[columns].copy()

    judged_df["filter_status"] = judged_df["flagged"].fillna(0).astype(int).apply(
        lambda x: "rejected" if x == 1 else "accepted"
    )

    judged_df["filter_reason"] = judged_df["baseline_judge_reason"]

    if "regeneration_attempt" not in judged_df.columns:
        judged_df["regeneration_attempt"] = 0
    else:
        judged_df["regeneration_attempt"] = (
            judged_df["regeneration_attempt"].fillna(0).astype(int)
        )

    if "source_rejected_id" not in judged_df.columns:
        judged_df["source_rejected_id"] = None

    if "was_regenerated" not in judged_df.columns:
        judged_df["was_regenerated"] = False

    judged_df.to_csv(output_csv_path, index=False)

    rejected_rows_count = 0
    if rejected_output_csv_path is not None:
        new_rejected_df = judged_df[
            (judged_df["flagged"].fillna(0).astype(int) == 1)
            & (judged_df["regeneration_attempt"].fillna(0).astype(int) == 0)
        ].copy()

        if os.path.exists(rejected_output_csv_path):
            old_rejected_df = pd.read_csv(rejected_output_csv_path)
            if "task" in old_rejected_df.columns and task_name is not None:
                old_rejected_df = old_rejected_df[
                    old_rejected_df["task"].astype(str).str.strip() != str(task_name).strip()
                ]
            rejected_df = pd.concat([old_rejected_df, new_rejected_df], ignore_index=True)
        else:
            rejected_df = new_rejected_df

        rejected_rows_count = int(len(new_rejected_df))
        os.makedirs(os.path.dirname(rejected_output_csv_path), exist_ok=True)
        rejected_df.to_csv(rejected_output_csv_path, index=False)
    print("[TOOL] judge_baseline_counterfactual_groups_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "baseline_csv_path": baseline_csv_path,
        "output_csv_path": output_csv_path,
        "num_refids_judged_this_run": int(len(refids)),
        "new_rows": int(len(judged_df)),
        "total_rows": int(len(judged_df)),
        "flagged_counts_this_run": judged_df["flagged"].fillna(0).astype(int).value_counts().to_dict(),
        "judge_results": judge_results[:10],
        "write_mode": "overwrite_output_with_latest_judged",
        "rejected_output_csv_path": rejected_output_csv_path,
        "rejected_rows_this_run": rejected_rows_count,
    }

@mcp.tool()
def generate_counterfactual_variants_with_llm(
    input_csv_path: str,
    bias_types_path: str,
    prompts_path: str,
    counterfactual_output_csv_path: str,
    baseline_output_csv_path: str,
    max_source_rows: int | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.3
) -> Dict[str, Any]:
    import sys
    import json
    import pandas as pd

    from engine.bias_types_loader import load_bias_types
    from engine.llm_counterfactual_generation import (
        build_counterfactual_prompt,
        call_llm_for_counterfactuals,
        parse_counterfactual_list,
    )

    print("[TOOL] generate_counterfactual_variants_with_llm: start", file=sys.stderr)

    df = pd.read_csv(input_csv_path)

    if "flagged" in df.columns:
        df = df[df["flagged"].astype(int) == 0].copy()

    if max_source_rows is not None:
        df = df.head(max_source_rows).copy()

    bias_types_data = load_bias_types(bias_types_path)

    bias_to_terms = {
        item["name"]: item["identity_terms"]
        for item in bias_types_data["bias_types"]
    }

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["counterfactual_generation_prompt"]["template"]

    baseline_rows = []
    counterfactual_rows = []
    raw_outputs = {}

    current_baseline_id = 1
    current_counterfactual_id = 1
    current_refid = 1
    skipped_non_reference_rows = 0

    for _, row in df.iterrows():
        bias_type = str(row["bias_type"])
        source_id_term = str(row["id_term"])
        source_sentence = str(row["sentence"])

        all_terms = bias_to_terms.get(bias_type, [])

        if source_id_term not in all_terms:
            all_terms = sorted(set(all_terms + [source_id_term]))

        if not all_terms:
            continue

        reference_term = all_terms[0]

        if source_id_term != reference_term:
            skipped_non_reference_rows += 1
            continue

        target_terms = [t for t in all_terms if t != source_id_term]

        original_row = {
            "id": current_baseline_id,
            "bias_type": bias_type,
            "label": row["label"],
            "task": row["task"],
            "id_term": source_id_term,
            "concept_term": row["concept_term"],
            "sentence": source_sentence,
            "flagged": 0,
            "refid": current_refid,
        }

        baseline_rows.append(original_row)
        current_baseline_id += 1

        for target_id_term in target_terms:
            prompt = build_counterfactual_prompt(
                prompt_template=prompt_template,
                source_id_term=source_id_term,
                target_id_term=target_id_term,
                sentences=[source_sentence],
            )

            print(
                f"[TOOL] counterfactual {source_id_term} -> {target_id_term}",
                file=sys.stderr,
            )

            raw_output = call_llm_for_counterfactuals(
                prompt=prompt,
                model=llm_model,
                temperature=temperature,
            )

            parsed = parse_counterfactual_list(raw_output)

            if len(parsed) != 1:
                raise ValueError(
                    f"Expected 1 counterfactual sentence, got {len(parsed)}"
                )

            counter_sentence = parsed[0]

            raw_outputs[f"{current_refid}_{source_id_term}_to_{target_id_term}"] = {
                "source_id_term": source_id_term,
                "target_id_term": target_id_term,
                "source_sentence": source_sentence,
                "raw_output": raw_output,
            }

            counterfactual_rows.append({
                "id": current_counterfactual_id,
                "bias_type": bias_type,
                "label": row["label"],
                "task": row["task"],
                "id_term": target_id_term,
                "concept_term": row["concept_term"],
                "sentence": counter_sentence,
                "flagged": 0,
                "refid": current_refid,
            })
            current_counterfactual_id += 1

            baseline_rows.append({
                "id": current_baseline_id,
                "bias_type": bias_type,
                "label": row["label"],
                "task": row["task"],
                "id_term": target_id_term,
                "concept_term": row["concept_term"],
                "sentence": counter_sentence,
                "flagged": 0,
                "refid": current_refid,
            })
            current_baseline_id += 1

        current_refid += 1

    columns = [
        "id",
        "bias_type",
        "label",
        "task",
        "id_term",
        "concept_term",
        "sentence",
        "flagged",
        "refid",
    ]

    counterfactual_df = pd.DataFrame(counterfactual_rows, columns=columns)
    baseline_df = pd.DataFrame(baseline_rows, columns=columns)
    latest_baseline_output_path = baseline_output_csv_path.replace(
    ".csv",
    "_latest.csv")

    baseline_df.to_csv(latest_baseline_output_path, index=False)
    latest_baseline_output_path = baseline_output_csv_path.replace(
    ".csv",
    "_latest.csv")

    baseline_df.to_csv(latest_baseline_output_path, index=False)

    final_counterfactual_df = append_dataframe_preserve_ids(
        new_df=counterfactual_df,
        output_path=counterfactual_output_csv_path,
        columns=columns,
    )

    final_baseline_df = append_dataframe_preserve_ids(
        new_df=baseline_df,
        output_path=baseline_output_csv_path,
        columns=columns,
    )

    raw_output_path = baseline_output_csv_path.replace(".csv", "_raw_outputs.json")

    try:
        with open(raw_output_path, "r", encoding="utf-8") as f:
            old_raw_outputs = json.load(f)
    except Exception:
        old_raw_outputs = {}

    old_raw_outputs.update(raw_outputs)

    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(old_raw_outputs, f, ensure_ascii=False, indent=2)

    print("[TOOL] generate_counterfactual_variants_with_llm: done", file=sys.stderr)

    return {
        "ok": True,
        "input_csv_path": input_csv_path,
        "counterfactual_output_csv_path": counterfactual_output_csv_path,
        "baseline_output_csv_path": baseline_output_csv_path,
        "raw_output_path": raw_output_path,
        "num_valid_input_rows": int(len(df)),
        "num_skipped_non_reference_rows": int(skipped_non_reference_rows),
        "new_counterfactual_rows": int(len(counterfactual_df)),
        "total_counterfactual_rows": int(len(final_counterfactual_df)),
        "new_baseline_rows": int(len(baseline_df)),
        "total_baseline_rows": int(len(final_baseline_df)),
        "columns": columns,
        "llm_model": llm_model,
        "temperature": temperature,
        "generation_strategy": "reference_term_only",
        "write_mode": "append",
        "latest_baseline_output_path": latest_baseline_output_path,
    }
@mcp.tool()
def generate_task_samples_with_llm(
    terms_csv_path: str,
    task_definitions_path: str,
    prompts_path: str,
    task_name: str,
    output_path: str,
    max_terms: int | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> Dict[str, Any]:
    import os
    import sys
    import json
    import pandas as pd

    from engine.terms_loader import load_terms_catalog
    from engine.task_definitions_loader import (
        load_task_definitions,
        get_task_labels,
    )
    from engine.llm_sample_generation import (
        build_sample_generation_prompt,
        call_llm_for_samples,
        parse_sample_generation_json,
    )

    print("[TOOL] generate_task_samples_with_llm: start", file=sys.stderr)
    print(f"[TOOL] task_name = {task_name}", file=sys.stderr)
    print(f"[TOOL] output_path = {output_path}", file=sys.stderr)

    columns = [
        "id", "bias_type", "label", "task", "id_term",
        "concept_term", "sentence", "flagged", "refid",
    ]

    latest_output_path = output_path.replace(".csv", "_latest.csv")

    if os.path.exists(latest_output_path):
        os.remove(latest_output_path)

    terms_df = load_terms_catalog(terms_csv_path)

    if max_terms is not None:
        terms_df = terms_df.head(max_terms).copy()

    task_definitions = load_task_definitions(task_definitions_path)
    labels = get_task_labels(task_definitions, task_name)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_template = prompts_data["sample_generation_prompt"]["template"]

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_df = existing_df[columns].copy()
        max_existing_id = int(existing_df["id"].max()) if len(existing_df) else 0
        max_existing_refid = int(existing_df["refid"].max()) if len(existing_df) else 0
    else:
        existing_df = pd.DataFrame(columns=columns)
        max_existing_id = 0
        max_existing_refid = 0

    rows = []
    raw_outputs = {}

    current_id = max_existing_id + 1
    current_refid = max_existing_refid + 1

    for _, term_row in terms_df.iterrows():
        bias_type = str(term_row["bias_type"])
        id_term = str(term_row["id_term"])
        concept_term = str(term_row["concept_term"])

        prompt = build_sample_generation_prompt(
            prompt_template=prompt_template,
            id_term=id_term,
            bias_type=bias_type,
            concept_term=concept_term,
            task_name=task_name,
            labels=labels,
        )

        print(
            f"[TOOL] generating sample for {bias_type} / {id_term} / {concept_term}",
            file=sys.stderr,
        )

        llm_output = call_llm_for_samples(
            prompt=prompt,
            model=llm_model,
            temperature=temperature,
        )

        raw_outputs[str(current_refid)] = {
            "bias_type": bias_type,
            "id_term": id_term,
            "concept_term": concept_term,
            "task_name": task_name,
            "raw_output": llm_output,
        }

        parsed = parse_sample_generation_json(llm_output)

        for label in labels:
            if label not in parsed:
                raise ValueError(
                    f"Missing label '{label}' in LLM output. "
                    f"Available labels: {list(parsed.keys())}"
                )

            rows.append({
                "id": current_id,
                "bias_type": bias_type,
                "label": label,
                "task": task_name,
                "id_term": id_term,
                "concept_term": concept_term,
                "sentence": parsed[label],
                "flagged": 0,
                "refid": current_refid,
            })

            current_id += 1

        current_refid += 1

    new_df = pd.DataFrame(rows, columns=columns)

    new_df.to_csv(latest_output_path, index=False)

    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df.to_csv(output_path, index=False)

    raw_output_path = output_path.replace(".csv", "_raw_outputs.json")

    if os.path.exists(raw_output_path):
        try:
            with open(raw_output_path, "r", encoding="utf-8") as f:
                existing_raw_outputs = json.load(f)
        except Exception:
            existing_raw_outputs = {}
    else:
        existing_raw_outputs = {}

    existing_raw_outputs.update(raw_outputs)

    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(existing_raw_outputs, f, ensure_ascii=False, indent=2)

    print("[TOOL] generate_task_samples_with_llm: done", file=sys.stderr)
    print(f"[TOOL] latest rows = {len(new_df)}", file=sys.stderr)
    print(f"[TOOL] total rows = {len(final_df)}", file=sys.stderr)

    return {
        "ok": True,
        "output_path": output_path,
        "latest_output_path": latest_output_path,
        "raw_output_path": raw_output_path,
        "new_rows": int(len(new_df)),
        "total_rows": int(len(final_df)),
        "columns": list(final_df.columns),
        "task_name": task_name,
        "labels": labels,
        "num_terms_used": int(len(terms_df)),
        "llm_model": llm_model,
        "temperature": temperature,
        "write_mode": "append_history_and_overwrite_latest",
    }
@mcp.tool()
def inspect_dataset_structure(
    csv_path: str
) -> Dict[str, Any]:
    from engine.data_loader import load_baseline

    import sys

    print("[TOOL] inspect_dataset_structure: start", file=sys.stderr)

    df = load_baseline(csv_path)

    columns = list(df.columns)

    required_counterfactual_columns = ["refid", "bias_type", "id_term", "sentence"]
    missing_counterfactual_columns = [
        col for col in required_counterfactual_columns if col not in columns
    ]

    has_refid = "refid" in columns
    has_bias_type = "bias_type" in columns
    has_id_term = "id_term" in columns
    has_sentence = "sentence" in columns
    has_task = "task" in columns
    has_label = "label" in columns

    task_counts = {}
    if has_task:
        task_counts = df["task"].value_counts().to_dict()

    bias_type_counts = {}
    if has_bias_type:
        bias_type_counts = df["bias_type"].value_counts().to_dict()

    refid_group_stats = {}
    if has_refid and has_id_term:
        group_sizes = df.groupby("refid")["id_term"].nunique()
        refid_group_stats = {
            "num_refids": int(group_sizes.shape[0]),
            "min_groups_per_refid": int(group_sizes.min()),
            "max_groups_per_refid": int(group_sizes.max()),
            "mean_groups_per_refid": float(group_sizes.mean())
        }

    is_counterfactual_dataset = (
        has_refid
        and has_bias_type
        and has_id_term
        and has_sentence
    )

    print("[TOOL] inspect_dataset_structure: done", file=sys.stderr)

    return {
        "ok": True,
        "csv_path": csv_path,
        "num_rows": int(len(df)),
        "columns": columns,
        "has_task_column": has_task,
        "has_label_column": has_label,
        "has_refid_column": has_refid,
        "has_bias_type_column": has_bias_type,
        "has_id_term_column": has_id_term,
        "has_sentence_column": has_sentence,
        "is_counterfactual_dataset": is_counterfactual_dataset,
        "missing_counterfactual_columns": missing_counterfactual_columns,
        "task_counts": task_counts,
        "bias_type_counts": bias_type_counts,
        "refid_group_stats": refid_group_stats,
        "interpretation": (
            "If is_counterfactual_dataset is true, refid can be treated as the identifier "
            "of matched counterfactual variants, id_term as the demographic group term, "
            "and bias_type as the protected attribute category."
        )
    }
@mcp.tool()
def compute_bias_score_with_config(
    csv_path: str,
    model_name: str,
    task_name: str,
    scoring_function: str,
    distance_function: str,
    fairness_idiom: str,
    contrasting_method: str,
    target_label: str | None = None,
    max_rows: int | None = 200,
    threshold: float = 0.20,
    output_csv_path: str | None = None
) -> Dict[str, Any]:
    from engine.data_loader import load_baseline, filter_dataset_by_task
    from engine.hf_model import build_pipeline
    from engine.configurable_bias import compute_bias_with_config

    import sys

    print("[TOOL] compute_bias_score_with_config: start", file=sys.stderr)

    df = load_baseline(csv_path)
    task_df = filter_dataset_by_task(df, task_name)

    print(f"[TOOL] rows for task: {len(task_df)}", file=sys.stderr)

    if max_rows is not None:
        task_df = task_df.head(max_rows).copy()
        print(f"[TOOL] rows after max_rows limit: {len(task_df)}", file=sys.stderr)

    print("[TOOL] building pipeline...", file=sys.stderr)
    clf = build_pipeline(model_name)

    print("[TOOL] executing requested metric configuration...", file=sys.stderr)

    result = compute_bias_with_config(
    task_df=task_df,
    clf=clf,
    target_label=target_label,
    scoring_function=scoring_function,
    distance_function=distance_function,
    fairness_idiom=fairness_idiom,
    contrasting_method=contrasting_method,
    threshold=threshold,
)
    print("[TOOL] compute_bias_score_with_config: done", file=sys.stderr)

    import pandas as pd

    if output_csv_path is not None:
      os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
      rows = []
      for case in result["all_cases"]:
        row = {
            "model_name": model_name,
            "task_name": task_name,
            "global_score": result["global_score"],
            "num_examples": result["num_examples"],
            "target_label": target_label,
            "scoring_function": scoring_function,
            "distance_function": distance_function,
            "fairness_idiom": fairness_idiom,
            "contrasting_method": contrasting_method,
            "threshold": threshold,
        }
        for k, v in case.items():
          row[k] = str(v) if isinstance(v, (list, dict, tuple)) else v
        rows.append(row)
      pd.DataFrame(rows).to_csv(output_csv_path, index=False)


    return {
        "ok": True,
        "csv_path": csv_path,
        "model_name": model_name,
        "task_name": task_name,
        "target_label": target_label,
        "threshold": threshold,
        "output_csv_path": output_csv_path,
        "metric_config_used": {
            "scoring_function": scoring_function,
            "distance_function": distance_function,
            "fairness_idiom": fairness_idiom,
            "contrasting_method": contrasting_method,
        },
        "global_score": result["global_score"],
        "top_cases": result["top_cases"],
        "num_examples": result["num_examples"],
        "used_max_rows": max_rows
    }

@mcp.tool()
def ping() -> Dict[str, Any]:
    return {"ok": True, "message": "pong"}


@mcp.tool()
def suggest_task(model_name: str, sample_texts: list[str] | None = None) -> Dict[str, Any]:
    from engine.hf_model import fetch_hf_model_metadata, build_pipeline
    from engine.label_mapper import extract_model_labels_from_sample_outputs
    from engine.task_suggester import suggest_task_from_metadata_and_labels

    import sys

    print("[TOOL] suggest_task: start", file=sys.stderr)

    metadata = fetch_hf_model_metadata(model_name)

    observed_model_labels = []

    if sample_texts is not None and len(sample_texts) > 0:
        print("[TOOL] suggest_task: building pipeline...", file=sys.stderr)
        clf = build_pipeline(model_name)

        print("[TOOL] suggest_task: running sample inference...", file=sys.stderr)
        outputs = clf(sample_texts)

        observed_model_labels = extract_model_labels_from_sample_outputs(outputs)

    result = suggest_task_from_metadata_and_labels(
        model_metadata=metadata,
        tasks_data=TASKS_DATA,
        model_labels=observed_model_labels
    )
    print("SUGGEST_TASK_RESULT =", result, file=sys.stderr)
    print("[TOOL] suggest_task: done", file=sys.stderr)

    return {
        "ok": True,
        "model_name": model_name,
        "pipeline_tag": metadata.get("pipeline_tag"),
        "observed_model_labels": observed_model_labels,
        **result
    }

@mcp.tool()
def get_model_metadata(model_name: str) -> Dict[str, Any]:
    from engine.hf_model import fetch_hf_model_metadata

    metadata = fetch_hf_model_metadata(model_name)
    return {
        "ok": True,
        "model_name": model_name,
        "metadata": metadata
    }


@mcp.tool()
def inspect_model_output(model_name: str, sample_texts: list[str]) -> Dict[str, Any]:
    from engine.hf_model import build_pipeline
    from engine.output_interpreter import normalize_single_output
    print("[TOOL] inspect_model_output: start", file=sys.stderr)
    print("[TOOL] building pipeline...", file=sys.stderr)
    clf = build_pipeline(model_name)

    print("[TOOL] running sample inference...", file=sys.stderr)
    outputs = clf(sample_texts)

    first_output = safe_first_output(outputs)
    output_type = "unknown"
    target_label_guess = None

    if first_output is not None:
      normalized = normalize_single_output(first_output)
      output_type = "multi_class_with_scores" if len(normalized["all_scores"]) > 1 else "single_label_score"
      target_label_guess = normalized["predicted_label"]
    print("[TOOL] inspect_model_output: done", file=sys.stderr)

    return {
        "ok": True,
        "model_name": model_name,
        "output_type": output_type,
        "sample_outputs": outputs[:3],
        "example_target_label_guess": target_label_guess
    }


@mcp.tool()
def get_label_alignment_evidence(
    csv_path: str,
    task_name: str,
    model_name: str,
    sample_size: int = 20
) -> Dict[str, Any]:
    from engine.data_loader import load_baseline, filter_dataset_by_task
    from engine.hf_model import build_pipeline, get_model_possible_labels
    from engine.output_interpreter import normalize_single_output

    print("[TOOL] get_label_alignment_evidence: start", file=sys.stderr)

    df = load_baseline(csv_path)
    sub = filter_dataset_by_task(df, task_name)

    has_label_column = "label" in sub.columns
    dataset_label_values = []
    if has_label_column:
        counts = sub["label"].dropna().value_counts()
        dataset_label_values = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ]

    clf = build_pipeline(model_name)
    model_possible_labels = get_model_possible_labels(clf)

    sample_n = min(sample_size, len(sub))
    sample_df = sub.head(sample_n)
    has_sentence_column = "sentence" in sub.columns
    sample_sentences = sample_df["sentence"].tolist() if has_sentence_column else []

    predicted_vs_dataset_label = []
    if sample_sentences:
        print("[TOOL] get_label_alignment_evidence: running sample inference...", file=sys.stderr)
        outputs = clf(sample_sentences)
        for i, output in enumerate(outputs):
            normalized = normalize_single_output(output)
            row = {
                "sentence": sample_sentences[i],
                "predicted_label": normalized["predicted_label"],
            }
            if has_label_column:
                row["dataset_label"] = str(sample_df["label"].iloc[i])
            predicted_vs_dataset_label.append(row)

    print("[TOOL] get_label_alignment_evidence: done", file=sys.stderr)

    return {
        "ok": True,
        "task_name": task_name,
        "model_name": model_name,
        "has_label_column": has_label_column,
        "dataset_label_values": dataset_label_values,
        "model_possible_labels": model_possible_labels,
        "sample_size_used": sample_n,
        "predicted_vs_dataset_label": predicted_vs_dataset_label,
        "note": (
            "This gives you the dataset's real label values, the model's real "
            "label set, and a side-by-side sample of what the model actually "
            "predicts versus what the dataset says for the same sentences. "
            "Decide for yourself, from this evidence, whether the two "
            "correspond closely enough to trust ground-truth-based metrics - "
            "do not assume alignment just because both sets contain "
            "human-readable names, and do not assume misalignment just "
            "because this evidence exists to check for it."
        )
    }


@mcp.tool()
def compute_bias_score(
    csv_path: str,
    model_name: str,
    task_name: str,
    target_label: str | None = None,
    max_rows: int | None = 200
) -> Dict[str, Any]:
    from engine.data_loader import load_baseline, filter_dataset_by_task
    from engine.hf_model import build_pipeline
    from engine.output_interpreter import detect_output_type, choose_target_label, get_target_score
    from engine.metric_selector import select_metric_configuration
    from engine.bias_engine import score_dataset, compute_global_counterfactual_pcm

    print("[TOOL] compute_bias_score: start", file=sys.stderr)
    print("[TOOL] loading dataset...", file=sys.stderr)
    df = load_baseline(csv_path)

    print("[TOOL] filtering dataset by task...", file=sys.stderr)
    task_df = filter_dataset_by_task(df, task_name)
    print(f"[TOOL] rows for task: {len(task_df)}", file=sys.stderr)

    if max_rows is not None:
        task_df = task_df.head(max_rows).copy()
        print(f"[TOOL] rows after max_rows limit: {len(task_df)}", file=sys.stderr)

    print("[TOOL] building pipeline...", file=sys.stderr)
    clf = build_pipeline(model_name)

    print("[TOOL] running first sample...", file=sys.stderr)
    sample_outputs = clf(task_df["sentence"].head(1).tolist())
    first_output = sample_outputs[0]
    output_type = detect_output_type(first_output)

    if target_label is None:
        target_label = choose_target_label(task_name, first_output)

    print(f"[TOOL] output_type = {output_type}", file=sys.stderr)
    print(f"[TOOL] target_label = {target_label}", file=sys.stderr)

    print("[TOOL] scoring dataset subset...", file=sys.stderr)
    scored_df = score_dataset(task_df, clf, get_target_score, target_label)

    print("[TOOL] computing global PCM...", file=sys.stderr)
    global_result = compute_global_counterfactual_pcm(scored_df)

    num_groups_example = int(scored_df.groupby("refid").size().iloc[0])

    metric_params = select_metric_configuration(
        output_type=output_type,
        label_mapping_known=False,
        dataset_structure="refid_counterfactual",
        num_groups=num_groups_example
    )

    per_refid = global_result["per_refid"]
    top_cases = sorted(per_refid, key=lambda x: x["pcm_score"], reverse=True)[:10]

    print("[TOOL] compute_bias_score: done", file=sys.stderr)

    return {
        "ok": True,
        "csv_path": csv_path,
        "model_name": model_name,
        "task_name": task_name,
        "output_type": output_type,
        "target_label": target_label,
        "metric_params": metric_params,
        "global_score": global_result["global_score"],
        "top_cases": top_cases,
        "num_examples": len(task_df),
        "used_max_rows": max_rows
    }
@mcp.tool()
def inspect_hf_model_task(
    model_name: str
) -> Dict[str, Any]:
    import sys

    from engine.hf_model import fetch_hf_model_metadata

    print("[TOOL] inspect_hf_model_task: start", file=sys.stderr)

    metadata = fetch_hf_model_metadata(model_name)

    model_id = str(
        metadata.get("modelId")
        or metadata.get("id")
        or model_name
    ).lower()

    pipeline_tag = str(metadata.get("pipeline_tag") or "").lower()

    task_name = None
    confidence = "unknown"
    reason = ""

    if "sentiment" in model_id:
        task_name = "sentiment analysis"
        confidence = "high"
        reason = "Model name indicates sentiment analysis."

    elif "spam" in model_id or "sms-spam" in model_id:
        task_name = "spam detection"
        confidence = "high"
        reason = "Model name indicates spam detection."

    elif "toxic" in model_id or "toxicity" in model_id:
        task_name = "toxicity detection"
        confidence = "high"
        reason = "Model name indicates toxicity detection."

    elif "hate" in model_id or "hatexplain" in model_id:
        task_name = "hate speech detection"
        confidence = "high"
        reason = "Model name indicates hate speech detection."

    elif "irony" in model_id or "sarcasm" in model_id:
        task_name = "irony detection"
        confidence = "high"
        reason = "Model name indicates irony/sarcasm detection."

    elif "stereotype" in model_id:
        task_name = "bias detection"
        confidence = "high"
        reason = "Model name indicates stereotype detection, treated as bias detection."

    elif "edu" in model_id or "educational" in model_id or "fineweb-edu" in model_id:
        task_name = "judging educational value of texts"
        confidence = "high"
        reason = "Model name indicates educational-value classification."
    elif "bias" in model_id:
        task_name = "bias detection"
        confidence = "high"
        reason = "Model name indicates bias detection."
    elif "fake-news" in model_id or "fake_news" in model_id or ("fake" in model_id and "news" in model_id):
        task_name = "fake news detection"
        confidence = "high"
        reason = "Model name indicates fake news detection."

    elif "clickbait" in model_id:
        task_name = "clickbait detection"
        confidence = "high"
        reason = "Model name indicates clickbait detection."
    elif "emotion" in model_id:
        task_name = "emotion detection"
        confidence = "high"
        reason = "Model name indicates emotion detection."
    else:
        confidence = "low"
        reason = (
            "Could not infer a specific task from Hugging Face metadata. "
            "Route should ask DataGenerationAgent or BiasTestingAgent to inspect more deeply."
        )

    print("[TOOL] inspect_hf_model_task: done", file=sys.stderr)

    return {
        "ok": True,
        "model_name": model_name,
        "pipeline_tag": pipeline_tag,
        "task_name": task_name,
        "confidence": confidence,
        "reason": reason,
    }

if __name__ == "__main__":
    mcp.run()
