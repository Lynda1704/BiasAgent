import json
import os
import re
import subprocess

from mcp_server.server import (
    replace_accepted_refid_groups,
    replace_accepted_rows_by_id,
    mark_rejected_as_regenerated,
    mark_final_dataset_rows_as_regenerated,
)


PROJECT_PATH = os.environ.get(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

ORCHESTRATOR_MODULE = "agent.orchestrator_agent"

AGENT_MODULES = {
    "BiasTestingAgent": "agent.bias_evaluation_agent",
    "DataGenerationAgent": "agent.data_generation_agent",
    "FilteringJudgeAgent": "agent.filtering_judge_agent",
}


def csv_has_rows(path: str) -> bool:
    import pandas as pd
    from pandas.errors import EmptyDataError

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return False
    except Exception:
        return False

    return len(df) > 0


def scope_rejected_csv_to_task(path: str, task_name: str, scoped_path: str) -> str:
    import pandas as pd

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return path

    try:
        df = pd.read_csv(path)
    except Exception:
        return path

    if "task" not in df.columns or len(df) == 0:
        return path

    scoped_df = df[df["task"].astype(str).str.strip() == str(task_name).strip()]

    os.makedirs(os.path.dirname(scoped_path), exist_ok=True)
    scoped_df.to_csv(scoped_path, index=False)
    return scoped_path


def has_pending_rejections(path: str) -> bool:
    import pandas as pd

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    if len(df) == 0:
        return False

    if "regeneration_attempt" not in df.columns:
        return True

    return (df["regeneration_attempt"].fillna(0).astype(int) == 0).any()


def run_module(module_name: str, env: dict | None = None) -> str:
    run_env = os.environ.copy()

    if env:
        run_env.update(env)

    result = subprocess.run(
        ["python", "-m", module_name],
        cwd=PROJECT_PATH,
        env=run_env,
        text=True,
        capture_output=True,
    )

    print(result.stdout)

    if result.stderr:
        print("\n=== STDERR ===\n")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Module {module_name} failed with exit code {result.returncode}"
        )

    return result.stdout


def extract_json_from_output(output: str) -> dict:
    match = re.search(r"\{[\s\S]*\}\s*$", output.strip())

    if not match:
        raise ValueError("Could not find JSON decision in orchestrator output.")

    return json.loads(match.group(0))


def main():
    print("\n=== RUNNING ORCHESTRATOR ===\n")

    orchestrator_output = run_module(ORCHESTRATOR_MODULE)
    decision = extract_json_from_output(orchestrator_output)

    print("\n=== ROUTING DECISION ===\n")
    print(json.dumps(decision, indent=2))

    model_name = decision["model_name"]
    paths = decision["paths"]

    data_generation_mode = decision.get("data_generation_mode", "full")
    filtering_mode = decision.get("filtering_mode", "full")

    latest_generated_samples_path = paths["generated_samples_path"].replace(".csv", "_latest.csv")
    latest_baseline_generated_path = paths["baseline_generated_path"].replace(".csv", "_latest.csv")
    tmp_baseline_deduped_path = paths["tmp_baseline_judged_path"].replace(".csv", "_deduped.csv")

    rejected_baseline_path = f"{PROJECT_PATH}/data/rejected_baseline_for_regeneration.csv"
    rejected_paraphrases_path = f"{PROJECT_PATH}/data/rejected_paraphrases_for_regeneration.csv"
    rejected_proxies_path = f"{PROJECT_PATH}/data/rejected_proxies_for_regeneration.csv"

    regenerated_baseline_path = f"{PROJECT_PATH}/data/regenerated_baseline.csv"
    regenerated_paraphrases_path = f"{PROJECT_PATH}/data/regenerated_paraphrases.csv"
    regenerated_proxies_path = f"{PROJECT_PATH}/data/regenerated_proxies.csv"

    regenerated_baseline_judged_path = f"{PROJECT_PATH}/data/regenerated_baseline_judged.csv"
    regenerated_paraphrases_judged_path = f"{PROJECT_PATH}/data/regenerated_paraphrases_judged.csv"
    regenerated_proxies_judged_path = f"{PROJECT_PATH}/data/regenerated_proxies_judged.csv"

    rejected_regenerated_baseline_path = f"{PROJECT_PATH}/data/rejected_regenerated_baseline.csv"
    rejected_regenerated_paraphrases_path = f"{PROJECT_PATH}/data/rejected_regenerated_paraphrases.csv"
    rejected_regenerated_proxies_path = f"{PROJECT_PATH}/data/rejected_regenerated_proxies.csv"

    ran_generation_or_filtering = False

    
    for agent_name in decision["route"]:
        module_name = AGENT_MODULES[agent_name]

        if agent_name == "DataGenerationAgent":
            ran_generation_or_filtering = True
            print("\n=== RUNNING DataGenerationAgent ===\n")
            inferred_task = decision.get("inferred_task")
            if not inferred_task:
                raise ValueError(
                    "inferred_task est vide/None: impossible de lancer DataGenerationAgent "
                    "sans nom de tache. Ajoute un mot-cle correspondant dans "
                    "inspect_hf_model_task (mcp_server/server.py) pour ce modele: "
                    f"{model_name}"
                )
            run_module(
                module_name,
                env={
                    "TASK_NAME": inferred_task,
                    "TERMS_CSV_PATH": paths["terms_csv_path"],
                    "PROMPTS_PATH": paths["prompts_path"],
                    "TASK_DEFINITIONS_PATH": paths["task_definitions_path"],
                    "GENERATED_SAMPLES_PATH": paths["generated_samples_path"],
                    "LATEST_GENERATED_SAMPLES_PATH": latest_generated_samples_path,
                    "COUNTERFACTUAL_SAMPLES_PATH": paths["counterfactual_samples_path"],
                    "BASELINE_GENERATED_PATH": paths["baseline_generated_path"],
                    "LATEST_BASELINE_GENERATED_PATH": latest_baseline_generated_path,
                    "CENTRAL_GENERATED_DATASET_PATH": paths["central_generated_dataset_path"],
                    "DATA_GENERATION_MODE": data_generation_mode,
                    "PARAPHRASES_PATH": paths["paraphrases_path"],
                    "PROXIES_PATH": paths["proxies_path"],
                    "REJECTED_BASELINE_PATH": rejected_baseline_path,
                    "REJECTED_PARAPHRASES_PATH": rejected_paraphrases_path,
                    "REJECTED_PROXIES_PATH": rejected_proxies_path,
                },
            )

        elif agent_name == "FilteringJudgeAgent":
            ran_generation_or_filtering = True
            print("\n=== RUNNING FilteringJudgeAgent ===\n")

            run_module(
                module_name,
                env={
                    "PROMPTS_PATH": paths["prompts_path"],
                    "LATEST_GENERATED_SAMPLES_PATH": latest_generated_samples_path,
                    "GENERATED_SAMPLES_JUDGED_PATH": paths["generated_samples_judged_path"],
                    "LATEST_BASELINE_GENERATED_PATH": latest_baseline_generated_path,
                    "TMP_BASELINE_JUDGED_PATH": paths["tmp_baseline_judged_path"],
                    "TMP_BASELINE_DEDUPED_PATH": tmp_baseline_deduped_path,
                    "CENTRAL_GENERATED_DATASET_PATH": paths["central_generated_dataset_path"],
                    "FILTERING_MODE": filtering_mode,
                    "PARAPHRASES_PATH": paths["paraphrases_path"],
                    "PARAPHRASES_JUDGED_PATH": paths["paraphrases_judged_path"],
                    "PROXIES_PATH": paths["proxies_path"],
                    "PROXIES_JUDGED_PATH": paths["proxies_judged_path"],
                    "TASK_NAME": decision["inferred_task"],
                    "REJECTED_BASELINE_PATH": rejected_baseline_path,
                    "REJECTED_PARAPHRASES_PATH": rejected_paraphrases_path,
                    "REJECTED_PROXIES_PATH": rejected_proxies_path,
                },
            )

    
    task_scoped_rejected_baseline_path = scope_rejected_csv_to_task(
        rejected_baseline_path, decision["inferred_task"],
        f"{PROJECT_PATH}/data/tmp_rejected_baseline_task_scoped.csv",
    )
    task_scoped_rejected_paraphrases_path = scope_rejected_csv_to_task(
        rejected_paraphrases_path, decision["inferred_task"],
        f"{PROJECT_PATH}/data/tmp_rejected_paraphrases_task_scoped.csv",
    )
    task_scoped_rejected_proxies_path = scope_rejected_csv_to_task(
        rejected_proxies_path, decision["inferred_task"],
        f"{PROJECT_PATH}/data/tmp_rejected_proxies_task_scoped.csv",
    )

    pending_rejections_exist = (
        has_pending_rejections(task_scoped_rejected_baseline_path)
        or has_pending_rejections(task_scoped_rejected_paraphrases_path)
        or has_pending_rejections(task_scoped_rejected_proxies_path)
    )

    if ran_generation_or_filtering and pending_rejections_exist:
        print("\n=== RUNNING ITERATIVE REGENERATION (up to 5 attempts) ===\n")

        import pandas as pd

        MAX_REGENERATION_ROUNDS = 5

        round_rejected_baseline_path = task_scoped_rejected_baseline_path
        round_rejected_paraphrases_path = task_scoped_rejected_paraphrases_path
        round_rejected_proxies_path = task_scoped_rejected_proxies_path

        accepted_baseline_frames = []
        accepted_paraphrases_frames = []
        accepted_proxies_frames = []

        def _still_rejected_path(stage: str, round_num: int) -> str:
            return f"{PROJECT_PATH}/data/tmp_still_rejected_{stage}_round{round_num}.csv"

        def _split_accepted_vs_rejected(judged_path: str, still_rejected_path: str):
            if not csv_has_rows(judged_path):
                return None, "NONE"
            judged_df = pd.read_csv(judged_path)
            flagged = judged_df["flagged"].fillna(0).astype(int)
            accepted_df = judged_df[flagged == 0].copy()
            still_rejected_df = judged_df[flagged == 1].copy()
            if len(still_rejected_df):
                still_rejected_df.to_csv(still_rejected_path, index=False)
                next_path = still_rejected_path
            else:
                next_path = "NONE"
            return (accepted_df if len(accepted_df) else None), next_path

        for round_num in range(1, MAX_REGENERATION_ROUNDS + 1):
            baseline_has_input = round_rejected_baseline_path != "NONE" and csv_has_rows(round_rejected_baseline_path)
            paraphrases_has_input = round_rejected_paraphrases_path != "NONE" and csv_has_rows(round_rejected_paraphrases_path)
            proxies_has_input = round_rejected_proxies_path != "NONE" and csv_has_rows(round_rejected_proxies_path)

            if not (baseline_has_input or paraphrases_has_input or proxies_has_input):
                print(f"\n--- Regeneration round {round_num}: nothing left to regenerate, stopping early ---\n")
                break

            print(f"\n--- Regeneration round {round_num}/{MAX_REGENERATION_ROUNDS} ---\n")

            for p in [
                regenerated_baseline_path, regenerated_paraphrases_path, regenerated_proxies_path,
                regenerated_baseline_judged_path, regenerated_paraphrases_judged_path, regenerated_proxies_judged_path,
            ]:
                if os.path.exists(p):
                    os.remove(p)

            run_module(
                AGENT_MODULES["DataGenerationAgent"],
                env={
                    "TASK_NAME": decision["inferred_task"],
                    "TERMS_CSV_PATH": paths["terms_csv_path"],
                    "PROMPTS_PATH": paths["prompts_path"],
                    "TASK_DEFINITIONS_PATH": paths["task_definitions_path"],
                    "GENERATED_SAMPLES_PATH": paths["generated_samples_path"],
                    "LATEST_GENERATED_SAMPLES_PATH": latest_generated_samples_path,
                    "COUNTERFACTUAL_SAMPLES_PATH": paths["counterfactual_samples_path"],
                    "BASELINE_GENERATED_PATH": paths["baseline_generated_path"],
                    "LATEST_BASELINE_GENERATED_PATH": latest_baseline_generated_path,
                    "CENTRAL_GENERATED_DATASET_PATH": paths["central_generated_dataset_path"],
                    "DATA_GENERATION_MODE": "regenerate_rejected_once",
                    "PARAPHRASES_PATH": paths["paraphrases_path"],
                    "PROXIES_PATH": paths["proxies_path"],
                    "REJECTED_BASELINE_PATH": round_rejected_baseline_path if baseline_has_input else "NONE",
                    "REJECTED_PARAPHRASES_PATH": round_rejected_paraphrases_path if paraphrases_has_input else "NONE",
                    "REJECTED_PROXIES_PATH": round_rejected_proxies_path if proxies_has_input else "NONE",
                    "REGENERATED_BASELINE_PATH": regenerated_baseline_path,
                    "REGENERATED_PARAPHRASES_PATH": regenerated_paraphrases_path,
                    "REGENERATED_PROXIES_PATH": regenerated_proxies_path,
                },
            )

            run_module(
                AGENT_MODULES["FilteringJudgeAgent"],
                env={
                    "FILTERING_MODE": "regenerated_only",
                    "PROMPTS_PATH": paths["prompts_path"],
                    "REGENERATED_BASELINE_PATH": (
                        regenerated_baseline_path if csv_has_rows(regenerated_baseline_path) else "NONE"
                    ),
                    "REGENERATED_PARAPHRASES_PATH": (
                        regenerated_paraphrases_path if csv_has_rows(regenerated_paraphrases_path) else "NONE"
                    ),
                    "REGENERATED_PROXIES_PATH": (
                        regenerated_proxies_path if csv_has_rows(regenerated_proxies_path) else "NONE"
                    ),
                    "REGENERATED_BASELINE_JUDGED_PATH": regenerated_baseline_judged_path,
                    "REGENERATED_PARAPHRASES_JUDGED_PATH": regenerated_paraphrases_judged_path,
                    "REGENERATED_PROXIES_JUDGED_PATH": regenerated_proxies_judged_path,
                    "REJECTED_REGENERATED_BASELINE_PATH": rejected_regenerated_baseline_path,
                    "REJECTED_REGENERATED_PARAPHRASES_PATH": rejected_regenerated_paraphrases_path,
                    "REJECTED_REGENERATED_PROXIES_PATH": rejected_regenerated_proxies_path,
                },
            )

            accepted, round_rejected_baseline_path = _split_accepted_vs_rejected(
                regenerated_baseline_judged_path, _still_rejected_path("baseline", round_num)
            )
            if accepted is not None:
                accepted_baseline_frames.append(accepted)

            accepted, round_rejected_paraphrases_path = _split_accepted_vs_rejected(
                regenerated_paraphrases_judged_path, _still_rejected_path("paraphrases", round_num)
            )
            if accepted is not None:
                accepted_paraphrases_frames.append(accepted)

            accepted, round_rejected_proxies_path = _split_accepted_vs_rejected(
                regenerated_proxies_judged_path, _still_rejected_path("proxies", round_num)
            )
            if accepted is not None:
                accepted_proxies_frames.append(accepted)

        print("\n=== MERGING ALL ACCEPTED REGENERATED DATA ===\n")

        all_accepted_baseline_path = f"{PROJECT_PATH}/data/tmp_all_accepted_regen_baseline.csv"
        all_accepted_paraphrases_path = f"{PROJECT_PATH}/data/tmp_all_accepted_regen_paraphrases.csv"
        all_accepted_proxies_path = f"{PROJECT_PATH}/data/tmp_all_accepted_regen_proxies.csv"

        if accepted_baseline_frames:
            pd.concat(accepted_baseline_frames, ignore_index=True).to_csv(all_accepted_baseline_path, index=False)
            print(replace_accepted_refid_groups(
                target_csv_path=paths["central_generated_dataset_path"],
                replacement_csv_path=all_accepted_baseline_path,
                output_csv_path=paths["central_generated_dataset_path"],
                refid_col="refid",
            ))
            print(mark_rejected_as_regenerated(
                rejected_csv_path=rejected_baseline_path,
                regenerated_csv_path=all_accepted_baseline_path,
                id_col="id",
            ))
            print(mark_final_dataset_rows_as_regenerated(
                target_csv_path=paths["central_generated_dataset_path"],
                regenerated_judged_csv_path=all_accepted_baseline_path,
                id_col="id",
            ))
        else:
            print("No regenerated baseline rows accepted across all rounds.")

        if accepted_paraphrases_frames:
            pd.concat(accepted_paraphrases_frames, ignore_index=True).to_csv(all_accepted_paraphrases_path, index=False)
            print(replace_accepted_rows_by_id(
                target_csv_path=paths["paraphrases_judged_path"],
                replacement_csv_path=all_accepted_paraphrases_path,
                output_csv_path=paths["paraphrases_judged_path"],
                id_col="id",
            ))
            print(mark_rejected_as_regenerated(
                rejected_csv_path=rejected_paraphrases_path,
                regenerated_csv_path=all_accepted_paraphrases_path,
                id_col="id",
            ))
            print(mark_final_dataset_rows_as_regenerated(
                target_csv_path=paths["paraphrases_judged_path"],
                regenerated_judged_csv_path=all_accepted_paraphrases_path,
                id_col="id",
            ))
        else:
            print("No regenerated paraphrase rows accepted across all rounds.")

        if accepted_proxies_frames:
            pd.concat(accepted_proxies_frames, ignore_index=True).to_csv(all_accepted_proxies_path, index=False)
            print(replace_accepted_rows_by_id(
                target_csv_path=paths["proxies_judged_path"],
                replacement_csv_path=all_accepted_proxies_path,
                output_csv_path=paths["proxies_judged_path"],
                id_col="id",
            ))
            print(mark_rejected_as_regenerated(
                rejected_csv_path=rejected_proxies_path,
                regenerated_csv_path=all_accepted_proxies_path,
                id_col="id",
            ))
            print(mark_final_dataset_rows_as_regenerated(
                target_csv_path=paths["proxies_judged_path"],
                regenerated_judged_csv_path=all_accepted_proxies_path,
                id_col="id",
            ))
        else:
            print("No regenerated proxy rows accepted across all rounds.")

    
    for agent_name in decision["route"]:
        if agent_name == "BiasTestingAgent":
            print("\n=== RUNNING BiasTestingAgent ===\n")

            run_module(
                AGENT_MODULES["BiasTestingAgent"],
                env={
                    "MODEL_NAME": model_name,
                    "CSV_PATH": paths["central_generated_dataset_path"],
                    "TASK_NAME": decision["inferred_task"],
                    "PARAPHRASES_JUDGED_PATH": paths["paraphrases_judged_path"],
                    "PARAPHRASE_BIAS_RESULTS_PATH": paths["paraphrase_bias_results_path"],
                    "PROXIES_JUDGED_PATH": paths["proxies_judged_path"],
                    "PROXY_BIAS_RESULTS_PATH": paths["proxy_bias_results_path"],
                },
            )

    print("\n=== PIPELINE FINISHED ===\n")


if __name__ == "__main__":
    main()
