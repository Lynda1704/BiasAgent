

import os
import sys
import json
import subprocess
import pandas as pd


PROJECT_PATH = os.environ.get(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def _load_api_keys_from_colab_secrets():
    try:
        from google.colab import userdata
    except ImportError:
        return  


    for key_name in ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]:
        if key_name in os.environ:
            continue  

        try:
            os.environ[key_name] = userdata.get(key_name)
            print(f"[run_consensus_evaluation] {key_name} loaded from Colab Secrets")
        except Exception:
            print(f"[run_consensus_evaluation] {key_name} not found in Colab Secrets (add it if needed)")


_load_api_keys_from_colab_secrets()
sys.path.insert(0, PROJECT_PATH)

REPORTS_DIR = os.path.join(PROJECT_PATH, "reports")
METRIC_CHOICES_PATH = os.path.join(PROJECT_PATH, "data", "metric_choices.csv")

LLM_CONFIGS = [
    {"label": "GPT-5", "AGENT_PROVIDER": "openai", "AGENT_LLM": "gpt-5"},
    {"label": "Claude Sonnet 5", "AGENT_PROVIDER": "openrouter", "AGENT_LLM": "anthropic/claude-sonnet-5"},
    {"label": "DeepSeek V4 Pro", "AGENT_PROVIDER": "openrouter", "AGENT_LLM": "deepseek/deepseek-v4-pro"},
    {"label": "Gemini 3.1 Pro", "AGENT_PROVIDER": "openrouter", "AGENT_LLM": "google/gemini-3.1-pro-preview"},
]


SCORING_FUNCTIONS = [
    "F1_Score", "Accuracy_Score", "Precision_Score", "Recall_Score",
    "ClassPrediction_Score", "AverageClassPrediction_Score",
    "AverageLikelihood_Score", "FailureRate",
    "AverageF1ForQA_Score", "AverageExactMatchForQA_Score",
]
DISTANCE_FUNCTIONS = [
    "AbsoluteDistance", "WassersteinDistance",
    "AbsoluteDivergenceFromExpectedOutcome",
    "PerGroupAbsoluteDivergenceFromExpectedOutcome", "FailureDistance",
]
CONTRASTING_METHODS = ["pcm", "bcm", "vbcm", "mcm", "vmcm", "ncm"]
FAIRNESS_IDIOMS = ["counterfactual", "group"]


AGREEMENT_FIELDS = ["scoring_function", "distance_function", "fairness_idiom", "contrasting_method"]


def _sanitize_model_name(model_name: str) -> str:
    return model_name.strip().replace("/", "_")


def _run_pipeline_once(model_name: str, agent_provider: str, agent_llm: str):
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["USER_INTENT"] = "evaluate_model"
    env["AGENT_PROVIDER"] = agent_provider
    env["AGENT_LLM"] = agent_llm
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        ["python", "-u", "-m", "agent.pipeline_runner"],
        cwd=PROJECT_PATH,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    for line in process.stdout:
        lines.append(line)
    process.wait()

    if process.returncode != 0:
        full_output = "".join(lines)
        tail = full_output[-3000:] if len(full_output) > 3000 else full_output


        log_path = os.path.join(
            PROJECT_PATH,
            "data",
            f"last_failed_run_{_sanitize_model_name(agent_llm)}.log",
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(full_output)

        raise RuntimeError(
            f"Pipeline failed for {agent_llm} (exit code {process.returncode})\n\n"
            f"--- Derniere sortie du sous-processus (tronquee - voir {log_path} pour le log complet) ---\n{tail}"
        )

    return "".join(lines)


def _get_last_metric_choice(model_name: str):
    df = pd.read_csv(METRIC_CHOICES_PATH)
    matches = df[df["model_under_test"] == model_name]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def _find_report(model_name: str):
    folder = os.path.join(REPORTS_DIR, _sanitize_model_name(model_name))
    if not os.path.isdir(folder):
        return None
    for task_dir in sorted(os.listdir(folder)):
        summary_path = os.path.join(folder, task_dir, "model_bias_report_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def run_four_llm_evaluation(model_name: str):
    choices = []

    for cfg in LLM_CONFIGS:
        yield f"Running evaluation with {cfg['label']}..."

        _run_pipeline_once(model_name, cfg["AGENT_PROVIDER"], cfg["AGENT_LLM"])
        choice = _get_last_metric_choice(model_name)

        if choice is None:
            raise RuntimeError(f"Aucun choix de metrique trouve apres le run avec {cfg['label']}")

        choices.append({
            "llm": cfg["label"],
            "scoring_function": choice.get("scoring_function"),
            "distance_function": choice.get("distance_function"),
            "fairness_idiom": choice.get("fairness_idiom"),
            "contrasting_method": choice.get("contrasting_method"),
            "target_label": choice.get("target_label"),
        })

        yield f"{cfg['label']} done."


    reference = {k: choices[0][k] for k in AGREEMENT_FIELDS}
    consensus = all(
        {k: c[k] for k in AGREEMENT_FIELDS} == reference
        for c in choices[1:]
    )

    report = _find_report(model_name) if consensus else None

    yield {
        "consensus": consensus,
        "choices": choices,
        "report": report,
    }


def compute_with_manual_config(model_name: str, task_name: str, scoring_function: str,
                                distance_function: str, fairness_idiom: str,
                                contrasting_method: str, target_label: str | None):
    from mcp_server.server import (
        compute_bias_score_with_config,
        compute_paraphrase_bias_after_counterfactual_filter,
        compute_proxy_bias_after_counterfactual_filter,
        generate_bias_report,
    )

    data_dir = os.path.join(PROJECT_PATH, "data")
    csv_path = os.path.join(data_dir, "baseline_generated_judged.csv")

    target_label = target_label if target_label not in ("", "null", "None") else None

    counterfactual_result = compute_bias_score_with_config(
        model_name=model_name,
        csv_path=csv_path,
        task_name=task_name,
        scoring_function=scoring_function,
        distance_function=distance_function,
        fairness_idiom=fairness_idiom,
        contrasting_method=contrasting_method,
        target_label=target_label,
        threshold=0.2,
        max_rows=200,
        output_csv_path=os.path.join(data_dir, "counterfactual_bias_results.csv"),
    )

    paraphrase_result = compute_paraphrase_bias_after_counterfactual_filter(
        model_name=model_name,
        counterfactual_dataset_path=csv_path,
        paraphrases_judged_path=os.path.join(data_dir, "baseline_paraphrases_judged.csv"),
        output_csv_path=os.path.join(data_dir, "paraphrase_bias_results.csv"),
        task_name=task_name,
        scoring_function=scoring_function,
        distance_function=distance_function,
        fairness_idiom=fairness_idiom,
        contrasting_method=contrasting_method,
        target_label=target_label,
        threshold=0.2,
        max_rows=200,
    )

    proxy_result = compute_proxy_bias_after_counterfactual_filter(
        model_name=model_name,
        counterfactual_dataset_path=csv_path,
        proxies_judged_path=os.path.join(data_dir, "baseline_proxies_judged.csv"),
        output_csv_path=os.path.join(data_dir, "proxy_bias_results.csv"),
        task_name=task_name,
        scoring_function=scoring_function,
        distance_function=distance_function,
        fairness_idiom=fairness_idiom,
        contrasting_method=contrasting_method,
        target_label=target_label,
        threshold=0.2,
        max_rows=200,
    )

    report_result = generate_bias_report(
        model_name=model_name,
        task_name=task_name,
        counterfactual_results_path=os.path.join(data_dir, "counterfactual_bias_results.csv"),
        paraphrase_results_path=os.path.join(data_dir, "paraphrase_bias_results.csv"),
        proxy_results_path=os.path.join(data_dir, "proxy_bias_results.csv"),
        output_dir=REPORTS_DIR,
    )

    return report_result
