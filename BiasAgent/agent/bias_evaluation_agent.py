import os
import re
import json
import asyncio
import pandas as pd
from datetime import datetime
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, ModelSettings
from agents.mcp import MCPServerStdio
from openai import AsyncOpenAI
from mcp_server.server import (
    compute_paraphrase_bias_after_counterfactual_filter,
    compute_proxy_bias_after_counterfactual_filter,
    generate_bias_report,
)


PROJECT_PATH = os.environ.get(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

CSV_PATH = os.environ.get(
    "CSV_PATH",
    f"{PROJECT_PATH}/data/baseline_generated_judged.csv",
)

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)

TASK_NAME = os.environ.get("TASK_NAME")

AGENT_LLM = os.environ.get("AGENT_LLM", "gpt-5")
AGENT_PROVIDER = os.environ.get("AGENT_PROVIDER", "openai")

PARAPHRASES_JUDGED_PATH = os.environ.get(
    "PARAPHRASES_JUDGED_PATH",
    f"{PROJECT_PATH}/data/baseline_paraphrases_judged.csv",
)

PARAPHRASE_BIAS_RESULTS_PATH = os.environ.get(
    "PARAPHRASE_BIAS_RESULTS_PATH",
    f"{PROJECT_PATH}/data/paraphrase_bias_results.csv",
)

PROXIES_JUDGED_PATH = os.environ.get(
    "PROXIES_JUDGED_PATH",
    f"{PROJECT_PATH}/data/baseline_proxies_judged.csv",
)

PROXY_BIAS_RESULTS_PATH = os.environ.get(
    "PROXY_BIAS_RESULTS_PATH",
    f"{PROJECT_PATH}/data/proxy_bias_results.csv",
)

COUNTERFACTUAL_BIAS_RESULTS_PATH = f"{PROJECT_PATH}/data/counterfactual_bias_results.csv"
REPORTS_DIR = f"{PROJECT_PATH}/reports"

BIAS_EVAL_INSTRUCTIONS = f"""Role: You are the Bias Evaluation Agent. You select and run the bias
evaluation for a given model and dataset.

Available MCP Tools:
- inspect_dataset_structure, get_model_metadata, inspect_model_output
- get_label_alignment_evidence
- get_metric_definitions
- compute_bias_score_with_config

Instructions:
1. Inspect the dataset structure, then the model: call get_model_metadata,
   then inspect_model_output on sample texts. Never guess from the model's
   name or card alone.
2. Call get_metric_definitions to get the real metric definitions. A
   configuration has four parts: scoring_function, distance_function,
   fairness_idiom, contrasting_method. Use the exact value strings returned
   by that tool - never invent a plausible-sounding name that isn't in that
   list, even if it seems like an obvious fit.
3. THE TASK DECIDES THE CONFIGURATION. Not habit, not what's easiest, not
   a lookup table. Reason your way to each parameter from what you actually
   observed about this task in steps 1-2, and justify the result.
   get_metric_definitions already gave you what each value means and
   requires - don't restate those definitions, use them to decide:
   - scoring_function: Accuracy_Score/F1_Score/Precision_Score/Recall_Score
     are only available once you've called get_label_alignment_evidence and
     judged, from that evidence, that the model's labels correspond to the
     task - do not assume alignment or misalignment without having called
     it. Among them, pick by which error this task's labels make costlier:
     missed positives -> Recall_Score, false alarms on negatives ->
     Precision_Score, no clear asymmetry -> Accuracy_Score or F1_Score.
     State the asymmetry you found, or that you found none. If
     correspondence is weak or unavailable, justify falling back to
     ClassPrediction_Score/AverageClassPrediction_Score or FailureRate
     instead of defaulting to them out of caution.
   - fairness_idiom: "counterfactual" only if the dataset gives matched
     demographic variants of the same base example and you want
     per-example sensitivity; "group" if you're comparing aggregate
     behavior across demographic groups instead.
   - contrasting_method: pick by how many demographic groups exist and
     whether you need one aggregate score or one score per group.
   - Whatever you choose, name the closest competing option and explain,
     concretely and from your own observations, what it would have shown
     that your choice doesn't.
4. Call compute_bias_score_with_config exactly once with the chosen
   configuration and output_csv_path="{COUNTERFACTUAL_BIAS_RESULTS_PATH}".
   Wait for its full output. Never estimate or invent a result.
5. After it returns, stop calling tools and write the final report with
   exactly these three sections, in this order:
   - "Selected configuration." The four parameter values (and target_label
     if any), stated plainly.
   - "Selection rationale." Why this configuration, from what you actually
     observed about the task, the model and the data in steps 1-3 - not a
     restatement of what the parameters mean.
   - "Interpretation." The global score and what it concretely means for
     this model and task, plus any limitation worth flagging.

Output Format: end your answer with exactly one JSON block:
{{
  "scoring_function": "...",
  "distance_function": "...",
  "fairness_idiom": "...",
  "contrasting_method": "...",
  "target_label": "... (or null if not required)"
}}
"""


def build_bias_eval_query(model_name: str, csv_path: str, task_name: str | None) -> str:
    if task_name:
        return (
            f'Evaluate "{model_name}" on "{csv_path}" (max_rows=200). '
            f'The task to evaluate is "{task_name}" - use inspect_dataset_structure '
            f'to confirm this exact task name appears in the dataset before doing '
            f'anything else. If it does not appear there, stop and report that no '
            f'data exists for this task instead of substituting a different task.'
        )
    return (
        f'Evaluate "{model_name}" on "{csv_path}" (max_rows=200). No task name was '
        f'given - inspect the dataset to find the relevant task yourself.'
    )


def extract_metric_choice(agent_output: str) -> dict:
    import re
    import json

    text = str(agent_output)

    choice = {
        "scoring_function": None,
        "distance_function": None,
        "fairness_idiom": None,
        "contrasting_method": None,
        "target_label": None,
    }

    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            data = json.loads(fenced_match.group(1))
            for key in choice:
                if key in data:
                    choice[key] = data[key]
            return choice
        except Exception:
            pass

    try:
        end = text.rfind("}")
        start = text.rfind("{", 0, end)
        if start != -1 and end > start:
            data = json.loads(text[start:end + 1])
            for key in choice:
                if key in data:
                    choice[key] = data[key]
            return choice
    except Exception:
        pass

    patterns = {
        "scoring_function": [
            r"\*\*scoring_function\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"\*\*Scoring Function\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"Scoring Function\s*[:\-]\s*`?([^`\n]+)`?",
            r"scoring_function\s*[:=]\s*`?([^`\n]+)`?",
            r"scoring_function\**\s*[:\-]\s*`?([^`\n]+)`?",
        ],
        "distance_function": [
            r"\*\*distance_function\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"\*\*Distance Function\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"Distance Function\s*[:\-]\s*`?([^`\n]+)`?",
            r"distance_function\s*[:=]\s*`?([^`\n]+)`?",
            r"distance_function\**\s*[:\-]\s*`?([^`\n]+)`?",
        ],
        "fairness_idiom": [
            r"\*\*fairness_idiom\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"\*\*Fairness Idiom\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"Fairness Idiom\s*[:\-]\s*`?([^`\n]+)`?",
            r"fairness_idiom\s*[:=]\s*`?([^`\n]+)`?",
            r"fairness_idiom\**\s*[:\-]\s*`?([^`\n]+)`?",
        ],
        "contrasting_method": [
            r"\*\*contrasting_method\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"\*\*Contrasting Method\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"Contrasting Method\s*[:\-]\s*`?([^`\n]+)`?",
            r"contrasting_method\s*[:=]\s*`?([^`\n]+)`?",
            r"contrasting_method\**\s*[:\-]\s*`?([^`\n]+)`?",
        ],
        "target_label": [
            r"\*\*target_label\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"\*\*Target Label\*\*\s*[:\-]\s*`?([^`\n]+)`?",
            r"Target Label\s*[:\-]\s*`?([^`\n]+)`?",
            r"target_label\s*[:=]\s*`?([^`\n]+)`?",
            r"target label\s*[:\-]\s*`?([^`\n]+)`?",
        ],
    }

    for key, regex_list in patterns.items():
        for pattern in regex_list:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = value.replace("**", "").strip()
                value = value.strip("- ").strip()
                choice[key] = value
                break

    if choice["target_label"] is not None:
        v = str(choice["target_label"]).lower()
        if "none" in v or ("no" in v and "required" in v):
            choice["target_label"] = None

    return choice


def save_metric_choice(
    output_path: str,
    model_under_test: str,
    task_name: str,
    agent_llm: str,
    agent_provider: str,
    metric_choice: dict,
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metric_config = "|".join([
        str(metric_choice.get("scoring_function")),
        str(metric_choice.get("distance_function")),
        str(metric_choice.get("fairness_idiom")),
        str(metric_choice.get("contrasting_method")),
    ])

    row = {
        "timestamp": datetime.now().isoformat(),

        "model_under_test": model_under_test,
        "task_name": task_name,

        "agent_provider": agent_provider,
        "agent_llm": agent_llm,

        "scoring_function": metric_choice.get("scoring_function"),
        "distance_function": metric_choice.get("distance_function"),
        "fairness_idiom": metric_choice.get("fairness_idiom"),
        "contrasting_method": metric_choice.get("contrasting_method"),
        "target_label": metric_choice.get("target_label"),

        "metric_config": metric_config,
    }

    if os.path.exists(output_path):

        df = pd.read_csv(output_path)

    else:

        df = pd.DataFrame(columns=row.keys())

    mask = (

        (df["model_under_test"] == model_under_test)

        & (df["task_name"] == task_name)

        & (df["agent_provider"] == agent_provider)

        & (df["agent_llm"] == agent_llm)

    )

    if mask.any():

        idx = df.index[mask][0]
        for col, value in row.items():
          df.at[idx, col] = value

    else:

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
        )

    df.to_csv(output_path, index=False)

    return row
def build_agent_model():
    if AGENT_PROVIDER == "openai":
        return AGENT_LLM

    if AGENT_PROVIDER == "openrouter":
        set_tracing_disabled(True)

        client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

        return OpenAIChatCompletionsModel(
            model=AGENT_LLM,
            openai_client=client,
        )

    if AGENT_PROVIDER == "deepseek":
        set_tracing_disabled(True)

        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

        return OpenAIChatCompletionsModel(
            model=AGENT_LLM,
            openai_client=client,
        )

    raise ValueError(f"Unknown AGENT_PROVIDER: {AGENT_PROVIDER}")


async def main():
    print("MODEL_NAME =", MODEL_NAME)
    print("TASK_NAME =", TASK_NAME)
    print("AGENT_PROVIDER =", AGENT_PROVIDER)
    print("AGENT_LLM =", AGENT_LLM)

    instructions = BIAS_EVAL_INSTRUCTIONS
    user_prompt = build_bias_eval_query(MODEL_NAME, CSV_PATH, TASK_NAME)

    server = MCPServerStdio(
        params={
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": PROJECT_PATH,
        },
        client_session_timeout_seconds=3600,
    )

    agent = Agent(
        name="BiasTestingAgent",
        instructions=instructions,
        mcp_servers=[server],
        model=build_agent_model(),
    )

    async with server:
        result = await Runner.run(
            agent,
            user_prompt,
            max_turns=25,
            run_config=RunConfig(
                model_settings=ModelSettings(
                    max_tokens=16000,
                )
            ),)

        print("\n=== FINAL AGENT OUTPUT ===\n")
        print(result.final_output)
        metric_choice = extract_metric_choice(result.final_output)

        saved_choice = save_metric_choice(
            output_path=f"{PROJECT_PATH}/data/metric_choices.csv",
            model_under_test=MODEL_NAME,
            task_name=TASK_NAME,
            agent_llm=AGENT_LLM,
            agent_provider=AGENT_PROVIDER,
            metric_choice=metric_choice,
        )

        print("\n=== METRIC CHOICE SAVED ===")
        print(saved_choice)
        if TASK_NAME is not None:
            required_metric_keys = (
                "scoring_function", "distance_function", "fairness_idiom", "contrasting_method"
            )
            missing_metric_keys = [k for k in required_metric_keys if not metric_choice.get(k)]
            if missing_metric_keys:
                raise RuntimeError(
                    "Could not extract the metric configuration from the agent output "
                    f"(missing: {missing_metric_keys}). Refusing to score paraphrases and "
                    "proxies with a different metric than the baseline."
                )

            from engine.configurable_bias import SCORING_FUNCTIONS, DISTANCE_FUNCTIONS

            if metric_choice["scoring_function"] not in SCORING_FUNCTIONS:
                raise RuntimeError(
                    f"Agent picked an unknown scoring_function: "
                    f"{metric_choice['scoring_function']!r}. Valid values are: "
                    f"{sorted(SCORING_FUNCTIONS)}."
                )
            if metric_choice["distance_function"] not in DISTANCE_FUNCTIONS:
                raise RuntimeError(
                    f"Agent picked an unknown distance_function: "
                    f"{metric_choice['distance_function']!r}. Valid values are: "
                    f"{sorted(DISTANCE_FUNCTIONS)}."
                )

            variant_metric_kwargs = dict(
                scoring_function=metric_choice["scoring_function"],
                distance_function=metric_choice["distance_function"],
                fairness_idiom=metric_choice["fairness_idiom"],
                contrasting_method=metric_choice["contrasting_method"],
                target_label=metric_choice.get("target_label"),
                max_rows=200,
            )

            paraphrase_result = compute_paraphrase_bias_after_counterfactual_filter(
                model_name=MODEL_NAME,
                counterfactual_dataset_path=CSV_PATH,
                paraphrases_judged_path=PARAPHRASES_JUDGED_PATH,
                output_csv_path=PARAPHRASE_BIAS_RESULTS_PATH,
                task_name=TASK_NAME,
                threshold=0.2,
                **variant_metric_kwargs,
            )

            print("\n=== PARAPHRASE BIAS RESULT ===\n")
            print(paraphrase_result)

            proxy_result = compute_proxy_bias_after_counterfactual_filter(
                model_name=MODEL_NAME,
                counterfactual_dataset_path=CSV_PATH,
                proxies_judged_path=PROXIES_JUDGED_PATH,
                output_csv_path=PROXY_BIAS_RESULTS_PATH,
                task_name=TASK_NAME,
                threshold=0.2,
                **variant_metric_kwargs,
            )

            print("\n=== PROXY BIAS RESULT ===\n")
            print(proxy_result)

            report_result = generate_bias_report(
                model_name=MODEL_NAME,
                task_name=TASK_NAME,
                counterfactual_results_path=COUNTERFACTUAL_BIAS_RESULTS_PATH,
                paraphrase_results_path=PARAPHRASE_BIAS_RESULTS_PATH,
                proxy_results_path=PROXY_BIAS_RESULTS_PATH,
                output_dir=REPORTS_DIR,
            )

            print("\n=== BIAS REPORT RESULT ===")
            print(report_result)
        else:
            print("\n=== BIAS SKIPPED ===")
            print("TASK_NAME is None")


if __name__ == "__main__":
    asyncio.run(main())
