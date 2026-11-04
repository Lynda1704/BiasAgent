import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


PROJECT_PATH = os.environ.get(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

CENTRAL_GENERATED_DATASET_PATH = f"{PROJECT_PATH}/data/baseline_generated_judged.csv"
PARAPHRASES_JUDGED_PATH = f"{PROJECT_PATH}/data/baseline_paraphrases_judged.csv"
PROXIES_JUDGED_PATH = f"{PROJECT_PATH}/data/baseline_proxies_judged.csv"

GENERATED_SAMPLES_PATH = f"{PROJECT_PATH}/data/generated_task_cases_llm.csv"
GENERATED_SAMPLES_JUDGED_PATH = f"{PROJECT_PATH}/data/generated_task_cases_judged.csv"
COUNTERFACTUAL_SAMPLES_PATH = f"{PROJECT_PATH}/data/counterfactual_samples.csv"
BASELINE_GENERATED_PATH = f"{PROJECT_PATH}/data/baseline_generated.csv"
TMP_BASELINE_JUDGED_PATH = f"{PROJECT_PATH}/data/tmp_baseline_generated_judged.csv"

PROMPTS_PATH = f"{PROJECT_PATH}/mcp_server/resources/prompts.json"
TASK_DEFINITIONS_PATH = f"{PROJECT_PATH}/mcp_server/resources/task_definitions.json"
TERMS_CSV_PATH = f"{PROJECT_PATH}/data/terms_catalog.csv"

PARAPHRASES_PATH = f"{PROJECT_PATH}/data/baseline_paraphrases.csv"
PARAPHRASE_BIAS_RESULTS_PATH = f"{PROJECT_PATH}/data/paraphrase_bias_results.csv"
PROXIES_PATH = f"{PROJECT_PATH}/data/baseline_proxies.csv"
PROXY_BIAS_RESULTS_PATH = f"{PROJECT_PATH}/data/proxy_bias_results.csv"

MODEL_NAME = os.environ.get("MODEL_NAME", "j-hartmann/emotion-english-distilroberta-base")
USER_INTENT = "evaluate_model"


def ensure_central_dataset_exists(path: str) -> None:

    if os.path.exists(path):
        return

    import pandas as pd

    columns = [
        "id", "bias_type", "label", "task", "id_term", "concept_term",
        "sentence", "flagged", "refid", "baseline_judge_valid",
        "baseline_judge_reason", "filter_status", "filter_reason",
        "regeneration_attempt", "source_rejected_id", "was_regenerated",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    print(f"[ensure_central_dataset_exists] created empty {path}")

ORCHESTRATOR_INSTRUCTIONS = """Role: You are the OrchestratorAgent. You decide which agent runs next; you
never generate data, judge samples, compute bias, select metrics, or
instantiate other agents yourself.

Available MCP Tools: inspect_hf_model_task, inspect_dataset_structure,
check_task_rows_in_csv

Instructions:
1. Call inspect_hf_model_task(model_name) and use its task_name exactly as
   inferred_task. Never call suggest_task, never override it.
2. Call inspect_dataset_structure(central_generated_dataset_path) and check
   whether inferred_task appears in task_counts.
3. Call check_task_rows_in_csv(paraphrases_judged_path, inferred_task) ->
   paraphrases_exist_for_task = has_task_rows.
4. Call check_task_rows_in_csv(proxies_judged_path, inferred_task) ->
   proxies_exist_for_task = has_task_rows. Never guess either from file
   names or from inspect_dataset_structure.
5. Decide route / data_generation_mode / filtering_mode:
   - user_intent=generate_data ->
     route=[DataGenerationAgent, FilteringJudgeAgent], mode=full/full
   - user_intent=evaluate_model:
     - task missing from dataset ->
       route=[DataGenerationAgent, FilteringJudgeAgent, BiasTestingAgent],
       mode=full/full
     - task exists, no paraphrases, no proxies -> mode=paraphrase_and_proxy_only
     - task exists, paraphrases only missing -> mode=proxy_only
     - task exists, proxies only missing -> mode=paraphrase_only
     - task exists, both already present ->
       route=[BiasTestingAgent], mode=none/none
   (mode = data_generation_mode/filtering_mode; route always starts with
   DataGenerationAgent+FilteringJudgeAgent unless mode is none)
6. Never call generation, judging, or bias-evaluation tools yourself.

Output Format: one JSON object only - echo every path below exactly as
given, unchanged, under "paths":
{
  "route": ["AgentName", ...],
  "data_generation_mode": "full"|"paraphrase_only"|"proxy_only"|"paraphrase_and_proxy_only"|"none",
  "filtering_mode": "full"|"paraphrase_only"|"proxy_only"|"paraphrase_and_proxy_only"|"none",
  "inferred_task": "...",
  "reason": "...",
  "model_name": "...",
  "paths": {
    "generated_samples_path": "...",
    "generated_samples_judged_path": "...",
    "counterfactual_samples_path": "...",
    "baseline_generated_path": "...",
    "tmp_baseline_judged_path": "...",
    "central_generated_dataset_path": "...",
    "prompts_path": "...",
    "task_definitions_path": "...",
    "terms_csv_path": "...",
    "paraphrases_path": "...",
    "paraphrases_judged_path": "...",
    "paraphrase_bias_results_path": "...",
    "proxies_path": "...",
    "proxies_judged_path": "...",
    "proxy_bias_results_path": "..."
  }
}
"""


def build_orchestrator_query(model_name: str, user_intent: str) -> str:
    return f"""model_name: {model_name}
user_intent: {user_intent}
central_generated_dataset_path: {CENTRAL_GENERATED_DATASET_PATH}
paraphrases_judged_path: {PARAPHRASES_JUDGED_PATH}
proxies_judged_path: {PROXIES_JUDGED_PATH}
generated_samples_path: {GENERATED_SAMPLES_PATH}
generated_samples_judged_path: {GENERATED_SAMPLES_JUDGED_PATH}
counterfactual_samples_path: {COUNTERFACTUAL_SAMPLES_PATH}
baseline_generated_path: {BASELINE_GENERATED_PATH}
tmp_baseline_judged_path: {TMP_BASELINE_JUDGED_PATH}
prompts_path: {PROMPTS_PATH}
task_definitions_path: {TASK_DEFINITIONS_PATH}
terms_csv_path: {TERMS_CSV_PATH}
paraphrases_path: {PARAPHRASES_PATH}
paraphrase_bias_results_path: {PARAPHRASE_BIAS_RESULTS_PATH}
proxies_path: {PROXIES_PATH}
proxy_bias_results_path: {PROXY_BIAS_RESULTS_PATH}
"""


async def main():
    ensure_central_dataset_exists(CENTRAL_GENERATED_DATASET_PATH)

    print("CENTRAL_GENERATED_DATASET_PATH =", CENTRAL_GENERATED_DATASET_PATH)
    print("CENTRAL FILE EXISTS =", os.path.exists(CENTRAL_GENERATED_DATASET_PATH))

    server = MCPServerStdio(
        params={
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": PROJECT_PATH,
        },
        client_session_timeout_seconds=60,
    )

    async with server:
        orchestrator_agent = Agent(
            name="OrchestratorAgent",
            model="gpt-5",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            mcp_servers=[server],
        )

        user_prompt = build_orchestrator_query(MODEL_NAME, USER_INTENT)

        result = await Runner.run(
            orchestrator_agent,
            user_prompt,
            max_turns=10,
        )

        print("\n=== ORCHESTRATOR ROUTING DECISION ===\n")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
