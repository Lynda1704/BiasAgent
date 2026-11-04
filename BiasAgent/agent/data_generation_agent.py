import os
import sys
import json
import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


PROJECT_PATH = os.environ.get(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

TASK_NAME = os.environ["TASK_NAME"]

TERMS_CSV_PATH = os.environ["TERMS_CSV_PATH"]
PROMPTS_PATH = os.environ["PROMPTS_PATH"]
TASK_DEFINITIONS_PATH = os.environ["TASK_DEFINITIONS_PATH"]

BIAS_TYPES_PATH = os.environ.get(
    "BIAS_TYPES_PATH",
    f"{PROJECT_PATH}/mcp_server/resources/bias_types.json"
)

GENERATED_SAMPLES_PATH = os.environ["GENERATED_SAMPLES_PATH"]
LATEST_GENERATED_SAMPLES_PATH = os.environ["LATEST_GENERATED_SAMPLES_PATH"]

COUNTERFACTUAL_SAMPLES_PATH = os.environ["COUNTERFACTUAL_SAMPLES_PATH"]
BASELINE_GENERATED_PATH = os.environ["BASELINE_GENERATED_PATH"]
LATEST_BASELINE_GENERATED_PATH = os.environ["LATEST_BASELINE_GENERATED_PATH"]

CENTRAL_GENERATED_DATASET_PATH = os.environ.get(
    "CENTRAL_GENERATED_DATASET_PATH",
    f"{PROJECT_PATH}/data/baseline_generated_judged.csv"
)

DATA_GENERATION_MODE = os.environ.get("DATA_GENERATION_MODE", "full")

PARAPHRASES_PATH = os.environ.get(
    "PARAPHRASES_PATH",
    f"{PROJECT_PATH}/data/baseline_paraphrases.csv"
)
PROXIES_PATH = os.environ.get(
    "PROXIES_PATH",
    f"{PROJECT_PATH}/data/baseline_proxies.csv"
)
REJECTED_BASELINE_PATH = os.environ.get(
    "REJECTED_BASELINE_PATH",
    f"{PROJECT_PATH}/data/rejected_baseline_for_regeneration.csv",
)

REJECTED_PARAPHRASES_PATH = os.environ.get(
    "REJECTED_PARAPHRASES_PATH",
    f"{PROJECT_PATH}/data/rejected_paraphrases_for_regeneration.csv",
)

REJECTED_PROXIES_PATH = os.environ.get(
    "REJECTED_PROXIES_PATH",
    f"{PROJECT_PATH}/data/rejected_proxies_for_regeneration.csv",
)

REGENERATED_BASELINE_PATH = os.environ.get(
    "REGENERATED_BASELINE_PATH",
    f"{PROJECT_PATH}/data/regenerated_baseline.csv",
)

REGENERATED_PARAPHRASES_PATH = os.environ.get(
    "REGENERATED_PARAPHRASES_PATH",
    f"{PROJECT_PATH}/data/regenerated_paraphrases.csv",
)

REGENERATED_PROXIES_PATH = os.environ.get(
    "REGENERATED_PROXIES_PATH",
    f"{PROJECT_PATH}/data/regenerated_proxies.csv",
)
DATA_GENERATION_INSTRUCTIONS = """Role: You are the DataGenerationAgent. You generate task samples, the
counterfactual baseline, paraphrases and proxies, and you regenerate
rejected rows after filtering feedback - depending on data_generation_mode.
You never judge or filter data yourself, and you never compute bias.

Available MCP Tools: generate_task_samples_with_llm,
generate_counterfactual_variants_with_llm, generate_paraphrases_with_llm,
generate_proxy_sentences_with_llm, regenerate_rejected_baseline_with_llm,
regenerate_rejected_paraphrases_with_llm, regenerate_rejected_proxies_with_llm

Instructions - run only the steps for the given data_generation_mode, in order:

full:
1. generate_task_samples_with_llm(terms_csv_path, task_definitions_path,
   prompts_path, task_name, output_path=generated_samples_path)
2. generate_counterfactual_variants_with_llm(input_csv_path=
   latest_generated_samples_path, bias_types_path, prompts_path,
   counterfactual_output_csv_path=counterfactual_samples_path,
   baseline_output_csv_path=baseline_generated_path)
3. generate_paraphrases_with_llm(input_csv_path=
   latest_baseline_generated_path, prompts_path, output_csv_path=
   paraphrases_path, task_name)
4. generate_proxy_sentences_with_llm(input_csv_path=
   latest_baseline_generated_path, prompts_path, output_csv_path=
   proxies_path, task_name)

paraphrase_only: run step 3 only, using central_generated_dataset_path as
input_csv_path instead of latest_baseline_generated_path.
proxy_only: run step 4 only, using central_generated_dataset_path as
input_csv_path instead of latest_baseline_generated_path.
paraphrase_and_proxy_only: run both of the above (with
central_generated_dataset_path as input), paraphrase then proxy.

regenerate_rejected_once: for each of the three stages below, run the call
only if its rejected_*_path value is not the literal string "NONE";
"NONE" means that stage has nothing to regenerate, skip it entirely:
- regenerate_rejected_baseline_with_llm(rejected_csv_path=
  rejected_baseline_path, prompts_path, output_csv_path=
  regenerated_baseline_path)
- regenerate_rejected_paraphrases_with_llm(rejected_csv_path=
  rejected_paraphrases_path, prompts_path, output_csv_path=
  regenerated_paraphrases_path)
- regenerate_rejected_proxies_with_llm(rejected_csv_path=
  rejected_proxies_path, prompts_path, output_csv_path=
  regenerated_proxies_path)

none: do nothing.

Output Format: end with a short JSON summary -
{"ok": true, "data_generation_mode": "...", "paths_written": ["..."]}
"""


def build_data_generation_query(mode: str) -> str:
    return f"""data_generation_mode: {mode}
task_name: {TASK_NAME}
terms_csv_path: {TERMS_CSV_PATH}
task_definitions_path: {TASK_DEFINITIONS_PATH}
bias_types_path: {BIAS_TYPES_PATH}
prompts_path: {PROMPTS_PATH}
generated_samples_path: {GENERATED_SAMPLES_PATH}
latest_generated_samples_path: {LATEST_GENERATED_SAMPLES_PATH}
counterfactual_samples_path: {COUNTERFACTUAL_SAMPLES_PATH}
baseline_generated_path: {BASELINE_GENERATED_PATH}
latest_baseline_generated_path: {LATEST_BASELINE_GENERATED_PATH}
central_generated_dataset_path: {CENTRAL_GENERATED_DATASET_PATH}
paraphrases_path: {PARAPHRASES_PATH}
proxies_path: {PROXIES_PATH}
rejected_baseline_path: {REJECTED_BASELINE_PATH}
rejected_paraphrases_path: {REJECTED_PARAPHRASES_PATH}
rejected_proxies_path: {REJECTED_PROXIES_PATH}
regenerated_baseline_path: {REGENERATED_BASELINE_PATH}
regenerated_paraphrases_path: {REGENERATED_PARAPHRASES_PATH}
regenerated_proxies_path: {REGENERATED_PROXIES_PATH}
"""


async def main():
    if DATA_GENERATION_MODE not in (
        "full", "paraphrase_only", "proxy_only",
        "paraphrase_and_proxy_only", "regenerate_rejected_once", "none",
    ):
        raise ValueError(
            f"Unknown DATA_GENERATION_MODE: {DATA_GENERATION_MODE}. "
            "Expected: full, paraphrase_only, proxy_only, "
            "paraphrase_and_proxy_only, regenerate_rejected_once, or none."
        )

    server = MCPServerStdio(
        params={
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": PROJECT_PATH,
        },
        client_session_timeout_seconds=14400,
    )

    async with server:
        agent = Agent(
            name="DataGenerationAgent",
            instructions=DATA_GENERATION_INSTRUCTIONS,
            mcp_servers=[server],
            model="gpt-5",
        )

        user_prompt = build_data_generation_query(DATA_GENERATION_MODE)

        result = await Runner.run(
            agent,
            user_prompt,
            max_turns=15,
        )

        print("\n=== DATA GENERATION AGENT OUTPUT ===\n")
        print(result.final_output)

        
        generation_ok = True
        try:
            text = str(result.final_output)
            end = text.rfind("}")
            start = text.rfind("{", 0, end)
            if start != -1 and end > start:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, dict) and "ok" in parsed:
                    generation_ok = bool(parsed["ok"])
        except Exception:
            pass

        if not generation_ok:
            print("\n=== DATA GENERATION AGENT FAILED - STOPPING PIPELINE ===\n")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
