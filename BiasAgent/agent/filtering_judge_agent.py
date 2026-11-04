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

TASK_NAME = os.environ.get("TASK_NAME")
FILTERING_MODE = os.environ.get("FILTERING_MODE", "full")

PROMPTS_PATH = os.environ.get(
    "PROMPTS_PATH",
    f"{PROJECT_PATH}/mcp_server/resources/prompts.json",
)

LATEST_BASELINE_GENERATED_PATH = os.environ.get(
    "LATEST_BASELINE_GENERATED_PATH",
    f"{PROJECT_PATH}/data/baseline_generated_latest.csv",
)

TMP_BASELINE_JUDGED_PATH = os.environ.get(
    "TMP_BASELINE_JUDGED_PATH",
    f"{PROJECT_PATH}/data/tmp_baseline_generated_judged.csv",
)

TMP_BASELINE_DEDUPED_PATH = os.environ.get(
    "TMP_BASELINE_DEDUPED_PATH",
    f"{PROJECT_PATH}/data/tmp_baseline_generated_judged_deduped.csv",
)

CENTRAL_GENERATED_DATASET_PATH = os.environ.get(
    "CENTRAL_GENERATED_DATASET_PATH",
    f"{PROJECT_PATH}/data/baseline_generated_judged.csv",
)

PARAPHRASES_PATH = os.environ.get(
    "PARAPHRASES_PATH",
    f"{PROJECT_PATH}/data/baseline_paraphrases.csv",
)

PARAPHRASES_JUDGED_PATH = os.environ.get(
    "PARAPHRASES_JUDGED_PATH",
    f"{PROJECT_PATH}/data/baseline_paraphrases_judged.csv",
)

PROXIES_PATH = os.environ.get(
    "PROXIES_PATH",
    f"{PROJECT_PATH}/data/baseline_proxies.csv",
)

PROXIES_JUDGED_PATH = os.environ.get(
    "PROXIES_JUDGED_PATH",
    f"{PROJECT_PATH}/data/baseline_proxies_judged.csv",
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

REGENERATED_BASELINE_JUDGED_PATH = os.environ.get(
    "REGENERATED_BASELINE_JUDGED_PATH",
    f"{PROJECT_PATH}/data/regenerated_baseline_judged.csv",
)

REGENERATED_PARAPHRASES_JUDGED_PATH = os.environ.get(
    "REGENERATED_PARAPHRASES_JUDGED_PATH",
    f"{PROJECT_PATH}/data/regenerated_paraphrases_judged.csv",
)

REGENERATED_PROXIES_JUDGED_PATH = os.environ.get(
    "REGENERATED_PROXIES_JUDGED_PATH",
    f"{PROJECT_PATH}/data/regenerated_proxies_judged.csv",
)

REJECTED_REGENERATED_BASELINE_PATH = os.environ.get(
    "REJECTED_REGENERATED_BASELINE_PATH",
    f"{PROJECT_PATH}/data/rejected_regenerated_baseline.csv",
)

REJECTED_REGENERATED_PARAPHRASES_PATH = os.environ.get(
    "REJECTED_REGENERATED_PARAPHRASES_PATH",
    f"{PROJECT_PATH}/data/rejected_regenerated_paraphrases.csv",
)

REJECTED_REGENERATED_PROXIES_PATH = os.environ.get(
    "REJECTED_REGENERATED_PROXIES_PATH",
    f"{PROJECT_PATH}/data/rejected_regenerated_proxies.csv",
)

FILTERING_INSTRUCTIONS = """Role: You are the FilteringJudgeAgent. You only judge and filter data
that was already generated. You never generate samples, counterfactuals,
paraphrases, or proxies; you never compute bias; you never select metrics.
Follow filtering_mode exactly.

Available MCP Tools: judge_baseline_counterfactual_groups_with_llm,
remove_duplicate_refid_groups_before_append, append_to_central_generated_dataset,
judge_paraphrases_with_llm, judge_proxy_sentences_with_llm

Instructions - run only the steps for the given filtering_mode, in order:

full:
1. judge_baseline_counterfactual_groups_with_llm(baseline_csv_path=
   latest_baseline_generated_path, prompts_path, output_csv_path=
   tmp_baseline_judged_path, task_name, rejected_output_csv_path=
   rejected_baseline_path)
2. remove_duplicate_refid_groups_before_append(central_dataset_path=
   central_generated_dataset_path, new_judged_baseline_path=
   tmp_baseline_judged_path, output_path=tmp_baseline_deduped_path)
3. append_to_central_generated_dataset(central_dataset_path=
   central_generated_dataset_path, new_judged_baseline_path=
   tmp_baseline_deduped_path, paraphrases_judged_path=paraphrases_path,
   proxies_judged_path=proxies_path). Pass paraphrases_path/proxies_path
   here (the raw, not-yet-judged files) - appending reassigns the
   baseline's id/refid to central-dataset-wide values, and this remaps
   original_id/original_refid (paraphrases) and baseline_id/baseline_refid
   (proxies) to match, before judging copies those columns through to the
   final judged files. Skipping this breaks the join between paraphrases/
   proxies and the baseline later, in BiasTestingAgent.
4. judge_paraphrases_with_llm(input_csv_path=paraphrases_path,
   prompts_path, output_csv_path=paraphrases_judged_path, task_name,
   rejected_output_csv_path=rejected_paraphrases_path)
5. judge_proxy_sentences_with_llm(input_csv_path=proxies_path,
   prompts_path, output_csv_path=proxies_judged_path, task_name,
   rejected_output_csv_path=rejected_proxies_path)

paraphrase_only: run step 4 only.
proxy_only: run step 5 only.
paraphrase_and_proxy_only: run step 4, then step 5.

regenerated_only: do not touch the original baseline/paraphrases/proxies
(skip steps 1-5 above). For each of the three stages below, run the call
only if its regenerated_*_path value is not the literal string "NONE";
"NONE" means that stage has nothing to judge, skip it entirely:
- judge_baseline_counterfactual_groups_with_llm(baseline_csv_path=
  regenerated_baseline_path, prompts_path, output_csv_path=
  regenerated_baseline_judged_path, rejected_output_csv_path=
  rejected_regenerated_baseline_path)
- judge_paraphrases_with_llm(input_csv_path=regenerated_paraphrases_path,
  prompts_path, output_csv_path=regenerated_paraphrases_judged_path,
  task_name=None, rejected_output_csv_path=
  rejected_regenerated_paraphrases_path)
- judge_proxy_sentences_with_llm(input_csv_path=regenerated_proxies_path,
  prompts_path, output_csv_path=regenerated_proxies_judged_path,
  task_name=None, rejected_output_csv_path=
  rejected_regenerated_proxies_path)

none: do nothing.

Return: filtering_mode, every input/output path used, row counts,
flagged counts, rejected counts.

If any step's tool call returns ok=false or an error, stop immediately
- do not continue to the next step. End your answer with exactly one JSON
object on its own line: {"ok": true} if every required step completed
successfully, or {"ok": false, "failed_step": "<tool name that failed>"}
otherwise.
"""


def build_filtering_query(
    task_name,
    filtering_mode,
    prompts_path,
    latest_baseline_generated_path,
    tmp_baseline_judged_path,
    tmp_baseline_deduped_path,
    central_generated_dataset_path,
    paraphrases_path,
    paraphrases_judged_path,
    proxies_path,
    proxies_judged_path,
    rejected_baseline_path,
    rejected_paraphrases_path,
    rejected_proxies_path,
    regenerated_baseline_path=None,
    regenerated_paraphrases_path=None,
    regenerated_proxies_path=None,
    regenerated_baseline_judged_path=None,
    regenerated_paraphrases_judged_path=None,
    regenerated_proxies_judged_path=None,
    rejected_regenerated_baseline_path=None,
    rejected_regenerated_paraphrases_path=None,
    rejected_regenerated_proxies_path=None,
) -> str:
    return f"""task_name: {task_name}
filtering_mode: {filtering_mode}
prompts_path: {prompts_path}
latest_baseline_generated_path: {latest_baseline_generated_path}
tmp_baseline_judged_path: {tmp_baseline_judged_path}
tmp_baseline_deduped_path: {tmp_baseline_deduped_path}
central_generated_dataset_path: {central_generated_dataset_path}
paraphrases_path: {paraphrases_path}
paraphrases_judged_path: {paraphrases_judged_path}
proxies_path: {proxies_path}
proxies_judged_path: {proxies_judged_path}
rejected_baseline_path: {rejected_baseline_path}
rejected_paraphrases_path: {rejected_paraphrases_path}
rejected_proxies_path: {rejected_proxies_path}
regenerated_baseline_path: {regenerated_baseline_path}
regenerated_paraphrases_path: {regenerated_paraphrases_path}
regenerated_proxies_path: {regenerated_proxies_path}
regenerated_baseline_judged_path: {regenerated_baseline_judged_path}
regenerated_paraphrases_judged_path: {regenerated_paraphrases_judged_path}
regenerated_proxies_judged_path: {regenerated_proxies_judged_path}
rejected_regenerated_baseline_path: {rejected_regenerated_baseline_path}
rejected_regenerated_paraphrases_path: {rejected_regenerated_paraphrases_path}
rejected_regenerated_proxies_path: {rejected_regenerated_proxies_path}
"""


async def main():
    server = MCPServerStdio(
        params={
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": PROJECT_PATH,
        },
        client_session_timeout_seconds=14400,
    )

    user_prompt = build_filtering_query(
        task_name=TASK_NAME,
        filtering_mode=FILTERING_MODE,
        prompts_path=PROMPTS_PATH,
        latest_baseline_generated_path=LATEST_BASELINE_GENERATED_PATH,
        tmp_baseline_judged_path=TMP_BASELINE_JUDGED_PATH,
        tmp_baseline_deduped_path=TMP_BASELINE_DEDUPED_PATH,
        central_generated_dataset_path=CENTRAL_GENERATED_DATASET_PATH,
        paraphrases_path=PARAPHRASES_PATH,
        paraphrases_judged_path=PARAPHRASES_JUDGED_PATH,
        proxies_path=PROXIES_PATH,
        proxies_judged_path=PROXIES_JUDGED_PATH,
        rejected_baseline_path=REJECTED_BASELINE_PATH,
        rejected_paraphrases_path=REJECTED_PARAPHRASES_PATH,
        rejected_proxies_path=REJECTED_PROXIES_PATH,
        regenerated_baseline_path=REGENERATED_BASELINE_PATH,
        regenerated_paraphrases_path=REGENERATED_PARAPHRASES_PATH,
        regenerated_proxies_path=REGENERATED_PROXIES_PATH,
        regenerated_baseline_judged_path=REGENERATED_BASELINE_JUDGED_PATH,
        regenerated_paraphrases_judged_path=REGENERATED_PARAPHRASES_JUDGED_PATH,
        regenerated_proxies_judged_path=REGENERATED_PROXIES_JUDGED_PATH,
        rejected_regenerated_baseline_path=REJECTED_REGENERATED_BASELINE_PATH,
        rejected_regenerated_paraphrases_path=REJECTED_REGENERATED_PARAPHRASES_PATH,
        rejected_regenerated_proxies_path=REJECTED_REGENERATED_PROXIES_PATH,
    )

    async with server:
        agent = Agent(
            name="FilteringJudgeAgent",
            instructions=FILTERING_INSTRUCTIONS,
            mcp_servers=[server],
            model="gpt-5",
        )

        result = await Runner.run(
            agent,
            user_prompt,
            max_turns=10,
        )

        print("\n=== FILTERING JUDGE AGENT OUTPUT ===\n")
        print(result.final_output)

        
        filtering_ok = True
        try:
            text = str(result.final_output)
            end = text.rfind("}")
            start = text.rfind("{", 0, end)
            if start != -1 and end > start:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, dict) and "ok" in parsed:
                    filtering_ok = bool(parsed["ok"])
        except Exception:
            pass

        if not filtering_ok:
            print("\n=== FILTERING JUDGE AGENT FAILED - STOPPING PIPELINE ===\n")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
