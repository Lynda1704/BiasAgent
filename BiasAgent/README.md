# BiasAgent

Multi-agent LLM pipeline for automated social bias testing of HuggingFace
text classification models.

## Installation

```bash
pip install -r requirement.txt -q
python -m spacy download en_core_web_sm -q
python -c "import benepar; benepar.download('benepar_en3')"
```

Set your API key(s) in a `.env` file at the project root (`OPENAI_API_KEY`
required; `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` only if using another
LLM provider).

## Structure

```
agent/      -- the 4 agents + pipeline_runner.py to run them in sequence
engine/     -- generation, judging, model inference, bias metric engine
mcp_server/ -- MCP server + resources
scripts/    -- bias_dashboard.py (Gradio UI) + run_consensus_evaluation.py
```

## Running it

```bash
MODEL_NAME="unitary/toxic-bert" python -m agent.pipeline_runner
```

Output: a bias report per model/task under `reports/<model_name>/<task_name>/`.
