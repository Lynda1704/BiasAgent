# BiasAgent

A multi-agent LLM pipeline for automated social bias testing of text
classification models, along with the dataset it relies on.

## About

BiasAgent evaluates whether a text classifier treats sentences differently
depending on the social group they mention (race, gender, age, etc.), using
a set of LLM agents to generate, judge, and select the right bias metric for
each task automatically.

## Repository structure

- [`BiasAgent/`](./BiasAgent) -- the pipeline itself: agents, bias metric
  engine, MCP server, dashboard.
- [`dataset/`](./dataset) -- the baseline sentences, paraphrases and proxies
  used to run the pipeline, with its own README explaining how they're
  built.