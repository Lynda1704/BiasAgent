# BiasAgent Dataset

This repository contains the bias evaluation dataset generated and validated
by **BiasAgent**, a multi-agent LLM-based framework for automated social bias
testing of text classification models.

The dataset consists of three complementary components: **baseline** test
cases that explicitly mention a social identity, **paraphrases** that
introduce linguistic variation while preserving meaning, and **proxies** that
replace explicit identity mentions with implicit, culturally associated cues.

## Files

| File | Description | Rows (total / accepted) |
|---|---|---|
| `baseline.csv` | Baseline test cases, with counterfactual identity variants grouped by `refid` | 1,817 / 1,652 |
| `paraphrases.csv` | Paraphrase-based variants of the baseline test cases | 13,788 / 9,060 |
| `proxies.csv` | Proxy-based variants of the baseline test cases (implicit identity cues) | 5,724 / 5,217 |

"Accepted" rows are those with `flagged = 0`, i.e., rows that passed the
automated validation performed by the Filtering Agent. Rows with `flagged = 1`
are retained for transparency (e.g., to support further validation research)
but were excluded from the bias evaluation reported in the accompanying
paper.

## Column reference

### `baseline.csv`

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `refid` | Groups counterfactual variants of the same base test case (same content, different identity) |
| `bias_type` | Protected attribute category (e.g., gender, religion, nationality) |
| `id_term` | The identity term used in this variant (e.g., "woman", "Muslim") |
| `concept_term` | The stereotype-associated concept term combined with the identity term |
| `task` | Target text classification task (e.g., "toxicity detection") |
| `label` | Ground-truth class label for the task |
| `sentence` | The generated test-case sentence |
| `flagged` | `0` if the sentence passed validation, `1` if rejected |
| `baseline_judge_valid` / `baseline_judge_reason` | Validation outcome and justification from the Filtering Agent |
| `filter_status` / `filter_reason` | Final filtering outcome after the regeneration process |
| `regeneration_attempt` | Number of regeneration attempts for this row (0 if accepted on first generation) |
| `source_rejected_id` | If this row is a regenerated replacement, the `id` of the originally rejected row |
| `was_regenerated` | Whether this row is the result of a regeneration attempt |

### `paraphrases.csv`

Same validation/regeneration columns as `baseline.csv`, plus:

| Column | Description |
|---|---|
| `original_id` / `original_refid` | The `id` / `refid` of the baseline row this paraphrase was derived from |
| `paraphrase_refid` | Identifier for this specific paraphrase (per-identity, not a counterfactual group key) |
| `original_sentence` | The baseline sentence before paraphrasing |
| `sentence` | The paraphrased sentence |
| `paraphrase_type` | Transformation category: `question`, `reorder`, `passive_active`, `conjunction`, `negation`, `idiomatic` (BiasAgent's own contribution), or `pattern_by_example`, `taboo_patterns`, `pivot_translation` (adapted from prior work; see paper) |
| `syntax_type` / `lexical_type` | Finer-grained categorization of the paraphrase transformation |

### `proxies.csv`

Same validation/regeneration columns as `baseline.csv`, plus:

| Column | Description |
|---|---|
| `baseline_id` / `baseline_refid` | The `id` / `refid` of the baseline row this proxy was derived from |
| `proxy_refid` | Identifier for this specific proxy (per-identity, not a counterfactual group key) |
| `baseline_sentence` | The baseline sentence before proxy substitution |
| `proxy_sentence` | The sentence with the identity term replaced by implicit cues |
| `proxy_type` | Confidence tier of the implicit identity signal: `weak`, `medium`, or `strong` |
| `phi`, `theta`, `prior`, `k` | Parameters of the proxy generation model (association strength, target confidence level, prior group probability, and number of required attributes; see paper for the underlying formulation) |





## Content notice

`concept_term` values in `baseline.csv` and the identity/concept associations
used to construct these test cases reflect common societal stereotypes used
to probe model behavior. They are included strictly for bias-testing purposes
and do not represent factual claims or endorsements by the authors.


