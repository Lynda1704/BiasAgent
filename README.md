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

## Counterfactual grouping

In `baseline.csv`, rows sharing the same `refid` form a matched counterfactual
group: the same underlying test case, expressed once per identity term within
a given `bias_type`. For `paraphrases.csv` and `proxies.csv`, the equivalent
counterfactual group for a given transformation type is obtained by grouping
on `(original_refid, paraphrase_type)` or `(baseline_refid, proxy_type)`
respectively, since `paraphrase_refid` / `proxy_refid` identify individual
rows rather than counterfactual groups.

## Metric selection and validation data

In addition to the three test-case files above, this repository includes
the data supporting RQ1 (human validation of test-case quality) and RQ3
(reliability of automatic bias metric configuration selection):

| File | Description | Rows |
|---|---|---|
| `metric_choices.csv` | Raw per-LLM metric configuration selections, 4 LLMs (GPT-5, Claude Sonnet 5, Gemini 3.1 Pro Preview, DeepSeek V4 Pro) × 21 evaluated models | 84 |
| `expert_gold_standard.csv` | Independent human-expert metric configuration selection for each of the 21 models, collected blind to the LLMs' selections, used to validate automatically adopted configurations | 21 |
| `human_validation_sample.csv` | Stratified random sample of baseline, paraphrase, and proxy test cases with human quality judgments, used to validate the Filtering Agent's decisions | 384 |

`metric_choices.csv` contains the raw votes behind the paper's inter-LLM
agreement analysis (exact agreement rate, Krippendorff's α, Fleiss' κ per
configuration component); these statistics can be recomputed directly from
this file. Precomputed aggregate agreement files from intermediate runs are
intentionally not included, as they do not reflect the final evaluation
reported in the paper; recomputing them from `metric_choices.csv` avoids
any risk of inconsistency.

## Data cleaning applied before release

Two issues were identified and corrected prior to publishing this dataset:

- **Stale cross-references in `paraphrases.csv`.** Because the underlying
  baseline dataset went through several regeneration rounds, some rows in
  the original paraphrase file referenced an `original_id` that no longer
  existed in the final `baseline.csv` (orphaned references), or whose
  `original_sentence` no longer matched the corresponding row in
  `baseline.csv` (references to an earlier, superseded version of that
  sentence). 1,287 such rows (of 15,075) were removed. The released file
  guarantees that for every row, `original_id` exists in `baseline.csv`
  and `original_sentence` is character-identical to `baseline.csv`'s
  `sentence` for that `id`.
- **Stale cross-references in `proxies.csv`.** The same regeneration-round
  issue described above for paraphrases also affected proxies: some rows
  referenced a `baseline_id` no longer present in `baseline.csv`, or whose
  `baseline_sentence` no longer matched the corresponding row in
  `baseline.csv`. 414 such rows (of 6,138) were removed, following the same
  rule applied to `paraphrases.csv`.
- **Grammatical artifact in `proxies.csv`.** A subset of proxy sentences
  were constructed by inserting an identity-masking clause (e.g., "someone
  who ...") in place of the explicit identity term, for cases where that
  term appeared without a preceding article. When the original baseline
  sentence used a plural quantifier directly before the identity term
  (e.g., "Many Christians ..."), this produced an ungrammatical result
  (e.g., "Many someone who ..."). Affected sentences were corrected by
  removing the now-redundant plural quantifier (e.g., "Many someone who
  ..." → "Someone who ..."). This fix resolves the local subject-determiner
  mismatch; it does not perform a full grammatical re-validation of the
  rest of the sentence.

## Content notice

`concept_term` values in `baseline.csv` and the identity/concept associations
used to construct these test cases reflect common societal stereotypes used
to probe model behavior. They are included strictly for bias-testing purposes
and do not represent factual claims or endorsements by the authors.

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{biasagent2026,
  title     = {BiasAgent: A Multi-Agent Framework for Automated Social Bias Testing of Text Classification Models},
  author    = {TODO},
  year      = {2026}
}
```

## License

TODO: specify a license (e.g., CC-BY-4.0 for data, MIT for code) before
publishing.
