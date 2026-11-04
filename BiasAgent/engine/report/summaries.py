def build_executive_summary(
    model_name,
    task_name,
    counterfactual_global,
    linguistic_rate,
    syntax_rate,
    pivot_rate,
    proxy_global,
    overall_df,
):
    top_family = overall_df.sort_values(
        "bias_rate",
        ascending=False,
    ).iloc[0]["method_family"]

    return (
        f"Model {model_name} was evaluated on task '{task_name}'. "
        f"The global counterfactual/group baseline score is {counterfactual_global:.4f}. "
        f"Linguistic Paraphrases reached {linguistic_rate:.4f}, "
        f"Syntax-Aware Patterns reached {syntax_rate:.4f}, "
        f"Pivot Translation reached {pivot_rate:.4f}, "
        f"and Proxy Testing reached {proxy_global:.4f}. "
        f"The highest reporting family is {top_family}."
    )
