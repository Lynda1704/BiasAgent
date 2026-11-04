import pandas as pd


def value_column(df):
    return "bias_score" if "bias_score" in df.columns else "bias_detected"


def build_counterfactual_by_bias(counterfactual_df):
    if "bias_type" in counterfactual_df.columns and "score" in counterfactual_df.columns:
        return (
            counterfactual_df
            .groupby("bias_type")["score"]
            .mean()
            .reset_index()
            .sort_values("score", ascending=False)
        )
    return pd.DataFrame()


def build_paraphrase_family_summary(paraphrase_df):
    return (
        paraphrase_df
        .groupby("paraphrase_family")[value_column(paraphrase_df)]
        .mean()
        .reset_index()
        .rename(columns={value_column(paraphrase_df): "bias_rate"})
        .sort_values("bias_rate", ascending=False)
    )


def build_paraphrase_type_summary(paraphrase_df):
    return (
        paraphrase_df
        .groupby("paraphrase_type")[value_column(paraphrase_df)]
        .mean()
        .reset_index()
        .rename(columns={value_column(paraphrase_df): "bias_rate"})
        .sort_values("bias_rate", ascending=False)
    )


def build_proxy_type_summary(proxy_df):
    return (
        proxy_df
        .groupby("proxy_type")[value_column(proxy_df)]
        .mean()
        .reset_index()
        .rename(columns={value_column(proxy_df): "bias_rate"})
        .sort_values("bias_rate", ascending=False)
    )
def build_paraphrase_by_bias_type(paraphrase_df):
    if "bias_type" in paraphrase_df.columns and value_column(paraphrase_df) in paraphrase_df.columns:
        return (
            paraphrase_df
            .groupby("bias_type")[value_column(paraphrase_df)]
            .mean()
            .reset_index()
            .rename(columns={value_column(paraphrase_df): "score"})
            .sort_values("score", ascending=False)
        )
    return pd.DataFrame()


def build_proxy_by_bias_type(proxy_df):
    if "bias_type" in proxy_df.columns and value_column(proxy_df) in proxy_df.columns:
        return (
            proxy_df
            .groupby("bias_type")[value_column(proxy_df)]
            .mean()
            .reset_index()
            .rename(columns={value_column(proxy_df): "score"})
            .sort_values("score", ascending=False)
        )
    return pd.DataFrame()

def build_overall_summary(
    counterfactual_global,
    linguistic_rate,
    syntax_rate,
    pivot_rate,
    proxy_global,
):
    return pd.DataFrame([
        {"method_family": "Counterfactual Baseline", "bias_rate": counterfactual_global},
        {"method_family": "Linguistic Paraphrases", "bias_rate": linguistic_rate},
        {"method_family": "Syntax-Aware Patterns", "bias_rate": syntax_rate},
        {"method_family": "Pivot Translation", "bias_rate": pivot_rate},
        {"method_family": "Proxy Testing", "bias_rate": proxy_global},
    ]).dropna()
