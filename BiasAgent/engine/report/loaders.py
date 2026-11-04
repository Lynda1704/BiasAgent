import pandas as pd


def load_counterfactual_results(path: str):
    return pd.read_csv(path)


def load_paraphrase_results(path: str):
    return pd.read_csv(path)


def load_proxy_results(path: str):
    return pd.read_csv(path)
