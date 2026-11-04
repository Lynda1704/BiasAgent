from typing import Dict, Any
import pandas as pd


REQUIRED_TERM_COLUMNS = [
    "id",
    "bias_type",
    "id_term",
    "topic",
    "concept_term",
    "flagged",
]


def load_terms_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    missing = [c for c in REQUIRED_TERM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in terms catalog: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_TERM_COLUMNS].copy()

    df["id"] = df["id"].astype(int)
    df["bias_type"] = df["bias_type"].astype(str)
    df["id_term"] = df["id_term"].astype(str)
    df["topic"] = df["topic"].astype(str)
    df["concept_term"] = df["concept_term"].astype(str)
    df["flagged"] = df["flagged"].astype(int)

    return df


def inspect_terms_catalog(path: str) -> Dict[str, Any]:
    df = load_terms_catalog(path)

    return {
        "ok": True,
        "path": path,
        "num_rows": int(len(df)),
        "columns": list(df.columns),
        "bias_types": sorted(df["bias_type"].unique().tolist()),
        "num_bias_types": int(df["bias_type"].nunique()),
        "bias_type_counts": df["bias_type"].value_counts().to_dict(),
        "num_unique_identity_terms": int(df["id_term"].nunique()),
        "num_unique_topics": int(df["topic"].nunique()),
        "num_unique_concept_terms": int(df["concept_term"].nunique()),
        "flagged_counts": df["flagged"].value_counts().to_dict(),
    }
