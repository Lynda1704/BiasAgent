import pandas as pd

REQUIRED_COLUMNS = [
    "id",
    "bias_type",
    "label",
    "task",
    "id_term",
    "concept_term",
    "sentence",
    "flagged",
    "refid",
]

def load_baseline(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(
        csv_path,
        sep=None,
        engine="python",
        encoding="utf-8"
    )


    df.columns = [
        c.strip().replace('"', "").replace("\ufeff", "")
        for c in df.columns
    ]


    if len(df.columns) == 1:
        first_col = str(df.columns[0])

        if "," in first_col:
            df = pd.read_csv(csv_path, sep=",", encoding="utf-8")

        elif ";" in first_col:
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8")

        df.columns = [
            c.strip().replace('"', "").replace("\ufeff", "")
            for c in df.columns
        ]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes: {missing}\n"
            f"Colonnes trouvées: {df.columns.tolist()}"
        )

    df["flagged"] = pd.to_numeric(
        df["flagged"],
        errors="coerce"
    )

    df = df[df["flagged"] == 0].copy()

    if df.empty:
        raise ValueError(
            "Le dataset est vide après filtrage flagged == 0"
        )

    return df


def filter_dataset_by_task(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    sub = df[df["task"].str.lower() == task_name.lower()].copy()
    if sub.empty:
        raise ValueError(f"Aucune ligne trouvée pour la tâche: {task_name}")
    return sub


def list_tasks(df: pd.DataFrame) -> list[str]:
    return sorted(df["task"].dropna().unique().tolist())
