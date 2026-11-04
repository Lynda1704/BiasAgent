import pandas as pd
import spacy
import benepar


_NLP = None


def get_parser():
    global _NLP

    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
        if "benepar" not in _NLP.pipe_names:
            _NLP.add_pipe("benepar", config={"model": "benepar_en3"})

    return _NLP


def extract_top2_syntax_pattern(sentence: str) -> str:
    nlp = get_parser()
    doc = nlp(str(sentence))

    sent = list(doc.sents)[0]
    tree = sent._.parse_string

    from nltk import Tree
    parsed_tree = Tree.fromstring(tree)

    labels = [parsed_tree.label()]

    for child in parsed_tree:
        if hasattr(child, "label"):
            labels.append(child.label())

            for grandchild in child:
                if hasattr(grandchild, "label"):
                    labels.append(grandchild.label())

    return "-".join(labels)


def compute_syntax_pattern_table(
    paraphrases_csv_path: str,
    task_name: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(paraphrases_csv_path)

    if task_name is not None and "task" in df.columns:
        df = df[df["task"].astype(str).str.strip() == str(task_name).strip()].copy()

    df = df.dropna(subset=["sentence"]).copy()

    df["syntax_pattern"] = df["sentence"].apply(extract_top2_syntax_pattern)

    counts = (
        df["syntax_pattern"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["syntax_pattern", "count"]
    return counts


def get_frequent_and_rare_patterns(
    paraphrases_csv_path: str,
    task_name: str | None = None,
    top_k: int = 3,
) -> dict:
    counts = compute_syntax_pattern_table(
        paraphrases_csv_path=paraphrases_csv_path,
        task_name=task_name,
    )

    frequent = counts.head(top_k)["syntax_pattern"].tolist()
    rare = counts.tail(top_k)["syntax_pattern"].tolist()

    return {
        "frequent_patterns": frequent,
        "rare_patterns": rare,
        "pattern_table": counts.to_dict(orient="records"),
    }
