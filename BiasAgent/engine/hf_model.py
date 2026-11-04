import json
import os
import threading
from typing import Any, Dict, List, Tuple

import torch
from huggingface_hub import model_info
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def get_model_possible_labels(clf):
    id2label = getattr(clf.model.config, "id2label", None)

    if not id2label:
        return []

    labels = []
    for key in sorted(id2label.keys(), key=lambda x: int(x)):
        labels.append(id2label[key])

    return labels
def fetch_hf_model_metadata(model_name: str) -> Dict[str, Any]:
    info = model_info(model_name)

    return {
        "model_id": model_name,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": getattr(info, "tags", None),
        "card_data": getattr(info, "card_data", None),
        "config": getattr(info, "config", None),
        "transformers_info": getattr(info, "transformers_info", None),
    }


_PIPELINE_CACHE = {}

_CACHE_LOCK = threading.Lock()


class _DiskCachedClassifier:

    def __init__(self, clf, model_name: str):
        self._clf = clf
        self.model = clf.model
        safe_name = model_name.strip().replace("/", "_")
        self._cache_path = os.path.join(
            "data", ".model_prediction_cache", f"{safe_name}.json"
        )
        self._cache: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with _CACHE_LOCK:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)

    def __call__(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            outputs = self._clf(missing)
            for text, output in zip(missing, outputs):
                self._cache[text] = output
            self._save()

        results = [self._cache[t] for t in texts]
        return results[0] if single else results


def build_pipeline(model_name: str):
    if model_name in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_name]


    device = 0 if torch.cuda.is_available() else -1


    clf = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,
        trust_remote_code=True,
        device=device,
        batch_size=32,
    )

    cached_clf = _DiskCachedClassifier(clf, model_name)
    _PIPELINE_CACHE[model_name] = cached_clf
    return cached_clf
def fetch_hf_model_metadata(model_name: str):
    info = model_info(model_name)

    return {
        "model_id": info.modelId,
        "pipeline_tag": info.pipeline_tag,
        "tags": info.tags,
        "card_data": info.card_data or {},
        "config": getattr(info, "config", {}) or {},
        "transformers_info": getattr(info, "transformers_info", {}) or {}
    }
def load_tokenizer_and_model(model_name: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def get_logits_and_probs(texts: List[str], tokenizer, model, batch_size: int = 16):
    all_logits = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        all_logits.extend(logits.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return all_logits, all_probs
