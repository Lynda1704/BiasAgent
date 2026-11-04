import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


_TRANSLATORS = {}


def get_translator(src_lang: str, tgt_lang: str):
    key = (src_lang, tgt_lang)

    if key not in _TRANSLATORS:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        _TRANSLATORS[key] = {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "model_name": model_name,
        }

    return _TRANSLATORS[key]


def translate_once(
    text: str,
    src_lang: str,
    tgt_lang: str,
    max_length: int = 256,
) -> str:
    translator = get_translator(src_lang, tgt_lang)

    tokenizer = translator["tokenizer"]
    model = translator["model"]
    device = translator["device"]

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )

    translated = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return translated.strip()


def pivot_translate(
    text: str,
    pivot_languages: list[str],
    source_lang: str = "en",
) -> str:
    current_text = str(text)
    current_lang = source_lang

    for next_lang in pivot_languages:
        current_text = translate_once(
            current_text,
            src_lang=current_lang,
            tgt_lang=next_lang,
        )
        current_lang = next_lang

    current_text = translate_once(
        current_text,
        src_lang=current_lang,
        tgt_lang=source_lang,
    )

    return current_text
