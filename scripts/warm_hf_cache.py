"""Warm the HuggingFace cache with the small baseline model at build time.

The full 7B LoRA model is intentionally not cached here because of its size;
it is downloaded automatically when the API first loads it.
"""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> int:
    model_name = "google/mt5-small"
    print(f"Warming HuggingFace cache for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"Cached {model_name} ({model.num_parameters()} parameters).")
    del tokenizer
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
