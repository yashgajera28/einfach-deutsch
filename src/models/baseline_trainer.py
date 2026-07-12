"""Baseline mT5-small trainer for German text simplification."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRANSFORMERS_AVAILABLE = False
_DATASETS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    _TRANSFORMERS_AVAILABLE = True
    _DATASETS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    logger.warning(
        "Training dependencies are missing (%s). "
        "Import the module for inspection, but call train() only after installing requirements.",
        exc,
    )
    AutoTokenizer = None  # type: ignore[misc,assignment]
    AutoModelForSeq2SeqLM = None  # type: ignore[misc,assignment]
    DataCollatorForSeq2Seq = None  # type: ignore[misc,assignment]
    EarlyStoppingCallback = None  # type: ignore[misc,assignment]
    Seq2SeqTrainer = None  # type: ignore[misc,assignment]
    Seq2SeqTrainingArguments = None  # type: ignore[misc,assignment]
    load_from_disk = None  # type: ignore[misc,assignment]

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning("torch is not installed; GPU detection will be unavailable.")
    torch = None  # type: ignore[misc,assignment]


def _require_training_deps() -> None:
    """Raise a clear error when training dependencies are missing."""
    missing: list[str] = []
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if not _DATASETS_AVAILABLE:
        missing.append("datasets")
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "Baseline training requires the following packages: "
            f"{', '.join(missing)}. Install them and try again."
        )


def _resolve_device(demo: bool) -> str:
    """Pick the compute device and warn when falling back to CPU."""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if demo:
        logger.info("Demo mode: CUDA unavailable, using CPU.")
    else:
        warnings.warn("CUDA is unavailable; falling back to CPU training.", UserWarning)
    return "cpu"


def _preprocess_function(
    examples: dict[str, Any],
    tokenizer: Any,
    source_column: str,
    target_column: str,
    prefix: str,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize a batch after prepending the simplification prefix."""
    inputs = [prefix + text for text in examples[source_column]]
    targets = [text for text in examples[target_column]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(
    config: dict[str, Any],
    demo: bool = False,
    output_dir: str | None = None,
) -> None:
    """Train the baseline mT5-small model for German text simplification.

    Args:
        config: Project configuration dictionary.
        demo: Run a tiny training loop for smoke testing.
        output_dir: Optional override for the checkpoint directory.
    """
    _require_training_deps()

    baseline_cfg = config.get("models", {}).get("baseline", {})
    train_cfg = config.get("training", {}).get("baseline", {})

    model_name: str = baseline_cfg.get("name", "google/mt5-small")
    max_length: int = baseline_cfg.get("max_length", 256)
    prefix: str = baseline_cfg.get("prefix", "vereinfachen: ")
    data_path: str = config["paths"]["data_processed"]

    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = Path(config["paths"]["checkpoints"]) / "baseline"
    out.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(demo)
    use_fp16 = device == "cuda" and not demo

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    dataset = load_from_disk(data_path)

    if demo:
        dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(
            range(min(10, len(dataset["validation"])))
        )

    column_names = dataset["train"].column_names
    source_column = "source" if "source" in column_names else column_names[0]
    target_column = "target" if "target" in column_names else column_names[1]

    def preprocess(examples: dict[str, Any]) -> dict[str, Any]:
        return _preprocess_function(
            examples,
            tokenizer,
            source_column,
            target_column,
            prefix,
            max_length,
        )

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    batch_size = train_cfg.get("batch_size", 16)
    num_epochs = train_cfg.get("num_epochs", 5)
    warmup_steps = train_cfg.get("warmup_steps", 500)
    eval_steps = train_cfg.get("eval_steps", 500)
    save_total_limit = train_cfg.get("save_total_limit", 2)

    if demo:
        batch_size = 2
        num_epochs = 1
        warmup_steps = 10

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=save_total_limit,
        predict_with_generate=True,
        generation_max_length=max_length,
        fp16=use_fp16,
        logging_steps=10 if demo else 100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(str(out / "final_model"))
    tokenizer.save_pretrained(str(out / "final_model"))
    logger.info("Baseline model saved to %s", out / "final_model")
