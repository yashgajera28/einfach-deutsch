"""LoRA fine-tuning script for a German 7B causal LM for text simplification."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRANSFORMERS_AVAILABLE = False
_DATASETS_AVAILABLE = False
_PEFT_AVAILABLE = False
_TRL_AVAILABLE = False
_TORCH_AVAILABLE = False
_BNB_AVAILABLE = False

AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None
BitsAndBytesConfig: Any = None
load_from_disk: Any = None
LoraConfig: Any = None
TaskType: Any = None
SFTTrainer: Any = None
torch: Any = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _TRANSFORMERS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    logger.warning("transformers is not installed: %s", exc)

try:
    from datasets import load_from_disk

    _DATASETS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    logger.warning("datasets is not installed: %s", exc)

try:
    from peft import LoraConfig, TaskType

    _PEFT_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    logger.warning("peft is not installed: %s", exc)

try:
    from trl import SFTTrainer

    _TRL_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    logger.warning("trl is not installed: %s", exc)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning("torch is not installed; GPU detection will be unavailable.")

try:
    import bitsandbytes  # noqa: F401

    _BNB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BNB_AVAILABLE = False


def _require_training_deps() -> None:
    """Raise a clear error when required training dependencies are missing."""
    missing: list[str] = []
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if not _DATASETS_AVAILABLE:
        missing.append("datasets")
    if not _PEFT_AVAILABLE:
        missing.append("peft")
    if not _TRL_AVAILABLE:
        missing.append("trl")
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "LoRA training requires the following packages: "
            f"{', '.join(missing)}. Install them and try again."
        )


def _has_gpu() -> bool:
    """Return True when a CUDA GPU is available."""
    return torch is not None and torch.cuda.is_available()


def _format_instruction(
    complex_text: str,
    simple_text: str,
    level_token: str,
    system_prompt: str,
) -> str:
    """Format one example as an instruction-following conversation."""
    return (
        f"SYSTEM: {system_prompt}\n"
        f"USER: {level_token} Vereinfache: {complex_text}\n"
        f"ASSISTANT: {simple_text}"
    )


def _format_dataset(
    dataset: Any,
    level_token: str,
    system_prompt: str,
    complex_column: str,
    simple_column: str,
) -> Any:
    """Add a formatted text column to a HuggingFace dataset."""

    def _format(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": _format_instruction(
                example[complex_column],
                example[simple_column],
                level_token,
                system_prompt,
            )
        }

    return dataset.map(_format)


def _load_tokenizer(
    model_name: str,
    control_tokens: list[str],
) -> Any:
    """Load a tokenizer, ensure a pad token, and add control tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(control_tokens)
    return tokenizer


def _load_model(model_name: str, tokenizer: Any) -> Any:
    """Load the causal LM, using 4-bit quantization when possible."""
    use_4bit = _BNB_AVAILABLE and _has_gpu()
    dtype = torch.float16 if _has_gpu() else torch.float32

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        if _has_gpu():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            warnings.warn(
                "No GPU detected; loading model on CPU with fp32. This will be very slow.",
                UserWarning,
                stacklevel=2,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

    model.resize_token_embeddings(len(tokenizer))
    return model


def _resolve_model_name(config: dict[str, Any]) -> str:
    """Return the primary model name or the fallback if primary load failed."""
    lora_cfg = config.get("models", {}).get("lora", {})
    primary: str = lora_cfg.get("name", "LeoLM/leo-hessianai-7b")
    fallback: str = lora_cfg.get(
        "fallback_name",
        "mistralai/Mistral-7B-Instruct-v0.2",
    )
    try:
        AutoTokenizer.from_pretrained(primary)
        return primary
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Failed to reach/load primary model %s (%s). Falling back to %s.",
            primary,
            exc,
            fallback,
        )
        return fallback


def train(
    config: dict[str, Any],
    demo: bool = False,
    output_dir: str | None = None,
) -> None:
    """Fine-tune a German 7B causal LM with LoRA for text simplification.

    Args:
        config: Project configuration dictionary.
        demo: Run a tiny training loop for smoke testing.
        output_dir: Optional override for the checkpoint directory.
    """
    _require_training_deps()

    lora_cfg = config.get("models", {}).get("lora", {})
    train_cfg = config.get("training", {}).get("lora", {})
    peft_cfg = config.get("lora", {})

    control_tokens: list[str] = lora_cfg.get("control_tokens", ["<A2>", "<B1>", "<B2>"])
    max_seq_length: int = lora_cfg.get("max_seq_length", 512)
    data_path: str = config["paths"]["data_processed"]

    default_level: str = config.get("simplification", {}).get("default_level", "B1")
    level_token: str = f"<{default_level}>"

    system_prompt = (
        "Du bist ein deutscher Textvereinfacher. Wandle komplexe Texte in einfache "
        "Sprache (Niveau B1) um. Behalte alle Fakten bei. Verwende kurze Sätze und "
        "einfache Wörter."
    )

    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = Path(config["paths"]["checkpoints"]) / "lora"
    out.mkdir(parents=True, exist_ok=True)

    model_name = _resolve_model_name(config)

    tokenizer = _load_tokenizer(model_name, control_tokens)
    model = _load_model(model_name, tokenizer)

    lora_config = LoraConfig(
        r=peft_cfg.get("r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        target_modules=peft_cfg.get(
            "target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias=peft_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = model if not hasattr(model, "add_adapter") else model

    dataset = load_from_disk(data_path)
    column_names = dataset["train"].column_names
    complex_column = "complex" if "complex" in column_names else column_names[0]
    simple_column = "simple" if "simple" in column_names else column_names[1]

    if demo:
        dataset["train"] = dataset["train"].select(range(min(30, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(
            range(min(10, len(dataset["validation"])))
        )

    formatted_train = _format_dataset(
        dataset["train"],
        level_token,
        system_prompt,
        complex_column,
        simple_column,
    )
    formatted_val = _format_dataset(
        dataset["validation"],
        level_token,
        system_prompt,
        complex_column,
        simple_column,
    )

    batch_size: int = train_cfg.get("batch_size", 4) if not demo else 1
    gradient_accumulation_steps: int = (
        train_cfg.get("gradient_accumulation_steps", 4) if not demo else 2
    )
    num_epochs: int = train_cfg.get("num_epochs", 3) if not demo else 1
    learning_rate: float = train_cfg.get("learning_rate", 2.0e-4)
    fp16: bool = train_cfg.get("fp16", True) if _has_gpu() else False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        peft_config=lora_config,
        max_seq_length=max_seq_length,
        args={
            "output_dir": str(out),
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "fp16": fp16,
            "logging_steps": train_cfg.get("logging_steps", 10),
            "eval_strategy": "steps",
            "eval_steps": 50,
            "save_strategy": "steps",
            "save_total_limit": 2,
            "report_to": [],
            "dataloader_num_workers": 0,
        },
    )

    trainer.train()
    trainer.save_model(str(out / "final_model"))
    tokenizer.save_pretrained(str(out / "final_model"))
    logger.info("LoRA adapter and tokenizer saved to %s", out / "final_model")
