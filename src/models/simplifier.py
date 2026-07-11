"""Unified inference wrapper for German text simplification."""

from __future__ import annotations

import difflib
import logging
import re
import warnings
from pathlib import Path
from typing import Any

from src.evaluation.readability import lix_score, wstf_score
from src.preprocessing import spacy_pipe

logger = logging.getLogger(__name__)

_TRANSFORMERS_AVAILABLE = False
_PEFT_AVAILABLE = False
_TORCH_AVAILABLE = False

AutoTokenizer: Any = None
AutoModelForSeq2SeqLM: Any = None
AutoModelForCausalLM: Any = None
PeftModel: Any = None
torch: Any = None

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    logger.warning("transformers is not installed: %s", exc)

try:
    from peft import PeftModel

    _PEFT_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    logger.warning("peft is not installed: %s", exc)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning("torch is not installed; inference will be unavailable.")


class Simplifier:
    """Load and run a German text simplification model.

    Supports both the mT5-small baseline and the LoRA-tuned causal LM.
    """

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        backend: str = "lora",
        model_path: str | None = None,
    ) -> None:
        """Initialize tokenizer and model for the chosen backend.

        Args:
            config: Project configuration dictionary or path to a YAML config file.
            backend: Either ``"baseline"`` (mT5) or ``"lora"`` (causal LM).
            model_path: Optional checkpoint or model name. Falls back to config defaults.
        """
        if backend not in {"baseline", "lora"}:
            raise ValueError("backend must be 'baseline' or 'lora'")

        self.backend = backend
        self.config = self._load_config(config)
        self.model_path = model_path or self._default_model_path()
        self.device = self._resolve_device()

        self.tokenizer: Any = None
        self.model: Any = None

        if _TRANSFORMERS_AVAILABLE:
            self._load_model()

    def _load_config(self, config: dict[str, Any] | str | Path) -> dict[str, Any]:
        """Normalize config input to a dictionary."""
        if isinstance(config, dict):
            return config

        from src.utils.config import load_config

        return load_config(config)

    def _default_model_path(self) -> str:
        """Return the default model identifier from config."""
        models = self.config.get("models", {})
        if self.backend == "baseline":
            return models.get("baseline", {}).get("name", "google/mt5-small")
        return models.get("lora", {}).get("name", "LeoLM/leo-hessianai-7b")

    def _resolve_device(self) -> str:
        """Pick GPU if available, otherwise CPU with a warning."""
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        warnings.warn("CUDA is unavailable; loading model on CPU.", UserWarning)
        return "cpu"

    def _require_inference_deps(self) -> None:
        """Raise a clear error when inference dependencies are missing."""
        missing: list[str] = []
        if not _TRANSFORMERS_AVAILABLE:
            missing.append("transformers")
        if self.backend == "lora" and not _PEFT_AVAILABLE:
            missing.append("peft")
        if not _TORCH_AVAILABLE:
            missing.append("torch")
        if missing:
            raise RuntimeError(
                "Inference requires the following packages: "
                f"{', '.join(missing)}. Install them and try again."
            )

    def _load_model(self) -> None:
        """Load tokenizer and model for the configured backend."""
        self._require_inference_deps()

        logger.info("Loading %s model from %s", self.backend, self.model_path)

        if self.backend == "baseline":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        control_tokens = (
            self.config.get("models", {}).get("lora", {}).get("control_tokens", [])
        )
        if control_tokens:
            self.tokenizer.add_tokens(control_tokens)

        adapter_config_path = Path(self.model_path) / "adapter_config.json"
        if adapter_config_path.exists():
            base_name = self._default_model_path()
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                base_model = base_model.to(self.device)
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def _format_baseline_input(self, text: str, level: str) -> str:
        """Format text for the mT5 baseline."""
        prefix = (
            self.config.get("models", {}).get("baseline", {}).get("prefix", "vereinfachen: ")
        )
        levels = self.config.get("simplification", {}).get("levels", ["A2", "B1", "B2"])
        if level in levels:
            return f"{prefix}<{level}> {text}"
        return f"{prefix}{text}"

    def _format_lora_input(self, text: str, level: str) -> str:
        """Format text as an instruction conversation for the LoRA model."""
        system = (
            "Du bist ein deutscher Textvereinfacher. Wandle komplexe Texte in einfache "
            "Sprache um. Behalte alle Fakten bei. Verwende kurze Sätze und einfache Wörter."
        )
        level_token = f"<{level}>"
        return (
            f"SYSTEM: {system}\n"
            f"USER: {level_token} Vereinfache: {text}\n"
            f"ASSISTANT:"
        )

    def _generate(
        self,
        prompt: str,
        max_length: int,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate text from a prompt and return decoded outputs."""
        self._require_inference_deps()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been loaded.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def _extract_answer(self, generated: str) -> str:
        """Strip the prompt/instruction prefix from a LoRA generation."""
        marker = "ASSISTANT:"
        if marker in generated:
            return generated.split(marker, 1)[-1].strip()
        return generated.strip()

    def _preserve_entities(self, original: str, simplified: str) -> tuple[str, list[str]]:
        """Re-inject named entities that disappeared during simplification."""
        original_ents = spacy_pipe.entities(original)
        if not original_ents:
            return simplified, []

        preserved: list[str] = []
        missing: list[str] = []
        for ent_text, _ in original_ents:
            if ent_text.lower() not in simplified.lower():
                missing.append(ent_text)
            else:
                preserved.append(ent_text)

        if missing:
            appendix = " (Wichtige Begriffe: " + ", ".join(missing) + ")"
            simplified = simplified.rstrip() + appendix
            preserved.extend(missing)

        return simplified, preserved

    def _build_explanation(self, original: str, simplified: str) -> list[str]:
        """Produce rule-based explanations for the observed changes."""
        explanations: list[str] = []

        original_lower = original.lower()
        simplified_lower = simplified.lower()

        if "wurde" in original_lower and "wurde" not in simplified_lower:
            explanations.append("Passive voice converted to active")

        original_words = re.findall(r"[a-zäöüßA-ZÄÖÜ]+", original)
        simplified_words = re.findall(r"[a-zäöüßA-ZÄÖÜ]+", simplified)
        long_original = [w for w in original_words if len(w) > 12]
        if long_original and len(simplified_words) >= len(original_words):
            explanations.append("Compound word split")

        legal_refs = spacy_pipe.extract_legal_references(original) + spacy_pipe.extract_legal_references(simplified)
        if legal_refs:
            explanations.append("Legal term explained")

        original_sents = len(spacy_pipe.sentences(original))
        simplified_sents = len(spacy_pipe.sentences(simplified))
        if simplified_sents > original_sents:
            explanations.append("Long sentence split")

        if not explanations:
            matcher = difflib.SequenceMatcher(None, original.split(), simplified.split())
            if any(tag != "equal" for tag, _, _, _, _ in matcher.get_opcodes()):
                explanations.append("Vocabulary simplified")

        return explanations

    def confidence_score(self, original: str, simplified: str) -> float:
        """Return a heuristic confidence score in ``[0, 1]``.

        Combines length ratio and generation stability.
        """
        original_len = max(len(original.strip()), 1)
        simplified_len = max(len(simplified.strip()), 1)
        ratio = simplified_len / original_len

        length_score = 1.0 - abs(ratio - 0.8) / 0.8
        length_score = max(0.0, min(1.0, length_score))

        stability = difflib.SequenceMatcher(None, original, simplified).ratio()

        confidence = 0.6 * length_score + 0.4 * stability
        return round(max(0.0, min(1.0, confidence)), 4)

    def simplify(
        self,
        text: str,
        level: str = "B1",
        preserve_entities: bool = True,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Simplify ``text`` and return a structured result dictionary.

        Args:
            text: The German text to simplify.
            level: Target language level (A2, B1, B2).
            preserve_entities: Re-inject named entities that were dropped.
            explain: Include rule-based explanations for changes.

        Returns:
            Dictionary with simplified text, metrics, and optional explanations.
        """
        self._require_inference_deps()

        if not text or not text.strip():
            return {
                "simplified": "",
                "level": level,
                "backend": self.backend,
                "confidence": 0.0,
                "readability_before": {"lix": 0.0, "wstf": 0.0},
                "readability_after": {"lix": 0.0, "wstf": 0.0},
                "entities_preserved": [],
                "explanation": [] if explain else None,
            }

        if self.backend == "baseline":
            prompt = self._format_baseline_input(text, level)
            max_length = self.config.get("models", {}).get("baseline", {}).get("max_length", 256)
        else:
            prompt = self._format_lora_input(text, level)
            max_length = self.config.get("models", {}).get("lora", {}).get("max_seq_length", 512)

        outputs = self._generate(prompt, max_length, num_return_sequences=1)
        simplified = outputs[0]
        if self.backend == "lora":
            simplified = self._extract_answer(simplified)

        entities_preserved: list[str] = []
        if preserve_entities:
            simplified, entities_preserved = self._preserve_entities(text, simplified)

        explanation = self._build_explanation(text, simplified) if explain else None

        return {
            "simplified": simplified,
            "level": level,
            "backend": self.backend,
            "confidence": self.confidence_score(text, simplified),
            "readability_before": {"lix": lix_score(text), "wstf": wstf_score(text)},
            "readability_after": {"lix": lix_score(simplified), "wstf": wstf_score(simplified)},
            "entities_preserved": entities_preserved,
            "explanation": explanation,
        }
