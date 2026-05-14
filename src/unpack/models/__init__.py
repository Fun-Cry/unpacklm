"""
unpack.models - Model adapter registry and auto-detection.

Supported architectures:
    - GPT-2 (sequential residual, Conv1D, learned position embeddings)
    - GPT-NeoX / Pythia (parallel residual, Linear, RoPE)
"""

from unpack.models.base import ModelAdapter, HookManager


class UnsupportedModelError(Exception):
    """Raised when no adapter exists for a model architecture."""
    pass


def get_adapter(model, **kwargs) -> ModelAdapter:
    """Auto-detect and return the right ModelAdapter for a model.

    Detection uses ``model.config.model_type`` which HuggingFace sets
    automatically when loading from a pretrained checkpoint.

    Args:
        model: a HuggingFace PreTrainedModel
        **kwargs: forwarded to the adapter constructor

    Returns:
        An initialized (but not yet hooked) ModelAdapter.

    Raises:
        UnsupportedModelError: if no adapter is registered for this model type.
    """
    model_type = getattr(model.config, "model_type", None)

    if model_type == "gpt2":
        from unpack.models.gpt2 import GPT2Adapter
        return GPT2Adapter(**kwargs)
    elif model_type == "gpt_neox":
        from unpack.models.gpt_neox import GPTNeoXAdapter
        return GPTNeoXAdapter(**kwargs)
    else:
        supported = ["gpt2", "gpt_neox"]
        raise UnsupportedModelError(
            f"No adapter for model_type={model_type!r}. "
            f"Supported: {supported}. "
            f"You can pass a custom adapter via Tracer(model=..., adapter=...)."
        )


def load_model(model_name_or_path, device="auto", cache_dir=None, **kwargs):
    """Load a model and tokenizer from a HuggingFace name or path.

    Handles the common short names (gpt2, gpt2-medium, etc.) and
    Pythia naming conventions (EleutherAI/pythia-410m-deduped).

    Returns:
        (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {"attn_implementation": "eager", "torch_dtype": torch.float32}
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    # Pythia step-specific loading
    step = kwargs.get("step")
    if step is not None:
        model_kwargs["revision"] = f"step{step}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        output_attentions=True,
        **model_kwargs,
    )

    tok_kwargs = {}
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    if step is not None:
        tok_kwargs["revision"] = f"step{step}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, **tok_kwargs)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "<|padding|>"

    # Device placement
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer


__all__ = [
    "ModelAdapter",
    "HookManager",
    "get_adapter",
    "load_model",
    "UnsupportedModelError",
]