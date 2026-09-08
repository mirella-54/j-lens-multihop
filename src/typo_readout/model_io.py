"""Loading the subject model.

Model-agnostic on purpose: the same `load_model` call has to work for a plain
`Qwen3ForCausalLM` (Qwen3-4B) and a multimodal-wrapped `Qwen3_5ForConditionalGeneration`
(Qwen3.5-0.8B, Qwen3.6-27B) without config changes beyond `model.hf_repo`. See README.md
("Model loading auto-class is not yet confirmed") for why the fallback below exists.
"""

from __future__ import annotations

import dataclasses
import logging

import torch
import transformers
from jlens.hf import HFLensModel, from_hf

from typo_readout.config import ModelConfig

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,  # not used by any shipped config, but not our job to forbid
    "float32": torch.float32,
}

# HARD CONSTRAINT (brief section 2): never quantised. Any of these present on the config
# (top-level or nested text_config, since the "3.5"/"3.6" line wraps text_config inside a
# ConditionalGeneration config) means the checkpoint isn't the fp16/bf16 function the lens
# was fit against, and the run must refuse rather than silently proceed on a shifted lens.
def _assert_not_quantized(model: torch.nn.Module) -> None:
    config = model.config
    culprits = []
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is not None and getattr(cfg, "quantization_config", None) is not None:
            culprits.append(getattr(cfg, "quantization_config"))
    # Also check the loaded parameters directly -- bitsandbytes/AWQ/GPTQ replace linear
    # layers with custom quantized modules whose parameters are int8/int4-packed, which
    # would not necessarily show up as a `quantization_config` if one were loaded through
    # a path that doesn't set it (belt and suspenders, not a substitute for the check above).
    for name, param in model.named_parameters():
        if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            culprits.append(f"parameter {name} has dtype {param.dtype}")
            break
    if culprits:
        raise RuntimeError(
            f"Refusing to run: model appears to be quantised ({culprits!r}). "
            "The brief forbids quantisation -- the lens is an average Jacobian of the "
            "fp16/bf16 function; a quantised build shifts both the residual stream and the "
            "unembedding, and the lens is no longer the validated artifact."
        )


def _assert_requested_dtype(model: torch.nn.Module, requested: torch.dtype) -> None:
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {requested}:
        raise RuntimeError(
            f"Refusing to run: expected all parameters in {requested}, found dtypes {dtypes}. "
            "Loading must be explicit bf16, never mixed or auto-selected."
        )


def resolve_device(requested: str) -> str:
    """"auto" deliberately does NOT consider MPS: PyTorch's MPS backend has known gaps in
    bf16 op coverage that can silently fall back to a different numeric path rather than
    raising, which is a worse failure mode for a numerically-sensitive sanity gate than
    just being slow on CPU. Pass device: "mps" explicitly to opt in (logged loudly)."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning(
            "device=auto: CUDA not available, defaulting to CPU (NOT mps) -- "
            "see resolve_device() docstring for why mps isn't auto-selected."
        )
        return "cpu"
    if requested == "mps":
        logger.warning(
            "device=mps requested explicitly: PyTorch's bf16 op coverage on MPS is not "
            "fully verified here. Stage B's identity check (gate check 1) is exactly the "
            "test that would catch silent numerical drift from this -- do not skip it."
        )
    return requested


@dataclasses.dataclass
class LoadedModel:
    lens_model: HFLensModel
    tokenizer: object
    hf_model: torch.nn.Module
    device: str
    auto_class_used: str


def load_model(cfg: ModelConfig) -> LoadedModel:
    dtype = _DTYPE_MAP[cfg.dtype]
    device = resolve_device(cfg.device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.hf_repo, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code
    )

    load_kwargs = dict(
        revision=cfg.revision,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )

    auto_classes: list[tuple[str, type]]
    if cfg.auto_class == "auto":
        auto_classes = [
            ("AutoModelForCausalLM", transformers.AutoModelForCausalLM),
            ("AutoModelForImageTextToText", transformers.AutoModelForImageTextToText),
        ]
    else:
        auto_classes = [(cfg.auto_class, getattr(transformers, cfg.auto_class))]

    hf_model = None
    auto_class_used = None
    errors = []
    for name, cls in auto_classes:
        try:
            hf_model = cls.from_pretrained(cfg.hf_repo, **load_kwargs)
            auto_class_used = name
            break
        except Exception as e:  # noqa: BLE001 -- deliberately broad, we try the next class
            errors.append(f"{name}: {e}")
    if hf_model is None:
        raise RuntimeError(
            f"Could not load {cfg.hf_repo} with any of {[n for n, _ in auto_classes]}:\n"
            + "\n".join(errors)
        )
    logger.info("loaded %s via transformers.%s", cfg.hf_repo, auto_class_used)

    _assert_not_quantized(hf_model)
    _assert_requested_dtype(hf_model, dtype)

    hf_model = hf_model.to(device)
    hf_model.eval()

    lens_model = from_hf(hf_model, tokenizer)
    logger.info(
        "%s: n_layers=%d d_model=%d device=%s auto_class=%s layout=%s",
        cfg.hf_repo, lens_model.n_layers, lens_model.d_model, device, auto_class_used,
        lens_model.layout,
    )

    return LoadedModel(
        lens_model=lens_model,
        tokenizer=tokenizer,
        hf_model=hf_model,
        device=device,
        auto_class_used=auto_class_used,
    )
