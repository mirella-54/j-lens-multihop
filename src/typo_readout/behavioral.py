"""Stage C: behavioral pre-check.

For each item, establishes whether the subject model actually solves the item -- with no
chain of thought -- BEFORE any lens is applied. Two modes, config-driven
(`stage_c_behavioral.mode`), because different families need genuinely different prompts:

- "chat_question" (typo): the bank's own `eval_render: "plain"` read tells us nothing
  about whether the model "knows" the correction, so this asks directly via a separate,
  chat-templated verification question (our own design, not a claimed reproduction of the
  bank's unpublished gate wording -- see `stage_c_behavioral.verification_template`'s
  comment in config).
- "plain_completion" (multihop): the item's own prompt already IS the two-hop question
  ("Fact: ... is "), so the natural, literal check is just completing it -- plain text, no
  chat template, matching the family's own `eval_render` and majority `gate_variant`. No
  chat wrapping is invented here.

Scored analysis (Stage D) runs only on the items that pass here: an item the model cannot
solve has no latent intermediate to find, and only adds noise to the eval.
"""

from __future__ import annotations

import dataclasses
import re

import torch

from typo_readout.config import StageCConfig
from typo_readout.item_fields import scored_string
from typo_readout.model_io import LoadedModel


@dataclasses.dataclass
class ItemBehavioralResult:
    name: str
    prompt: str
    target: str
    greedy_text: str
    greedy_correct: bool
    sampled_texts: list[str]
    n_sampled_correct: int
    n_trials: int
    stage_c_pass: bool
    first_token_correct: bool | None = None


def _build_prompt_text(tokenizer, cfg: StageCConfig, item_prompt: str) -> tuple[str, bool]:
    """Returns (text_to_tokenize, add_special_tokens)."""
    if cfg.mode == "chat_question":
        user_content = cfg.verification_template.format(prompt=item_prompt)
        messages = [{"role": "user", "content": user_content}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=cfg.chat_enable_thinking,
        )
        # False: the chat template text above already embeds its own special tokens
        # (<|im_start|> etc.) as literal text -- re-adding them would double them up.
        return text, False
    if cfg.mode == "plain_completion":
        # Default (True): matches HFLensModel.encode()'s convention (used throughout
        # Stage D's actual lens reads), so Stage C sees the same token sequence Stage D
        # will read from -- no chat wrapping invented for a family that doesn't use one.
        return item_prompt, True
    raise ValueError(f"stage_c_behavioral.mode={cfg.mode!r} not implemented")


def _generate(hf_model, tokenizer, text: str, *, add_special_tokens: bool, max_new_tokens: int,
              do_sample: bool, temperature: float | None, num_return_sequences: int,
              device: str) -> list[str]:
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0
    with torch.no_grad():
        out = hf_model.generate(**inputs, **gen_kwargs)
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


def _hits(texts: list[str], target: str, *, answer_prefix: bool = False) -> int:
    """Reuses the upstream word+exact matcher (do not reimplement scoring): each
    generated text is one "sample", scored exactly as the family's own mechanical metric
    scores lens/AO samples (see baseline_evals/single_token/README.md)."""
    from global_workspace.olens_suite.bank import matching  # local import: needs sys.path bootstrap

    if answer_prefix:
        # Only the initial answer counts, not a later option list or explanation.
        pattern = re.compile(r"^\s*" + re.escape(target) + r"(?!\w|[.,]\d)", re.IGNORECASE)
        texts = [t for t in texts if pattern.match(t)]
    return sum(1 for t in texts if matching.hit_any([t], [target]))


def run_stage_c(
    loaded: LoadedModel,
    items: list[dict],
    cfg: StageCConfig,
) -> list[ItemBehavioralResult]:
    results: list[ItemBehavioralResult] = []
    for item in items:
        target = scored_string(item, cfg.scored_field)
        text, add_special_tokens = _build_prompt_text(loaded.tokenizer, cfg, item["prompt"])

        greedy_texts = _generate(
            loaded.hf_model, loaded.tokenizer, text, add_special_tokens=add_special_tokens,
            max_new_tokens=cfg.max_new_tokens, do_sample=False, temperature=None,
            num_return_sequences=1, device=loaded.device,
        )
        greedy_text = greedy_texts[0]
        plain = cfg.mode == "plain_completion"
        greedy_correct = _hits([greedy_text], target, answer_prefix=plain) > 0
        first_token_correct = None
        if plain:
            from typo_readout.causal import compute_unpatched_logits
            from typo_readout.probe import resolve_continuation_token
            token = resolve_continuation_token(loaded.lens_model, item["prompt"], target)
            logits, _ = compute_unpatched_logits(loaded.lens_model, item["prompt"])
            first_token_correct = token.is_single_token and int(logits[-1].argmax()) == token.token_id

        sampled_texts = _generate(
            loaded.hf_model, loaded.tokenizer, text, add_special_tokens=add_special_tokens,
            max_new_tokens=cfg.max_new_tokens, do_sample=True, temperature=cfg.temperature,
            num_return_sequences=cfg.n_trials, device=loaded.device,
        )
        n_sampled_correct = _hits(sampled_texts, target, answer_prefix=plain)

        stage_c_pass = greedy_correct and (first_token_correct is not False) and (n_sampled_correct >= cfg.pass_threshold)

        results.append(
            ItemBehavioralResult(
                name=item["name"],
                prompt=item["prompt"],
                target=target,
                greedy_text=greedy_text,
                greedy_correct=greedy_correct,
                sampled_texts=sampled_texts,
                n_sampled_correct=n_sampled_correct,
                n_trials=cfg.n_trials,
                stage_c_pass=stage_c_pass,
                first_token_correct=first_token_correct,
            )
        )
    return results


def retention_rate(results: list[ItemBehavioralResult]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.stage_c_pass) / len(results)


def render_markdown(results: list[ItemBehavioralResult], *, model_repo: str, cfg: StageCConfig) -> str:
    passed = [r for r in results if r.stage_c_pass]
    if cfg.mode == "chat_question":
        prompt_desc = (
            f"Verification prompt (our own design, chat-templated, enable_thinking="
            f"{cfg.chat_enable_thinking}): {cfg.verification_template.strip()!r}"
        )
    else:
        prompt_desc = (
            f"Mode: plain completion of the item's own prompt (no chat template), "
            f"scored against `{cfg.scored_field}`. No one-hop sub-check: the bank provides "
            "no one-hop prompt field (see Stage A characterization) -- retention below "
            "confirms end-to-end composability only, not hop-by-hop factual knowledge."
        )
    lines = [
        "# Stage C — behavioral pre-check",
        "",
        f"Model: `{model_repo}`",
        prompt_desc,
        f"Pass rule: greedy answer names the target AND >= {cfg.pass_threshold}/{cfg.n_trials} "
        f"sampled answers (temp={cfg.temperature}) name it.",
        "",
        f"## Retention rate: {len(passed)}/{len(results)} = {retention_rate(results):.1%}",
        "",
        "Scored analysis (Stage D) runs on the passing subset only -- items the model "
        "cannot solve have no latent intermediate to find and only add noise. A low "
        "retention rate is a finding, not something to fix by loosening this check.",
        "",
        "| item | target | greedy pass | sampled pass | overall |",
        "|---|---|:---:|---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.target!r} | {'✓' if r.greedy_correct else '✗'} | "
            f"{r.n_sampled_correct}/{r.n_trials} | {'PASS' if r.stage_c_pass else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"
