"""Stage C: behavioral pre-check.

For each item, establishes whether the subject model actually recovers the intended word
from the misspelling -- with no chain of thought -- BEFORE any lens is applied. This is
deliberately a different prompt than the one the lens reads in Stage D: the bank's own
`eval_render: "plain"` read (a bare completion prompt, scored at one position) tells us
nothing about whether the model "knows" the correction; the question here is direct and
chat-templated, matching the bench's own `gate_variant: "family_chat"` curation gate in
spirit (that gate's exact wording isn't published anywhere we could find -- see
`stage_c_behavioral.verification_template`'s comment in config, confirmed with the user as
our own explicit design choice rather than a guess at reproducing theirs).

Scored analysis (Stage D) runs only on the items that pass here: an item the model cannot
solve has no latent intermediate to find, and only adds noise to the eval.
"""

from __future__ import annotations

import dataclasses

import torch

from typo_readout.config import StageCConfig
from typo_readout.model_io import LoadedModel

MAX_NEW_TOKENS = 60


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


def _build_chat_text(tokenizer, verification_template: str, item_prompt: str, enable_thinking: bool) -> str:
    user_content = verification_template.format(prompt=item_prompt)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def _generate(hf_model, tokenizer, chat_text: str, *, do_sample: bool, temperature: float | None,
              num_return_sequences: int, device: str) -> list[str]:
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
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
    texts = [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out
    ]
    return texts


def _hits(texts: list[str], target: str) -> int:
    """Reuses the upstream word+exact matcher (do not reimplement scoring): each
    generated text is one "sample", scored exactly as the typo family's own mechanical
    metric scores lens/AO samples (see baseline_evals/single_token/README.md)."""
    from global_workspace.olens_suite.bank import matching  # local import: needs sys.path bootstrap

    return sum(1 for t in texts if matching.hit_any([t], [target]))


def run_stage_c(
    loaded: LoadedModel,
    items: list[dict],
    cfg: StageCConfig,
) -> list[ItemBehavioralResult]:
    results: list[ItemBehavioralResult] = []
    for item in items:
        target = item["intermediates"][0]
        chat_text = _build_chat_text(
            loaded.tokenizer, cfg.verification_template, item["prompt"], cfg.chat_enable_thinking
        )

        greedy_texts = _generate(
            loaded.hf_model, loaded.tokenizer, chat_text,
            do_sample=False, temperature=None, num_return_sequences=1, device=loaded.device,
        )
        greedy_text = greedy_texts[0]
        greedy_correct = _hits([greedy_text], target) > 0

        sampled_texts = _generate(
            loaded.hf_model, loaded.tokenizer, chat_text,
            do_sample=True, temperature=cfg.temperature, num_return_sequences=cfg.n_trials,
            device=loaded.device,
        )
        n_sampled_correct = _hits(sampled_texts, target)

        stage_c_pass = greedy_correct and (n_sampled_correct >= cfg.pass_threshold)

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
            )
        )
    return results


def retention_rate(results: list[ItemBehavioralResult]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.stage_c_pass) / len(results)


def render_markdown(results: list[ItemBehavioralResult], *, model_repo: str, cfg: StageCConfig) -> str:
    passed = [r for r in results if r.stage_c_pass]
    lines = [
        "# Stage C — behavioral pre-check",
        "",
        f"Model: `{model_repo}`",
        f"Verification prompt (our own design, chat-templated, enable_thinking="
        f"{cfg.chat_enable_thinking}): {cfg.verification_template.strip()!r}",
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
