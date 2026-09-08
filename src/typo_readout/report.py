"""Stage F: scoring and report.

Pure aggregation over files already written by Stages C-E -- no model, no lens, no GPU.
Metrics: harmonic mean of the rank (== 1/MRR) and pass@k, both as per-layer curves (never
collapsed to a single number until the final verdict). The workspace band is marked on
every plot; layer indices are converted to the paper's 0-100% depth scale ONLY here, at
the plotting boundary -- every stored artifact upstream of this file uses raw indices.
"""

from __future__ import annotations

import dataclasses

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def group_by_layer(rows: list[dict]) -> dict[int, list[dict]]:
    by_layer: dict[int, list[dict]] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
    return by_layer


def harmonic_mean_rank_curve(rows: list[dict]) -> dict[int, float]:
    """Per layer: harmonic_mean(target_rank) == 1/MRR over items at that layer. The
    brief's primary probing metric -- reported in rank units (interpretable directly),
    not the 0-1 MRR itself, though the two carry identical information."""
    curve = {}
    for layer, layer_rows in group_by_layer(rows).items():
        ranks = [r["target_rank_lens"] for r in layer_rows]
        curve[layer] = len(ranks) / sum(1.0 / rk for rk in ranks)
    return curve


def pass_at_k_curve(rows: list[dict], k: int) -> dict[int, float]:
    """Per layer: fraction of items whose target rank at THAT layer is <= k. (Distinct
    from the family's own "best-over-grid" pass@k used for the permutation-null ratio in
    Stage E -- this is the per-layer curve the brief asks for; that one is a single
    per-item boolean over the whole grid, needed for an apples-to-apples chance-line
    comparison.)"""
    curve = {}
    for layer, layer_rows in group_by_layer(rows).items():
        curve[layer] = sum(1 for r in layer_rows if r["target_rank_lens"] <= k) / len(layer_rows)
    return curve


def layer_to_pct_depth(layer: int, n_layers_total: int) -> float:
    """The paper's 0-100% rescaling, applied ONLY here (the plotting boundary) -- every
    stored artifact upstream uses raw 0-indexed layer numbers. `n_layers_total - 1` so the
    true final layer maps to 100%, matching the paper's own convention."""
    return 100.0 * layer / (n_layers_total - 1)


def _workspace_band_layers(workspace_band_pct: list[float], n_layers_total: int) -> tuple[float, float]:
    lo_pct, hi_pct = workspace_band_pct
    return (lo_pct / 100.0) * (n_layers_total - 1), (hi_pct / 100.0) * (n_layers_total - 1)


def _shade_workspace_band(ax, workspace_band_pct: list[float], n_layers_total: int) -> None:
    lo, hi = _workspace_band_layers(workspace_band_pct, n_layers_total)
    ax.axvspan(lo, hi, color="tab:green", alpha=0.08, zorder=0, label="workspace band (paper)")


def _secondary_pct_axis(ax, n_layers_total: int) -> None:
    def fwd(layer):
        return 100.0 * layer / (n_layers_total - 1)

    def inv(pct):
        return pct / 100.0 * (n_layers_total - 1)

    secax = ax.secondary_xaxis("top", functions=(fwd, inv))
    secax.set_xlabel("% depth (paper's 0-100 rescaling)")
    secax.xaxis.set_major_formatter(mticker.PercentFormatter())


def plot_harmonic_mean_rank(
    curves: dict[str, dict[int, float]],
    *,
    n_layers_total: int,
    workspace_band_pct: list[float],
    out_path: str,
) -> None:
    """`curves`: {"j_lens": {layer: value}, "logit_lens": {layer: value}}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"j_lens": "tab:blue", "logit_lens": "tab:orange"}
    for method, curve in curves.items():
        layers = sorted(curve)
        ax.plot(layers, [curve[l] for l in layers], marker="o", markersize=3,
                 label=method, color=colors.get(method))
    _shade_workspace_band(ax, workspace_band_pct, n_layers_total)
    _secondary_pct_axis(ax, n_layers_total)
    ax.set_yscale("log")
    ax.set_xlabel("layer (raw index)")
    ax.set_ylabel("harmonic mean of target rank (== 1/MRR; lower is better)")
    ax.set_title("Primary probing metric: J-lens vs logit lens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pass_at_k(
    curves_by_k: dict[int, dict[str, dict[int, float]]],
    *,
    n_layers_total: int,
    workspace_band_pct: list[float],
    out_path: str,
) -> None:
    """`curves_by_k`: {k: {"j_lens": {layer: rate}, "logit_lens": {layer: rate}}}."""
    ks = sorted(curves_by_k)
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.5), sharey=True)
    if len(ks) == 1:
        axes = [axes]
    colors = {"j_lens": "tab:blue", "logit_lens": "tab:orange"}
    for ax, k in zip(axes, ks, strict=True):
        for method, curve in curves_by_k[k].items():
            layers = sorted(curve)
            ax.plot(layers, [curve[l] for l in layers], marker="o", markersize=3,
                     label=method, color=colors.get(method))
        _shade_workspace_band(ax, workspace_band_pct, n_layers_total)
        _secondary_pct_axis(ax, n_layers_total)
        ax.set_xlabel("layer (raw index)")
        ax.set_title(f"pass@{k}")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("fraction of items")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclasses.dataclass
class Verdict:
    k: int
    observed_pass_rate: float
    chance_rate_measured: float
    chance_rate_used: float
    ratio_vs_chance: float
    required_multiple: float
    passes: bool


def compute_verdicts(permutation_results: dict, chance_multiple_required: float) -> list[Verdict]:
    verdicts = []
    for k, p in sorted(permutation_results.items()):
        ratio = p["ratio_vs_chance"]
        verdicts.append(
            Verdict(
                k=k,
                observed_pass_rate=p["observed_pass_rate"],
                chance_rate_measured=p["rate"],
                chance_rate_used=p["chance_rate_used_for_ratio"],
                ratio_vs_chance=ratio,
                required_multiple=chance_multiple_required,
                passes=(ratio is not None and ratio >= chance_multiple_required),
            )
        )
    return verdicts
