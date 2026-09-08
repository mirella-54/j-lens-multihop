"""Downloading and loading the pre-fitted Jacobian lens.

Stage A only downloads and hashes the file (`download_lens_files`); Stage B is what
actually deserialises it into a `jlens.JacobianLens` and runs the sanity gate.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from jlens import JacobianLens

from typo_readout.config import REPO_ROOT, LensConfig
from typo_readout.provenance import ArtifactRecord

LENS_CACHE_DIR = REPO_ROOT / "data" / "lenses"


def _sibling(filename: str, new_name: str) -> str:
    """Same directory as `filename`, different basename."""
    return str(Path(filename).with_name(new_name))


def download_lens_files(lens: LensConfig) -> list[ArtifactRecord]:
    """Downloads the lens checkpoint plus its sidecar provenance files (config.yaml /
    CREDIT.md / convergence.csv -- whichever exist next to it; not every lens directory
    has all three). Returns one ArtifactRecord per file actually present.

    Does NOT deserialize the .pt -- that happens in Stage B via jlens.JacobianLens.load,
    which also validates the checkpoint shape.
    """
    records: list[ArtifactRecord] = []
    local_dir = LENS_CACHE_DIR / lens.repo.replace("/", "__")

    main_path = hf_hub_download(
        repo_id=lens.repo,
        repo_type="model",
        filename=lens.filename,
        revision=lens.revision,
        local_dir=str(local_dir),
    )
    records.append(
        ArtifactRecord.from_file(
            main_path,
            source=f"hf:{lens.repo}@{lens.revision or 'main'}/{lens.filename}",
        )
    )

    for sidecar_name in ("config.yaml", "CREDIT.md"):
        sidecar_filename = _sibling(lens.filename, sidecar_name)
        try:
            sidecar_path = hf_hub_download(
                repo_id=lens.repo,
                repo_type="model",
                filename=sidecar_filename,
                revision=lens.revision,
                local_dir=str(local_dir),
            )
        except Exception:
            continue  # not every lens directory ships every sidecar file
        records.append(
            ArtifactRecord.from_file(
                sidecar_path,
                source=f"hf:{lens.repo}@{lens.revision or 'main'}/{sidecar_filename}",
            )
        )

    return records


def resolved_lens_path(lens: LensConfig) -> Path:
    local_dir = LENS_CACHE_DIR / lens.repo.replace("/", "__")
    return local_dir / lens.filename


def load_lens(lens: LensConfig) -> JacobianLens:
    """Deserialize the lens already fetched by `download_lens_files`. Does not touch the
    network -- run download_lens_files (Stage A) first, or this raises FileNotFoundError."""
    path = resolved_lens_path(lens)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/download_artifacts.py against this config first."
        )
    jlens_obj = JacobianLens.load(str(path))
    if lens.layers is not None:
        missing = sorted(set(lens.layers) - set(jlens_obj.source_layers))
        if missing:
            raise ValueError(
                f"config.lens.layers={lens.layers} requests layers {missing} not in this "
                f"lens's source_layers={jlens_obj.source_layers}"
            )
    return jlens_obj
