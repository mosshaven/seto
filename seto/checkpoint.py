"""Seto checkpoint management — streaming ZIP, no intermediate directories.

Peak disk per checkpoint: single ZIP file (~5.5GB for small).
No directory + ZIP duplication.
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: dict,
    save_dir: str,
    keep_last_n: int = 1,
    scheduler=None,
    scaler=None,
    rng_state: Optional[dict] = None,
    tokens_seen: int = 0,
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"step_{step:08d}"
    zip_path = save_dir / f"seto_{ckpt_name}.zip"
    tmp_path = save_dir / f"seto_{ckpt_name}.tmp"

    # Unwrap DDP
    state_dict = model.state_dict()
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()

    # Low-disk: remove old checkpoints BEFORE writing new one
    _cleanup_old_checkpoints(save_dir, keep_last_n, exclude=tmp_path)

    # Write everything directly into ZIP via tmp file
    # (atomic-ish: if crash mid-write, .tmp won't match seto_step_*.zip glob)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
        with zf.open("model.pt", "w") as f:
            torch.save(state_dict, f)

        with zf.open("optimizer.pt", "w") as f:
            torch.save(optimizer.state_dict(), f)

        if scheduler is not None:
            with zf.open("scheduler.pt", "w") as f:
                torch.save(scheduler.state_dict(), f)

        if scaler is not None and hasattr(scaler, "state_dict"):
            with zf.open("scaler.pt", "w") as f:
                torch.save(scaler.state_dict(), f)

        if rng_state is not None:
            with zf.open("rng.pt", "w") as f:
                torch.save(rng_state, f)

        meta = {
            "step": step,
            "loss": loss,
            "tokens_seen": tokens_seen,
            "config": config,
        }
        zf.writestr("meta.json", json.dumps(meta, indent=2))

    # Atomic rename: .tmp → final name
    tmp_path.rename(zip_path)

    return str(zip_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    path = Path(checkpoint_path)

    if path.suffix == ".zip":
        is_distributed = torch.distributed.is_initialized()
        is_main = (not is_distributed) or torch.distributed.get_rank() == 0

        with zipfile.ZipFile(path, "r") as zf:
            # Load model
            with zf.open("model.pt") as f:
                state_dict = torch.load(f, map_location=device, weights_only=True)
            if hasattr(model, "module"):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

            # Load optimizer
            if optimizer is not None and "optimizer.pt" in zf.namelist():
                with zf.open("optimizer.pt") as f:
                    optimizer.load_state_dict(
                        torch.load(f, map_location=device, weights_only=True)
                    )

            # Load metadata
            meta = {}
            if "meta.json" in zf.namelist():
                meta = json.loads(zf.read("meta.json").decode())

            if "scheduler.pt" in zf.namelist():
                with zf.open("scheduler.pt") as f:
                    meta["scheduler"] = torch.load(f, map_location=device, weights_only=True)

            if "scaler.pt" in zf.namelist():
                with zf.open("scaler.pt") as f:
                    meta["scaler"] = torch.load(f, map_location=device, weights_only=True)

            if "rng.pt" in zf.namelist():
                with zf.open("rng.pt") as f:
                    meta["rng"] = torch.load(f, map_location=device, weights_only=False)

        return meta

    # Legacy: plain directory
    state_dict = torch.load(path / "model.pt", map_location=device, weights_only=True)
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    if optimizer is not None and (path / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location=device, weights_only=True)
        )

    meta = {}
    if (path / "meta.json").exists():
        with open(path / "meta.json") as f:
            meta = json.load(f)

    if (path / "scheduler.pt").exists():
        meta["scheduler"] = torch.load(path / "scheduler.pt", map_location=device, weights_only=True)

    if (path / "scaler.pt").exists():
        meta["scaler"] = torch.load(path / "scaler.pt", map_location=device, weights_only=True)

    if (path / "rng.pt").exists():
        meta["rng"] = torch.load(path / "rng.pt", map_location=device, weights_only=False)

    return meta


def _cleanup_old_checkpoints(save_dir: Path, keep_last_n: int = 1, exclude: Optional[Path] = None):
    """Keep only the N most recent checkpoint zips. Also cleans legacy dirs."""
    zips = sorted(save_dir.glob("seto_step_*.zip"), key=lambda x: x.stat().st_mtime)
    for old in zips[: len(zips) - keep_last_n]:
        if exclude and old == exclude:
            continue
        old.unlink(missing_ok=True)

    # Clean legacy directories
    for d in save_dir.glob("step_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    save_dir = Path(save_dir)

    # Prefer ZIPs (native format)
    zips = sorted(save_dir.glob("seto_step_*.zip"), key=lambda x: x.stat().st_mtime)
    if zips:
        return str(zips[-1])

    # Fallback to legacy directories
    dirs = sorted(save_dir.glob("step_*"), key=lambda x: x.stat().st_mtime)
    if dirs:
        return str(dirs[-1])

    return None


def clean_checkpoints(save_dir: str):
    """Remove all checkpoint zips and legacy dirs."""
    save_dir = Path(save_dir)
    for f in save_dir.glob("seto_step_*.zip"):
        f.unlink(missing_ok=True)
    for d in save_dir.glob("step_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def zip_checkpoint(ckpt_dir: str, output_path: str) -> str:
    """ZIP a plain directory for export. Returns path to zip."""
    ckpt_dir = Path(ckpt_dir)
    output_path = Path(output_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        for fp in ckpt_dir.iterdir():
            zf.write(fp, f"{ckpt_dir.name}/{fp.name}")

    return str(output_path)
