"""Seto checkpoint management with ZIP packaging — full state."""

import json
import os
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
    zip_it: bool = True,
    keep_last_n: int = 3,
    scheduler=None,
    rng_state: Optional[dict] = None,
    tokens_seen: int = 0,
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"step_{step:08d}"
    ckpt_dir = save_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    state_dict = model.state_dict()
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    torch.save(state_dict, ckpt_dir / "model.pt")

    # Save optimizer
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    # Save scheduler
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    # Save RNG state
    if rng_state is not None:
        torch.save(rng_state, ckpt_dir / "rng.pt")

    # Save metadata
    meta = {
        "step": step,
        "loss": loss,
        "tokens_seen": tokens_seen,
        "config": config,
    }
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    zip_path = None
    if zip_it:
        zip_path = save_dir / f"seto_{ckpt_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in ckpt_dir.iterdir():
                arcname = f"{ckpt_name}/{file_path.name}"
                zf.write(file_path, arcname)

        shutil.rmtree(ckpt_dir)

    _cleanup_old_checkpoints(save_dir, keep_last_n)

    return str(zip_path or ckpt_dir)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    path = Path(checkpoint_path)

    if path.suffix == ".zip":
        # Extract ZIP to parent directory
        extract_dir = path.parent / path.stem
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        # The ZIP contains step_XXXXXXXX/model.pt etc.
        # Find the actual checkpoint directory
        subdirs = list(extract_dir.iterdir())
        if subdirs and subdirs[0].is_dir():
            path = subdirs[0]
        else:
            path = extract_dir

    # Load model
    state_dict = torch.load(path / "model.pt", map_location=device, weights_only=True)
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # Load optimizer
    if optimizer is not None and (path / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location=device, weights_only=True)
        )

    # Load metadata
    meta = {}
    if (path / "meta.json").exists():
        with open(path / "meta.json") as f:
            meta = json.load(f)

    # Load scheduler state (caller must restore separately)
    if (path / "scheduler.pt").exists():
        meta["scheduler"] = torch.load(path / "scheduler.pt", map_location=device, weights_only=True)

    # Load RNG state
    if (path / "rng.pt").exists():
        meta["rng"] = torch.load(path / "rng.pt", map_location=device, weights_only=False)

    return meta


def _cleanup_old_checkpoints(save_dir: Path, keep_last_n: int):
    zips = sorted(save_dir.glob("seto_step_*.zip"), key=lambda x: x.stat().st_mtime)
    if len(zips) > keep_last_n:
        for old_zip in zips[: len(zips) - keep_last_n]:
            old_zip.unlink(missing_ok=True)


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    save_dir = Path(save_dir)
    zips = sorted(save_dir.glob("seto_step_*.zip"), key=lambda x: x.stat().st_mtime)
    if zips:
        return str(zips[-1])

    dirs = sorted(save_dir.glob("step_*"), key=lambda x: x.stat().st_mtime)
    if dirs:
        return str(dirs[-1])

    return None
