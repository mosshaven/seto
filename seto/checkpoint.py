"""Seto checkpoint management with ZIP packaging."""

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
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = save_dir / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()

    torch.save(state_dict, ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    meta = {
        "step": step,
        "loss": loss,
        "config": config,
        "model_class": model.__class__.__name__,
    }
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    zip_path = None
    if zip_it:
        zip_path = save_dir / f"seto_step_{step:08d}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(ckpt_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(save_dir)
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
        extract_dir = path.parent / path.stem
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        path = extract_dir

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
