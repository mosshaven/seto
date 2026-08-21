"""Seto checkpoint management — directory-based, ZIP only for final export."""

import json
import shutil
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
    ckpt_dir = save_dir / ckpt_name

    # Delete target dir if it already exists (atomic-ish replace)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model (unwrap DDP)
    state_dict = model.state_dict()
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    torch.save(state_dict, ckpt_dir / "model.pt")

    # Save optimizer
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    # Save scheduler
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    # Save GradScaler
    if scaler is not None and hasattr(scaler, "state_dict"):
        torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")

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

    # Clean old checkpoints BEFORE returning (keeps disk usage low)
    _cleanup_old_checkpoints(save_dir, keep_last_n)

    return str(ckpt_dir)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    path = Path(checkpoint_path)

    if path.suffix == ".zip":
        # Extract ZIP — only rank 0, then barrier
        extract_dir = path.parent / path.stem
        is_distributed = torch.distributed.is_initialized()
        is_main = (not is_distributed) or torch.distributed.get_rank() == 0

        if is_main:
            extract_dir.mkdir(parents=True, exist_ok=True)
            import zipfile
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extract_dir)

        if is_distributed:
            torch.distributed.barrier()

        # ZIP contains step_XXXXXXXX/model.pt
        subdirs = sorted(extract_dir.iterdir()) if extract_dir.exists() else []
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

    # Load scheduler state
    if (path / "scheduler.pt").exists():
        meta["scheduler"] = torch.load(path / "scheduler.pt", map_location=device, weights_only=True)

    # Load GradScaler state
    if (path / "scaler.pt").exists():
        meta["scaler"] = torch.load(path / "scaler.pt", map_location=device, weights_only=True)

    # Load RNG state
    if (path / "rng.pt").exists():
        meta["rng"] = torch.load(path / "rng.pt", map_location=device, weights_only=False)

    return meta


def _cleanup_old_checkpoints(save_dir: Path, keep_last_n: int = 1):
    """Keep only the N most recent checkpoint dirs/zips."""
    # Directories
    dirs = sorted(save_dir.glob("step_*"), key=lambda x: x.stat().st_mtime)
    for old in dirs[: len(dirs) - keep_last_n]:
        shutil.rmtree(old, ignore_errors=True)

    # Legacy ZIPs
    for old in save_dir.glob("seto_step_*.zip"):
        old.unlink(missing_ok=True)


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    save_dir = Path(save_dir)

    # Prefer directories (our native format)
    dirs = sorted(save_dir.glob("step_*"), key=lambda x: x.stat().st_mtime)
    if dirs:
        return str(dirs[-1])

    # Fallback to ZIPs (legacy or exported)
    zips = sorted(save_dir.glob("seto_step_*.zip"), key=lambda x: x.stat().st_mtime)
    if zips:
        return str(zips[-1])

    return None


def clean_checkpoints(save_dir: str):
    """Remove all checkpoint dirs and zips."""
    save_dir = Path(save_dir)
    for d in save_dir.glob("step_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    for f in save_dir.glob("seto_step_*.zip"):
        f.unlink(missing_ok=True)


def zip_checkpoint(ckpt_dir: str, output_path: str) -> str:
    """ZIP a checkpoint directory for export. Returns path to zip."""
    import zipfile
    ckpt_dir = Path(ckpt_dir)
    output_path = Path(output_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        for fp in ckpt_dir.iterdir():
            zf.write(fp, f"{ckpt_dir.name}/{fp.name}")

    return str(output_path)
