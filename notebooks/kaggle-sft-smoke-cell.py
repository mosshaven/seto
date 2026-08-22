# Paste this file into one Kaggle notebook cell.
# Required input: /kaggle/input/notebooks/alleydick/*/seto/

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


INPUT_ROOTS = (
    Path("/kaggle/input/notebooks/alleydick"),
    Path("/kaggle/input/alleydick"),
)
WORK_ROOT = Path("/kaggle/working")
REPO = WORK_ROOT / "seto"
BASE_MODEL = WORK_ROOT / "seto-pretrain"
TOKENIZER = WORK_ROOT / "seto-tokenizer"
SMOKE_OUTPUT = WORK_ROOT / "seto-sft-smoke"


def run(*command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


source_candidates = sorted(
    path
    for root in INPUT_ROOTS
    if root.exists()
    for path in root.glob("*/seto")
)
if not source_candidates:
    raise FileNotFoundError(
        "Missing /kaggle/input/notebooks/alleydick/*/seto"
    )
complete_sources = [
    path for path in source_candidates
    if (path / "seto-small" / "final_pretrain.zip").is_file()
]
if not complete_sources:
    raise FileNotFoundError("No notebook snapshot contains final_pretrain.zip")
source = max(complete_sources, key=lambda path: path.stat().st_mtime)
pretrain_zip = source / "seto-small" / "final_pretrain.zip"
print(f"Source: {source}")
print(f"Pretrain ZIP: {pretrain_zip}")

# Recreate writable code copy, excluding every large previous-training artifact.
for path in (REPO, BASE_MODEL, TOKENIZER, SMOKE_OUTPUT):
    if path.exists():
        shutil.rmtree(path)


def ignore_large_artifacts(_directory, names):
    blocked = {
        "seto-small",
        "seto-output",
        "output",
        "checkpoints",
        ".cache",
        "__pycache__",
    }
    return [name for name in names if name in blocked]


shutil.copytree(source, REPO, ignore=ignore_large_artifacts)
if not (REPO / ".git").exists():
    shutil.rmtree(REPO)
    run("git", "clone", "https://github.com/mosshaven/seto.git", REPO)
else:
    run("git", "pull", "--ff-only", cwd=REPO)

# Read ZIP directly from read-only Kaggle input. Never copy its ~2GB payload.
BASE_MODEL.mkdir(parents=True)
tokenizer_members = {}
with zipfile.ZipFile(pretrain_zip) as archive:
    names = archive.namelist()
    for filename in ("model.pt", "config.json"):
        matches = [
            name for name in names
            if name.endswith("/" + filename) and "/tokenizer/" not in name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {filename} in ZIP, found {matches}")
        with archive.open(matches[0]) as src, (BASE_MODEL / filename).open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    for filename in ("tokenizer.json", "config.json"):
        matches = [
            name for name in names
            if name.endswith("/tokenizer/" + filename)
        ]
        if len(matches) == 1:
            tokenizer_members[filename] = matches[0]
    if "tokenizer.json" in tokenizer_members:
        TOKENIZER.mkdir(parents=True)
        for filename, member in tokenizer_members.items():
            with archive.open(member) as src, (TOKENIZER / filename).open("wb") as dst:
                shutil.copyfileobj(src, dst)

# Older final ZIPs did not package tokenizer recursively. Copy only tokenizer
# files from attached input in that case, never old checkpoints.
if not (TOKENIZER / "tokenizer.json").exists():
    tokenizer_candidates = [
        path for path in source.glob("**/tokenizer.json")
        if "checkpoints" not in path.parts
    ]
    if not tokenizer_candidates:
        raise FileNotFoundError("No tokenizer.json found in final ZIP or notebook input")
    tokenizer_json = max(tokenizer_candidates, key=lambda path: path.stat().st_mtime)
    TOKENIZER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_json, TOKENIZER / "tokenizer.json")
    tokenizer_config = tokenizer_json.with_name("config.json")
    if tokenizer_config.exists():
        shutil.copy2(tokenizer_config, TOKENIZER / "config.json")

run(sys.executable, "-m", "pip", "install", "-q", "tokenizers", "datasets")
run(
    sys.executable,
    "scripts/check_sft_batch.py",
    "--dataset", "datasets/test-sft.jsonl",
    "--tokenizer", TOKENIZER,
    "--seq-len", "1024",
    cwd=REPO,
)
run(
    "torchrun", "--standalone", "--nproc_per_node=2",
    "scripts/train.py",
    "--stage", "sft",
    "--model-config", "small",
    "--model-config-file", BASE_MODEL / "config.json",
    "--dataset", "datasets/test-sft.jsonl",
    "--tokenizer", TOKENIZER,
    "--init-from", BASE_MODEL,
    "--delete-init-from-after-load",
    "--output-dir", SMOKE_OUTPUT,
    "--batch-size", "1",
    "--grad-accum", "2",
    "--seq-len", "1024",
    "--warmup-steps", "2",
    "--max-steps", "10",
    "--save-every", "100",
    "--log-every", "1",
    cwd=REPO,
)

# Base weights were deleted immediately after all ranks loaded init_from.
assert not BASE_MODEL.exists()
usage = shutil.disk_usage(WORK_ROOT)
print(f"Smoke output: {SMOKE_OUTPUT}")
print(f"Working disk free: {usage.free / 1024**3:.2f} GiB")
