# Paste this file into one Kaggle notebook cell.
# Input: /kaggle/input/notebooks/alleydick/*/seto/seto-small/final_pretrain.zip

import json
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
WORK = Path("/kaggle/working")
REPO = WORK / "seto"
HF_CACHE = WORK / "hf-sft-cache"
SOURCES = WORK / "seto-sft-sources"
MIX = WORK / "seto-sft-mix"
BASE_MODEL = WORK / "seto-pretrain"
TOKENIZER = WORK / "seto-tokenizer"
OUTPUT = WORK / "seto-sft-v1"
OLD_SMOKE_OUTPUT = WORK / "seto-sft-smoke"


def run(*command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


source_candidates = sorted(
    path
    for root in INPUT_ROOTS
    if root.exists()
    for path in root.glob("*/seto")
    if (path / "seto-small" / "final_pretrain.zip").is_file()
)
if not source_candidates:
    raise FileNotFoundError(
        "Missing /kaggle/input/notebooks/alleydick/*/seto/seto-small/final_pretrain.zip"
    )
source = max(source_candidates, key=lambda path: path.stat().st_mtime)
pretrain_zip = source / "seto-small" / "final_pretrain.zip"
print(f"Source: {source}")
print(f"Pretrain ZIP: {pretrain_zip}")

# Clean only disposable artifacts produced by this cell.
for path in (
    REPO,
    HF_CACHE,
    SOURCES,
    MIX,
    BASE_MODEL,
    TOKENIZER,
    OUTPUT,
    OLD_SMOKE_OUTPUT,
):
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
if (REPO / ".git").exists():
    run("git", "pull", "--ff-only", cwd=REPO)
else:
    shutil.rmtree(REPO)
    run("git", "clone", "https://github.com/mosshaven/seto.git", REPO)

run(sys.executable, "-m", "pip", "install", "-q", "tokenizers", "datasets")
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")

# Stream public data into compact canonical JSONL files.
run(
    sys.executable,
    "scripts/prepare_sft_sources.py",
    "--output-dir", SOURCES,
    "--seed", "42",
    cwd=REPO,
)
run(
    sys.executable,
    "scripts/prepare_oasst.py",
    "--languages", "ru,en,uk",
    "--max-turns", "12",
    "--max-chars", "8000",
    "--min-reviews", "1",
    "--output", SOURCES / "oasst1.jsonl",
    "--metadata-output", SOURCES / "oasst1.meta.jsonl",
    "--manifest-output", SOURCES / "oasst1.manifest.json",
    cwd=REPO,
)

# Download/cache payload is no longer needed after canonical JSONL exists.
shutil.rmtree(HF_CACHE, ignore_errors=True)

MIX.mkdir(parents=True)
run(
    sys.executable,
    "scripts/mix_sft.py",
    "--config", "configs/sft-v1-first-run.json",
    "--output", MIX / "seto-sft-v1.jsonl",
    "--metadata-output", MIX / "seto-sft-v1.meta.jsonl",
    "--manifest-output", MIX / "seto-sft-v1.manifest.json",
    cwd=REPO,
)
manifest = json.loads((MIX / "seto-sft-v1.manifest.json").read_text())
print(json.dumps({
    "selected_records": manifest["selected_records"],
    "selected_targets": manifest["selected_targets"],
    "target_deficit": manifest["target_deficit"],
    "weights": {
        name: round(info["actual_target_weight"], 4)
        for name, info in manifest["sources"].items()
    },
}, indent=2))
if manifest["target_deficit"] != 0:
    raise RuntimeError("SFT mix did not reach exactly 50,000 trainable targets")

# Stream model/config from read-only input ZIP; never copy ZIP or old checkpoints.
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
        matches = [name for name in names if name.endswith("/tokenizer/" + filename)]
        if len(matches) == 1:
            tokenizer_members[filename] = matches[0]
    if "tokenizer.json" in tokenizer_members:
        TOKENIZER.mkdir(parents=True)
        for filename, member in tokenizer_members.items():
            with archive.open(member) as src, (TOKENIZER / filename).open("wb") as dst:
                shutil.copyfileobj(src, dst)

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

run(
    sys.executable,
    "scripts/check_sft_batch.py",
    "--dataset", MIX / "seto-sft-v1.jsonl",
    "--tokenizer", TOKENIZER,
    "--seq-len", "1024",
    cwd=REPO,
)

# About one epoch: 50k targets / (2 GPUs * batch 1 * accumulation 8) ~= 3125.
run(
    "torchrun", "--standalone", "--nproc_per_node=2",
    "scripts/train.py",
    "--stage", "sft",
    "--model-config", "small",
    "--model-config-file", BASE_MODEL / "config.json",
    "--dataset", MIX / "seto-sft-v1.jsonl",
    "--tokenizer", TOKENIZER,
    "--init-from", BASE_MODEL,
    "--delete-init-from-after-load",
    "--output-dir", OUTPUT,
    "--batch-size", "1",
    "--grad-accum", "8",
    "--seq-len", "1024",
    "--lr", "2e-5",
    "--warmup-steps", "100",
    "--max-steps", "3000",
    "--save-every", "500",
    "--log-every", "10",
    cwd=REPO,
)

final_zip = OUTPUT / "final_sft.zip"
if not final_zip.exists():
    raise FileNotFoundError(f"Training finished without {final_zip}")

# Successful final ZIP supersedes rotating optimizer checkpoint and train data.
shutil.copy2(MIX / "seto-sft-v1.manifest.json", OUTPUT / "data_manifest.json")
shutil.copy2(SOURCES / "sources.manifest.json", OUTPUT / "sources_manifest.json")
shutil.copy2(SOURCES / "oasst1.manifest.json", OUTPUT / "oasst1_manifest.json")
shutil.rmtree(OUTPUT / "checkpoints_sft", ignore_errors=True)
shutil.rmtree(SOURCES, ignore_errors=True)
shutil.rmtree(MIX, ignore_errors=True)
shutil.rmtree(TOKENIZER, ignore_errors=True)
usage = shutil.disk_usage(WORK)
print(f"Final model: {final_zip}")
print(f"Final size: {final_zip.stat().st_size / 1024**3:.2f} GiB")
print(f"Working disk free: {usage.free / 1024**3:.2f} GiB")
