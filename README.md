# Seto

Tiny language model for mobile deployment. Bilingual: English + Russian.

## Architecture

| Spec | small (~600M) | base (~1B) |
|------|---------------|------------|
| d_model | 1408 | 2048 |
| Layers | 24 | 22 |
| Heads | 22 (Q) / 4 (KV) | 16 (Q) / 4 (KV) |
| FFN | 3840 (SwiGLU) | 5504 (SwiGLU) |
| Context | 4096 | 2048 |
| Vocab | 40k | 32k |
| Norm | RMSNorm | RMSNorm |
| Position | RoPE | RoPE |
| Embeddings | Tied | Tied |

All configs: decoder-only Transformer + GQA + SwiGLU + RoPE + RMSNorm.

## Training Pipeline

```
raw text
  ↓
quality filter (lang detect, dedup, scoring)
  ↓
┌─────────────────────────────────────────┐
│ Stage 1: PRETRAIN                       │
│ next-token prediction on web data       │
│ 50-300B tokens, cosine LR, bf16        │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Stage 2: COOLDOWN (optional)            │
│ higher quality data, lower LR           │
│ math, code, textbooks, educational      │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Stage 3: SFT                            │
│ instruction following on chat data      │
│ 200k-2M conversations                   │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Stage 4: DPO                            │
│ preference optimization                 │
│ chosen vs rejected response pairs       │
└─────────────────────────────────────────┘
  ↓
seto-final.zip
```

## Data Mixture (pretrain)

| Domain | Weight |
|--------|--------|
| Web / educational | 50% |
| Code | 15% |
| Books | 10% |
| Math / science | 10% |
| Wikipedia | 10% |
| Synthetic | 5% |

Language split: 70% English, 30% Russian (adjustable).

## Kaggle (2x GPU)

1. Upload `seto/` as Kaggle dataset
2. Add training data (FineWeb-Edu, SlimPajama, etc.)
3. Open `notebooks/seto-train.ipynb`
4. Set `PRETRAIN_DATA`, `SFT_DATA`, `DPO_DATA` paths
5. Run stages in order

## Local

```bash
# Pretrain
torchrun --nproc_per_node=2 scripts/train.py \
  --stage pretrain --data-dir ./data \
  --output-dir ./output --max-steps 100000

# Cooldown
torchrun --nproc_per_node=2 scripts/train.py \
  --stage cooldown --data-dir ./curated_data \
  --output-dir ./output --resume output/checkpoints_pretrain/best

# SFT
python scripts/train.py \
  --stage sft --data-dir ./sft_data \
  --output-dir ./output

# DPO
python scripts/train.py \
  --stage dpo --data-dir ./dpo_data \
  --ref-model output/final_sft/model.pt \
  --output-dir ./output
```

## Checkpoints

ZIP archives every N steps:
```
checkpoints_pretrain/
  seto_step_0001000.zip
  seto_step_0002000.zip
  best/
checkpoints_sft/
  seto_step_0000500.zip
```

Resume from any:
```bash
python scripts/train.py --stage pretrain --resume checkpoints/seto_step_0005000.zip
```

## DPO Data Format

```json
{"prompt": "Write a sorting function", "chosen": "def sort(arr): ...", "rejected": "I can't do that"}
```

## SFT Data Format

```json
{"messages": [
  {"role": "system", "content": "You are Seto"},
  {"role": "user", "content": "What is Python?"},
  {"role": "assistant", "content": "Python is a programming language..."}
]}
```

## License

MIT
