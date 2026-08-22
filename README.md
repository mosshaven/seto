# Seto

Tiny language model for mobile deployment. Bilingual: Russian + English.

## Architecture

| Spec | tiny (~200M) | small (~500M) | base (~1B) |
|------|-------------|---------------|------------|
| d_model | 1024 | 1280 | 2048 |
| Layers | 14 | 24 | 22 |
| Heads | 16 (Q) / 4 (KV) | 20 (Q) / 5 (KV) | 16 (Q) / 4 (KV) |
| FFN | 2816 (SwiGLU) | 3584 (SwiGLU) | 5504 (SwiGLU) |
| Context | 1024 | 2048 | 2048 |
| Vocab | 48k | 48k | 48k |
| Norm | RMSNorm | RMSNorm | RMSNorm |
| Position | RoPE | RoPE | RoPE |
| Attention | GQA + SDPA | GQA + SDPA | GQA + SDPA |
| Embeddings | Tied | Tied | Tied |

## Pipeline

```
1. Train tokenizer (48k BPE, RU/UK/EN)
        ↓
2. Download data (FineWeb2 RU + Wikipedia RU)
        ↓
3. Tokenize → pack uint16 shards
        ↓
4. Pretrain (~200M for testing, ~500M for real)
        ↓
5. SFT (weighted UltraChat + WildChat + OASST + RU + curated synthetic)
        ↓
6. (optional) DPO
        ↓
seto-final.zip
```

## Quick Start

```bash
# 1. Train tokenizer
python scripts/prepare_data.py --output-dir data --tokenizer-dir seto-tokenizer

# 2. Prepare shards
python scripts/prepare_data.py --output-dir data --tokenizer-dir seto-tokenizer --skip-tokenizer

# 3. Pretrain
torchrun --nproc_per_node=2 scripts/train.py \
  --stage pretrain --model-config tiny \
  --data-dir data/shards --output-dir output

# 4. Chat
python scripts/chat.py --model output/final_pretrain --tokenizer seto-tokenizer
```

Final exports package model config and tokenizer, so SFT ZIPs can be opened
directly without selecting architecture manually:

```bash
python scripts/chat.py --model output/final_sft.zip
python scripts/chat.py --model output/final_sft/model.pt
python scripts/chat.py --model output/final_sft.zip --prompt "Привет, кто ты?"
```

Use `--tokenizer PATH` only for older exports that did not package tokenizer
files recursively.

## Data

### Pretrain
- FineWeb2 `rus_Cyrl` (85%)
- Wikipedia `20231101.ru` (15%)

### SFT
- UltraChat (40%)
- WildChat cleaned (25%)
- OpenAssistant reviewed paths (15%)
- Russian instruction/dialogue data (10%)
- Curated synthetic data (10%)

### Tokenizer
- 48k BPE vocab
- Trained on RU (80%) + UK (10%) + EN (10%)
- Special tokens: `<bos>`, `<eos>`, `<pad>`, `<|system|>`, `<|user|>`,
  `<|assistant|>`, `<|tool_call|>`, `<|tool_result|>`

### SFT data preparation

SFT records use one canonical format:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Convert pinned OASST1 trees into one deterministic, reviewed path per tree:

```bash
python scripts/prepare_oasst.py \
  --languages ru,en --max-turns 12 --max-chars 12000 \
  --min-reviews 1 \
  --output datasets/sft/oasst1.jsonl \
  --metadata-output datasets/sft/oasst1.meta.jsonl \
  --manifest-output datasets/sft/oasst1.manifest.json
```

Filter general SFT data while preserving useful bounded refusals. Valid
roleplay can be routed into a separate future personality dataset:

```bash
python scripts/sft_filter.py \
  --input datasets/sft/raw.jsonl \
  --output datasets/sft/clean.jsonl \
  --roleplay-output datasets/sft/roleplay.jsonl --stats
```

Validation is curated separately and never sampled from OASST training trees.
See `datasets/validation/README.md`.

Build the weighted v1 corpus after every source has been converted and
filtered. Weights apply to trainable assistant/tool-call turns rather than raw
conversation rows. Sampling never repeats records; unavailable quota is
redistributed and reported in the manifest.

```bash
python scripts/mix_sft.py \
  --config configs/sft-v1.json \
  --output datasets/sft/seto-sft-v1.jsonl \
  --metadata-output datasets/sft/seto-sft-v1.meta.jsonl \
  --manifest-output datasets/sft/seto-sft-v1.manifest.json
```

### SFT smoke test

Run on Kaggle or Colab before using the full mix. The fixture contains
multi-turn dialogue and one complete tool-call/result/final-answer sequence.

```bash
python scripts/check_sft_batch.py \
  --dataset datasets/test-sft.jsonl \
  --tokenizer seto-tokenizer --seq-len 1024

python scripts/train.py \
  --stage sft --model-config small \
  --dataset datasets/test-sft.jsonl \
  --tokenizer seto-tokenizer \
  --init-from seto-small/final_pretrain.zip \
  --output-dir smoke-sft --max-steps 10 --save-every 10
```

Preflight must report 15 SFT targets: 14 assistant and one tool call. Training
must load the old checkpoint, resize vocabulary when needed, avoid shape
mismatches, and produce finite loss.

## Training Notes

- **FP16** for T4 (Turing) — no bf16 on T4
- **SDPA** attention — auto-selects best backend
- **uint16 shards** — 1B tokens ≈ 2GB
- **Full checkpoints** — model + optimizer + scheduler + RNG + tokens_seen
- **Resume** between Kaggle sessions

## Kaggle

1. Upload `seto/` as dataset
2. Run `scripts/prepare_data.py` (tokenizer + shards)
3. Run `scripts/train.py --stage pretrain`
4. Checkpoints auto-saved to `/kaggle/working/`

## License

MIT
