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
5. SFT (OASST2 RU + Saiga Scored + Easy Instructions)
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

## Data

### Pretrain
- FineWeb2 `rus_Cyrl` (85%)
- Wikipedia `20231101.ru` (15%)

### SFT
- `IlyaGusev/saiga_scored` (45%, score >= 8)
- `IlyaGusev/oasst2_ru_main_branch` (30%)
- `attn-signs/russian-easy-instructions` (25%)

### Tokenizer
- 48k BPE vocab
- Trained on RU (80%) + UK (10%) + EN (10%)
- Special tokens: `<bos>`, `<eos>`, `<pad>`, `<|system|>`, `<|user|>`, `<|assistant|>`

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
