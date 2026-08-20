# Seto

Tiny language model (~1B params) designed for mobile deployment.

## Architecture

| Spec | Value |
|------|-------|
| Type | Decoder-only Transformer |
| Parameters | ~1.04B |
| Vocab | 32,000 |
| d_model | 2048 |
| Layers | 22 |
| Heads | 16 (Q) / 4 (KV, GQA) |
| FFN dim | 5504 (SwiGLU) |
| Context | 2048 tokens |
| Norm | RMSNorm |
| Position | RoPE (θ=10000) |
| Embeddings | Tied |

After INT4 quantization: ~520MB — runs on phones.

## Training

### Kaggle (2x GPU)

1. Upload `seto/` folder to Kaggle dataset
2. Add training data (FineWeb-Edu, SlimPajama, etc.)
3. Open `notebooks/seto-train.ipynb`
4. Run all cells

### Local

```bash
# Single GPU
python scripts/train.py --data-dir ./data --output-dir ./output

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 scripts/train.py --data-dir ./data --output-dir ./output
```

## Checkpoints

Checkpoints saved every 1000 steps as ZIP archives:

```
checkpoints/
  seto_step_0001000.zip
  seto_step_0002000.zip
  best/
    seto_step_0005000.zip
```

Resume training from any checkpoint:
```bash
python scripts/train.py --data-dir ./data --resume checkpoints/seto_step_0005000.zip
```

## Quantization (planned)

- INT8 / INT4 via `llama.cpp` GGUF format
- ONNX export for mobile inference
- Core ML / TFLite wrappers

## License

MIT
