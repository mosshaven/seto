# Seto SFT Validation

Validation v1 contains exactly 1,000 curated conversations. It is evaluation
data, never SFT input.

## Quotas

Each category contains 100 records: 50 Russian and 50 English.

| Category | Coverage |
|---|---|
| `math` | arithmetic, word problems, quantitative reasoning |
| `code` | generation, debugging, explanation, edge cases |
| `explanation` | factual and conceptual explanations |
| `dialogue` | context retention, correction, personality |
| `safety` | bounded refusal plus useful safe alternative |
| `tools` | valid calls, results, missing-tool handling |
| `ambiguity` | clarification instead of invented assumptions |
| `toxicity` | calm handling without mirroring abuse |
| `rewriting` | summary, translation, tone and format changes |
| `colloquial` | informal Russian and English, typos, slang |

Every category must mix `easy`, `medium`, and `hard` examples. Dataset should
contain at least 25% multi-turn conversations and 20% cases where clarification,
bounded refusal, or explicit uncertainty is correct.

## Files

- `seto-sft-val-v1.jsonl`: model-facing records containing only `messages`.
- `seto-sft-val-v1.meta.jsonl`: provenance and scoring metadata, one row per
  model record.

Model-facing format:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Required metadata:

```json
{
  "id": "val-ru-math-0001",
  "line": 1,
  "language": "ru",
  "category": "math",
  "difficulty": "medium",
  "expected_behavior": "correct_result_with_concise_reasoning",
  "origin_type": "human_authored",
  "review_status": "manually_checked",
  "source": "internal",
  "license": "internal",
  "content_sha256": "..."
}
```

Allowed `origin_type` values are `human_authored`, `public_benchmark`,
`adapted_human`, and `model_draft_human_edited`. Generated drafts qualify only
after manual editing. `review_status` is `curated` during construction and
`manually_checked` after final review. Source URL, original item ID, and
adaptation notes should be retained for public benchmark records.

## Freeze And Leakage

1. Curate prompts and reference answers.
2. Manually check every final item.
3. Generate canonical SHA-256 values and freeze v1.
4. Run exact normalized leakage checks against every SFT source.
5. Manually inspect near-duplicate candidates before training.

Validation must not be sampled from OASST training trees. Common prompts alone
are weak evaluation; prefer cases with explicit acceptance criteria.

Run structural, quota, provenance, duplicate, and exact leakage checks:

```bash
python scripts/validate_sft_val.py \
  --data datasets/validation/seto-sft-val-v1.jsonl \
  --metadata datasets/validation/seto-sft-val-v1.meta.jsonl \
  --train datasets/sft/seto-sft-v1.jsonl
```
