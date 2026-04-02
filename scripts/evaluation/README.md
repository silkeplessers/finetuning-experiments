# Evaluation Scripts

Scripts for evaluating inference results using LLM-as-a-judge.

## `run_evaluation.py`

Evaluates model inference results by sending each (prompt, expected output, predicted output) triple to an Azure OpenAI judge model. The judge scores each response on two criteria:

1. **Dutch language quality** (1–5) — grammar, fluency, vocabulary
2. **Instruction following** (1–5) — faithfulness to the expected output

Results are uploaded back to Azure Blob Storage alongside the inference results.

### Prerequisites

- Azure credentials (via `DefaultAzureCredential`) for both blob storage and the Azure OpenAI judge endpoint
- Inference results must already exist in blob storage (produced by `run_inference.py`) or be available as a local file

### Usage

```bash
# Evaluate baseline results
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label baseline \
    --azure-endpoint https://your-endpoint.openai.azure.com \
    --judge-model grok-4-fast-reasoning

# Evaluate finetuned results
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label run_r16_a16_e1_b16 \
    --azure-endpoint https://your-endpoint.openai.azure.com \
    --judge-model grok-4-fast-reasoning

# Evaluate from a local file instead of blob storage
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label baseline \
    --azure-endpoint https://your-endpoint.openai.azure.com \
    --local-results outputs/my_inference_results.jsonl

# Increase judge concurrency
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label baseline \
    --azure-endpoint https://your-endpoint.openai.azure.com \
    --max-workers 8
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | — | Path to `qlora_config.json` (validated but not deeply used) |
| `--model-label` | Yes | — | Label identifying which model's results to evaluate (e.g. `baseline` or the W&B run name) |
| `--azure-endpoint` | Yes | — | Azure OpenAI endpoint URL for the judge model |
| `--judge-model` | No | `grok-4-fast-reasoning` | Deployment name of the judge model |
| `--local-results` | No | — | Path to a local JSONL file; skips downloading from blob storage |
| `--storage-account` | No | `llmaml5615532443` | Azure Storage account name |
| `--results-container` | No | `inference-results` | Blob container where inference results are stored |
| `--max-workers` | No | `4` | Number of concurrent judge API calls |

### How it works

1. Downloads `<model-label>/inference_results.jsonl` from the `inference-results` blob container (or reads from `--local-results`).
2. For each row, sends the prompt, expected output, and predicted output to the judge model in a single API call.
3. The judge returns a JSON object with scores and justifications for both criteria.
4. Prints a summary (mean scores) to the console.
5. Uploads `<model-label>/evaluation_results.jsonl` to the same blob container.

### Output format

The evaluation JSONL extends the inference results with additional columns:

| Field | Description |
|---|---|
| `input` | The prompt sent to the model |
| `expected_output` | The reference answer from the test set |
| `predicted_output` | The model's generated response |
| `model` | Model label |
| `dutch_quality_score` | Judge score 1–5 for Dutch language quality |
| `dutch_quality_justification` | One-sentence justification from the judge |
| `instruction_following_score` | Judge score 1–5 for instruction following |
| `instruction_following_justification` | One-sentence justification from the judge |

### Scoring rubric

**Dutch language quality:**
| Score | Meaning |
|---|---|
| 1 | Very poor: major grammar errors, largely incomprehensible or not Dutch |
| 2 | Poor: frequent grammar mistakes, unnatural phrasing |
| 3 | Acceptable: understandable but contains noticeable errors |
| 4 | Good: mostly fluent with only minor mistakes |
| 5 | Excellent: fluent, grammatically correct, natural vocabulary |

**Instruction following:**
| Score | Meaning |
|---|---|
| 1 | Completely irrelevant or fails to address the instruction |
| 2 | Partially addresses the instruction but misses key elements |
| 3 | Addresses the instruction with notable omissions or inaccuracies |
| 4 | Follows instructions well with only minor deviations |
| 5 | Perfectly follows instructions; comprehensive and accurate |
