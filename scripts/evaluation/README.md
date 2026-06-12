# Evaluation Scripts

Dual-judge evaluation of inference results using Azure OpenAI. Both `JUDGE_LLM_1` and `JUDGE_LLM_2` independently score every row, with inter-judge agreement (Cohen's Kappa) computed automatically.

## `run_evaluation.py`

Main evaluation CLI. Runs absolute scoring + pairwise comparison, saves results to blob storage, generates charts, and logs to MLflow.

### Architecture

- **2 API calls per row per judge** (absolute): Dutch quality (grammar, fluency, vocabulary, language mixing) + instruction following
- **2 API calls per row per judge** (pairwise): Dutch-quality comparison and instruction-following comparison are now issued as separate single-dimension calls (independent position randomisation each) to avoid halo effects.
- **Structured outputs**: all judge calls use Pydantic response models via `client.beta.chat.completions.parse()`
- **Parallel execution**: all (row × judge) tasks submitted to a single thread pool
- **Deterministic position swap**: pairwise A/B layout is seeded per (pair_idx, dimension), so reruns are reproducible while still removing first-position bias on aggregate.
- **Length-bias mitigation**: every judge prompt (absolute and pairwise) explicitly tells the judge to ignore response length and judge only on quality / coverage.
- **Baseline caching**: baseline row-level scores are cached in blob and reused across experiments

### Prerequisites

- `.env` file with `ENDPOINT`, `JUDGE_LLM_1`, `JUDGE_LLM_2`
- Inference results in blob: `inference-results/{model_label}_results.jsonl`
- Azure credentials (via `DefaultAzureCredential`)

### Usage

```bash
# Evaluate baseline (scores are cached for reuse)
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label baseline

# Evaluate finetuned model (runs pairwise vs cached baseline)
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label mistral_r16_a16_e1_b16_w30

# Use local file instead of blob, skip MLflow
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label baseline \
    --local-results outputs/local_test/baseline_results.jsonl \
    --skip-mlflow
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--config` | required | Path to `qlora_config.json` |
| `--model-label` | required | `baseline` or wandb run name |
| `--local-results` | None | Local JSONL file override |
| `--storage-account` | `llmaml5615532443` | Azure storage account |
| `--max-workers` | 4 | Concurrent API calls |
| `--skip-mlflow` | false | Skip MLflow logging |

### How it works

1. Loads `inference-results/{model_label}_results.jsonl` from blob (or `--local-results`).
2. **Absolute scoring**: each row is scored by both judges in parallel — Dutch quality (grammar, fluency, vocabulary, language mixing) and instruction following, returning Pydantic-validated results with `j1_`/`j2_` prefixed columns.
3. **Pairwise comparison** (finetuned only): for each row, two independent single-dimension judge calls (Dutch quality + instruction following) are made by both judges, each with its own randomised A/B position derived deterministically from the pair index.
4. **Aggregation**: per-judge and combined means, plus inter-judge agreement rate and Cohen's Kappa.
5. **Charts**: per-judge score distributions, baseline-vs-finetuned bars, pairwise win charts, language mixing rates, and agreement heatmap.
6. **Persistence**: row scores, pairwise results, aggregate JSON, and chart PNGs uploaded to blob under `eval-results/{model_label}/`.
7. **MLflow**: aggregate metrics logged to the experiment.

### Output (blob storage)

```
eval-results/
  baseline/
    row_scores.jsonl          # Per-row j1_/j2_ prefixed scores
    aggregate.json            # Per-judge + combined metrics + Cohen's Kappa
    charts/                   # PNG charts per judge + agreement heatmap
  mistral_r16_a16_e1_b16_w30/
    row_scores.jsonl
    pairwise.jsonl            # Pairwise win/tie/loss per judge
    aggregate.json
    charts/
```

## `dashboard.py`

Streamlit dashboard for cross-experiment comparison. Auto-discovers all experiments from blob storage.

### Usage

```bash
streamlit run scripts/evaluation/dashboard.py
```

### Features

- Summary table with all experiments and key metrics
- Combined + per-judge score bar charts
- Score deltas vs baseline
- Pairwise win rate comparison
- Language mixing rate charts
- Inter-judge agreement table (agreement rate + Cohen's Kappa)
- Experiment detail drill-down
- Auto-refreshes every 2 minutes from blob storage

## Library Modules

| Module | Description |
|---|---|
| `finetuning/evaluation.py` | Judge calls, parallel runners, aggregation, persistence, MLflow |
| `finetuning/eval_visualization.py` | Per-judge charts + agreement heatmap generation |
| `finetuning/judge_prompts.py` | System prompts (Dutch quality, instruction following, pairwise quality, pairwise instruction) |
| `finetuning/schemas.py` | Pydantic response models (`DutchQualityResult`, `InstructionFollowingResult`, `PairwiseSingleResult`) |

---

## Azure ML Job Submission

Run evaluation as an Azure ML job for distributed compute, integrated monitoring, and automatic logging.

### Scripts

- **`submit_eval_job.py`**: Submits an evaluation job to Azure ML with configurable baseline handling
- **`job_eval.py`**: Azure ML job entry point (runs on compute cluster) — orchestrates the full evaluation pipeline
- **`azureml/eval_job_config.json`**: Configuration template for job defaults (compute, timeout, storage paths)

### Setup

1. **Ensure environment file exists**:
   ```bash
   # .env must contain:
   ENDPOINT=https://your-azure-openai-endpoint.openai.azure.com/
   JUDGE_LLM_1=gpt-4-judge-1-deployment
   JUDGE_LLM_2=gpt-4-judge-2-deployment
   ```

2. **Inference results prepared**:
   - Baseline inference results in blob: `inference-results/baseline_results.jsonl`
   - Finetuned model inference results in blob: `inference-results/{model_label}_results.jsonl`

3. **Baseline scores cached** (if `--baseline-eval false`):
   - Run baseline evaluation first OR use `--baseline-eval true` on first finetuned job

### Usage

#### Option 1: Reuse Cached Baseline Scores (Faster)

After running baseline evaluation once:

```bash
python scripts/evaluation/submit_eval_job.py \
    --model-label mistral_r16_a16_e1_b16_w30 \
    --baseline-eval false
```

**Flow**:
1. Job loads cached baseline scores from blob
2. Evaluates finetuned model absolutely
3. Runs pairwise comparison (finetuned vs cached baseline)
4. Uploads results to blob and logs to MLflow

**Typical duration**: ~20–30 minutes (2 judges × 1000 rows)

#### Option 2: Run Baseline Evaluation First (Initial Setup)

```bash
python scripts/evaluation/submit_eval_job.py \
    --model-label baseline \
    --baseline-eval true
```

**Flow**:
1. Job evaluates baseline model absolutely (both judges in parallel)
2. Caches baseline scores to blob
3. Uploads results to blob and logs to MLflow

**Typical duration**: ~20–30 minutes (2 judges × 1000 rows)

#### Option 3: Dry-Run (Test)

```bash
python scripts/evaluation/submit_eval_job.py \
    --model-label mistral_r16_a16_e1_b16_w30 \
    --baseline-eval false \
    --test-size 5
```

**Behavior**:
- Evaluates only 5 samples (quick test)
- Skips blob storage save and MLflow logging
- Saves results locally to `outputs/eval_dry_run/`

**Typical duration**: ~2–3 minutes

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--model-label` | ✓ | — | Model to evaluate (`baseline` or wandb run name) |
| `--baseline-eval` | | `false` | Whether to run baseline evaluation (`true`/`false`) |
| `--test-size` | | None | Dry-run: evaluate only N samples, skip blob + MLflow |
| `--config` | | `azureml/eval_job_config.json` | Job configuration file |
| `--qlora-config` | | `configs/qlora_config.json` | QLoRA config (for judge endpoints) |

### Configuration

Edit `azureml/eval_job_config.json` to customize:

```json
{
  "azureml": {
    "compute": "gpu-cluster",
    "timeout_minutes": 480,
    "instance_count": 1
  ## Environment Setup
  },
  ### Conda Environments
  "evaluation": {
  The project uses **separate conda environments** for inference and evaluation:
    "max_workers": 4
  - **`azureml/conda_inference.yml`** (lightweight): For inference-only jobs
    - Unsloth, transformers, CUDA
    - Azure storage (blob, identity)
    - Minimal dependencies (~300 MB)

  - **`azureml/conda_eval.yml`** (evaluation-specific): For evaluation jobs
    - All inference dependencies +
    - **Judge dependencies**: `openai`, `anthropic`
    - **Evaluation**: `scikit-learn` (Cohen's Kappa), `aiohttp` (async calls)
    - **Visualization**: `matplotlib`, `seaborn`
    - Total size ~500 MB (adds judge models + charting)
  }
  **Why separate?**
  - Inference jobs don't need judge models or scikit-learn
  - Keeps inference environment lean
  - Evaluation environment ready for Azure ML job submissions
  - Different scaling considerations: inference is compute-heavy; evaluation is I/O-heavy
}

| Setting | Default | Description |
|---|---|---|
| `azureml.compute` | `gpu-cluster` | Compute target name |
| `azureml.timeout_minutes` | `480` | Job timeout (8 hours) |
| `azureml.instance_count` | `1` | Number of compute instances |
| `evaluation.max_workers` | `4` | Concurrent judge API calls |

### Job Monitoring

After submission, you can:

1. **Stream logs** (default): Logs printed to terminal in real-time
2. **View in Azure ML Studio**: Link printed at submission
3. **Check blob storage**: Results appear under `eval-results/{model_label}/` once job completes
4. **Query MLflow**: Metrics logged to `dutch-mistral` experiment

### Typical Workflow

```bash
# Step 1: Submit baseline evaluation (one-time setup)
python scripts/evaluation/submit_eval_job.py \
    --model-label baseline \
    --baseline-eval true
# → Waits for completion, baseline scores cached to blob

# Step 2: After finetuning, evaluate new model
python scripts/evaluation/submit_eval_job.py \
    --model-label mistral_r16_a16_e1_b16_w30 \
    --baseline-eval false
# → Uses cached baseline, fast pairwise comparison

# Step 3: Evaluate another finetuned model variant
python scripts/evaluation/submit_eval_job.py \
    --model-label mistral_r16_a16_e3_b32_w50 \
    --baseline-eval false
# → Same baseline cached, can compare multiple runs

# Step 4: View all results in dashboard
streamlit run scripts/evaluation/dashboard.py
```

### Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `FileNotFoundError: No blobs found under inference-results/` | Inference results not in blob | Run inference job first; ensure results in `inference-results/{model_label}_results.jsonl` |
| `Baseline scores not cached` | Using `--baseline-eval false` but no cached baseline | Run `--baseline-eval true` first to cache baseline scores |
| Job timeout | Too many samples or slow judge API | Increase `azureml.timeout_minutes` in config or use `--test-size 10` to test locally first |
| `ENDPOINT` environment variable not set | Missing .env file | Create `.env` with `ENDPOINT=...`, `JUDGE_LLM_1=...`, `JUDGE_LLM_2=...` |
| Job fails with missing packages | Conda environment not installed | Verify `azureml/conda_inference.yml` has all dependencies (openai, anthropic, scikit-learn, aiohttp) |

### Performance Notes

- **Baseline evaluation**: ~20–30 min per 1000 rows (2 judges × 3 calls each)
- **Finetuned + pairwise**: ~30–40 min per 1000 rows (finetuned absolute + pairwise for both judges)
- **Cost**: GPU cluster charges during job; typical job ~\$5–10 USD
- **Parallel judges**: Both judges run concurrently; doubling judges roughly doubles time (not quadruples)
