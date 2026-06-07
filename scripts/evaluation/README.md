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
