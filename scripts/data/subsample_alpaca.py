"""Subsample the cleaned Alpaca dataset to high-quality Dutch examples.

Two-stage pipeline:
  Stage 1: Heuristic pre-filter (fast, free) — narrows ~47K to ~15K candidates
  Stage 2: LLM quality scoring with gpt-5.4-mini — scores Dutch fluency,
           naturalness, and completeness on the survivors

Usage:
    python scripts/data/subsample_alpaca.py
    python scripts/data/subsample_alpaca.py --num-examples 5000
    python scripts/data/subsample_alpaca.py --dry-run          # stage 1 only, no API calls
    python scripts/data/subsample_alpaca.py --heuristic-top 10000 --concurrency 20
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from finetuning.data import heuristic_quality_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress fasttext warnings
logging.getLogger("fasttext").setLevel(logging.WARNING)

JUDGE_MODEL = "gpt-5.4-mini"

INPUT_PATH = Path("datasets/alpaca_data_cleaned-dutch-clean.jsonl")
OUTPUT_PATH = Path("datasets/alpaca_high_quality.jsonl")
SCORING_LOG_PATH = Path("datasets/subsample_scoring_log.jsonl")

BATCH_SIZE = 20  # examples per LLM call

JUDGE_SYSTEM_PROMPT = """\
Je bent een expert in de Nederlandse taal. Je beoordeelt trainingsvoorbeelden \
op drie criteria, elk met een score van 1-5.

1. **dutch_fluency**: Hoe correct en vloeiend is het Nederlands?
   1 = Grotendeels onbegrijpelijk of niet-Nederlands
   2 = Veel grammaticale fouten, onnatuurlijke zinsbouw
   3 = Begrijpelijk maar duidelijk vertaald/onnatuurlijk
   4 = Goed Nederlands met kleine onvolkomenheden
   5 = Uitstekend, vloeiend, natuurlijk Nederlands

2. **naturalness**: Hoe natuurlijk klinkt het? (Geen vertaaltaal/anglicismen)
   1 = Duidelijk woord-voor-woord vertaald uit het Engels
   2 = Vaak onnatuurlijke woordkeuze of zinsconstructies
   3 = Gemengd — sommige delen natuurlijk, andere niet
   4 = Overwegend natuurlijk met sporadische anglicismen
   5 = Volledig idiomatisch Nederlands, geen vertaaltaal

3. **completeness**: Is het antwoord volledig en bruikbaar?
   1 = Afgekapt, onvolledig of onzinnig
   2 = Grotendeels onvolledig, mist essentiële informatie
   3 = Redelijk maar mist context of nuance
   4 = Goed antwoord met kleine hiaten
   5 = Volledig, informatief en goed afgerond"""


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Stage 1: Heuristic pre-filter
# ---------------------------------------------------------------------------
def heuristic_prefilter(rows: list[dict], top_n: int) -> tuple[list[dict], list[dict]]:
    """Score all rows with fastText + Dutch GPT-2 perplexity and return (candidates, all_scored).

    all_scored is a list of dicts with id, instruction (truncated), scores, and kept/rejected status.
    """
    logger.info("Stage 1: scoring %d rows with fastText LID + Dutch GPT-2 perplexity...", len(rows))

    all_scored = []
    row_scores = []
    for r in tqdm(rows, desc="Scoring"):
        scores = heuristic_quality_score(r)
        row_scores.append((scores["total"], r, scores))

    row_scores.sort(key=lambda x: x[0], reverse=True)

    # Build log entries with kept/rejected status
    kept_ids = set()
    for _, r, _ in row_scores[:top_n]:
        kept_ids.add(r["id"])

    for total, r, scores in row_scores:
        all_scored.append({
            "id": r["id"],
            "instruction": r["instruction"][:120],
            "stage1_status": "kept" if r["id"] in kept_ids else "rejected",
            **scores,
        })

    totals = [s for s, _, _ in row_scores]
    logger.info("Heuristic score distribution (%d rows):", len(totals))
    for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        count = sum(1 for s in totals if s >= threshold)
        logger.info("  >= %.1f: %6d rows", threshold, count)

    candidates = [r for _, r, _ in row_scores[:top_n]]
    cutoff = row_scores[top_n - 1][0] if top_n <= len(row_scores) else 0
    logger.info("Stage 1: kept %d candidates (min heuristic score: %.2f)", len(candidates), cutoff)
    return candidates, all_scored


# ---------------------------------------------------------------------------
# Stage 2: LLM quality scoring
# ---------------------------------------------------------------------------
def build_judge_prompt(batch: list[dict]) -> str:
    """Build a prompt asking the judge to score a batch of examples."""
    parts = []
    for i, row in enumerate(batch, 1):
        parts.append(
            f"--- Voorbeeld {i} ---\n"
            f"Instructie: {row['instruction']}\n"
            f"Input: {row.get('input', '')}\n"
            f"Output: {row['output']}"
        )
    return (
        f"Beoordeel de volgende {len(batch)} voorbeelden. "
        f"Geef voor elk voorbeeld exact drie scores.\n\n"
        + "\n\n".join(parts)
    )


async def score_batch(
    client,
    semaphore: asyncio.Semaphore,
    batch_id: int,
    batch: list[dict],
    QualityScoreBatch,
) -> list[dict | None]:
    """Score a batch of examples via the LLM judge. Returns score dicts or None on failure."""
    prompt = build_judge_prompt(batch)

    async with semaphore:
        try:
            response = await client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=QualityScoreBatch,
                max_completion_tokens=8192,
            )
            message = response.choices[0].message

            if message.refusal:
                logger.warning("Batch %d: model refused: %s", batch_id, message.refusal)
                return [None] * len(batch)

            parsed = message.parsed
            if not parsed or len(parsed.scores) != len(batch):
                logger.warning(
                    "Batch %d: expected %d scores, got %d",
                    batch_id, len(batch), len(parsed.scores) if parsed else 0,
                )
                return [None] * len(batch)

            return [s.model_dump() for s in parsed.scores]

        except Exception as e:
            logger.error("Batch %d failed: %s", batch_id, e)
            return [None] * len(batch)


async def llm_score_all(
    candidates: list[dict],
    concurrency: int,
) -> list[tuple[float, dict]]:
    """Score all candidates with the LLM judge and return (combined_score, row) pairs."""
    from azure.identity import (DefaultAzureCredential,
                                get_bearer_token_provider)
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from finetuning.schemas import QualityScoreBatch

    load_dotenv(PROJECT_ROOT / ".env")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = AsyncOpenAI(base_url=os.environ["ENDPOINT"], api_key=token_provider())
    semaphore = asyncio.Semaphore(concurrency)

    # Split into batches
    batches = [
        candidates[i : i + BATCH_SIZE]
        for i in range(0, len(candidates), BATCH_SIZE)
    ]
    logger.info(
        "Stage 2: scoring %d candidates in %d batches (concurrency=%d)",
        len(candidates), len(batches), concurrency,
    )

    start = time.time()
    tasks = [
        score_batch(client, semaphore, i + 1, batch, QualityScoreBatch)
        for i, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # Flatten and pair with rows
    scored = []
    scored_count = 0
    failed_count = 0
    for batch, score_list in zip(batches, results):
        for row, score_dict in zip(batch, score_list):
            if score_dict is None:
                failed_count += 1
                continue
            combined = (
                score_dict["dutch_fluency"]
                + score_dict["naturalness"]
                + score_dict["completeness"]
            )
            row["_quality"] = score_dict
            row["_quality_total"] = combined
            scored.append((combined, row))
            scored_count += 1

    logger.info(
        "Stage 2: scored %d, failed %d in %.1fs",
        scored_count, failed_count, elapsed,
    )
    return scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage quality subsampling of the cleaned Alpaca dataset."
    )
    parser.add_argument(
        "--input", type=str, default=str(INPUT_PATH),
        help=f"Input JSONL path (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help=f"Output JSONL path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--scoring-log", type=str, default=str(SCORING_LOG_PATH),
        help=f"Scoring log JSONL path (default: {SCORING_LOG_PATH})",
    )
    parser.add_argument(
        "--num-examples", type=int, default=4500,
        help="Final number of examples to select (default: 4500)",
    )
    parser.add_argument(
        "--heuristic-top", type=int, default=10000,
        help="Number of candidates to keep after heuristic pre-filter (default: 10000)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent LLM scoring calls (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run stage 1 only (no API calls), print stats and exit",
    )
    return parser.parse_args()


def write_scoring_log(scoring_log: list[dict], path: str) -> None:
    """Write the full scoring log to a JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for entry in scoring_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Scoring log: %d entries written to %s", len(scoring_log), out)


async def run(args):
    rows = load_rows(args.input)
    logger.info("Loaded %d rows from %s", len(rows), args.input)

    # Stage 1: heuristic pre-filter
    candidates, scoring_log = heuristic_prefilter(rows, args.heuristic_top)

    if args.dry_run:
        logger.info("Dry run — skipping LLM scoring. Would score %d candidates.", len(candidates))
        write_scoring_log(scoring_log, args.scoring_log)
        return

    # Stage 2: LLM quality scoring
    scored = await llm_score_all(candidates, args.concurrency)

    # Sort by combined LLM score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top N
    selected_ids = set()
    selected = []
    for _, r in scored[: args.num_examples]:
        selected_ids.add(r["id"])
        selected.append(r)

    # Update scoring log with stage 2 results
    log_by_id = {entry["id"]: entry for entry in scoring_log}
    for llm_total, r in scored:
        entry = log_by_id.get(r["id"])
        if entry:
            quality = r.get("_quality", {})
            entry["llm_dutch_fluency"] = quality.get("dutch_fluency")
            entry["llm_naturalness"] = quality.get("naturalness")
            entry["llm_completeness"] = quality.get("completeness")
            entry["llm_total"] = llm_total
            entry["final_status"] = "selected" if r["id"] in selected_ids else "rejected_stage2"

    # Mark stage1-rejected entries as not scored by LLM
    for entry in scoring_log:
        if "final_status" not in entry:
            entry["final_status"] = entry["stage1_status"]

    write_scoring_log(scoring_log, args.scoring_log)

    # Clean internal fields and re-index
    for i, r in enumerate(selected, 1):
        r.pop("_quality", None)
        r.pop("_quality_total", None)
        r["id"] = i

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    output_lengths = [len(r["output"].split()) for r in selected]
    avg_len = sum(output_lengths) / len(output_lengths)
    cutoff = scored[args.num_examples - 1][0] if args.num_examples <= len(scored) else 0

    logger.info("Selected %d examples (min LLM score: %d/15)", len(selected), cutoff)
    logger.info(
        "Output word lengths: min=%d, avg=%.0f, max=%d",
        min(output_lengths), avg_len, max(output_lengths),
    )
    logger.info("Written to %s", out_path)


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
