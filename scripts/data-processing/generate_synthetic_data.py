"""
Generate synthetic Dutch instruction-following data via Azure OpenAI.

Uses async batching for throughput. Each API call generates multiple examples
to reduce cost. Outputs JSONL in the same format as alpaca_train.jsonl.

Usage:
    python scripts/generate_synthetic_data.py --num-examples 5000 --output datasets/synthetic_dutch.jsonl
    python scripts/generate_synthetic_data.py --num-examples 100 --concurrency 5 --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Add project root to path so we can import finetuning helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from finetuning.blob_storage import upload_file_to_blob

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure OpenAI config (loaded from .env)
# ---------------------------------------------------------------------------
ENDPOINT = os.environ["ENDPOINT"]
DEPLOYMENT = os.environ["DEPLOYMENT"]
STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT"]
CONTAINER_NAME = os.environ["CONTAINER_NAME"]

# ---------------------------------------------------------------------------
# Topic and task variety
# ---------------------------------------------------------------------------
TOPICS = [
    # Wetenschap & Technologie
    "klimaatverandering en duurzaamheid",
    "kunstmatige intelligentie en ethiek",
    "ruimtevaart en astronomie",
    "biologie en ecosystemen",
    "medische wetenschap en gezondheid",
    "hernieuwbare energie",
    "cybersecurity en privacybescherming",
    "quantumcomputing",
    "genetica en biotechnologie",
    "robotica en automatisering",
    # Maatschappij & Cultuur
    "Nederlandse geschiedenis en tradities",
    "onderwijs en leren",
    "filosofie en ethiek",
    "psychologie en mentale gezondheid",
    "sociale media en communicatie",
    "kunst en literatuur",
    "muziek en film",
    "sport en beweging",
    "reizen en toerisme in Europa",
    "culinaire cultuur en voeding",
    # Economie & Werk
    "ondernemerschap en startups",
    "financiële planning en beleggen",
    "arbeidsmarkt en loopbaanontwikkeling",
    "internationale handel en globalisering",
    "vastgoed en woningmarkt",
    # Dagelijks leven
    "opvoeding en gezinsleven",
    "huisdieren en dierenverzorging",
    "tuinieren en natuur in de stad",
    "productiviteit en tijdmanagement",
    "vrijwilligerswerk en gemeenschap",
    # Politiek & Recht
    "democratie en burgerschap",
    "Europese Unie en samenwerking",
    "mensenrechten en gelijkheid",
    "milieuwetgeving en beleid",
    "mediavrijheid en journalistiek",
]

TASK_TYPES = [
    {
        "type": "uitleg",
        "prompt": "Maak een instructie die vraagt om een concept of onderwerp uit te leggen.",
    },
    {
        "type": "creatief schrijven",
        "prompt": "Maak een instructie die vraagt om iets creatiefs te schrijven (verhaal, gedicht, brief, dialoog).",
    },
    {
        "type": "samenvatting",
        "prompt": "Maak een instructie die vraagt om informatie samen te vatten of de kern weer te geven.",
    },
    {
        "type": "mening/argumentatie",
        "prompt": "Maak een instructie die vraagt om een mening te geven of argumenten te formuleren.",
    },
    {
        "type": "vergelijking",
        "prompt": "Maak een instructie die vraagt om twee of meer dingen te vergelijken.",
    },
    {
        "type": "advies",
        "prompt": "Maak een instructie die vraagt om praktisch advies of tips te geven.",
    },
    {
        "type": "lijst",
        "prompt": "Maak een instructie die vraagt om een lijst te maken van items, ideeën of stappen.",
    },
    {
        "type": "analyse",
        "prompt": "Maak een instructie die vraagt om iets te analyseren of kritisch te bekijken.",
    },
    {
        "type": "herschrijving",
        "prompt": "Maak een instructie die vraagt om tekst te herschrijven, verbeteren of aan te passen.",
    },
    {
        "type": "vraag en antwoord",
        "prompt": "Maak een feitelijke vraag waar een informatief antwoord op gegeven moet worden.",
    },
]

LENGTH_INSTRUCTIONS = [
    ("kort", "Het antwoord moet kort en bondig zijn (1-3 zinnen)."),
    ("middellang", "Het antwoord moet middellang zijn (1-2 alinea's, ongeveer 50-100 woorden)."),
    ("uitgebreid", "Het antwoord moet uitgebreid en gedetailleerd zijn (2-4 alinea's, ongeveer 150-300 woorden)."),
    ("zeer uitgebreid", "Het antwoord moet zeer uitgebreid zijn met meerdere alinea's, voorbeelden en nuance (300+ woorden)."),
]

# Weights: more medium/long to balance the short Alpaca data
LENGTH_WEIGHTS = [0.15, 0.30, 0.35, 0.20]

EXAMPLES_PER_CALL = 5

SYSTEM_PROMPT = """Je bent een expert in het genereren van hoogwaardige Nederlandstalige trainingsdata voor het finetunen van een taalmodel.

Je taak is om instructie-antwoord paren te genereren die:
- In natuurlijk, vloeiend Nederlands zijn geschreven (geen vertaald Engels)
- Gevarieerd zijn in onderwerp, stijl en lengte
- Feitelijk correct en informatief zijn
- Compleet en afgerond zijn (het antwoord moet een duidelijk einde hebben)

Genereer EXACT het gevraagde aantal voorbeelden in het opgegeven JSON-formaat.
Gebruik GEEN markdown codeblokken om de JSON heen. Geef alleen de raw JSON array terug."""


def build_generation_prompt(batch_topics, batch_tasks, batch_lengths):
    """Build a prompt that asks for multiple instruction-output pairs."""
    examples_spec = []
    for i, (topic, task, (length_label, length_desc)) in enumerate(
        zip(batch_topics, batch_tasks, batch_lengths), 1
    ):
        examples_spec.append(
            f"  {i}. Onderwerp: {topic}\n"
            f"     Taaktype: {task['type']} — {task['prompt']}\n"
            f"     Lengte: {length_label} — {length_desc}"
        )

    prompt = f"""Genereer {len(batch_topics)} Nederlandstalige instructie-antwoord paren volgens de specificaties hieronder.

Specificaties per voorbeeld:
{chr(10).join(examples_spec)}

Geef het resultaat als een JSON array met objecten die elk de volgende velden hebben:
- "instruction": de instructie/vraag (in het Nederlands)
- "input": optionele extra context bij de instructie (laat leeg als niet nodig, gebruik "" als waarde)
- "output": het volledige antwoord (in het Nederlands)

Belangrijk:
- Schrijf in natuurlijk Nederlands, niet in vertaald Engels
- Elk antwoord moet compleet zijn en een duidelijk einde hebben
- Varieer in schrijfstijl en toon
- Geef ALLEEN de JSON array terug, geen andere tekst"""

    return prompt


def sample_batch_params(n):
    """Sample random topics, tasks, and lengths for a batch."""
    topics = random.choices(TOPICS, k=n)
    tasks = random.choices(TASK_TYPES, k=n)
    lengths = random.choices(LENGTH_INSTRUCTIONS, weights=LENGTH_WEIGHTS, k=n)
    return topics, tasks, lengths


def parse_response(text):
    """Parse the JSON array from the model response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        items = json.loads(text)
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse response as JSON array")
    return []


async def generate_batch(client, semaphore, batch_id, n=EXAMPLES_PER_CALL):
    """Generate a batch of examples with concurrency control."""
    topics, tasks, lengths = sample_batch_params(n)
    prompt = build_generation_prompt(topics, tasks, lengths)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=2048,
            )
            text = response.choices[0].message.content
            items = parse_response(text)

            valid = []
            for item in items:
                if (
                    isinstance(item, dict)
                    and "instruction" in item
                    and "output" in item
                    and item["instruction"].strip()
                    and item["output"].strip()
                ):
                    valid.append(
                        {
                            "instruction": item["instruction"].strip(),
                            "input": item.get("input", "").strip(),
                            "output": item["output"].strip(),
                        }
                    )

            logger.info(
                f"Batch {batch_id}: generated {len(valid)}/{n} valid examples"
            )
            return valid

        except Exception as e:
            logger.error(f"Batch {batch_id} failed: {e}")
            return []


async def generate_all(num_examples, concurrency, output_path, dry_run=False, no_upload=False):
    """Generate all examples with async batching."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    client = AsyncOpenAI(
        base_url=ENDPOINT,
        api_key=token_provider(),
    )

    num_batches = (num_examples + EXAMPLES_PER_CALL - 1) // EXAMPLES_PER_CALL
    semaphore = asyncio.Semaphore(concurrency)

    logger.info(
        f"Generating ~{num_examples} examples in {num_batches} batches "
        f"({EXAMPLES_PER_CALL} per batch, concurrency={concurrency})"
    )

    if dry_run:
        topics, tasks, lengths = sample_batch_params(EXAMPLES_PER_CALL)
        prompt = build_generation_prompt(topics, tasks, lengths)
        print("\n=== DRY RUN: Sample prompt ===\n")
        print(prompt)
        print(f"\n=== Would make {num_batches} API calls ===")
        return

    start = time.time()

    # Launch all batches concurrently (semaphore limits actual concurrency)
    tasks = [
        generate_batch(client, semaphore, i + 1) for i in range(num_batches)
    ]
    results = await asyncio.gather(*tasks)

    # Flatten and assign IDs
    all_examples = []
    for batch in results:
        all_examples.extend(batch)

    # Trim to exact count and assign IDs
    all_examples = all_examples[:num_examples]
    for i, ex in enumerate(all_examples, 1):
        ex["id"] = i
        ex["prompt"] = (
            ex["instruction"] + "\ninput: " + ex["input"]
        ).strip()

    elapsed = time.time() - start

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(all_examples)} examples in {elapsed:.1f}s")
    logger.info(f"Written to {output_path}")

    # Quick stats
    output_lengths = [len(ex["output"].split()) for ex in all_examples]
    if output_lengths:
        avg_len = sum(output_lengths) / len(output_lengths)
        logger.info(
            f"Output word length: min={min(output_lengths)}, "
            f"avg={avg_len:.0f}, max={max(output_lengths)}"
        )

    # Upload to blob storage
    if not no_upload:
        blob_name = output_path.name
        try:
            url = upload_file_to_blob(
                storage_account=STORAGE_ACCOUNT,
                container_name=CONTAINER_NAME,
                blob_name=blob_name,
                local_path=str(output_path),
            )
            logger.info(f"Uploaded to blob storage: {url}")
        except Exception as e:
            logger.error(f"Failed to upload to blob storage: {e}")
    else:
        logger.info("Skipping blob storage upload (--no-upload)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Dutch instruction-following data via Azure OpenAI."
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5000,
        help="Number of examples to generate (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/synthetic_dutch.jsonl",
        help="Output JSONL path (default: datasets/synthetic_dutch.jsonl)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a sample prompt without making API calls",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to blob storage",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(
        generate_all(
            args.num_examples,
            args.concurrency,
            args.output,
            args.dry_run,
            args.no_upload,
        )
    )


if __name__ == "__main__":
    main()
