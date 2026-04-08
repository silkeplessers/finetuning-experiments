"""
Clean the alpaca_data_cleaned-dutch.jsonl dataset.

Removes:
  1. Duplicate (instruction + input) pairs
  2. English / untranslated rows
  3. Coding examples (code blocks, programming tasks)

Writes cleaned output to datasets/alpaca_data_cleaned-dutch-clean.jsonl
and prints a summary of what was removed.
"""

import json
import re
import sys
from pathlib import Path

INPUT_PATH = Path("datasets/alpaca_data_cleaned-dutch.jsonl")
OUTPUT_PATH = Path("datasets/alpaca_data_cleaned-dutch-clean.jsonl")

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
rows = []
with INPUT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # skip malformed lines

total = len(rows)
print(f"Loaded {total:,} rows from {INPUT_PATH}")

# Track removal reasons (a row can match multiple, but is counted once)
reasons: dict[int, list[str]] = {}  # id -> [reason, ...]


def flag(row, reason):
    rid = row["id"]
    reasons.setdefault(rid, []).append(reason)


# ---------------------------------------------------------------------------
# 2. Flag duplicates on (instruction, input)
# ---------------------------------------------------------------------------
seen = set()
for r in rows:
    key = (r["instruction"].strip(), r.get("input", "").strip())
    if key in seen:
        flag(r, "duplicate")
    else:
        seen.add(key)

# ---------------------------------------------------------------------------
# 3. Flag English / untranslated rows
# ---------------------------------------------------------------------------

# 3a. Explicit "no translation" markers
NO_TRANS = re.compile(
    r"no translation needed|no translation required|"
    r"niet vertaald|geen vertaling nodig",
    re.IGNORECASE,
)

for r in rows:
    if NO_TRANS.search(r["instruction"]) or NO_TRANS.search(r["output"]):
        flag(r, "no_translation")

# 3b. Instruction is fully English (not translated at all)
#     Heuristic: high ratio of English-only function words
DUTCH_COMMON = set(
    "de het een van en in is dat op te voor met als aan er zijn was worden "
    "niet door ook uit nog kan naar maar om bij al wel dan zou hun hem haar "
    "meer geen dit werd tot ze had wat heeft die ik je we".split()
)
ENGLISH_ONLY = (
    set(
        "the a an are were be been being have has had do does did will would "
        "shall should can could may might must to of and that it for on with "
        "as at by from or but not this which what when where who how all each "
        "every some any no".split()
    )
    - DUTCH_COMMON
)  # subtract words shared with Dutch


def english_word_ratio(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 5:
        return 0.0
    return sum(1 for w in words if w in ENGLISH_ONLY) / len(words)


for r in rows:
    # Flag if the instruction itself is English (>20% English function words)
    if english_word_ratio(r["instruction"]) > 0.20:
        flag(r, "english_instruction")
    # Flag if the output is predominantly English (>25% English function words)
    # Use a slightly higher threshold for output to avoid flagging rows
    # that quote English titles/names in otherwise Dutch text
    if english_word_ratio(r["output"]) > 0.25:
        flag(r, "english_output")

# 3c. Rows about translating FROM another language where the output is not Dutch
TRANSLATION_TASK = re.compile(
    r"vertaal.*(naar het engels|to english|from dutch|van het nederlands naar)",
    re.IGNORECASE,
)
for r in rows:
    if TRANSLATION_TASK.search(r["instruction"]):
        flag(r, "translation_to_english")

# ---------------------------------------------------------------------------
# 4. Flag coding examples
# ---------------------------------------------------------------------------

# 4a. Output contains code blocks
CODE_BLOCK = re.compile(r"```")

# 4b. Output contains programming language constructs
CODE_KEYWORDS = re.compile(
    r"(?:^|\s)(?:"
    r"def \w+\(|class \w+[:\(]|import \w+|from \w+ import|"
    r"function\s+\w+|var \w+\s*=|let \w+\s*=|const \w+\s*=|"
    r"public\s+(?:static|class|void)|private\s+\w+|void\s+main|"
    r"#include\s*<|System\.out|console\.log|printf\(|scanf\(|"
    r"SELECT\s+\*|CREATE\s+TABLE|INSERT\s+INTO|"
    r"<html|<div\s|<script|<body|<head|<!DOCTYPE"
    r")"
)

# 4c. Instruction explicitly asks for code/programming
CODE_INSTRUCTION = re.compile(
    r"schrijf\s+.{0,30}(?:code|functie|programma|script|algoritme)|"
    r"maak\s+.{0,30}(?:code|functie|programma|script)|"
    r"genereer\s+.{0,30}(?:code|functie|programma|script)|"
    r"(?:python|javascript|java|c\+\+|html|css|sql|regex)\s|"
    r"(?:code|functie|programma|script)\s+(?:schrijven|maken|genereren)|"
    r"codefragment|programmeertaal",
    re.IGNORECASE,
)

for r in rows:
    combined = r["instruction"] + " " + r.get("input", "") + " " + r["output"]
    if CODE_BLOCK.search(combined) or CODE_KEYWORDS.search(combined):
        flag(r, "code_in_content")
    if CODE_INSTRUCTION.search(r["instruction"]):
        flag(r, "code_instruction")

# ---------------------------------------------------------------------------
# 5. Flag math questions
# ---------------------------------------------------------------------------

# 5a. Instruction asks for math/calculation
MATH_INSTRUCTION = re.compile(
    r"bereken|berekening|optellen|aftrekken|vermenigvuldig|delen door|"
    r"vierkants\s*wortel|wortel van|kwadraat|macht van|"
    r"wat is \d+\s*[+\-×÷xX*/]\s*\d+|los op|vergelijking|"
    r"wiskundig|rekenkundig|breuk|noemer|teller|decimaal|"
    r"percentage|procent van|hoeveel is \d+|\d+\s*%\s*van|"
    r"oppervlakte|omtrek|volume|straal|diameter|"
    r"priemgetal|deelbaar|deler|veelvoud|faculteit|"
    r"gemiddelde|mediaan|modus|standaardafwijking|"
    r"sinus|cosinus|tangens|logaritme|exponent",
    re.IGNORECASE,
)

# 5b. Output is predominantly numeric (math result)
MATH_OUTPUT = re.compile(r"^\s*-?[\d.,/%°]+\s*$")  # output is just a number/percentage

# 5c. Content has math expressions
MATH_EXPRESSIONS = re.compile(
    r"\d+\s*[+\-×÷]\s*\d+\s*=|\d+\s*\*\s*\d+\s*=|\d+\s*/\s*\d+\s*=|"
    r"\^\d+|√\d+|\d+\s*mod\s*\d+|\bx\s*[=+\-]\s*\d+|"
    r"\d+\s*>\s*\d+.*\d+\s*<\s*\d+"
)

for r in rows:
    instr = r["instruction"]
    output = r["output"]
    inp = r.get("input", "")
    combined = instr + " " + inp + " " + output

    if MATH_INSTRUCTION.search(instr):
        flag(r, "math_instruction")
    elif MATH_OUTPUT.match(output):
        flag(r, "math_output")
    elif MATH_EXPRESSIONS.search(combined) and len(output) < 50:
        # Short outputs with math expressions are likely pure math problems
        flag(r, "math_expression")

# ---------------------------------------------------------------------------
# 6. Build cleaned dataset
# ---------------------------------------------------------------------------
flagged_ids = set(reasons.keys())
cleaned = [r for r in rows if r["id"] not in flagged_ids]

# Re-index IDs starting from 1
for i, r in enumerate(cleaned, start=1):
    r["id"] = i

# ---------------------------------------------------------------------------
# 7. Write output
# ---------------------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for r in cleaned:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 55}")
print(f"CLEANING SUMMARY")
print(f"{'=' * 55}")

# Count by reason category
from collections import Counter

reason_counts = Counter()
for rid, rlist in reasons.items():
    for reason in rlist:
        reason_counts[reason] += 1

category_labels = {
    "duplicate": "Duplicate (instruction + input)",
    "no_translation": "No translation / untranslated",
    "english_instruction": "English instruction (>20%)",
    "english_output": "English output (>25%)",
    "translation_to_english": "Translation-to-English task",
    "code_in_content": "Code in content",
    "code_instruction": "Code in instruction",
    "math_instruction": "Math in instruction",
    "math_output": "Math-only output (numeric)",
    "math_expression": "Math expression (short output)",
}

print(f"\nRows flagged by reason (one row can match multiple):")
for reason, label in category_labels.items():
    count = reason_counts.get(reason, 0)
    if count > 0:
        print(f"  {label:<40s} {count:>5,}")

print(f"\n{'─' * 55}")
print(f"  Total rows removed (unique):           {len(flagged_ids):>5,}")
print(f"  Rows remaining:                        {len(cleaned):>5,}")
print(
    f"  Removal rate:                          {100 * len(flagged_ids) / total:>5.1f}%"
)
print(f"{'=' * 55}")
print(f"\nCleaned dataset written to: {OUTPUT_PATH}")
