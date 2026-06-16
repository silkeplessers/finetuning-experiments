from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    instruction: str
    input: str
    output: str


class TrainingExamples(BaseModel):
    examples: list[TrainingExample]


class QualityScore(BaseModel):
    dutch_fluency: int = Field(ge=1, le=5)
    naturalness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)


class QualityScoreBatch(BaseModel):
    scores: list[QualityScore]


# ── Judge response models ────────────────────────────────────────────────────


class DutchQualityResult(BaseModel):
    """Structured response from the Dutch quality judge."""

    grammar_score: int = Field(ge=0, le=5)
    grammar_justification: str
    fluency_score: int = Field(ge=0, le=5)
    fluency_justification: str
    vocabulary_score: int = Field(ge=0, le=2)
    vocabulary_justification: str
    language_mixing: bool
    language_mixing_examples: str


class InstructionFollowingResult(BaseModel):
    """Structured response from the instruction-following judge."""

    instruction_following_score: int = Field(ge=0, le=3)
    instruction_following_justification: str


class CorrectnessResult(BaseModel):
    """Structured response from the correctness judge."""

    correctness_score: int = Field(ge=0, le=5)
    correctness_justification: str


class PairwiseWinner(str, Enum):
    A = "A"
    B = "B"
    tie = "tie"


class PairwiseResult(BaseModel):
    """Structured response from the pairwise comparison judge (combined, legacy)."""

    quality_winner: PairwiseWinner
    quality_justification: str
    instruction_winner: PairwiseWinner
    instruction_justification: str


class PairwiseSingleResult(BaseModel):
    """Structured response from a single-dimension pairwise judge call."""

    winner: PairwiseWinner
    justification: str
