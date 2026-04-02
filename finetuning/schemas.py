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
