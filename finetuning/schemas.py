from pydantic import BaseModel


class TrainingExample(BaseModel):
    instruction: str
    input: str
    output: str


class TrainingExamples(BaseModel):
    examples: list[TrainingExample]
