from pydantic import BaseModel, Field
from typing import List


class ReasoningStep(BaseModel):
    step_id: int = Field(..., description="Step number starting from 1")
    content: str = Field(..., description="The reasoning content of this step")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model self-reported confidence (0-1)")


class CoTDraft(BaseModel):
    question: str = Field(..., description="The original question")
    dataset: str = Field(default="musique", description="Source dataset")
    complexity: str = Field(..., description="simple | medium | complex")
    steps: List[ReasoningStep] = Field(..., description="Ordered list of reasoning steps")
    final_answer: str = Field(..., description="Final answer derived from the steps")
    raw_output: str = Field(default="", description="Raw model output before parsing")
    parse_method: str = Field(default="schema", description="schema | fallback_sentence | fallback_delimiter")


# JSON schema for outlines constrained generation
COT_JSON_SCHEMA = CoTDraft.model_json_schema()
