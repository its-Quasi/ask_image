"""
Pydantic schemas for LLM query parsing and validation.
Defines structured outputs for different vision pipeline intents.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class ObjectAttributes(BaseModel):
    """Attributes that describe an object (color, condition, etc.)."""
    attributes: Optional[List[str]] = Field(None)


class ObjectDescriptor(BaseModel):
    """Describes objects with their attributes for detection."""

    objects: List[str] = Field(..., description="List of object names to detect")
    attributes: Optional[ObjectAttributes] = Field(
        None, description="Visual attributes of the objects"
    )


class SimpleQueryResult(BaseModel):
    """Result schema for detect/count/exists intents."""

    intent: Literal["detect", "count", "exists"] = Field(
        ..., description="Type of operation"
    )
    dino_prompt: str = Field(..., description="Optimized prompt for Grounding DINO")
    objects: List[str] = Field(..., description="List of object names")
    attributes: Optional[ObjectAttributes] = Field(
        None, description="Object attributes"
    )



class CompareQueryResult(BaseModel):
    """Result schema for compare_count intent."""

    intent: Literal["compare_count"] = Field(..., description="Comparison operation")
    dino_prompt: str = Field(..., description="Combined prompt for both object types")
    left: ObjectDescriptor = Field(..., description="First object to compare")
    right: ObjectDescriptor = Field(..., description="Second object to compare")
    operator: Literal[">", "<", "=="] = Field(..., description="Comparison operator")


class DetectionResult(BaseModel):
    """Structured result from DINO detections."""

    count: int = Field(..., description="Number of detected objects")
    class_names: List[str] = Field(..., description="Names of detected classes")
    confidences: List[float] = Field(
        ..., description="Confidence scores for each detection"
    )
    bounding_boxes: Optional[List[List[float]]] = Field(
        None, description="Bounding box coordinates"
    )


class AnswerResponse(BaseModel):
    """Final answer to user's question."""

    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Natural language answer")
    detections: DetectionResult = Field(..., description="Detection results from DINO")
    intent: str = Field(..., description="Detected intent")
