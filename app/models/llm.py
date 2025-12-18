"""
LLM wrapper for parsing natural language queries and generating answers.
Uses Ollama for local LLM inference.
"""

import json
import logging
from typing import Dict, Any
from ollama import Client
from pydantic import ValidationError
from supervision import Detections
import numbers

from app.core.config import SYSTEM_PROMPT, MODEL
from app.models.schemas import DetectionResult, CompareQueryResult, SimpleQueryResult

logger = logging.getLogger(__name__)


class LlmModel:
    """
    LLM wrapper used for:
    1. Parsing natural language queries into structured inputs for Grounding DINO
    2. Generating natural language answers from detection results
    """

    def __init__(self) -> None:
        self.client = Client()
        self.model = MODEL
        logger.info(f"LlmModel initialized with model: {MODEL}")

    def parse_query(self, question: str):
        """
        Converts a natural language question into a structured, validated schema
        that can be consumed by the vision pipeline.

        Args:
            question: User's natural language question

        Returns:
            Validated Pydantic model based on intent type

        Raises:
            ValueError: If LLM response is invalid or cannot be parsed
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        try:
            response = self.client.chat(model=self.model, messages=messages)
            content = response["message"]["content"]
            logger.debug(f"Raw LLM response: {content}")

            # Parse JSON from response
            parsed_json = self._extract_json(content)

            # Validate with Pydantic based on intent
            return self._validate_response(parsed_json)

        except Exception as e:
            logger.error(f"Error parsing query '{question}': {str(e)}")
            raise ValueError(f"Failed to parse query: {str(e)}")

    def generate_answer(
        self,
        question: str,
        intent: str,
        detections: Detections,
        parsed_query: SimpleQueryResult | CompareQueryResult,
        class_names: list[str],
    ) -> str:
        """
        Generates a natural language answer based on detection results.

        Args:
            question: Original user question
            intent: Detected intent (detect, count, compare_count, etc.)
            detections: Supervision Detections object with detection results
            parsed_query: Parsed query structure
            class_names: List of class names where index matches class_id

        Returns:
            Natural language answer string
        """
        # Build context for answer generation
        context = self._build_answer_context(intent, detections, parsed_query, class_names)

        answer_prompt = f"""
        Based on the following detection results, answer the user's question naturally and concisely.

        User question: "{question}"

        Detection results:
        {context}

        Provide a direct answer in 1-2 sentences. Be specific about numbers and objects detected.
        Do NOT use markdown. Just plain text.
        """

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about image detections.",
                },
                {"role": "user", "content": answer_prompt},
            ]

            response = self.client.chat(model=self.model, messages=messages)
            answer = response["message"]["content"].strip()
            logger.debug(f"Generated answer: {answer}")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            # Fallback to simple answer
            return self._generate_fallback_answer(intent, detections)

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """
        Extracts and parses JSON from LLM response.
        Handles cases where JSON might be wrapped in markdown or extra text.
        """
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        # Try to find JSON in the content
        start_idx = content.find("{")
        end_idx = content.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx : end_idx + 1]
            return json.loads(json_str)

        raise ValueError(f"No valid JSON found in response: {content}")

    def _validate_response(
        self, parsed_json: dict[str, any]
    ) -> SimpleQueryResult | CompareQueryResult:
        """
        Validates parsed JSON against appropriate Pydantic schema based on intent.
        """
        intent = parsed_json.get("intent")

        if not intent:
            raise ValueError("Missing 'intent' field in LLM response")

        try:
            if intent in ["detect", "count", "exists"]:
                return SimpleQueryResult(**parsed_json)
            elif intent == "compare_count":
                return CompareQueryResult(**parsed_json)
            else:
                raise ValueError(f"Unknown intent: {intent}")

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise ValueError(f"Invalid response schema: {e}")

    def _build_answer_context(
        self,
        intent: str,
        detections: Detections,
        parsed_query,
        class_names: list[str],
    ) -> str:
        """
        Builds context string for answer generation based on intent.
        """

        if detections is None or detections.xyxy is None or len(detections.xyxy) == 0:
            return "No objects were detected in the image."

        detected_objects = {}

        for i, class_id in enumerate(detections.class_id):

            if class_id is None:
                continue

            # Ensure class_id is valid integer
            if not isinstance(class_id, numbers.Integral):
                continue

            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = "unknown"

            confidence = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 0.0
            )

            if class_name not in detected_objects:
                detected_objects[class_name] = {
                    "count": 0,
                    "confidences": [],
                }

            detected_objects[class_name]["count"] += 1
            detected_objects[class_name]["confidences"].append(confidence)

        if not detected_objects:
            return "Objects were detected, but none matched the requested classes."

        total_count = sum(info["count"] for info in detected_objects.values())

        # -------------------------------
        # Intent-specific context
        # -------------------------------
        if intent in {"detect", "count", "exists"}:
            parts = []
            for class_name, info in detected_objects.items():
                avg_conf = sum(info["confidences"]) / len(info["confidences"]) * 100
                parts.append(
                    f"{info['count']} {class_name}(s) with {avg_conf:.1f}% confidence"
                )

            return (
                f"- Total classified detections: {total_count}\n"
                f"- Detected: {', '.join(parts)}"
            )

        elif intent == "compare_count":
            left_objects = parsed_query.left.objects
            right_objects = parsed_query.right.objects

            left_count = sum(
                detected_objects.get(obj, {}).get("count", 0)
                for obj in left_objects
            )
            right_count = sum(
                detected_objects.get(obj, {}).get("count", 0)
                for obj in right_objects
            )

            return (
                f"- {', '.join(left_objects)}: {left_count} detected\n"
                f"- {', '.join(right_objects)}: {right_count} detected\n"
                f"- Comparison: {left_count} vs {right_count}"
            )

        else:
            summary = ", ".join(
                f"{info['count']} {name}(s)"
                for name, info in detected_objects.items()
            )
            return f"- Total detections: {total_count}\n- Found: {summary}"

    def _generate_fallback_answer(
        self, intent: str, detection_result: DetectionResult
    ) -> str:
        """
        Generates a simple fallback answer if LLM answer generation fails.
        """
        count = detection_result.count
        classes = (
            ", ".join(detection_result.class_names)
            if detection_result.class_names
            else "objects"
        )

        if intent == "count":
            return f"I detected {count} instance(s) of {classes}."
        elif intent == "exists":
            return (
                f"Yes, I found {count} instance(s) of {classes}."
                if count > 0
                else f"No, I did not find any {classes}."
            )
        elif intent == "detect":
            return f"I detected {count} object(s): {classes}."
        else:
            return f"Detection complete. Found {count} object(s)."
