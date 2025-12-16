"""
LLM wrapper for parsing natural language queries and generating answers.
Uses Ollama for local LLM inference.
"""

import json
import logging
from typing import Dict, Any
from ollama import Client
from pydantic import ValidationError

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
        detection_result: DetectionResult,
        parsed_query: SimpleQueryResult | CompareQueryResult,
    ) -> str:
        """
        Generates a natural language answer based on detection results.

        Args:
            question: Original user question
            intent: Detected intent (detect, count, compare_count, etc.)
            detection_result: Results from DINO detection
            parsed_query: Parsed query structure

        Returns:
            Natural language answer string
        """
        # Build context for answer generation
        context = self._build_answer_context(intent, detection_result, parsed_query)

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
            return self._generate_fallback_answer(intent, detection_result)

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
        detection_result: DetectionResult,
        parsed_query: SimpleQueryResult | CompareQueryResult,
    ) -> str:
        """
        Builds context string for answer generation based on intent.
        """
        if intent in ["detect", "count", "exists"]:
            objects = ", ".join(parsed_query.objects)
            return f"- Detected {detection_result.count} instance(s) of: {objects}\n- Classes found: {', '.join(detection_result.class_names)}"

        elif intent == "compare_count":
            # For compare, we need to split detections by object type
            # This is a simplified version - you might want more sophisticated logic
            return f"- Total detections: {detection_result.count}\n- Classes: {', '.join(detection_result.class_names)}"

        else:
            return f"- Total detections: {detection_result.count}\n- Classes: {', '.join(detection_result.class_names)}"

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
