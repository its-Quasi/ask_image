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

    def analyze_classes(self, classes: list[str]) -> dict[str, any]:
        """
        Validates and normalizes class names for object detection.

        Args:
            classes: List of class names (can be in Spanish or English)

        Returns:
            Dictionary with:
                - success: Boolean indicating if validation was successful
                - dino_prompt: Formatted string for DINO (e.g., "cat. dog. person.")
                - normalized_classes: List of normalized class names in English
                - error: Error message if validation failed
        """
        if not classes or len(classes) == 0:
            return {
                "success": False,
                "error": "No se proporcionaron clases para validar",
                "dino_prompt": "",
                "normalized_classes": []
            }

        # Create prompt for LLM to validate and normalize classes
        validation_prompt = f"""
        You are a class name validator and normalizer for an object detection system.

        Task: Validate and normalize the following class names to English.

        Input classes: {', '.join(classes)}

        Rules:
        1. Translate Spanish class names to English
        2. Normalize to lowercase
        3. Use singular form when appropriate for object detection (e.g., "cat" not "cats")
        4. Discard nonsensical or invalid class names
        5. Keep only valid object/animal/thing names that can be detected in images

        Output ONLY valid JSON (no markdown, no explanation):
        {{
            "valid": true/false,
            "normalized_classes": [list of valid English class names],
            "discarded": [list of invalid/nonsensical class names that were removed],
            "reason": "explanation if any classes were discarded or if validation failed"
        }}

        Examples:
        Input: ["gato", "perro", "persona"]
        Output: {{"valid": true, "normalized_classes": ["cat", "dog", "person"], "discarded": [], "reason": ""}}

        Input: ["asdfgh", "cat", "xyzabc"]
        Output: {{"valid": true, "normalized_classes": ["cat"], "discarded": ["asdfgh", "xyzabc"], "reason": "Removed nonsensical class names"}}

        Input: ["qwerty", "zxcvbn"]
        Output: {{"valid": false, "normalized_classes": [], "discarded": ["qwerty", "zxcvbn"], "reason": "All class names are invalid or nonsensical"}}
        """

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that validates and normalizes class names for object detection."
                },
                {"role": "user", "content": validation_prompt}
            ]

            response = self.client.chat(model=self.model, messages=messages)
            content = response["message"]["content"]
            logger.debug(f"Class validation LLM response: {content}")

            # Parse JSON from response
            parsed_json = self._extract_json(content)

            # Check if validation was successful
            if not parsed_json.get("valid", False):
                return {
                    "success": False,
                    "error": parsed_json.get("reason", "Las clases proporcionadas no son válidas"),
                    "dino_prompt": "",
                    "normalized_classes": []
                }

            normalized_classes = parsed_json.get("normalized_classes", [])

            if len(normalized_classes) == 0:
                return {
                    "success": False,
                    "error": "No se encontraron clases válidas después de la normalización",
                    "dino_prompt": "",
                    "normalized_classes": []
                }

            # Format as DINO prompt: "class1. class2. class3."
            dino_prompt = ". ".join(normalized_classes) + "."

            logger.info(f"Normalized classes: {normalized_classes} -> DINO prompt: '{dino_prompt}'")

            return {
                "success": True,
                "dino_prompt": dino_prompt,
                "normalized_classes": normalized_classes,
                "discarded": parsed_json.get("discarded", []),
                "reason": parsed_json.get("reason", "")
            }

        except Exception as e:
            logger.error(f"Error validating classes: {str(e)}")
            return {
                "success": False,
                "error": f"Error al validar clases: {str(e)}",
                "dino_prompt": "",
                "normalized_classes": []
            }

    def validate_question(self, question: str) -> dict[str, any]:
        """
        Validates if a question makes sense for image analysis.

        Args:
            question: Natural language question from user

        Returns:
            Dictionary with:
                - valid: Boolean indicating if question is valid
                - error: Error message if question is invalid
        """
        if not question or len(question.strip()) == 0:
            return {
                "valid": False,
                "error": "La pregunta no puede estar vacía"
            }

        validation_prompt = f"""
        Determine if the following question makes sense for analyzing an image.

        Question: "{question}"

        A valid question should:
        - Ask about objects, people, animals, or things that can be visually detected
        - Ask about quantities, existence, comparisons, or attributes
        - Be related to visual content in an image

        Invalid questions:
        - Questions about audio, smell, taste, or other non-visual properties
        - Questions about abstract concepts that cannot be seen
        - Nonsensical or gibberish text
        - Questions unrelated to image content

        Output ONLY valid JSON (no markdown):
        {{
            "valid": true/false,
            "reason": "brief explanation why the question is valid or invalid"
        }}
        """

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a validator for image analysis questions."
                },
                {"role": "user", "content": validation_prompt}
            ]

            response = self.client.chat(model=self.model, messages=messages)
            content = response["message"]["content"]
            logger.debug(f"Question validation LLM response: {content}")

            parsed_json = self._extract_json(content)

            is_valid = parsed_json.get("valid", False)
            reason = parsed_json.get("reason", "")

            if not is_valid:
                return {
                    "valid": False,
                    "error": f"La pregunta no es válida para análisis de imágenes: {reason}"
                }

            return {
                "valid": True,
                "error": ""
            }

        except Exception as e:
            logger.error(f"Error validating question: {str(e)}")
            # If validation fails, assume question is valid to avoid blocking
            logger.warning("Question validation failed, assuming question is valid")
            return {
                "valid": True,
                "error": ""
            }
