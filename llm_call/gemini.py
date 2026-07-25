"""Google Gemini client for structured LLM-as-a-Judge evaluation."""

from __future__ import annotations

import os
from typing import Optional

from google import genai
import pydantic


class JudgeScore(pydantic.BaseModel):
    personalization_score: int = pydantic.Field(ge=1, le=5)
    personalization_reason: str
    explanation_score: int = pydantic.Field(ge=1, le=5)
    explanation_reason: str


class Gemini_LLM_Client:
    """Gemini text generation client with schema-constrained output."""

    def __init__(
        self,
        model_name: str = "gemini-3.0-flash",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def chat_completion(self, query: str, schema_type: str = "judge") -> str:
        if schema_type == "judge":
            response_schema = JudgeScore
        else:
            response_schema = None
        model_output_text = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
        )
        if not model_output_text.text:
            raise ValueError("Gemini returned an empty response.")
        return model_output_text.text
