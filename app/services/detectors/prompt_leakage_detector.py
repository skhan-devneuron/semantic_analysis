"""Semantic detector for prompt leakage"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify prompt leakage vulnerabilities
PROMPT_LEAKAGE_QUESTIONS = [
    "Are system prompts, internal instructions, or prompt templates exposed in API responses, error messages, or visible to end users?",
    "Do error handling, logging, or exception messages reveal system prompts, internal instructions, or sensitive prompt templates?",
    "Can users see or extract system prompts, internal instructions, or prompt engineering details that should remain hidden?"
]


class PromptLeakageDetector(BaseSemanticDetector):
    """Detect prompt leakage vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, PROMPT_LEAKAGE_QUESTIONS, codebase_id=codebase_id)

