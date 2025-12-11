"""Semantic detector for prompt injection"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify prompt injection vulnerabilities
PROMPT_INJECTION_QUESTIONS = [
    "Is user input or external data directly concatenated or injected into system prompts without validation, sanitization, or proper separation?",
    "Are user inputs used in prompt templates or LLM calls without escaping, tagging, or isolation from system instructions?",
    "Can malicious input manipulate or override the intended behavior of the AI system by injecting instructions into prompts?"
]


class PromptInjectionDetector(BaseSemanticDetector):
    """Detect prompt injection vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, PROMPT_INJECTION_QUESTIONS, codebase_id=codebase_id)
