"""Semantic detector for insecure output handling"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify insecure output handling vulnerabilities
INSECURE_OUTPUT_QUESTIONS = [
    "Is agent or LLM output used directly in system operations (database queries, file operations, system commands) without validation or sanitization?",
    "Are LLM responses executed as code, passed to eval(), or used in SQL queries without proper escaping or parameterization?",
    "Is agent output trusted and used in security-sensitive operations without checking for malicious content or validating against expected formats?"
]


class InsecureOutputDetector(BaseSemanticDetector):
    """Detect insecure output handling vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, INSECURE_OUTPUT_QUESTIONS, codebase_id=codebase_id)

