"""Semantic detector for secret leakage"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify secret leakage vulnerabilities
SECRET_LEAKAGE_QUESTIONS = [
    "Are API keys, passwords, tokens, or other sensitive credentials hardcoded directly in source code, configuration files, or committed to version control?",
    "Are secrets, credentials, or sensitive authentication information exposed in logs, error messages, console output, or API responses?",
    "Are sensitive credentials stored in plain text format without encryption, or accessible in ways that could be leaked to unauthorized parties?"
]


class SecretLeakageDetector(BaseSemanticDetector):
    """Detect secret leakage vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, SECRET_LEAKAGE_QUESTIONS, codebase_id=codebase_id)

