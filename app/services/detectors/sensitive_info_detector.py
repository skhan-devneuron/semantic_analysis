"""Semantic detector for sensitive information disclosure"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify sensitive information disclosure vulnerabilities
SENSITIVE_INFO_QUESTIONS = [
    "Are sensitive operations, system calls, or privileged actions performed without proper authorization checks or permission validation?",
    "Is sensitive data (personal information, credentials, system details) exposed in API responses, error messages, or returned to unauthorized users?",
    "Can the system access sensitive resources, perform privileged operations, or retrieve confidential data without verifying user permissions or access rights?"
]


class SensitiveInfoDetector(BaseSemanticDetector):
    """Detect sensitive information disclosure vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, SENSITIVE_INFO_QUESTIONS, codebase_id=codebase_id)

