"""Semantic detector for unbounded consumption"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify unbounded consumption vulnerabilities
UNBOUNDED_CONSUMPTION_QUESTIONS = [
    "Does any part of the code make outbound API calls, network requests, or cloud-service requests without validation or rate limiting?",
    "Does the code execute shell commands, spawn subprocesses, or access system resources that could consume CPU, memory, or disk?",
    "Is the system reading or writing files, logs, or large artifacts in a way that could grow unbounded or consume storage?",
    "Are there any continuous agent loops, recursive calls, or background tasks that can trigger repeated actions without user control?",
    "Do any agent tools perform outbound actions (API calls, DB queries, network operations, subprocess runs) that can consume external resources without user confirmation?"
]


class UnboundedConsumptionDetector(BaseSemanticDetector):
    """Detect unbounded consumption vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, UNBOUNDED_CONSUMPTION_QUESTIONS, codebase_id=codebase_id)

