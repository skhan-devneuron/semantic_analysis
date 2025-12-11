"""Semantic detector for excessive agency"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify excessive agency vulnerabilities
# Focus on code structure patterns, not keywords in strings
EXCESSIVE_AGENCY_QUESTIONS = [
    "Find code that defines agent classes, agent runners, or multi-agent orchestration systems with autonomous execution loops.",
    "Find code that registers tools, tasks, or actions to agents, including function decorators, tool registration methods, or capability exposure.",
    "Find code where agent or LLM can execute API calls, subprocess commands, file writes, network requests, or tool runs without user confirmation or validation checks."
]


class ExcessiveAgencyDetector(BaseSemanticDetector):
    """Detect excessive agency vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, EXCESSIVE_AGENCY_QUESTIONS, codebase_id=codebase_id)

