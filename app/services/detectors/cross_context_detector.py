"""Semantic detector for cross-context leakage"""
from typing import List
from app.services.base_detector import BaseSemanticDetector

# Questions to identify cross-context leakage vulnerabilities
CROSS_CONTEXT_QUESTIONS = [
    "Are vector database queries, embeddings, or shared storage accessed without user isolation, namespace separation, or session-based filtering?",
    "Can data from one user's context, session, or namespace leak into another user's queries or responses through shared vector stores or memory?",
    "Is there shared state, memory, or vector storage between different users or sessions without proper isolation mechanisms to prevent cross-context data leakage?"
]


class CrossContextDetector(BaseSemanticDetector):
    """Detect cross-context leakage vulnerabilities"""
    
    def __init__(self, user_id: str, codebase_id: str = None):
        super().__init__(user_id, CROSS_CONTEXT_QUESTIONS, codebase_id=codebase_id)

