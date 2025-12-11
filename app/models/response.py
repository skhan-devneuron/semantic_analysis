"""Response models for API"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class VulnerabilityInfo(BaseModel):
    """Vulnerability information"""
    name: str
    type: str
    file_path: str
    line_start: int
    line_end: int
    code: str
    similarity_score: float
    gemini_confidence: float
    vulnerability_detected: bool


class DetectionResponse(BaseModel):
    """Response for detection"""
    status: str
    vulnerability_detected: bool  # Overall: was any vulnerability detected?
    vulnerabilities_found: List[VulnerabilityInfo]
    summary: Dict[str, Any]
    reasoning: Optional[str] = None  # Explanation when no vulnerabilities found
    question_summary: Optional[str] = None  # Summary of findings for all 3 questions
    gemini_reasoning: Optional[str] = None  # Final Gemini analysis and reasoning
    processing_time: Optional[float] = None
    error: Optional[str] = None


class VectorizationResponse(BaseModel):
    """Response for vectorization"""
    status: str
    codebase_id: str
    total_chunks: int
    total_files: int
    processing_time: Optional[float] = None
    error: Optional[str] = None


class BatchDetectionResponse(BaseModel):
    """Response for batch detection"""
    status: str
    results: Dict[str, DetectionResponse]
    total_processing_time: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    embedding_model_loaded: bool
    vector_db_connected: bool

