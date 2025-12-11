"""Request models for API"""
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path


class DetectionRequest(BaseModel):
    """Base request for detection"""
    user_id: str = Field(..., description="User ID")
    codebase_path: Optional[str] = Field(None, description="Path to codebase directory")
    github_url: Optional[str] = Field(None, description="GitHub repository URL")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "github_url": "https://github.com/example/repo",
                "similarity_threshold": 0.7
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request for batch detection"""
    user_id: str = Field(..., description="User ID")
    codebase_path: Optional[str] = Field(None, description="Path to codebase directory")
    github_url: Optional[str] = Field(None, description="GitHub repository URL")
    modules: List[str] = Field(..., description="List of modules to run")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "github_url": "https://github.com/example/repo",
                "modules": ["excessive_agency", "prompt_injection"],
                "similarity_threshold": 0.7
            }
        }


class VectorizationRequest(BaseModel):
    """Request for vectorizing a codebase"""
    user_id: str = Field(..., description="User ID")
    github_url: Optional[str] = Field(None, description="GitHub repository URL")
    codebase_path: Optional[str] = Field(None, description="Path to codebase directory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "github_url": "https://github.com/example/repo"
            }
        }


class EvaluationRequest(BaseModel):
    """Request for evaluating vulnerabilities using existing vectors"""
    user_id: str = Field(..., description="User ID")
    modules: List[str] = Field(..., description="List of modules to evaluate")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "modules": ["excessive_agency", "prompt_injection"],
                "similarity_threshold": 0.7
            }
        }

