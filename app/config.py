"""Configuration for Semantic Detection Service"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env at project root (explicit path for uvicorn reloads)
DOTENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=False)

VECTOR_DB_BASE_DIR = BASE_DIR / "vector_databases"
VECTOR_DB_SHARED_DIR = VECTOR_DB_BASE_DIR / "shared"
VECTOR_DB_USERS_DIR = VECTOR_DB_BASE_DIR / "users"

# Temporary codebase storage
TEMP_CODEBASE_DIR = BASE_DIR / "temp_codebases"

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "microsoft/codebert-base")
EMBEDDING_DIMENSION = 768  # CodeBERT dimension

# Detection thresholds
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
INITIAL_THRESHOLD = float(os.getenv("INITIAL_THRESHOLD", "0.5"))  # Lower threshold for stage 1
CONTEXT_EXPANSION_THRESHOLD = float(os.getenv("CONTEXT_EXPANSION_THRESHOLD", "0.6"))  # Stage 2 threshold
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.6"))  # For iterative refinement

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "codebase")
PINECONE_SHARED_NAMESPACE = "shared_patterns"

if not PINECONE_API_KEY:
    raise RuntimeError(f"PINECONE_API_KEY not set. Ensure it exists in {DOTENV_PATH}")
if not PINECONE_HOST:
    raise RuntimeError(f"PINECONE_HOST not set. Ensure it exists in {DOTENV_PATH}")

# Semantic chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))  # Tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Overlap tokens

# Supported file extensions
SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}

# Exclude patterns
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "*.pyc",
    "*.pyo",
    "*.pyd",
]

# GitHub configuration
GITHUB_CLONE_TIMEOUT = int(os.getenv("GITHUB_CLONE_TIMEOUT", "300"))  # 5 minutes
CLEANUP_AFTER_SECONDS = int(os.getenv("CLEANUP_AFTER_SECONDS", "3600"))  # 1 hour

# API configuration
API_VERSION = "v1"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Try gemini-2.5-pro first, fallback to gemini-1.5-pro if not available
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "3"))  # Top K matches to send to Gemini

if not GEMINI_API_KEY:
    raise RuntimeError(f"GEMINI_API_KEY not set. Ensure it exists in {DOTENV_PATH}")

def get_user_vector_db_path(user_id: str) -> Path:
    """Get vector database path for a specific user"""
    path = VECTOR_DB_USERS_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_shared_vector_db_path() -> Path:
    """Get shared vector database path"""
    VECTOR_DB_SHARED_DIR.mkdir(parents=True, exist_ok=True)
    return VECTOR_DB_SHARED_DIR

def get_temp_codebase_path(user_id: str, codebase_id: str) -> Path:
    """Get temporary codebase path"""
    path = TEMP_CODEBASE_DIR / user_id / codebase_id
    path.mkdir(parents=True, exist_ok=True)
    return path

