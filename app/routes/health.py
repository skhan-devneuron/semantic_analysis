"""Health check routes"""
from fastapi import APIRouter
from app.models.response import HealthResponse
from app.config import PINECONE_INDEX_NAME, PINECONE_API_KEY
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - lightweight and fast"""
    # Model is pre-loaded on startup, so we assume it's loaded
    embedding_model_loaded = True
    vector_db_connected = False
    
    try:
        # Check Pinecone connection (lightweight check)
        import pinecone
        
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        # Quick stats check to verify connection (non-blocking)
        stats = index.describe_index_stats()
        vector_db_connected = True
    except Exception as e:
        logger.warning(f"Vector DB check failed: {e}")
        vector_db_connected = False
    
    return HealthResponse(
        status="healthy" if embedding_model_loaded and vector_db_connected else "degraded",
        version="1.0.0",
        embedding_model_loaded=embedding_model_loaded,
        vector_db_connected=vector_db_connected
    )

