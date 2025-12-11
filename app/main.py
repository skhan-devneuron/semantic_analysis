"""Main FastAPI application for Semantic Detection Service"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.routes import detection, health, vectorization
from app.config import LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Semantic Detection Service",
    description="Microservice for semantic-based security vulnerability detection in AI agent codebases",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(vectorization.router, prefix="/api/v1", tags=["vectorization"])
app.include_router(detection.router, prefix="/api/v1/detect", tags=["detection"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Semantic Detection Service...")
    logger.info("Loading embedding model (this may take a moment)...")
    
    try:
        # Pre-load embedding model
        from app.services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService(user_id="startup", codebase_id=None)
        embedding_service.embed_code("test")
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
    
    logger.info("Semantic Detection Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Semantic Detection Service...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Semantic Detection Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

