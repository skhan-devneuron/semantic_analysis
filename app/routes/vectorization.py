"""Vectorization API routes"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
import time
import logging
from pathlib import Path
import hashlib

from app.models.request import VectorizationRequest
from app.models.response import VectorizationResponse
from app.utils.github_utils import clone_github_repo, cleanup_codebase, generate_codebase_id
from app.services.embedding_service import EmbeddingService
from app.utils.code_parser import SemanticCodeChunker, find_code_files
from app.config import EXCLUDE_PATTERNS

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/vectorize", response_model=VectorizationResponse)
async def vectorize_codebase(
    request: VectorizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Vectorize a codebase: chunk code and store embeddings in Pinecone.
    Returns codebase_id that can be used for evaluation.
    """
    start_time = time.time()
    codebase_path = None
    codebase_id = None
    
    try:
        # Clean up existing namespaces for this user before starting
        logger.info(f"Cleaning up existing Pinecone namespaces for user: {request.user_id}")
        deleted_count = EmbeddingService.delete_user_namespaces(request.user_id)
        logger.info(f"Deleted {deleted_count} existing namespaces for user {request.user_id}")
        
        # Handle GitHub URL or codebase path
        if request.github_url:
            codebase_id = generate_codebase_id(request.github_url)
            codebase_path = clone_github_repo(request.github_url, request.user_id, codebase_id)
        elif request.codebase_path:
            codebase_path = Path(request.codebase_path)
            # Generate codebase_id for local paths (use path hash)
            codebase_id = hashlib.md5(str(codebase_path).encode()).hexdigest()[:12]
        else:
            raise HTTPException(status_code=400, detail="Either github_url or codebase_path must be provided")
        
        # Initialize embedding service with codebase_id
        embedding_service = EmbeddingService(user_id=request.user_id, codebase_id=codebase_id)
        chunker = SemanticCodeChunker()
        
        # Find all code files
        code_files = find_code_files(codebase_path, EXCLUDE_PATTERNS)
        logger.info(f"Found {len(code_files)} code files")
        
        # Chunk and embed all code
        all_chunks = []
        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                chunks = chunker.chunk_file(file_path, code)
                for chunk in chunks:
                    chunk['file_path'] = str(file_path.relative_to(codebase_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Created {len(all_chunks)} code chunks")
        
        # Embed all chunks
        code_texts = [chunk['code'] for chunk in all_chunks]
        chunk_embeddings = embedding_service.embed_batch(code_texts)
        
        # Store embeddings in Pinecone
        embedding_service.store_embeddings(
            all_chunks,
            chunk_embeddings,
            metadata={"codebase_path": str(codebase_path)}
        )
        
        processing_time = time.time() - start_time
        
        # Schedule cleanup
        if codebase_id and request.github_url:
            background_tasks.add_task(cleanup_codebase, request.user_id, codebase_id)
        
        logger.info(f"Vectorization complete: {len(all_chunks)} chunks stored for codebase_id: {codebase_id}")
        
        return VectorizationResponse(
            status="success",
            codebase_id=codebase_id,
            total_chunks=len(all_chunks),
            total_files=len(code_files),
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error vectorizing codebase: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Immediate cleanup if not using background tasks
        if codebase_id and request.github_url and not background_tasks:
            cleanup_codebase(request.user_id, codebase_id)


