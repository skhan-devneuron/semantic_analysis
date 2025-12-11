"""Detection API routes"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import time
import logging
from pathlib import Path

from app.models.request import DetectionRequest, BatchDetectionRequest, EvaluationRequest
from app.models.response import DetectionResponse, BatchDetectionResponse, VulnerabilityInfo, VectorizationResponse
from app.services.detectors.excessive_agency_detector import ExcessiveAgencyDetector
from app.services.detectors.prompt_injection_detector import PromptInjectionDetector
from app.services.detectors.insecure_output_detector import InsecureOutputDetector
from app.services.detectors.secret_leakage_detector import SecretLeakageDetector
from app.services.detectors.unbounded_consumption_detector import UnboundedConsumptionDetector
from app.services.detectors.prompt_leakage_detector import PromptLeakageDetector
from app.services.detectors.sensitive_info_detector import SensitiveInfoDetector
from app.services.detectors.cross_context_detector import CrossContextDetector
from app.utils.github_utils import clone_github_repo, cleanup_codebase, generate_codebase_id
from app.services.embedding_service import EmbeddingService
from app.config import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

router = APIRouter()


# Detector mapping
DETECTOR_MAP = {
    "excessive_agency": ExcessiveAgencyDetector,
    "prompt_injection": PromptInjectionDetector,
    "insecure_output": InsecureOutputDetector,
    "secret_leakage": SecretLeakageDetector,
    "unbounded_consumption": UnboundedConsumptionDetector,
    "prompt_leakage": PromptLeakageDetector,
    "sensitive_info": SensitiveInfoDetector,
    "cross_context": CrossContextDetector,
}


@router.post("/batch", response_model=BatchDetectionResponse)
async def batch_detect(request: BatchDetectionRequest, background_tasks: BackgroundTasks):
    """Batch detection for multiple modules"""
    start_time = time.time()
    codebase_path = None
    codebase_id = None
    
    try:
        # Handle GitHub URL or codebase path
        if request.github_url:
            codebase_id = generate_codebase_id(request.github_url)
            codebase_path = clone_github_repo(request.github_url, request.user_id, codebase_id)
        elif request.codebase_path:
            codebase_path = Path(request.codebase_path)
            # Generate codebase_id for local paths (use path hash)
            import hashlib
            codebase_id = hashlib.md5(str(codebase_path).encode()).hexdigest()[:12]
        else:
            raise HTTPException(status_code=400, detail="Either github_url or codebase_path must be provided")
        
        results = {}
        
        # Process each module
        for module in request.modules:
            try:
                if module not in DETECTOR_MAP:
                    results[module] = DetectionResponse(
                        status="error",
                        vulnerability_detected=False,
                        vulnerabilities_found=[],
                        summary={},
                        error=f"Unknown module: {module}. Available: {list(DETECTOR_MAP.keys())}"
                    )
                    continue
                
                # Initialize detector with only user_id (no codebase_id for simplicity)
                DetectorClass = DETECTOR_MAP[module]
                detector = DetectorClass(user_id=request.user_id, codebase_id=None)
                
                # Detect vulnerabilities (with Gemini verification)
                vulnerabilities, potential_count, question_summary, gemini_reasoning = detector.detect(
                    codebase_path, 
                    request.similarity_threshold,
                    vulnerability_type=module
                )
                
                # Convert to response format
                vulnerabilities_found = [
                    VulnerabilityInfo(
                        name=vuln['name'],
                        type=vuln['type'],
                        file_path=vuln['file_path'],
                        line_start=vuln['line_start'],
                        line_end=vuln['line_end'],
                        code=vuln['code'],
                        similarity_score=vuln['similarity_score'],
                        gemini_confidence=vuln['gemini_confidence'],
                        vulnerability_detected=vuln['vulnerability_detected']
                    )
                    for vuln in vulnerabilities
                ]
                
                # Generate reasoning when no vulnerabilities found
                reasoning = None
                if len(vulnerabilities_found) == 0:
                    if potential_count == 0:
                        reasoning = f"No potential vulnerabilities found. The code did not match any {module} vulnerability questions with similarity >= {request.similarity_threshold}. This suggests the codebase follows security best practices for {module}."
                    else:
                        reasoning = f"Found {potential_count} potential {module} vulnerabilities, but Gemini verification determined they are false positives. The code appears secure after detailed analysis."
                
                results[module] = DetectionResponse(
                    status="success",
                    vulnerability_detected=len(vulnerabilities_found) > 0,
                    vulnerabilities_found=vulnerabilities_found,
                    summary={
                        "total_vulnerabilities": len(vulnerabilities_found),
                        "user_id": request.user_id
                    },
                    reasoning=reasoning,
                    question_summary=question_summary if question_summary else None,
                    gemini_reasoning=gemini_reasoning
                )
                
            except Exception as e:
                logger.error(f"Error processing module {module}: {e}")
                results[module] = DetectionResponse(
                    status="error",
                    vulnerability_detected=False,
                    vulnerabilities_found=[],
                    summary={},
                    error=str(e)
                )
        
        total_time = time.time() - start_time
        
        # Schedule cleanup
        if codebase_id and request.github_url:
            background_tasks.add_task(cleanup_codebase, request.user_id, codebase_id)
        
        return BatchDetectionResponse(
            status="success",
            results=results,
            total_processing_time=total_time
        )
    
    except Exception as e:
        logger.error(f"Error in batch detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if codebase_id and request.github_url and not background_tasks:
            cleanup_codebase(request.user_id, codebase_id)


@router.post("/evaluate", response_model=BatchDetectionResponse)
async def evaluate_vulnerabilities(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Evaluate vulnerabilities using existing vectors.
    Uses user_id for namespace (no codebase_id needed).
    """
    start_time = time.time()
    
    try:
        # Initialize embedding service with user_id namespace
        embedding_service = EmbeddingService(user_id=request.user_id, codebase_id=None)
        logger.info(f"Using namespace: {embedding_service.namespace} for user_id: {request.user_id}")
        
        # Try to query the namespace directly to verify vectors exist
        # Use a dummy zero vector to check if namespace has any vectors
        import numpy as np
        dummy_vector = np.zeros(EMBEDDING_DIMENSION).tolist()
        try:
            test_query = embedding_service.index.query(
                vector=dummy_vector,
                top_k=1,
                include_metadata=False,
                namespace=embedding_service.namespace
            )
            if not hasattr(test_query, 'matches') or not test_query.matches or len(test_query.matches) == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"No vectors found in namespace '{embedding_service.namespace}' for user_id: {request.user_id}. Please vectorize the codebase first using /api/v1/detect/batch or /api/v1/detect/{{detector_type}}"
                )
            logger.info(f"Verified vectors exist in namespace: {embedding_service.namespace}")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.error(f"Error querying namespace {embedding_service.namespace}: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Error accessing namespace '{embedding_service.namespace}' for user_id: {request.user_id}. Error: {str(e)}"
            )
        
        results = {}
        
        # Process each module
        for module in request.modules:
            try:
                if module not in DETECTOR_MAP:
                    results[module] = DetectionResponse(
                        status="error",
                        vulnerability_detected=False,
                        vulnerabilities_found=[],
                        summary={},
                        reasoning=None,
                        error=f"Unknown module: {module}. Available: {list(DETECTOR_MAP.keys())}"
                    )
                    continue
                
                # Initialize detector with only user_id (no codebase_id)
                DetectorClass = DETECTOR_MAP[module]
                detector = DetectorClass(user_id=request.user_id, codebase_id=None)
                
                # Detect vulnerabilities using existing vectors (no vectorization)
                vulnerabilities, potential_count, question_summary, gemini_reasoning = detector.detect(
                    codebase_path=None,  # Not needed when using existing vectors
                    similarity_threshold=request.similarity_threshold,
                    vulnerability_type=module,
                    use_existing_vectors=True
                )
                
                # Convert to response format
                vulnerabilities_found = [
                    VulnerabilityInfo(
                        name=vuln['name'],
                        type=vuln['type'],
                        file_path=vuln['file_path'],
                        line_start=vuln['line_start'],
                        line_end=vuln['line_end'],
                        code=vuln['code'],
                        similarity_score=vuln['similarity_score'],
                        gemini_confidence=vuln['gemini_confidence'],
                        vulnerability_detected=vuln['vulnerability_detected']
                    )
                    for vuln in vulnerabilities
                ]
                
                # Generate reasoning when no vulnerabilities found
                reasoning = None
                if len(vulnerabilities_found) == 0:
                    if potential_count == 0:
                        reasoning = f"No potential vulnerabilities found. The code did not match any {module} vulnerability questions with similarity >= {request.similarity_threshold}. This suggests the codebase follows security best practices for {module}."
                    else:
                        reasoning = f"Found {potential_count} potential {module} vulnerabilities, but Gemini verification determined they are false positives. The code appears secure after detailed analysis."
                
                results[module] = DetectionResponse(
                    status="success",
                    vulnerability_detected=len(vulnerabilities_found) > 0,
                    vulnerabilities_found=vulnerabilities_found,
                    summary={
                        "total_vulnerabilities": len(vulnerabilities_found),
                        "user_id": request.user_id
                    },
                    reasoning=reasoning,
                    question_summary=question_summary if question_summary else None,
                    gemini_reasoning=gemini_reasoning
                )
                
            except Exception as e:
                logger.error(f"Error processing module {module}: {e}")
                results[module] = DetectionResponse(
                    status="error",
                    vulnerability_detected=False,
                    vulnerabilities_found=[],
                    summary={},
                    reasoning=None,
                    error=str(e)
                )
        
        total_time = time.time() - start_time
        
        return BatchDetectionResponse(
            status="success",
            results=results,
            total_processing_time=total_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{detector_type}", response_model=DetectionResponse)
async def detect_vulnerability(
    detector_type: str,
    request: DetectionRequest,
    background_tasks: BackgroundTasks
):
    """Detect vulnerabilities using specified detector"""
    start_time = time.time()
    codebase_path = None
    codebase_id = None
    
    try:
        # Clean up existing namespaces for this user before starting
        logger.info(f"Cleaning up existing Pinecone namespaces for user: {request.user_id}")
        deleted_count = EmbeddingService.delete_user_namespaces(request.user_id)
        logger.info(f"Deleted {deleted_count} existing namespaces for user {request.user_id}")
        
        # Check if detector type is valid
        if detector_type not in DETECTOR_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown detector type: {detector_type}. Available: {list(DETECTOR_MAP.keys())}"
            )
        
        # Handle GitHub URL or codebase path
        if request.github_url:
            codebase_id = generate_codebase_id(request.github_url)
            codebase_path = clone_github_repo(request.github_url, request.user_id, codebase_id)
        elif request.codebase_path:
            codebase_path = Path(request.codebase_path)
            # Generate codebase_id for local paths (use path hash)
            import hashlib
            codebase_id = hashlib.md5(str(codebase_path).encode()).hexdigest()[:12]
        else:
            raise HTTPException(status_code=400, detail="Either github_url or codebase_path must be provided")
        
        # Initialize detector with only user_id (no codebase_id for simplicity)
        DetectorClass = DETECTOR_MAP[detector_type]
        detector = DetectorClass(user_id=request.user_id, codebase_id=None)
        
        # Detect vulnerabilities (with Gemini verification)
        vulnerabilities, potential_count, question_summary, gemini_reasoning = detector.detect(
            codebase_path, 
            request.similarity_threshold,
            vulnerability_type=detector_type
        )
        
        # Convert to response format
        vulnerabilities_found = [
            VulnerabilityInfo(
                name=vuln['name'],
                type=vuln['type'],
                file_path=vuln['file_path'],
                line_start=vuln['line_start'],
                line_end=vuln['line_end'],
                code=vuln['code'],
                similarity_score=vuln['similarity_score'],
                gemini_confidence=vuln['gemini_confidence'],
                vulnerability_detected=vuln['vulnerability_detected']
            )
            for vuln in vulnerabilities
        ]
        
        processing_time = time.time() - start_time
        
        # Schedule cleanup
        if codebase_id and request.github_url:
            background_tasks.add_task(cleanup_codebase, request.user_id, codebase_id)
        
        # Generate reasoning when no vulnerabilities found
        reasoning = None
        if len(vulnerabilities_found) == 0:
            if potential_count == 0:
                reasoning = f"No potential vulnerabilities found. The code did not match any {detector_type} vulnerability questions with similarity >= {request.similarity_threshold}. This suggests the codebase follows security best practices for {detector_type}."
            else:
                reasoning = f"Found {potential_count} potential {detector_type} vulnerabilities, but Gemini verification determined they are false positives. The code appears secure after detailed analysis."
        
        return DetectionResponse(
            status="success",
            vulnerability_detected=len(vulnerabilities_found) > 0,
            vulnerabilities_found=vulnerabilities_found,
            summary={
                "total_vulnerabilities": len(vulnerabilities_found),
                "user_id": request.user_id
            },
            reasoning=reasoning,
            question_summary=question_summary if question_summary else None,
            gemini_reasoning=gemini_reasoning,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting vulnerabilities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Immediate cleanup if not using background tasks
        if codebase_id and request.github_url and not background_tasks:
            cleanup_codebase(request.user_id, codebase_id)

