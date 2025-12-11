"""Base semantic detector for all vulnerability types"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiVerificationService
from app.utils.code_parser import SemanticCodeChunker, find_code_files
from app.config import SIMILARITY_THRESHOLD, EXCLUDE_PATTERNS, EMBEDDING_DIMENSION, GEMINI_TOP_K

logger = logging.getLogger(__name__)

# Multi-stage detection thresholds
INITIAL_THRESHOLD = 0.5  # Lower threshold for stage 1 (cast wider net)
CONTEXT_EXPANSION_THRESHOLD = 0.6  # Threshold for context expansion
FINAL_THRESHOLD = 0.7  # Final threshold for Gemini verification
LOW_CONFIDENCE_THRESHOLD = 0.6  # If Gemini confidence < this, do iterative refinement


class BaseSemanticDetector:
    """Base class for all semantic vulnerability detectors"""
    
    def __init__(self, user_id: str, vulnerability_questions: List[str], codebase_id: Optional[str] = None):
        """
        Initialize base detector.
        
        Args:
            user_id: User ID for embedding service
            vulnerability_questions: List of 3 questions to identify vulnerabilities
            codebase_id: Codebase ID for namespace isolation
        """
        self.user_id = user_id
        self.codebase_id = codebase_id
        self.embedding_service = EmbeddingService(user_id=user_id, codebase_id=codebase_id)
        self.gemini_service = GeminiVerificationService()
        self.chunker = SemanticCodeChunker()
        self.vulnerability_questions = vulnerability_questions
        
        if len(vulnerability_questions) < 3:
            raise ValueError(f"At least 3 questions required, got {len(vulnerability_questions)}")
        
        # Embed questions for detection and context gathering (use shared namespace)
        shared_embedding_service = EmbeddingService(user_id=None, codebase_id=None)
        logger.info(f"Embedding {len(vulnerability_questions)} vulnerability detection questions...")
        self.question_embeddings = shared_embedding_service.embed_batch(vulnerability_questions)
        logger.info("Vulnerability questions embedded successfully")
    
    def detect(
        self,
        codebase_path: Optional[Path] = None,
        similarity_threshold: Optional[float] = None,
        vulnerability_type: Optional[str] = None,
        use_existing_vectors: bool = False
    ) -> Tuple[List[Dict], int, str, Optional[str]]:
        """
        Simple RAG-based detection: Query Pinecone with questions, get top results, send to Gemini.
        
        Args:
            codebase_path: Path to codebase directory (only used if use_existing_vectors=False)
            similarity_threshold: Minimum similarity threshold (default from config)
            vulnerability_type: Type of vulnerability being detected
            use_existing_vectors: If True, use existing vectors from Pinecone
        
        Returns:
            Tuple of (verified_vulnerabilities, total_candidates_found, question_summary, gemini_reasoning)
        """
        if similarity_threshold is None:
            similarity_threshold = SIMILARITY_THRESHOLD
        
        if vulnerability_type is None:
            vulnerability_type = self.__class__.__name__.replace("Detector", "").lower()
        
        logger.info(f"Detecting {vulnerability_type} vulnerabilities using simple RAG approach")
        
        # If not using existing vectors, vectorize the codebase first
        if not use_existing_vectors:
            if codebase_path is None:
                raise ValueError("codebase_path is required when use_existing_vectors=False")
            
            logger.info(f"Vectorizing codebase: {codebase_path}")
            code_files = find_code_files(codebase_path, EXCLUDE_PATTERNS)
            logger.info(f"Found {len(code_files)} code files")
            
            # Chunk and embed all code
            all_chunks = []
            for file_path in code_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    chunks = self.chunker.chunk_file(file_path, code)
                    for chunk in chunks:
                        chunk['file_path'] = str(file_path.relative_to(codebase_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
            
            logger.info(f"Created {len(all_chunks)} code chunks")
            
            # Embed all chunks
            code_texts = [chunk['code'] for chunk in all_chunks]
            chunk_embeddings = self.embedding_service.embed_batch(code_texts)
            
            # Store embeddings in Pinecone
            self.embedding_service.store_embeddings(
                all_chunks,
                chunk_embeddings,
                metadata={"codebase_path": str(codebase_path)}
            )
            logger.info(f"Stored {len(all_chunks)} embeddings in Pinecone")
        
        # STEP 1: Query Pinecone with each question to get relevant code chunks
        logger.info(f"Querying Pinecone with {len(self.vulnerability_questions)} questions")
        # Use a dict to track chunks by ID, allowing same chunk for multiple questions
        chunks_by_id = {}
        
        # Patterns to exclude (health checks, utilities, etc.)
        EXCLUDE_FUNCTION_PATTERNS = [
            'health_check', 'ping', 'ping_twice', 'status', 'health',
            'ready', 'liveness', 'readiness', 'heartbeat', 'alive'
        ]
        
        # Query Pinecone with each question embedding
        for i, question in enumerate(self.vulnerability_questions):
            question_embedding = self.question_embeddings[i]
            
            # Query Pinecone for top K results for this question (increase to get more variety)
            top_k = max(GEMINI_TOP_K * 2, 10)  # Get more results to avoid duplicates
            results = self.embedding_service.search_similar(
                query_embedding=question_embedding,
                n_results=top_k,
                filter_metadata=None
            )
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {i+1}: {question}")
            logger.info(f"Retrieved {len(results)} code chunks from Pinecone (threshold: {similarity_threshold}):")
            
            chunks_passed = 0
            chunks_filtered = 0
            chunks_skipped_duplicate = 0
            
            for j, result in enumerate(results, 1):
                chunk_id = result.get('id', '')
                score = result.get('score', 0.0)
                metadata = result.get('metadata', {})
                code = result.get('code', '')
                
                # Log all chunks, showing which pass threshold
                passed_threshold = score >= similarity_threshold
                status = "✓ PASSED" if passed_threshold else f"✗ FILTERED (score {score:.3f} < {similarity_threshold})"
                
                # Check if this chunk was already seen for this question
                if chunk_id in chunks_by_id:
                    # Chunk already exists, just add this question to it
                    if passed_threshold:
                        chunks_by_id[chunk_id]['question_indices'].add(i)
                        chunks_by_id[chunk_id]['questions'].append(question)
                        # Update similarity if this question has higher score
                        if score > chunks_by_id[chunk_id]['similarity_score']:
                            chunks_by_id[chunk_id]['similarity_score'] = score
                        chunks_skipped_duplicate += 1
                        status += " (already seen, added to question list)"
                    logger.info(f"  Chunk {j}: {status} (duplicate across questions)")
                else:
                    logger.info(f"  Chunk {j}: {status}")
                    logger.info(f"    File: {metadata.get('file_path', 'unknown')}")
                    logger.info(f"    Lines: {metadata.get('line_start', 0)}-{metadata.get('line_end', 0)}")
                    logger.info(f"    Similarity: {score:.3f}")
                    logger.info(f"    Code (first 300 chars):\n{code[:300]}...")
                    
                    # Only include if similarity meets threshold
                    if passed_threshold:
                        # Filter out health check and utility functions
                        chunk_name = metadata.get('name', '').lower()
                        code_lower = code.lower()
                        
                        # Skip health check functions
                        if any(pattern in chunk_name for pattern in EXCLUDE_FUNCTION_PATTERNS):
                            chunks_filtered += 1
                            logger.info(f"    → FILTERED: Health check/utility function '{chunk_name}'")
                            continue
                        
                        # Skip functions that only return static JSON/dicts (likely health checks)
                        if ('return {' in code_lower or 'return {' in code_lower) and len(code.split('\n')) < 5:
                            # Check if it's just a simple return statement
                            lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
                            if len(lines) <= 3 and any('return' in l and '{' in l for l in lines):
                                chunks_filtered += 1
                                logger.info(f"    → FILTERED: Simple return-only function (likely health check)")
                                continue
                        
                        chunks_passed += 1
                        chunks_by_id[chunk_id] = {
                            "id": chunk_id,
                            "code": code,
                            "file_path": metadata.get('file_path', 'unknown'),
                            "type": metadata.get('type', 'code_block'),
                            "name": metadata.get('name', 'chunk'),
                            "line_start": int(metadata.get('line_start', 0)),
                            "line_end": int(metadata.get('line_end', 0)),
                            "similarity_score": score,
                            "question_indices": {i},  # Set of question indices
                            "questions": [question]  # List of questions this chunk answers
                        }
                    else:
                        chunks_filtered += 1
            
            logger.info(f"  Summary: {chunks_passed} new chunks passed, {chunks_filtered} filtered, {chunks_skipped_duplicate} duplicates added to existing chunks")
            logger.info(f"{'='*80}\n")
        
        # Convert dict to list format
        all_retrieved_chunks = []
        for chunk_id, chunk_data in chunks_by_id.items():
            # Create a chunk entry for each question it's associated with
            for q_idx in chunk_data['question_indices']:
                all_retrieved_chunks.append({
                    "id": chunk_data['id'],
                    "code": chunk_data['code'],
                    "file_path": chunk_data['file_path'],
                    "type": chunk_data['type'],
                    "name": chunk_data['name'],
                    "line_start": chunk_data['line_start'],
                    "line_end": chunk_data['line_end'],
                    "similarity_score": chunk_data['similarity_score'],
                    "question": self.vulnerability_questions[q_idx],
                    "question_index": q_idx
                })
        
        # Get unique chunks count
        unique_chunk_ids = set(chunk['id'] for chunk in all_retrieved_chunks)
        logger.info(f"Total unique chunks retrieved: {len(unique_chunk_ids)} (across {len(all_retrieved_chunks)} question-chunk associations)")
        
        if len(all_retrieved_chunks) == 0:
            logger.info("No chunks found above similarity threshold")
            return [], 0, "", None
        
        # STEP 2: Group chunks by question for context
        question_contexts = []
        for i, question in enumerate(self.vulnerability_questions):
            question_chunks = [
                chunk for chunk in all_retrieved_chunks 
                if chunk.get('question_index') == i
            ]
            
            question_contexts.append({
                "question": question,
                "code_snippets": [
                    {
                        "code": chunk['code'],
                        "file_path": chunk['file_path'],
                        "similarity": chunk['similarity_score']
                    }
                    for chunk in question_chunks
                ]
            })
        
        # STEP 3: Send all retrieved chunks to Gemini for final detection
        logger.info(f"\n{'='*80}")
        logger.info(f"SENDING TO GEMINI FOR FINAL DETECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Total chunks to analyze: {len(all_retrieved_chunks)}")
        logger.info(f"Questions: {len(self.vulnerability_questions)}")
        for i, q_ctx in enumerate(question_contexts, 1):
            logger.info(f"  Question {i}: {len(q_ctx['code_snippets'])} chunks")
        logger.info(f"{'='*80}\n")
        
        # Filter out health checks before sending to Gemini
        filtered_chunks_for_gemini = []
        seen_chunk_ids_gemini = set()
        for chunk in all_retrieved_chunks:
            chunk_id = chunk.get('id', f"{chunk.get('file_path')}_{chunk.get('line_start')}")
            
            # Skip duplicates
            if chunk_id in seen_chunk_ids_gemini:
                continue
            seen_chunk_ids_gemini.add(chunk_id)
            
            # Filter out health checks
            chunk_name = chunk.get('name', '').lower()
            code_lower = chunk.get('code', '').lower()
            
            if any(pattern in chunk_name for pattern in EXCLUDE_FUNCTION_PATTERNS):
                logger.info(f"Excluding health check function '{chunk_name}' from Gemini analysis")
                continue
            
            # Skip simple return-only functions
            if ('return {' in code_lower or 'return {' in code_lower) and len(chunk.get('code', '').split('\n')) < 5:
                lines = [l.strip() for l in chunk.get('code', '').split('\n') if l.strip() and not l.strip().startswith('#')]
                if len(lines) <= 3 and any('return' in l and '{' in l for l in lines):
                    logger.info(f"Excluding simple return-only function '{chunk_name}' from Gemini analysis")
                    continue
            
            filtered_chunks_for_gemini.append(chunk)
        
        logger.info(f"After filtering health checks: {len(filtered_chunks_for_gemini)} chunks to send to Gemini (from {len(all_retrieved_chunks)} total)")
        
        if len(filtered_chunks_for_gemini) == 0:
            logger.info("No relevant chunks after filtering health checks")
            return [], len(unique_chunk_ids)
        
        # Combine filtered chunks into a single context for Gemini
        combined_code_snippet = "\n\n".join([
            f"// From {chunk['file_path']} (lines {chunk['line_start']}-{chunk['line_end']})\n{chunk['code']}"
            for chunk in filtered_chunks_for_gemini[:20]  # Limit to top 20 chunks to avoid token limits
        ])
        
        # Update question_contexts to only include filtered chunks
        filtered_question_contexts = []
        for i, q_ctx in enumerate(question_contexts):
            filtered_snippets = []
            for snippet in q_ctx['code_snippets']:
                # Check if this snippet is in our filtered list
                snippet_code = snippet.get('code', '')
                snippet_file = snippet.get('file_path', '')
                if any(
                    chunk.get('code', '') == snippet_code and chunk.get('file_path', '') == snippet_file
                    for chunk in filtered_chunks_for_gemini
                ):
                    filtered_snippets.append(snippet)
            
            filtered_question_contexts.append({
                "question": q_ctx['question'],
                "code_snippets": filtered_snippets
            })
        
        # Verify with Gemini
        gemini_result = self.gemini_service.verify_vulnerability(
            vulnerability_type=vulnerability_type,
            code_snippet=combined_code_snippet,
            questions=self.vulnerability_questions,
            question_contexts=filtered_question_contexts,
            context=f"Analyzing {len(filtered_chunks_for_gemini)} code chunks from codebase (health checks filtered out)"
        )
        
        # STEP 4: Create summary of findings for all questions
        question_summary_parts = []
        for i, q_ctx in enumerate(filtered_question_contexts, 1):
            question = q_ctx['question']
            snippets = q_ctx['code_snippets']
            if snippets:
                file_names = list(set(s['file_path'] for s in snippets))
                question_summary_parts.append(
                    f"Question {i}: Found {len(snippets)} relevant code snippet(s) in {len(file_names)} file(s) "
                    f"({', '.join(file_names[:3])}{'...' if len(file_names) > 3 else ''})"
                )
            else:
                question_summary_parts.append(f"Question {i}: No relevant code found")
        
        question_summary = " | ".join(question_summary_parts)
        
        # STEP 5: Format results
        verified_vulnerabilities = []
        
        if gemini_result["vulnerability_detected"]:
            # If Gemini detected vulnerabilities, only include the filtered chunks we sent to Gemini
            # (already deduplicated and health checks removed)
            seen_chunk_ids_results = set()
            for chunk in filtered_chunks_for_gemini:
                chunk_id = chunk.get('id', f"{chunk.get('file_path')}_{chunk.get('line_start')}")
                
                # Skip duplicates (shouldn't happen but just in case)
                if chunk_id in seen_chunk_ids_results:
                    continue
                seen_chunk_ids_results.add(chunk_id)
                
                verified_vulnerabilities.append({
                    "name": chunk.get('name', 'unknown'),
                    "type": chunk.get('type', 'code_block'),
                    "file_path": chunk.get('file_path', ''),
                    "line_start": chunk.get('line_start', 0),
                    "line_end": chunk.get('line_end', 0),
                    "code": chunk['code'],
                    "similarity_score": chunk['similarity_score'],
                    "vulnerability_detected": True,
                    "gemini_confidence": gemini_result["confidence"]
                })
        
        logger.info(f"Gemini result: vulnerability_detected={gemini_result['vulnerability_detected']}, confidence={gemini_result['confidence']:.2f}")
        logger.info(f"Final: {len(verified_vulnerabilities)} vulnerabilities detected from {len(all_retrieved_chunks)} retrieved chunks")
        
        # Return vulnerabilities along with summary and reasoning (to be added at response level)
        return verified_vulnerabilities, len(all_retrieved_chunks), question_summary, gemini_result["reasoning"] if gemini_result["vulnerability_detected"] else None
    
    def _load_existing_vectors(self) -> tuple[List[Dict], np.ndarray]:
        """
        Load existing chunks and embeddings from Pinecone.
        
        Returns:
            Tuple of (all_chunks, chunk_embeddings)
        """
        try:
            # Get stats first to know how many vectors we have
            stats = self.embedding_service.index.describe_index_stats()
            
            # Debug: Log all available namespaces
            all_namespaces = {}
            if isinstance(stats, dict):
                all_namespaces = stats.get('namespaces', {})
                logger.info(f"Available namespaces in index: {list(all_namespaces.keys())}")
                namespace_stats = all_namespaces.get(self.embedding_service.namespace, {})
                vector_count = namespace_stats.get('vector_count', 0)
            else:
                namespaces = getattr(stats, 'namespaces', {})
                if isinstance(namespaces, dict):
                    all_namespaces = namespaces
                    logger.info(f"Available namespaces in index: {list(all_namespaces.keys())}")
                namespace_stats = all_namespaces.get(self.embedding_service.namespace, {}) if isinstance(all_namespaces, dict) else {}
                vector_count = namespace_stats.get('vector_count', 0) if isinstance(namespace_stats, dict) else 0
            
            logger.info(f"Looking for vectors in namespace: '{self.embedding_service.namespace}'")
            logger.info(f"Found {vector_count} vectors in namespace {self.embedding_service.namespace}")
            
            if vector_count == 0:
                logger.warning(f"No vectors found in namespace '{self.embedding_service.namespace}'. Available namespaces: {list(all_namespaces.keys())}")
                # Try to find any namespace that starts with user_id
                if all_namespaces:
                    matching_namespaces = [ns for ns in all_namespaces.keys() if ns.startswith(self.user_id)]
                    if matching_namespaces:
                        logger.info(f"Found {len(matching_namespaces)} namespace(s) matching user_id '{self.user_id}': {matching_namespaces}")
                        # Use the first matching namespace
                        matching_ns = matching_namespaces[0]
                        logger.info(f"Attempting to use namespace: '{matching_ns}'")
                        self.embedding_service.namespace = matching_ns
                        namespace_stats = all_namespaces.get(matching_ns, {})
                        vector_count = namespace_stats.get('vector_count', 0)
                        logger.info(f"Found {vector_count} vectors in namespace '{matching_ns}'")
                    else:
                        return [], np.array([])
                else:
                    return [], np.array([])
            
            # Use multiple random queries to get all vectors (more reliable than zero vector)
            # Generate multiple random normalized vectors and query with each
            all_chunks = []
            seen_ids = set()
            import random
            
            # Try multiple random vectors to ensure we get all vectors
            num_queries = min(5, max(1, vector_count // 1000))  # More queries for larger datasets
            
            for query_num in range(num_queries):
                # Generate a random normalized vector
                random_vector = [random.gauss(0, 1) for _ in range(EMBEDDING_DIMENSION)]
                norm = sum(x*x for x in random_vector) ** 0.5
                if norm > 0:
                    random_vector = [x/norm for x in random_vector]
                else:
                    random_vector = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
                
                try:
                    # Query with high top_k
                    query_response = self.embedding_service.index.query(
                        vector=random_vector,
                        top_k=min(10000, vector_count),
                        include_metadata=True,
                        namespace=self.embedding_service.namespace
                    )
                    
                    if query_response and hasattr(query_response, 'matches') and query_response.matches:
                        for match in query_response.matches:
                            # Get match ID
                            if isinstance(match, dict):
                                match_id = match.get('id', '')
                                metadata = match.get('metadata', {})
                            elif hasattr(match, 'id'):
                                match_id = match.id
                                metadata = match.metadata if hasattr(match, 'metadata') else {}
                            else:
                                continue
                            
                            # Skip if we've already seen this ID
                            if match_id and match_id in seen_ids:
                                continue
                            
                            seen_ids.add(match_id)
                            
                            # Extract code from metadata
                            code = metadata.get('code', '') if isinstance(metadata, dict) else ''
                            if code:
                                chunk = {
                                    'code': code,
                                    'file_path': metadata.get('file_path', 'unknown') if isinstance(metadata, dict) else 'unknown',
                                    'type': metadata.get('type', 'code_block') if isinstance(metadata, dict) else 'code_block',
                                    'name': metadata.get('name', 'chunk') if isinstance(metadata, dict) else 'chunk',
                                    'line_start': int(metadata.get('line_start', 0)) if isinstance(metadata, dict) else 0,
                                    'line_end': int(metadata.get('line_end', 0)) if isinstance(metadata, dict) else 0
                                }
                                all_chunks.append(chunk)
                
                except Exception as e:
                    logger.warning(f"Error in query {query_num + 1}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(all_chunks)} unique chunks from {num_queries} queries")
            
            logger.info(f"Loaded {len(all_chunks)} chunks from Pinecone")
            
            # Log sample of retrieved chunks
            if all_chunks:
                logger.info(f"\n{'='*80}")
                logger.info(f"RETRIEVED CHUNKS FROM PINECONE ({len(all_chunks)} total)")
                logger.info(f"{'='*80}")
                for i, chunk in enumerate(all_chunks[:10], 1):  # Log first 10
                    logger.info(f"\nChunk {i}:")
                    logger.info(f"  File: {chunk.get('file_path', 'unknown')}")
                    logger.info(f"  Lines: {chunk.get('line_start', 0)}-{chunk.get('line_end', 0)}")
                    logger.info(f"  Type: {chunk.get('type', 'unknown')}")
                    logger.info(f"  Name: {chunk.get('name', 'unknown')}")
                    logger.info(f"  Code Preview (first 300 chars):\n{chunk.get('code', '')[:300]}...")
                if len(all_chunks) > 10:
                    logger.info(f"\n... and {len(all_chunks) - 10} more chunks")
                logger.info(f"{'='*80}\n")
            
            # Re-embed the code (since we need embeddings for similarity calculation)
            if all_chunks:
                code_texts = [chunk['code'] for chunk in all_chunks]
                chunk_embeddings = self.embedding_service.embed_batch(code_texts)
                logger.info(f"Re-embedded {len(all_chunks)} chunks for similarity calculation")
                return all_chunks, chunk_embeddings
            
            return [], np.array([])
            
        except Exception as e:
            logger.error(f"Error loading existing vectors: {e}", exc_info=True)
            return [], np.array([])
    
    def _gather_question_contexts(
        self,
        target_embedding: np.ndarray,
        all_chunks: List[Dict],
        chunk_embeddings: np.ndarray,
        top_k_per_question: int = 1
    ) -> List[Dict[str, str]]:
        """
        Gather code snippets that answer each question.
        
        Args:
            target_embedding: Embedding of the potential vulnerability chunk
            all_chunks: All code chunks
            chunk_embeddings: Embeddings of all chunks
            top_k_per_question: Number of code snippets to retrieve per question
        
        Returns:
            List of dicts with 'question' and 'code_snippets' (list of relevant code)
        """
        question_contexts = []
        
        for i, question in enumerate(self.vulnerability_questions):
            question_embedding = self.question_embeddings[i]
            
            # Find code chunks most relevant to this question
            # Combine question embedding with target embedding for better context
            combined_query = (question_embedding + target_embedding) / 2.0
            
            # Calculate similarities with all chunks
            similarities = self._calculate_similarities(combined_query, chunk_embeddings)
            
            # Get top K chunks for this question
            top_k_indices = np.argsort(similarities)[-top_k_per_question:][::-1]
            
            relevant_snippets = []
            for idx in top_k_indices:
                # Lower minimum relevance threshold for iterative refinement
                min_threshold = 0.4 if top_k_per_question > 3 else 0.5
                if similarities[idx] > min_threshold:
                    relevant_snippets.append({
                        "code": all_chunks[idx]['code'],
                        "file_path": all_chunks[idx].get('file_path', 'unknown'),
                        "similarity": float(similarities[idx])
                    })
            
            question_contexts.append({
                "question": question,
                "code_snippets": relevant_snippets
            })
        
        return question_contexts
    
    def _calculate_similarities(self, embedding: np.ndarray, query_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between embedding and query embeddings"""
        # Normalize embeddings
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        query_norms = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity
        similarities = np.dot(query_norms, embedding_norm)
        return similarities
    
    def _expand_context_with_ast(self, chunk: Dict, codebase_path: Optional[Path]) -> Dict:
        """
        Expand context using AST relationships (callers, callees, related functions).
        
        Args:
            chunk: Code chunk to expand context for
            codebase_path: Path to codebase (for AST analyzer)
        
        Returns:
            Dict with expanded context (callers, callees, related, surrounding)
        """
        expanded = {
            "callers": [],
            "callees": [],
            "related": [],
            "surrounding": ""
        }
        
        if not self.ast_analyzer or not codebase_path:
            return expanded
        
        try:
            file_path = codebase_path / chunk.get('file_path', '')
            function_name = chunk.get('name', '')
            line_start = chunk.get('line_start', 0)
            line_end = chunk.get('line_end', 0)
            
            # Try to find function in AST analyzer
            full_function_name = f"{file_path}:{function_name}"
            
            # Get callers (functions that call this one)
            callers = self.ast_analyzer.get_callers(full_function_name)
            expanded["callers"] = callers[:5]  # Limit to top 5
            
            # Get callees (functions called by this one)
            callees = self.ast_analyzer.get_callees(full_function_name)
            expanded["callees"] = callees[:5]  # Limit to top 5
            
            # Get related functions (same file, same class)
            related = self.ast_analyzer.get_related_functions(full_function_name, same_file=True)
            expanded["related"] = related[:5]  # Limit to top 5
            
            # Get surrounding context
            surrounding = self.ast_analyzer.get_surrounding_context(
                str(file_path),
                line_start,
                line_end,
                context_lines=15
            )
            expanded["surrounding"] = surrounding
            
        except Exception as e:
            logger.debug(f"Error expanding AST context: {e}")
        
        return expanded
    
    def _score_with_expanded_context(
        self,
        embedding: np.ndarray,
        similarities: np.ndarray,
        expanded_context: Dict,
        all_chunks: List[Dict],
        chunk_embeddings: np.ndarray
    ) -> float:
        """
        Re-score candidate using expanded AST context.
        
        Args:
            embedding: Original chunk embedding
            similarities: Similarities with questions
            expanded_context: Expanded context from AST
            all_chunks: All code chunks
            chunk_embeddings: All chunk embeddings
        
        Returns:
            Updated score considering expanded context
        """
        base_score = float(np.max(similarities))
        
        # Boost score if we found related code in AST
        context_boost = 0.0
        if expanded_context.get("callers"):
            context_boost += 0.05
        if expanded_context.get("callees"):
            context_boost += 0.05
        if expanded_context.get("related"):
            context_boost += 0.05
        
        # Cap boost at 0.15
        context_boost = min(context_boost, 0.15)
        
        return base_score + context_boost
    
    def _merge_ast_context(
        self,
        question_contexts: List[Dict],
        expanded_context: Dict
    ) -> List[Dict]:
        """
        Merge AST-expanded context into question contexts.
        
        Args:
            question_contexts: Original question contexts
            expanded_context: AST-expanded context
        
        Returns:
            Merged question contexts with AST information
        """
        merged = []
        
        for q_ctx in question_contexts:
            merged_ctx = q_ctx.copy()
            
            # Add AST context as additional snippets
            ast_snippets = []
            
            # Add callers
            for caller in expanded_context.get("callers", [])[:2]:
                ast_snippets.append({
                    "code": caller.get("code", ""),
                    "file_path": caller.get("file_path", "unknown"),
                    "similarity": 0.8,  # High relevance for callers
                    "type": "caller"
                })
            
            # Add callees
            for callee in expanded_context.get("callees", [])[:2]:
                ast_snippets.append({
                    "code": callee.get("code", ""),
                    "file_path": callee.get("file_path", "unknown"),
                    "similarity": 0.8,  # High relevance for callees
                    "type": "callee"
                })
            
            # Add related functions
            for related in expanded_context.get("related", [])[:2]:
                ast_snippets.append({
                    "code": related.get("code", ""),
                    "file_path": related.get("file_path", "unknown"),
                    "similarity": 0.7,  # Medium relevance for related
                    "type": "related"
                })
            
            # Merge with existing snippets
            merged_ctx["code_snippets"] = merged_ctx.get("code_snippets", []) + ast_snippets
            merged_ctx["ast_context"] = {
                "surrounding": expanded_context.get("surrounding", "")
            }
            
            merged.append(merged_ctx)
        
        return merged
    
    def _iterative_refinement(
        self,
        vulnerability_type: str,
        chunk: Dict,
        question_contexts: List[Dict],
        all_chunks: List[Dict],
        chunk_embeddings: np.ndarray,
        chunk_embedding: np.ndarray
    ) -> Dict:
        """
        Iteratively refine analysis by gathering more context for low-confidence results.
        
        Args:
            vulnerability_type: Type of vulnerability
            chunk: Code chunk being analyzed
            question_contexts: Current question contexts
            all_chunks: All code chunks
            chunk_embeddings: All chunk embeddings
            chunk_embedding: Current chunk embedding
        
        Returns:
            Updated Gemini result with refined analysis
        """
        logger.info(f"Gathering expanded context for iterative refinement")
        
        # Expand question contexts with more snippets (top 5 instead of 3)
        expanded_question_contexts = self._gather_question_contexts(
            chunk_embedding,
            all_chunks,
            chunk_embeddings,
            top_k_per_question=5
        )
        
        # Add surrounding code context
        surrounding_context = self._get_surrounding_code_context(chunk, all_chunks)
        
        # Re-query Gemini with expanded context
        gemini_result = self.gemini_service.verify_vulnerability(
            vulnerability_type=vulnerability_type,
            code_snippet=chunk['code'],
            questions=self.vulnerability_questions,
            question_contexts=expanded_question_contexts,
            context=f"File: {chunk.get('file_path', 'unknown')}, Lines: {chunk.get('line_start', 0)}-{chunk.get('line_end', 0)}\n\nSurrounding Code:\n{surrounding_context}"
        )
        
        return gemini_result
    
    def _get_surrounding_code_context(self, chunk: Dict, all_chunks: List[Dict]) -> str:
        """Get surrounding code context from same file"""
        file_path = chunk.get('file_path', '')
        line_start = chunk.get('line_start', 0)
        line_end = chunk.get('line_end', 0)
        
        # Find chunks from same file
        same_file_chunks = [
            c for c in all_chunks
            if c.get('file_path') == file_path
        ]
        
        # Find chunks that are near this one
        nearby_chunks = []
        for c in same_file_chunks:
            c_start = c.get('line_start', 0)
            c_end = c.get('line_end', 0)
            
            # Check if chunk is within 20 lines
            if (c_start <= line_end + 20 and c_end >= line_start - 20):
                nearby_chunks.append(c)
        
        # Sort by line number
        nearby_chunks.sort(key=lambda x: x.get('line_start', 0))
        
        # Combine into context
        context_lines = []
        for c in nearby_chunks[:5]:  # Limit to 5 nearby chunks
            context_lines.append(f"Lines {c.get('line_start', 0)}-{c.get('line_end', 0)}:\n{c.get('code', '')}")
        
        return "\n\n".join(context_lines)
