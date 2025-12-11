"""Code embedding service using sentence-transformers and Pinecone"""
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging
import pinecone
from sentence_transformers import SentenceTransformer
from app.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_HOST,
    PINECONE_INDEX_NAME,
    PINECONE_SHARED_NAMESPACE
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and storing code embeddings"""
    
    def __init__(self, user_id: Optional[str] = None, codebase_id: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            user_id: User ID for user-specific namespace
            codebase_id: Codebase ID for codebase-specific namespace (creates separate collection per codebase)
        """
        self.user_id = user_id
        self.codebase_id = codebase_id
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
        
        # Initialize Pinecone
        try:
            # Initialize Pinecone client (v3+ API)
            pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            
            # Connect to index
            self.index = pc.Index(PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            
            # Set namespace based on codebase_id (separate namespace per codebase)
            if codebase_id:
                # Use codebase_id as namespace for isolation
                self.namespace = f"{user_id}_{codebase_id}" if user_id else codebase_id
            elif user_id:
                # Fallback to user_id if no codebase_id
                self.namespace = user_id
            else:
                # Shared namespace for vulnerability signatures
                self.namespace = PINECONE_SHARED_NAMESPACE
            
            logger.info(f"Using namespace: {self.namespace}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def _preprocess_code_for_embedding(self, code: str) -> str:
        """
        Preprocess code to remove string literals and focus on code structure.
        This prevents matching on string content (like "ChatBot" in messages).
        
        Args:
            code: Raw code snippet
        
        Returns:
            Preprocessed code with string literals replaced
        """
        import re
        
        # Replace string literals with placeholders to focus on code structure
        # This prevents matching on text content in strings
        processed = code
        
        # Replace single-quoted strings
        processed = re.sub(r"'[^']*'", "'STRING'", processed)
        # Replace double-quoted strings
        processed = re.sub(r'"[^"]*"', '"STRING"', processed)
        # Replace triple-quoted strings (docstrings, multi-line strings)
        processed = re.sub(r'"""[^"]*"""', '"""STRING"""', processed, flags=re.DOTALL)
        processed = re.sub(r"'''[^']*'''", "'''STRING'''", processed, flags=re.DOTALL)
        
        return processed
    
    def embed_code(self, code: str) -> np.ndarray:
        """
        Embed a single code snippet.
        
        Args:
            code: Code snippet to embed
        
        Returns:
            Embedding vector (768 dimensions for CodeBERT)
        """
        try:
            # Preprocess to remove string literals before embedding
            processed_code = self._preprocess_code_for_embedding(code)
            embedding = self.model.encode(processed_code, show_progress_bar=False)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding code: {e}")
            raise
    
    def embed_batch(self, code_snippets: List[str]) -> np.ndarray:
        """
        Embed multiple code snippets in batch.
        
        Args:
            code_snippets: List of code snippets
        
        Returns:
            Array of embedding vectors
        """
        try:
            # Preprocess all code snippets to remove string literals
            processed_snippets = [self._preprocess_code_for_embedding(code) for code in code_snippets]
            
            embeddings = self.model.encode(
                processed_snippets,
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def store_embeddings(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Store embeddings in Pinecone.
        
        Args:
            chunks: List of code chunks with metadata
            embeddings: Embedding vectors
            metadata: Additional metadata
        """
        try:
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{chunk.get('file_path', 'unknown')}_{chunk.get('name', 'chunk')}_{i}"
                # Clean chunk_id to be Pinecone-compatible (alphanumeric, hyphens, underscores)
                chunk_id = chunk_id.replace('/', '_').replace('\\', '_').replace('.', '_')
                
                # Prepare metadata (Pinecone metadata must be dict with string values)
                # Pinecone has a 40KB limit per metadata object, so we truncate code if needed
                code_snippet = str(chunk.get('code', ''))
                # Truncate to ~35KB to leave room for other metadata (safety margin)
                max_code_length = 35000
                if len(code_snippet) > max_code_length:
                    code_snippet = code_snippet[:max_code_length] + "... [truncated]"
                
                chunk_metadata = {
                    "file_path": str(chunk.get('file_path', '')),
                    "type": str(chunk.get('type', 'code_block')),
                    "name": str(chunk.get('name', 'chunk')),
                    "line_start": str(chunk.get('line_start', 0)),
                    "line_end": str(chunk.get('line_end', 0)),
                    "code": code_snippet
                }
                
                if metadata:
                    # Convert all metadata values to strings
                    for key, value in metadata.items():
                        chunk_metadata[str(key)] = str(value)
                
                # Prepare vector for upsert
                vector_data = {
                    "id": chunk_id,
                    "values": embeddings[i].tolist(),
                    "metadata": chunk_metadata
                }
                vectors_to_upsert.append(vector_data)
            
            # Upsert in batches (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)
            
            logger.info(f"Stored {len(chunks)} embeddings in namespace {self.namespace}")
        
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar code using embeddings.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (Pinecone filter format)
        
        Returns:
            List of similar code chunks with metadata
        """
        try:
            # Convert filter_metadata to Pinecone filter format if provided
            pinecone_filter = None
            if filter_metadata:
                # Pinecone uses $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin operators
                # Convert simple dict to Pinecone filter format
                pinecone_filter = {}
                for key, value in filter_metadata.items():
                    if isinstance(value, (list, tuple)):
                        pinecone_filter[key] = {"$in": [str(v) for v in value]}
                    else:
                        pinecone_filter[key] = {"$eq": str(value)}
            
            # Query Pinecone
            query_response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=n_results,
                include_metadata=True,
                namespace=self.namespace,
                filter=pinecone_filter
            )
            
            # Format results (Pinecone v3+ returns matches in query_response.matches)
            formatted_results = []
            if hasattr(query_response, 'matches') and query_response.matches:
                for match in query_response.matches:
                    # Handle both dict and object formats
                    if isinstance(match, dict):
                        match_id = match.get('id', '')
                        match_score = match.get('score', 0.0)
                        match_metadata = match.get('metadata', {})
                    else:
                        match_id = match.id
                        match_score = match.score if hasattr(match, 'score') else 0.0
                        match_metadata = match.metadata if hasattr(match, 'metadata') else {}
                    
                    formatted_results.append({
                        "id": match_id,
                        "code": match_metadata.get("code", ""),
                        "metadata": match_metadata,
                        "distance": 1.0 - match_score if match_score else None,  # Convert similarity to distance
                        "score": match_score  # Keep similarity score
                    })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all embeddings from namespace"""
        try:
            logger.warning(f"Clearing namespace {self.namespace} - this may take time for large namespaces")
            
            # Get stats to see how many vectors we need to delete
            stats = self.index.describe_index_stats()
            
            # Handle both dict and object response formats
            if isinstance(stats, dict):
                namespace_stats = stats.get('namespaces', {}).get(self.namespace, {})
                vector_count = namespace_stats.get('vector_count', 0)
            else:
                # Object format
                namespaces = getattr(stats, 'namespaces', {})
                namespace_stats = namespaces.get(self.namespace, {}) if isinstance(namespaces, dict) else {}
                vector_count = namespace_stats.get('vector_count', 0) if isinstance(namespace_stats, dict) else 0
            
            if vector_count == 0:
                logger.info(f"Namespace {self.namespace} is already empty")
                return
            
            # Pinecone v3+ supports delete_all for namespace
            try:
                # Try delete_all first (if supported)
                self.index.delete(delete_all=True, namespace=self.namespace)
                logger.info(f"Cleared all vectors from namespace {self.namespace} using delete_all")
            except Exception:
                # Fallback: delete by querying and deleting in batches
                logger.warning(f"delete_all not supported, using batch deletion for {vector_count} vectors")
                batch_size = 1000
                deleted = 0
                
                # Query to get IDs (using a zero vector)
                zero_vector = [0.0] * EMBEDDING_DIMENSION
                query_response = self.index.query(
                    vector=zero_vector,
                    top_k=min(batch_size, vector_count),
                    include_metadata=False,
                    namespace=self.namespace
                )
                
                while query_response and hasattr(query_response, 'matches') and query_response.matches:
                    ids_to_delete = []
                    for match in query_response.matches:
                        if isinstance(match, dict):
                            ids_to_delete.append(match.get('id'))
                        else:
                            ids_to_delete.append(match.id)
                    
                    if ids_to_delete:
                        self.index.delete(ids=ids_to_delete, namespace=self.namespace)
                        deleted += len(ids_to_delete)
                    
                    if deleted >= vector_count:
                        break
                    
                    # Query again for more
                    query_response = self.index.query(
                        vector=zero_vector,
                        top_k=min(batch_size, vector_count - deleted),
                        include_metadata=False,
                        namespace=self.namespace
                    )
                
                logger.info(f"Cleared {deleted} vectors from namespace {self.namespace}")
        
        except Exception as e:
            logger.error(f"Error clearing namespace: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the namespace"""
        try:
            stats = self.index.describe_index_stats()
            
            # Handle both dict and object response formats
            if isinstance(stats, dict):
                namespace_stats = stats.get('namespaces', {}).get(self.namespace, {})
                vector_count = namespace_stats.get('vector_count', 0)
            else:
                # Object format
                namespaces = getattr(stats, 'namespaces', {})
                namespace_stats = namespaces.get(self.namespace, {}) if isinstance(namespaces, dict) else {}
                vector_count = namespace_stats.get('vector_count', 0) if isinstance(namespace_stats, dict) else 0
            
            return {
                "namespace": self.namespace,
                "index_name": PINECONE_INDEX_NAME,
                "total_embeddings": vector_count,
                "user_id": self.user_id
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def delete_user_namespaces(user_id: str) -> int:
        """
        Delete all namespaces for a specific user from Pinecone.
        
        Args:
            user_id: User ID whose namespaces should be deleted
        
        Returns:
            Number of namespaces deleted
        """
        try:
            import pinecone
            from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION
            
            # Initialize Pinecone
            pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            
            # Get all namespaces
            stats = index.describe_index_stats()
            
            # Handle both dict and object response formats
            if isinstance(stats, dict):
                all_namespaces = stats.get('namespaces', {})
            else:
                all_namespaces = getattr(stats, 'namespaces', {})
                if not isinstance(all_namespaces, dict):
                    all_namespaces = {}
            
            # Find namespaces that belong to this user
            # Namespaces can be: "user123" or "user123_codebase_id"
            user_namespaces = []
            for namespace_name in all_namespaces.keys():
                if namespace_name == user_id or namespace_name.startswith(f"{user_id}_"):
                    user_namespaces.append(namespace_name)
            
            logger.info(f"Found {len(user_namespaces)} namespaces for user {user_id}: {user_namespaces}")
            
            # Delete each namespace
            deleted_count = 0
            for namespace in user_namespaces:
                try:
                    # Get vector count for this namespace
                    if isinstance(stats, dict):
                        ns_stats = all_namespaces.get(namespace, {})
                        vector_count = ns_stats.get('vector_count', 0) if isinstance(ns_stats, dict) else 0
                    else:
                        ns_stats = all_namespaces.get(namespace, {})
                        vector_count = ns_stats.get('vector_count', 0) if isinstance(ns_stats, dict) else 0
                    
                    if vector_count == 0:
                        logger.info(f"Namespace {namespace} is already empty, skipping")
                        deleted_count += 1
                        continue
                    
                    # Delete all vectors in namespace
                    try:
                        index.delete(delete_all=True, namespace=namespace)
                        logger.info(f"Deleted namespace {namespace} ({vector_count} vectors)")
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"delete_all failed for {namespace}, trying batch deletion: {e}")
                        # Fallback to batch deletion
                        zero_vector = [0.0] * EMBEDDING_DIMENSION
                        deleted = 0
                        batch_size = 1000
                        
                        while deleted < vector_count:
                            query_response = index.query(
                                vector=zero_vector,
                                top_k=min(batch_size, vector_count - deleted),
                                include_metadata=False,
                                namespace=namespace
                            )
                            
                            if not query_response or not hasattr(query_response, 'matches') or not query_response.matches:
                                break
                            
                            ids_to_delete = []
                            for match in query_response.matches:
                                if isinstance(match, dict):
                                    ids_to_delete.append(match.get('id'))
                                else:
                                    ids_to_delete.append(match.id)
                            
                            if ids_to_delete:
                                index.delete(ids=ids_to_delete, namespace=namespace)
                                deleted += len(ids_to_delete)
                            else:
                                break
                        
                        logger.info(f"Deleted namespace {namespace} ({deleted} vectors via batch deletion)")
                        deleted_count += 1
                        
                except Exception as e:
                    logger.error(f"Error deleting namespace {namespace}: {e}")
                    continue
            
            logger.info(f"Deleted {deleted_count} namespaces for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting user namespaces: {e}")
            raise
