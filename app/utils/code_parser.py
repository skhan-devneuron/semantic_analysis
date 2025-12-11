"""Code parsing utilities for semantic chunking"""
import ast
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticCodeChunker:
    """Semantic code chunker that preserves code structure"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_file(self, file_path: Path, code: str) -> List[Dict]:
        """
        Chunk a code file semantically.
        
        Args:
            file_path: Path to file
            code: File content
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Try to parse as Python AST for better chunking
        if file_path.suffix == ".py":
            chunks = self._chunk_python_ast(file_path, code)
        
        # Fallback to line-based chunking
        if not chunks:
            chunks = self._chunk_by_lines(file_path, code)
        
        return chunks
    
    def _chunk_python_ast(self, file_path: Path, code: str) -> List[Dict]:
        """Chunk Python code using AST structure"""
        chunks = []
        
        try:
            tree = ast.parse(code)
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chunk_code = ast.get_source_segment(code, node) or ""
                    
                    if len(chunk_code.split()) > self.chunk_size:
                        # Large function/class - split further
                        sub_chunks = self._split_large_code(chunk_code, node.lineno)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({
                            "code": chunk_code,
                            "type": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno or node.lineno,
                            "file_path": str(file_path)
                        })
            
            # If no functions/classes found, use line-based
            if not chunks:
                return []
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}, using line-based chunking")
            return []
        except Exception as e:
            logger.warning(f"Error parsing AST for {file_path}: {e}, using line-based chunking")
            return []
        
        return chunks
    
    def _chunk_by_lines(self, file_path: Path, code: str) -> List[Dict]:
        """Chunk code by lines (fallback method)"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_line = 1
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            # Approximate token count (rough estimate: 1 token ≈ 0.75 words)
            chunk_text = '\n'.join(current_chunk)
            word_count = len(chunk_text.split())
            
            if word_count >= self.chunk_size:
                chunks.append({
                    "code": '\n'.join(current_chunk),
                    "type": "code_block",
                    "name": f"block_{len(chunks) + 1}",
                    "line_start": current_line,
                    "line_end": i + 1,
                    "file_path": str(file_path)
                })
                
                # Overlap: keep last chunk_overlap lines
                overlap_lines = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_lines
                current_line = i + 1 - len(overlap_lines) + 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                "code": '\n'.join(current_chunk),
                "type": "code_block",
                "name": f"block_{len(chunks) + 1}",
                "line_start": current_line,
                "line_end": len(lines),
                "file_path": str(file_path)
            })
        
        return chunks
    
    def _split_large_code(self, code: str, start_line: int) -> List[Dict]:
        """Split large code block into smaller chunks"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_line = start_line
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            word_count = len('\n'.join(current_chunk).split())
            
            if word_count >= self.chunk_size:
                chunks.append({
                    "code": '\n'.join(current_chunk),
                    "type": "code_block",
                    "name": f"sub_block_{len(chunks) + 1}",
                    "line_start": current_line,
                    "line_end": start_line + i,
                    "file_path": ""  # Will be set by caller
                })
                
                overlap_lines = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_lines
                current_line = start_line + i - len(overlap_lines) + 1
        
        if current_chunk:
            chunks.append({
                "code": '\n'.join(current_chunk),
                "type": "code_block",
                "name": f"sub_block_{len(chunks) + 1}",
                "line_start": current_line,
                "line_end": start_line + len(lines),
                "file_path": ""
            })
        
        return chunks


def find_code_files(directory: Path, exclude_patterns: List[str]) -> List[Path]:
    """
    Find all code files in directory.
    
    Args:
        directory: Root directory to search
        exclude_patterns: Patterns to exclude
    
    Returns:
        List of code file paths
    """
    from app.config import SUPPORTED_EXTENSIONS
    
    code_files = []
    
    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        
        # Check extension
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        
        # Check exclude patterns
        file_str = str(file_path)
        if any(pattern in file_str for pattern in exclude_patterns):
            continue
        
        code_files.append(file_path)
    
    return code_files

