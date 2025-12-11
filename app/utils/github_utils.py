"""GitHub repository cloning and cleanup utilities"""
import os
import shutil
import subprocess
import logging
import hashlib
import stat
import time
import platform
from pathlib import Path
from typing import Optional
from app.config import TEMP_CODEBASE_DIR, GITHUB_CLONE_TIMEOUT

logger = logging.getLogger(__name__)


def _remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree that removes read-only files on Windows.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logger.warning(f"Could not remove {path}: {e}")


def _rmtree_windows_safe(path: Path, max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    """
    Windows-safe recursive directory removal with retry logic.
    
    Args:
        path: Path to remove
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        True if successful, False otherwise
    """
    if not path.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            # On Windows, handle read-only files
            if platform.system() == "Windows":
                # First, try to make all files writable
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        try:
                            os.chmod(os.path.join(root, d), stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                        except Exception:
                            pass
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)
                        except Exception:
                            pass
                
                # Use error handler for read-only files
                shutil.rmtree(path, onerror=_remove_readonly)
            else:
                shutil.rmtree(path)
            
            return True
            
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Permission error removing {path} (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to remove {path} after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error removing {path} (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to remove {path} after {max_retries} attempts: {e}")
                return False
    
    return False


def generate_codebase_id(github_url: str) -> str:
    """Generate unique ID for codebase from GitHub URL"""
    return hashlib.md5(github_url.encode()).hexdigest()[:12]


def clone_github_repo(github_url: str, user_id: str, codebase_id: Optional[str] = None) -> Path:
    """
    Clone GitHub repository to temporary directory.
    
    Args:
        github_url: GitHub repository URL
        user_id: User ID for directory organization
        codebase_id: Optional codebase ID (generated if not provided)
    
    Returns:
        Path to cloned repository
    """
    if not codebase_id:
        codebase_id = generate_codebase_id(github_url)
    
    clone_path = TEMP_CODEBASE_DIR / user_id / codebase_id
    
    # Clean up if exists
    if clone_path.exists():
        logger.warning(f"Cleaning up existing clone at {clone_path}")
        _rmtree_windows_safe(clone_path)
    
    clone_path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Cloning {github_url} to {clone_path}")
        
        # Clone repository
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(clone_path)],
            timeout=GITHUB_CLONE_TIMEOUT,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Successfully cloned repository to {clone_path}")
        return clone_path
        
    except subprocess.TimeoutExpired:
        logger.error(f"Git clone timeout for {github_url}")
        if clone_path.exists():
            _rmtree_windows_safe(clone_path)
        raise Exception(f"Git clone timeout after {GITHUB_CLONE_TIMEOUT} seconds")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e.stderr}")
        if clone_path.exists():
            _rmtree_windows_safe(clone_path)
        raise Exception(f"Git clone failed: {e.stderr}")
    
    except Exception as e:
        logger.error(f"Unexpected error during git clone: {e}")
        if clone_path.exists():
            _rmtree_windows_safe(clone_path)
        raise


def cleanup_codebase(user_id: str, codebase_id: str) -> bool:
    """
    Clean up cloned codebase.
    
    Args:
        user_id: User ID
        codebase_id: Codebase ID
    
    Returns:
        True if cleanup successful
    """
    codebase_path = TEMP_CODEBASE_DIR / user_id / codebase_id
    
    if not codebase_path.exists():
        logger.warning(f"Codebase path does not exist: {codebase_path}")
        return False
    
    try:
        logger.info(f"Cleaning up codebase at {codebase_path}")
        success = _rmtree_windows_safe(codebase_path)
        if success:
            logger.info(f"Successfully cleaned up {codebase_path}")
        else:
            logger.warning(f"Partial cleanup of {codebase_path} - some files may remain")
        return success
    except Exception as e:
        logger.error(f"Error cleaning up codebase: {e}")
        return False


def cleanup_user_codebases(user_id: str) -> int:
    """
    Clean up all codebases for a user.
    
    Args:
        user_id: User ID
    
    Returns:
        Number of codebases cleaned up
    """
    user_dir = TEMP_CODEBASE_DIR / user_id
    
    if not user_dir.exists():
        return 0
    
    count = 0
    try:
        for item in user_dir.iterdir():
            if item.is_dir():
                if _rmtree_windows_safe(item):
                    count += 1
        logger.info(f"Cleaned up {count} codebases for user {user_id}")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up user codebases: {e}")
        return count


def cleanup_old_codebases(max_age_seconds: int = 3600) -> int:
    """
    Clean up codebases older than max_age_seconds.
    
    Args:
        max_age_seconds: Maximum age in seconds
    
    Returns:
        Number of codebases cleaned up
    """
    if not TEMP_CODEBASE_DIR.exists():
        return 0
    
    count = 0
    current_time = time.time()
    
    try:
        for user_dir in TEMP_CODEBASE_DIR.iterdir():
            if not user_dir.is_dir():
                continue
            
            for codebase_dir in user_dir.iterdir():
                if not codebase_dir.is_dir():
                    continue
                
                # Check modification time
                mtime = codebase_dir.stat().st_mtime
                age = current_time - mtime
                
                if age > max_age_seconds:
                    logger.info(f"Cleaning up old codebase: {codebase_dir} (age: {age:.0f}s)")
                    if _rmtree_windows_safe(codebase_dir):
                        count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} old codebases")
        
        return count
    except Exception as e:
        logger.error(f"Error cleaning up old codebases: {e}")
        return count

