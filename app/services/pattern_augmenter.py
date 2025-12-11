"""Pattern augmentation layer for known vulnerability patterns"""
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PatternAugmenter:
    """Augments semantic detection with known vulnerability patterns"""
    
    # Common vulnerability patterns (regex-based)
    EXCESSIVE_AGENCY_PATTERNS = [
        r"subprocess\.(run|call|Popen)",
        r"os\.(system|popen|exec)",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"compile\s*\(",
        r"open\s*\([^)]*['\"]w",
        r"shutil\.(rmtree|copytree)",
        r"requests\.(get|post|put|delete)",
        r"urllib\.(urlopen|request)",
    ]
    
    PROMPT_INJECTION_PATTERNS = [
        r"f['\"]\s*\{.*user.*\}",
        r"format\s*\(.*user",
        r"%s.*user",
        r"\.format\s*\(.*input",
        r"eval\s*\(.*input",
        r"exec\s*\(.*input",
        r"LLM\(.*user",
        r"prompt\s*=\s*.*input",
        r"system_prompt.*\+.*user",
    ]
    
    INSECURE_OUTPUT_PATTERNS = [
        r"print\s*\(.*response",
        r"return\s+.*response",
        r"json\.dumps\s*\(.*llm",
        r"render_template.*response",
        r"HttpResponse\s*\(.*ai",
        r"send\s*\(.*output",
    ]
    
    SECRET_LEAKAGE_PATTERNS = [
        r"(api_key|apikey|API_KEY)\s*=\s*['\"][^'\"]+['\"]",
        r"(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]",
        r"(secret|SECRET)\s*=\s*['\"][^'\"]+['\"]",
        r"(token|TOKEN)\s*=\s*['\"][^'\"]+['\"]",
        r"aws_access_key",
        r"aws_secret_key",
        r"private_key",
        r"PRIVATE_KEY",
    ]
    
    UNBOUNDED_CONSUMPTION_PATTERNS = [
        r"for\s+.*in\s+.*range\s*\([^)]*\):",
        r"while\s+True:",
        r"while\s+.*:",
        r"\.generate\s*\([^)]*max_tokens\s*=\s*None",
        r"max_tokens\s*=\s*999999",
        r"limit\s*=\s*None",
        r"\.stream\s*\([^)]*\)",
    ]
    
    PATTERN_MAP = {
        "excessive_agency": EXCESSIVE_AGENCY_PATTERNS,
        "prompt_injection": PROMPT_INJECTION_PATTERNS,
        "insecure_output": INSECURE_OUTPUT_PATTERNS,
        "secret_leakage": SECRET_LEAKAGE_PATTERNS,
        "unbounded_consumption": UNBOUNDED_CONSUMPTION_PATTERNS,
    }
    
    def __init__(self, vulnerability_type: str):
        """
        Initialize pattern augmenter.
        
        Args:
            vulnerability_type: Type of vulnerability to detect patterns for
        """
        self.vulnerability_type = vulnerability_type
        self.patterns = self.PATTERN_MAP.get(vulnerability_type, [])
    
    def score_code(self, code: str) -> float:
        """
        Score code based on pattern matches.
        
        Args:
            code: Code snippet to score
        
        Returns:
            Pattern match score (0.0 to 1.0)
        """
        if not self.patterns:
            return 0.0
        
        matches = 0
        for pattern in self.patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                matches += 1
        
        # Normalize score: more matches = higher score, capped at 1.0
        score = min(matches / len(self.patterns), 1.0)
        return score
    
    def get_matched_patterns(self, code: str) -> List[str]:
        """
        Get list of matched patterns.
        
        Args:
            code: Code snippet to check
        
        Returns:
            List of matched pattern descriptions
        """
        matched = []
        for pattern in self.patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                # Extract readable pattern name
                pattern_name = pattern.replace(r"\s*", " ").replace(r"\(", "(").replace(r"\)", ")")
                matched.append(pattern_name)
        
        return matched
    
    def augment_similarity_score(self, semantic_score: float, code: str) -> float:
        """
        Augment semantic similarity score with pattern matching.
        
        Args:
            semantic_score: Original semantic similarity score
            code: Code snippet
        
        Returns:
            Augmented score (weighted combination)
        """
        pattern_score = self.score_code(code)
        
        # Weighted combination: 70% semantic, 30% pattern
        augmented_score = 0.7 * semantic_score + 0.3 * pattern_score
        
        return min(augmented_score, 1.0)


