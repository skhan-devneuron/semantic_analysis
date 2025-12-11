"""Gemini AI service for vulnerability verification"""
import logging
import os
from typing import List, Dict, Optional, Any
import google.generativeai as genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


class GeminiVerificationService:
    """Service for using Gemini to verify vulnerability detections"""
    
    def __init__(self):
        """Initialize Gemini service"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Set it in environment variables.")
        
        # Configure google-generativeai
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different model name formats
        model_name = GEMINI_MODEL
        if model_name.startswith("gemini/"):
            # Remove "gemini/" prefix for google-generativeai
            model_name = model_name.replace("gemini/", "")
        
        # Try to initialize the model
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize {model_name}, trying gemini-1.5-pro: {e}")
            # Fallback to gemini-1.5-pro
            try:
                self.model = genai.GenerativeModel("gemini-1.5-pro")
                logger.info("Initialized Gemini model: gemini-1.5-pro (fallback)")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback model: {e2}")
                # Last resort: try gemini-pro
                self.model = genai.GenerativeModel("gemini-pro")
                logger.info("Initialized Gemini model: gemini-pro (last resort)")
    
    def verify_vulnerability(
        self,
        vulnerability_type: str,
        code_snippet: str,
        questions: List[str],
        question_contexts: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify if a vulnerability actually exists using Gemini with question-based context.
        
        Args:
            vulnerability_type: Type of vulnerability (e.g., "excessive_agency", "prompt_injection")
            code_snippet: The code snippet to analyze
            questions: List of 3 contextual questions
            question_contexts: List of dicts with 'question' and 'code_snippets' for each question
            context: Optional additional context about the codebase
        
        Returns:
            Dict with:
                - vulnerability_detected: bool
                - confidence: float (0.0 to 1.0)
                - reasoning: str (explanation)
        """
        try:
            # Build prompt for Gemini
            prompt = self._build_verification_prompt(
                vulnerability_type,
                code_snippet,
                questions,
                question_contexts,
                context
            )
            
            # Log what we're sending to Gemini
            logger.info(f"\n{'='*80}")
            logger.info(f"GEMINI VERIFICATION REQUEST")
            logger.info(f"{'='*80}")
            logger.info(f"Vulnerability Type: {vulnerability_type}")
            logger.info(f"Code Snippet (first 500 chars):\n{code_snippet[:500]}...")
            logger.info(f"Number of question contexts: {len(question_contexts)}")
            for i, q_ctx in enumerate(question_contexts, 1):
                snippets = q_ctx.get('code_snippets', [])
                logger.info(f"  Question {i}: {q_ctx.get('question', questions[i-1] if i <= len(questions) else '')}")
                logger.info(f"    Found {len(snippets)} code snippets for this question:")
                for j, snippet in enumerate(snippets, 1):
                    logger.info(f"      Snippet {j}:")
                    logger.info(f"        File: {snippet.get('file_path', 'unknown')}")
                    logger.info(f"        Similarity: {snippet.get('similarity', 0):.3f}")
                    logger.info(f"        Code (first 300 chars):\n{snippet.get('code', '')[:300]}...")
            logger.info(f"{'='*80}\n")
            
            # Generate response using google-generativeai
            generation_config = {
                "temperature": 1.0,
                "max_output_tokens": 16000,
            }
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            response_text = response.text.strip()
            
            # Log raw response
            logger.info(f"\n{'='*80}")
            logger.info(f"GEMINI RAW RESPONSE")
            logger.info(f"{'='*80}")
            logger.info(f"{response_text}")
            logger.info(f"{'='*80}\n")
            
            # Parse response
            result = self._parse_response(response_text)
            
            logger.info(f"Gemini verification result: {result['vulnerability_detected']} (confidence: {result['confidence']:.2f})")
            logger.info(f"Reasoning: {result['reasoning'][:200]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini verification: {e}", exc_info=True)
            # Fallback: return uncertain result
            return {
                "vulnerability_detected": False,
                "confidence": 0.0,
                "reasoning": f"Verification failed: {str(e)}"
            }
    
    def _build_verification_prompt(
        self,
        vulnerability_type: str,
        code_snippet: str,
        questions: List[str],
        question_contexts: List[Dict[str, Any]],
        context: Optional[str]
    ) -> str:
        """Build the prompt for Gemini verification using question-based context"""
        
        vulnerability_descriptions = {
            "excessive_agency": "Excessive Agency - An AI agent has too much autonomy or permissions, allowing it to perform actions beyond its intended scope",
            "prompt_injection": "Prompt Injection - Malicious input designed to manipulate or override the intended behavior of an AI system",
            "insecure_output": "Insecure Output Handling - The system fails to properly validate, sanitize, or filter outputs from AI models before presenting them to users",
            "secret_leakage": "Secret Leakage - Sensitive information like API keys, passwords, or tokens are exposed in code or outputs",
            "unbounded_consumption": "Unbounded Consumption - The system allows unlimited resource consumption (API calls, tokens, compute) without proper limits",
            "prompt_leakage": "Prompt Leakage - System prompts or sensitive instructions are exposed to end users",
            "sensitive_info": "Sensitive Information Disclosure - Personal data, credentials, or other sensitive information is improperly handled or exposed",
            "cross_context": "Cross-Context Leakage - Information from one context or session leaks into another unauthorized context"
        }
        
        vuln_description = vulnerability_descriptions.get(
            vulnerability_type,
            f"{vulnerability_type} vulnerability"
        )
        
        # Build question-based context section
        context_section = "\n\nCONTEXTUAL EVIDENCE (Code snippets that answer key questions):\n"
        for i, q_context in enumerate(question_contexts, 1):
            question = q_context.get('question', questions[i-1] if i <= len(questions) else '')
            code_snippets = q_context.get('code_snippets', [])
            
            context_section += f"\nQuestion {i}: {question}\n"
            if code_snippets:
                context_section += "Relevant code snippets:\n"
                for snippet_info in code_snippets:
                    snippet_code = snippet_info.get('code', '')
                    snippet_file = snippet_info.get('file_path', 'unknown')
                    context_section += f"  From {snippet_file}:\n```\n{snippet_code}\n```\n"
            else:
                context_section += "  (No relevant code snippets found)\n"
        
        prompt = f"""You are a security expert analyzing code for vulnerabilities. Your task is to determine if a specific vulnerability exists in the provided code snippet, using the contextual evidence gathered from the codebase.

VULNERABILITY TYPE: {vuln_description}

PRIMARY CODE SNIPPET TO ANALYZE:
```python
{code_snippet}
```
{context_section}
"""
        
        if context:
            prompt += f"\nADDITIONAL CONTEXT:\n{context}\n"
        
        prompt += """
INSTRUCTIONS:
1. Carefully analyze the primary code snippet
2. Review the contextual evidence (code snippets that answer the key questions)
3. Answer each of the 3 questions based on the evidence provided
4. Determine if the vulnerability actually exists based on your answers
5. Provide a clear, definitive assessment

RESPONSE FORMAT (you must respond in this exact format):
VULNERABILITY_DETECTED: [YES/NO]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation of your decision based on the evidence, 2-3 sentences]

Your response:"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured format"""
        vulnerability_detected = False
        confidence = 0.0
        reasoning = response_text
        
        # Try to parse structured response
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('VULNERABILITY_DETECTED:'):
                value = line.split(':', 1)[1].strip().upper()
                vulnerability_detected = value in ['YES', 'TRUE', '1', 'DETECTED']
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except ValueError:
                    pass
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        # If parsing failed, try to infer from text
        if not any(line.startswith('VULNERABILITY_DETECTED:') for line in lines):
            text_lower = response_text.lower()
            if any(word in text_lower for word in ['yes', 'detected', 'vulnerable', 'exists', 'present']):
                vulnerability_detected = True
                confidence = 0.7
            elif any(word in text_lower for word in ['no', 'not detected', 'safe', 'not vulnerable', 'absent']):
                vulnerability_detected = False
                confidence = 0.7
        
        return {
            "vulnerability_detected": vulnerability_detected,
            "confidence": confidence,
            "reasoning": reasoning
        }
