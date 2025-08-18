"""
Enhanced LLM system with fallback - extracted from main_app.py
Integrates with existing workflow system
"""

import os
import time
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

MODEL_HIERARCHY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
]

MAX_RETRIES_PER_KEY = 2
TIMEOUT = 30
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]


class EnhancedLLMWithFallback:
    """
    Enhanced LLM wrapper with multiple API keys and model fallback
    Can be used as drop-in replacement for existing ChatGoogleGenerativeAI
    """
    def __init__(self, keys=None, models=None, temperature=0):
        self.keys = keys or GEMINI_KEYS or [os.getenv("GOOGLE_API_KEY")]
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature
        self.slow_keys_log = defaultdict(list)
        self.failing_keys_log = defaultdict(int)
        self.current_llm = None
        
        if not self.keys:
            raise RuntimeError("No API keys found. Set GOOGLE_API_KEY or gemini_api_1, gemini_api_2, etc.")

    def _get_llm_instance(self):
        """Get working LLM instance with fallback"""
        last_error = None
        for model in self.models:
            for key in self.keys:
                try:
                    llm_instance = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                        google_api_key=key
                    )
                    self.current_llm = llm_instance
                    return llm_instance
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    if any(qk in msg for qk in QUOTA_KEYWORDS):
                        self.slow_keys_log[key].append(model)
                    self.failing_keys_log[key] += 1
                    time.sleep(0.5)
        raise RuntimeError(f"All models/keys failed. Last error: {last_error}")

    def invoke(self, prompt, **kwargs):
        """Invoke with automatic fallback"""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                llm_instance = self._get_llm_instance()
                return llm_instance.invoke(prompt, **kwargs)
            except Exception as e:
                last_error = e
                time.sleep(1)  # Brief pause between retries
                
        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}")

    def bind_tools(self, tools):
        """For LangChain agent compatibility"""
        llm_instance = self._get_llm_instance()
        return llm_instance.bind_tools(tools)


def create_enhanced_llm(temperature=0):
    """Factory function to create enhanced LLM"""
    return EnhancedLLMWithFallback(temperature=temperature)
