"""LLM utility functions for initializing language models."""

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


def get_llm(provider: Optional[str] = None, model: Optional[str] = None):
    """
    Initialize and return an LLM based on provider.
    
    Args:
        provider: LLM provider name (openai, anthropic, ollama)
        model: Model name to use
    
    Returns:
        Initialized LLM instance
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(model=model, temperature=0, api_key=api_key)
    
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return ChatAnthropic(model=model, temperature=0, api_key=api_key)
    
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_for_planning(provider: Optional[str] = None, model: Optional[str] = None):
    """
    Initialize and return an LLM for planning/reviewing tasks.
    Uses GPT-5 by default for advanced reasoning and planning.
    
    Args:
        provider: LLM provider name (openai, anthropic, ollama)
        model: Model name to use (default: gpt-5)
    
    Returns:
        Initialized LLM instance for planning
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("PLANNING_MODEL", "gpt-5")
    
    return get_llm(provider=provider, model=model)


def get_llm_for_coding(provider: Optional[str] = None, model: Optional[str] = None):
    """
    Initialize and return an LLM for coding/code review tasks.
    Uses GPT-5 by default for advanced coding capabilities.
    
    Args:
        provider: LLM provider name (openai, anthropic, ollama)
        model: Model name to use (default: gpt-5)
    
    Returns:
        Initialized LLM instance for coding
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("CODING_MODEL", "gpt-5")
    
    return get_llm(provider=provider, model=model)

