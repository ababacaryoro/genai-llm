"""
Utility functions for the LLM Chatbot application
"""

from typing import List, Dict, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The input text
        model: The model name for tokenizer selection
    
    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Estimate the API cost for a request.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: The model name
    
    Returns:
        Estimated cost in USD
    """
    # Pricing per 1M tokens (as of 2024)
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    
    input_cost = (input_tokens / 1000000) * model_pricing["input"]
    output_cost = (output_tokens / 1000000) * model_pricing["output"]
    
    return input_cost + output_cost


def format_conversation_history(messages: List[Dict]) -> str:
    """
    Format conversation history for display or logging.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Formatted string representation
    """
    formatted = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        formatted.append(f"[{role}]: {content}")
    
    return "\n\n".join(formatted)


def truncate_history(messages: List[Dict], max_tokens: int = 4000, model: str = "gpt-4o-mini") -> List[Dict]:
    """
    Truncate conversation history to fit within token limits.
    Keeps the most recent messages.
    
    Args:
        messages: List of message dictionaries
        max_tokens: Maximum allowed tokens
        model: Model for tokenization
    
    Returns:
        Truncated message list
    """
    if not messages:
        return []
    
    total_tokens = 0
    truncated = []
    
    # Process messages from most recent to oldest
    for message in reversed(messages):
        msg_tokens = count_tokens(message["content"], model)
        if total_tokens + msg_tokens <= max_tokens:
            truncated.insert(0, message)
            total_tokens += msg_tokens
        else:
            break
    
    return truncated