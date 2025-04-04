"""
Utility functions for SEC Filing Q&A App
"""
import re
from typing import Dict, Any, List, Optional

def clean_text(text):
    """Clean text from HTML content"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n\n', text)
    return text.strip()

def format_sources(sources, max_length=300):
    """Format source documents for display"""
    formatted = []
    for source in sources:
        if len(source) > max_length:
            formatted.append(f"{source[:max_length]}...")
        else:
            formatted.append(source)
    return formatted