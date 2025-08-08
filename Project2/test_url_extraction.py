#!/usr/bin/env python3
"""Test URL extraction regex fix"""

import re

def extract_url_from_task(task_description: str) -> str:
    """Extract URL from task description, handling Markdown links properly"""
    
    # First try to extract from Markdown links [text](url)
    markdown_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    markdown_matches = re.findall(markdown_pattern, task_description)
    if markdown_matches:
        return markdown_matches[0][1]  # Return the URL part
    
    # Fallback to regular URL extraction, but exclude common punctuation
    url_pattern = r"https?://[^\s\)\]\},]+"
    urls = re.findall(url_pattern, task_description)
    return urls[0] if urls else ""

# Test cases
test_cases = [
    "Test from [ecourts website](https://judgments.ecourts.gov.in/).",
    "Visit https://judgments.ecourts.gov.in/ for more info.",
    "Check https://judgments.ecourts.gov.in/), it's useful.",
    "Website: https://judgments.ecourts.gov.in/",
]

print("Testing URL extraction fix:")
for i, test in enumerate(test_cases, 1):
    url = extract_url_from_task(test)
    print(f"{i}. Input: {test}")
    print(f"   Output: {url}")
    print(f"   Valid: {'✅' if url == 'https://judgments.ecourts.gov.in/' else '❌'}")
    print()
