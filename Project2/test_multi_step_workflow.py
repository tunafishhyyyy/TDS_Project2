#!/usr/bin/env python3
"""
Test script for multi-step web scraping workflow
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chains.workflows import AdvancedWorkflowOrchestrator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Mock LLM for demo purposes ---
def mock_llm(prompt):
    # Always return the hardcoded plan for GDP scraping
    return '''[
      {"step": "scrape_table", "url": "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"},
      {"step": "inspect_table"},
      {"step": "clean_data"},
      {"step": "analyze_data", "top_n": 10},
      {"step": "visualize"},
      {"step": "answer", "questions": [
        "Which country ranks 5th by GDP?",
        "What is the total GDP of the top 10 countries?"
      ]}
    ]'''

if __name__ == '__main__':
    # Read the user request from questions.txt
    questions_path = os.path.join(os.path.dirname(__file__), 'questions.txt')
    with open(questions_path, 'r') as f:
        user_request = f.read()

    # Use the mock LLM for demo/testing
    result = run_llm_planned_workflow(user_request, llm=mock_llm)
    print("\n--- LLM-driven Modular Workflow Result ---")
    print(result)

    # --- To use a real LLM, replace mock_llm with your LLM instance ---
    # from your_llm_module import my_llm
    # result = run_llm_planned_workflow(user_request, llm=my_llm)
    # print(result) 