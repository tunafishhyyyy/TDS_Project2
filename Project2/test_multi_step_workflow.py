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

async def test_orchestrator():
    """Test the orchestrator initialization and workflow execution"""
    try:
        # Test orchestrator creation
        orchestrator = AdvancedWorkflowOrchestrator()
        print('✓ Orchestrator created successfully')
        print('Available workflows:', list(orchestrator.workflows.keys()))
        print('LLM available:', orchestrator.llm is not None)
        
        # Read the user request from questions.txt
        questions_path = os.path.join(os.path.dirname(__file__), 'questions.txt')
        if os.path.exists(questions_path):
            with open(questions_path, 'r') as f:
                user_request = f.read()
            
            # Test workflow execution
            workflow_input = {
                "task_description": user_request,
                "questions": user_request,
                "additional_files": {},
                "processed_files_info": {},
                "workflow_type": "multi_step_web_scraping"
            }
            
            result = await orchestrator.execute_workflow("multi_step_web_scraping", workflow_input)
            print("\n--- Workflow Execution Result ---")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Workflow Type: {result.get('workflow_type', 'unknown')}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("✓ Workflow executed successfully")
        else:
            print("questions.txt not found, skipping workflow test")
            
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_orchestrator())