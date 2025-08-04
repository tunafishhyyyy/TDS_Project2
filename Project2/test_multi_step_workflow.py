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

async def test_multi_step_workflow():
    """Test the multi-step web scraping workflow"""
    
    # Read the questions.txt file
    with open('questions.txt', 'r', encoding='utf-8') as f:
        questions_text = f.read()
    
    logger.info(f"Testing with questions: {questions_text[:200]}...")
    
    # Initialize the orchestrator
    orchestrator = AdvancedWorkflowOrchestrator()
    
    # Prepare workflow input
    workflow_input = {
        "task_description": questions_text,
        "questions": questions_text,
        "additional_files": {},
        "processed_files_info": {},
        "workflow_type": "multi_step_web_scraping",
        "parameters": {
            "data_requirements": "Extract table data and perform analysis",
            "output_format": "structured data with visualizations",
            "special_instructions": "Execute all steps and provide final answers"
        },
        "output_requirements": {
            "format": "structured",
            "include_visualizations": True,
            "answer_questions": True
        }
    }
    
    logger.info("Starting multi-step web scraping workflow test")
    
    try:
        # Execute the workflow
        result = await orchestrator.execute_workflow("multi_step_web_scraping", workflow_input)
        
        logger.info("Workflow execution completed")
        logger.info(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            logger.info(f"Result keys: {list(result.keys())}")
            
            # Print key results
            if "scraping_plan" in result:
                logger.info("Scraping plan generated")
                print("\n=== SCRAPING PLAN ===")
                print(result["scraping_plan"][:500] + "..." if len(result["scraping_plan"]) > 500 else result["scraping_plan"])
            
            if "execution_result" in result:
                logger.info("Execution results available")
                print("\n=== EXECUTION RESULTS ===")
                print(f"Total code blocks: {result['execution_result'].get('total_blocks', 0)}")
                print(f"Successful blocks: {result['execution_result'].get('successful_blocks', 0)}")
                
                if "execution_results" in result["execution_result"]:
                    for i, exec_result in enumerate(result["execution_result"]["execution_results"]):
                        print(f"\nBlock {i+1}: {exec_result['status']}")
                        if exec_result['status'] == 'success' and 'result' in exec_result:
                            if 'output_variables' in exec_result['result']:
                                for var_name, var_value in exec_result['result']['output_variables'].items():
                                    print(f"  {var_name}: {var_value[:100]}...")
            
            if "steps_executed" in result:
                print(f"\n=== STEPS EXECUTED ===")
                for step in result["steps_executed"]:
                    print(f"  ✓ {step}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(test_multi_step_workflow()) 