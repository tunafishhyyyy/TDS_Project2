#!/usr/bin/env python3
"""
Improved test script for multi-step web scraping workflow with dependency handling
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

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import requests
        print("✓ requests available")
    except ImportError:
        missing_deps.append("requests")
        print("✗ requests missing")
    
    try:
        import pandas
        print("✓ pandas available")
    except ImportError:
        missing_deps.append("pandas")
        print("✗ pandas missing")
    
    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        missing_deps.append("matplotlib")
        print("✗ matplotlib missing")
    
    try:
        import bs4
        print("✓ beautifulsoup4 available")
    except ImportError:
        missing_deps.append("beautifulsoup4")
        print("✗ beautifulsoup4 missing")
    
    try:
        import numpy
        print("✓ numpy available")
    except ImportError:
        missing_deps.append("numpy")
        print("✗ numpy missing")
    
    if missing_deps:
        print(f"\nMissing dependencies: {missing_deps}")
        print("Run: python install_dependencies.py")
        return False
    
    print("\n✓ All dependencies available!")
    return True

async def test_multi_step_workflow():
    """Test the multi-step web scraping workflow"""
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies first.")
        return
    
    # Read the questions.txt file
    try:
        with open('questions.txt', 'r', encoding='utf-8') as f:
            questions_text = f.read()
    except FileNotFoundError:
        print("questions.txt not found. Please create it with your questions.")
        return
    
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
                print("\n" + "="*50)
                print("SCRAPING PLAN")
                print("="*50)
                print(result["scraping_plan"])
            
            if "execution_result" in result:
                logger.info("Execution results available")
                print("\n" + "="*50)
                print("EXECUTION RESULTS")
                print("="*50)
                exec_result = result["execution_result"]
                print(f"Total code blocks: {exec_result.get('total_blocks', 0)}")
                print(f"Successful blocks: {exec_result.get('successful_blocks', 0)}")
                
                if "execution_results" in exec_result:
                    for i, exec_result_item in enumerate(exec_result["execution_results"]):
                        print(f"\n--- Block {i+1} ---")
                        print(f"Status: {exec_result_item['status']}")
                        
                        if exec_result_item['status'] == 'success' and 'result' in exec_result_item:
                            result_data = exec_result_item['result']
                            if 'output_variables' in result_data:
                                print("Output Variables:")
                                for var_name, var_value in result_data['output_variables'].items():
                                    print(f"  {var_name}: {var_value[:200]}...")
                            elif 'error' in result_data:
                                print(f"Error: {result_data['error']}")
                        elif 'error' in exec_result_item:
                            print(f"Error: {exec_result_item['error']}")
            
            if "steps_executed" in result:
                print("\n" + "="*50)
                print("STEPS EXECUTED")
                print("="*50)
                for step in result["steps_executed"]:
                    print(f"  ✓ {step}")
            
            if "target_url" in result:
                print(f"\nTarget URL: {result['target_url']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(test_multi_step_workflow()) 