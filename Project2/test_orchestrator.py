#!/usr/bin/env python3
"""
Test script for the updated orchestrator
"""

try:
    from chains.workflows import AdvancedWorkflowOrchestrator
    orchestrator = AdvancedWorkflowOrchestrator()
    print('✓ Orchestrator created successfully')
    print('Available workflows:', list(orchestrator.workflows.keys()))
    print('LLM available:', orchestrator.llm is not None)
    
    # Test if multi_step_web_scraping workflow is available
    if 'multi_step_web_scraping' in orchestrator.workflows:
        workflow = orchestrator.workflows['multi_step_web_scraping']
        print('✓ Multi-step web scraping workflow available')
        print('  Type:', type(workflow).__name__)
    else:
        print('✗ Multi-step web scraping workflow not found')
        
except Exception as e:
    print('✗ Error creating orchestrator:', e)
    import traceback
    traceback.print_exc()
