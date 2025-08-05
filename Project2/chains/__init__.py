# LangChain workflows package
"""
This package contains LangChain workflows for data analysis tasks.

Available workflows:
- Data Analysis: General data analysis and recommendations
- Code Generation: Generate Python code for data analysis
- Report Generation: Create comprehensive analysis reports
- Exploratory Data Analysis: Comprehensive EDA planning
- Predictive Modeling: ML model development guidance
- Data Visualization: Visualization recommendations

Usage:
    from chains.workflows import AdvancedWorkflowOrchestrator
    
    orchestrator = AdvancedWorkflowOrchestrator()
    result = await orchestrator.execute_workflow("data_analysis", input_data)
"""

from .base import (
    BaseWorkflow,
    DataAnalysisChain,
    CodeGenerationChain,
    ReportGenerationChain,
    WorkflowOrchestrator
)

from .workflows import (
    ExploratoryDataAnalysisWorkflow,
    PredictiveModelingWorkflow,
    DataVisualizationWorkflow,
    AdvancedWorkflowOrchestrator,
    MultiStepWebScrapingWorkflow,
    ModularWebScrapingWorkflow
)

__all__ = [
    "BaseWorkflow",
    "DataAnalysisChain", 
    "CodeGenerationChain",
    "ReportGenerationChain",
    "WorkflowOrchestrator",
    "ExploratoryDataAnalysisWorkflow",
    "PredictiveModelingWorkflow", 
    "DataVisualizationWorkflow",
    "AdvancedWorkflowOrchestrator",
    "MultiStepWebScrapingWorkflow",
    "ModularWebScrapingWorkflow"
]
