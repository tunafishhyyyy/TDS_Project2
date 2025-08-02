"""
Specific workflow implementations for data analysis tasks
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import json
from datetime import datetime
import logging
from .base import BaseWorkflow, WorkflowOrchestrator
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

logger = logging.getLogger(__name__)

class ExploratoryDataAnalysisWorkflow(BaseWorkflow):
    """Workflow for Exploratory Data Analysis (EDA)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_eda_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_eda_prompt(self) -> ChatPromptTemplate:
        """Create EDA-specific prompt"""
        system_message = """You are an expert data scientist specializing in Exploratory Data Analysis (EDA).
Your task is to provide a comprehensive EDA plan and insights based on the provided dataset information.

Focus on:
1. Data quality assessment
2. Distribution analysis
3. Correlation analysis
4. Outlier detection
5. Missing value analysis
6. Feature engineering opportunities
7. Visualization recommendations
"""
        
        human_message = """
Dataset Information:
- Description: {dataset_description}
- Columns: {columns_info}
- Data Types: {data_types}
- Sample Size: {sample_size}
- Business Context: {business_context}

Additional Parameters: {parameters}

Provide a structured EDA plan including:
1. Initial data inspection steps
2. Statistical summaries to compute
3. Visualizations to create
4. Data quality checks
5. Feature relationships to explore
6. Potential data issues to investigate
7. Python code snippets for key analyses

Format your response with clear sections and actionable recommendations.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute EDA workflow"""
        try:
            # Extract dataset information
            dataset_info = input_data.get("dataset_info", {})
            
            result = self.chain.run(
                dataset_description=dataset_info.get("description", "Unknown dataset"),
                columns_info=json.dumps(dataset_info.get("columns", []), indent=2),
                data_types=json.dumps(dataset_info.get("data_types", {}), indent=2),
                sample_size=dataset_info.get("sample_size", "Unknown"),
                business_context=input_data.get("business_context", "General analysis"),
                parameters=json.dumps(input_data.get("parameters", {}), indent=2)
            )
            
            return {
                "eda_plan": result,
                "workflow_type": "exploratory_data_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "dataset_summary": dataset_info
            }
            
        except Exception as e:
            logger.error(f"Error in EDA workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "exploratory_data_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }

class PredictiveModelingWorkflow(BaseWorkflow):
    """Workflow for predictive modeling tasks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_modeling_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_modeling_prompt(self) -> ChatPromptTemplate:
        """Create predictive modeling prompt"""
        system_message = """You are a machine learning expert specializing in predictive modeling.
Your task is to design an appropriate modeling approach based on the problem description and data characteristics.

Consider:
1. Problem type (classification, regression, clustering, etc.)
2. Data characteristics and quality
3. Model selection and evaluation metrics
4. Feature engineering requirements
5. Cross-validation strategy
6. Model interpretability needs
7. Production deployment considerations
"""
        
        human_message = """
Problem Statement: {problem_statement}
Target Variable: {target_variable}
Dataset Characteristics: {dataset_characteristics}
Business Requirements: {business_requirements}
Performance Requirements: {performance_requirements}

Provide a comprehensive modeling approach including:
1. Problem formulation
2. Recommended algorithms
3. Feature engineering strategy
4. Model evaluation approach
5. Cross-validation strategy
6. Performance metrics
7. Implementation roadmap
8. Potential challenges and mitigation strategies

Include Python code examples using scikit-learn and other relevant libraries.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute predictive modeling workflow"""
        try:
            result = self.chain.run(
                problem_statement=input_data.get("problem_statement", ""),
                target_variable=input_data.get("target_variable", ""),
                dataset_characteristics=json.dumps(input_data.get("dataset_characteristics", {}), indent=2),
                business_requirements=input_data.get("business_requirements", ""),
                performance_requirements=input_data.get("performance_requirements", "")
            )
            
            return {
                "modeling_plan": result,
                "workflow_type": "predictive_modeling",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "problem_type": self._identify_problem_type(input_data)
            }
            
        except Exception as e:
            logger.error(f"Error in predictive modeling workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "predictive_modeling",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _identify_problem_type(self, input_data: Dict[str, Any]) -> str:
        """Identify the type of ML problem"""
        problem_statement = input_data.get("problem_statement", "").lower()
        target_variable = input_data.get("target_variable", "").lower()
        
        if any(keyword in problem_statement for keyword in ["classify", "classification", "category", "class"]):
            return "classification"
        elif any(keyword in problem_statement for keyword in ["predict", "regression", "forecast", "continuous"]):
            return "regression"
        elif any(keyword in problem_statement for keyword in ["cluster", "segment", "group"]):
            return "clustering"
        elif any(keyword in problem_statement for keyword in ["recommend", "recommendation"]):
            return "recommendation"
        else:
            return "unknown"

class DataVisualizationWorkflow(BaseWorkflow):
    """Workflow for data visualization recommendations"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_visualization_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_visualization_prompt(self) -> ChatPromptTemplate:
        """Create visualization prompt"""
        system_message = """You are a data visualization expert specializing in creating effective and insightful charts and graphs.
Your task is to recommend appropriate visualizations based on the data characteristics and analysis goals.

Consider:
1. Data types (categorical, numerical, temporal)
2. Number of variables and relationships
3. Target audience
4. Story to tell with the data
5. Interactive vs static requirements
6. Best practices for clarity and aesthetics
"""
        
        human_message = """
Data Description: {data_description}
Variables: {variables}
Analysis Goals: {analysis_goals}
Target Audience: {target_audience}
Platform/Tools: {platform}

Recommend appropriate visualizations including:
1. Chart types for each analysis goal
2. Layout and design considerations
3. Interactive features (if applicable)
4. Color schemes and styling
5. Python code using matplotlib, seaborn, or plotly
6. Dashboard structure (if multiple charts)

Provide detailed rationale for each recommendation.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization workflow"""
        try:
            result = self.chain.run(
                data_description=input_data.get("data_description", ""),
                variables=json.dumps(input_data.get("variables", []), indent=2),
                analysis_goals=input_data.get("analysis_goals", ""),
                target_audience=input_data.get("target_audience", "technical team"),
                platform=input_data.get("platform", "Python (matplotlib/seaborn)")
            )
            
            return {
                "visualization_plan": result,
                "workflow_type": "data_visualization",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "recommended_tools": ["matplotlib", "seaborn", "plotly"]
            }
            
        except Exception as e:
            logger.error(f"Error in visualization workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "data_visualization",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }

class AdvancedWorkflowOrchestrator(WorkflowOrchestrator):
    """Enhanced orchestrator with domain-specific workflows"""
    
    def __init__(self):
        super().__init__()
        # Add specialized workflows
        self.workflows.update({
            "exploratory_data_analysis": ExploratoryDataAnalysisWorkflow(),
            "predictive_modeling": PredictiveModelingWorkflow(),
            "data_visualization": DataVisualizationWorkflow()
        })
    
    async def execute_complete_analysis_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete data analysis pipeline"""
        
        # Define the analysis pipeline steps
        pipeline_steps = [
            {
                "workflow_type": "exploratory_data_analysis",
                "input_data": {
                    "dataset_info": input_data.get("dataset_info", {}),
                    "business_context": input_data.get("business_context", ""),
                    "parameters": input_data.get("eda_parameters", {})
                }
            },
            {
                "workflow_type": "data_visualization",
                "input_data": {
                    "data_description": input_data.get("data_description", ""),
                    "variables": input_data.get("variables", []),
                    "analysis_goals": "Exploratory data analysis and pattern discovery",
                    "target_audience": input_data.get("target_audience", "technical team")
                }
            }
        ]
        
        # Add predictive modeling if specified
        if input_data.get("include_modeling", False):
            pipeline_steps.append({
                "workflow_type": "predictive_modeling",
                "input_data": {
                    "problem_statement": input_data.get("problem_statement", ""),
                    "target_variable": input_data.get("target_variable", ""),
                    "dataset_characteristics": input_data.get("dataset_info", {}),
                    "business_requirements": input_data.get("business_requirements", ""),
                    "performance_requirements": input_data.get("performance_requirements", "")
                }
            })
        
        # Add report generation
        pipeline_steps.append({
            "workflow_type": "report_generation",
            "input_data": {
                "analysis_results": "Will be populated from previous steps",
                "data_summary": json.dumps(input_data.get("dataset_info", {})),
                "key_findings": "Will be extracted from analysis",
                "audience": input_data.get("target_audience", "technical team")
            }
        })
        
        # Execute the pipeline
        result = await self.execute_multi_step_workflow(pipeline_steps)
        
        return {
            "pipeline_result": result,
            "pipeline_type": "complete_analysis",
            "timestamp": datetime.now().isoformat(),
            "input_summary": {
                "dataset_info": input_data.get("dataset_info", {}),
                "include_modeling": input_data.get("include_modeling", False),
                "target_audience": input_data.get("target_audience", "technical team")
            }
        }
    
    def get_workflow_capabilities(self) -> Dict[str, Any]:
        """Return information about available workflows and their capabilities"""
        return {
            "available_workflows": list(self.workflows.keys()),
            "workflow_descriptions": {
                "data_analysis": "General data analysis and recommendations",
                "code_generation": "Generate Python code for data analysis tasks",
                "report_generation": "Create comprehensive analysis reports",
                "exploratory_data_analysis": "Comprehensive EDA planning and execution",
                "predictive_modeling": "Machine learning model development guidance",
                "data_visualization": "Visualization recommendations and code generation"
            },
            "pipeline_capabilities": [
                "complete_analysis_pipeline",
                "multi_step_workflow"
            ],
            "supported_features": [
                "Memory management across conversations",
                "Error handling and recovery",
                "Execution history tracking",
                "Flexible input/output formats",
                "Integration with multiple LLM providers"
            ]
        }
