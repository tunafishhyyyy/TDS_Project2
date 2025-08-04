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

class WebScrapingWorkflow(BaseWorkflow):
    """Workflow for web scraping and data extraction"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_scraping_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_scraping_prompt(self) -> ChatPromptTemplate:
        """Create web scraping prompt"""
        system_message = """You are a web scraping expert specializing in data extraction from websites.
Your task is to provide Python code and analysis for web scraping tasks.

Focus on:
1. URL analysis and data structure identification
2. HTML parsing strategies using BeautifulSoup/Selenium
3. Data cleaning and transformation
4. Handling pagination and dynamic content
5. Error handling and rate limiting
6. Data validation and quality checks
"""
        
        human_message = """
Scraping Task: {task_description}
Target URL: {url}
Data Requirements: {data_requirements}
Output Format: {output_format}
Special Instructions: {special_instructions}

Provide a complete solution including:
1. Python code for scraping the data
2. Data cleaning and processing steps
3. Analysis of the extracted data
4. Visualization code if requested
5. Error handling strategies
6. Expected output format

Format your response with clear code blocks and explanations.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web scraping workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                url=input_data.get("url", ""),
                data_requirements=input_data.get("data_requirements", ""),
                output_format=input_data.get("output_format", "structured data"),
                special_instructions=input_data.get("special_instructions", "")
            )
            
            return {
                "scraping_plan": result,
                "workflow_type": "web_scraping",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "target_url": input_data.get("url", "")
            }
            
        except Exception as e:
            logger.error(f"Error in web scraping workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "web_scraping",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }


class DatabaseAnalysisWorkflow(BaseWorkflow):
    """Workflow for database analysis using DuckDB and SQL"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_database_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_database_prompt(self) -> ChatPromptTemplate:
        """Create database analysis prompt"""
        system_message = """You are a database analysis expert specializing in SQL queries and data analysis.
Your task is to provide SQL code and analysis strategies for complex datasets.

Focus on:
1. SQL query optimization for large datasets
2. DuckDB-specific features and functions
3. Data aggregation and statistical analysis
4. Performance optimization strategies
5. Cloud storage integration (S3, etc.)
6. Data visualization and reporting
"""
        
        human_message = """
Analysis Task: {task_description}
Database/Dataset: {database_info}
Data Schema: {schema_info}
Analysis Goals: {analysis_goals}
Performance Requirements: {performance_requirements}

Provide a comprehensive solution including:
1. SQL queries for data analysis
2. Performance optimization strategies
3. Data processing pipeline
4. Statistical analysis methods
5. Visualization recommendations
6. Expected insights and outputs

Use DuckDB syntax and best practices for cloud data access.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database analysis workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                database_info=json.dumps(input_data.get("database_info", {}), indent=2),
                schema_info=json.dumps(input_data.get("schema_info", {}), indent=2),
                analysis_goals=input_data.get("analysis_goals", ""),
                performance_requirements=input_data.get("performance_requirements", "")
            )
            
            return {
                "analysis_plan": result,
                "workflow_type": "database_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "database_type": "DuckDB"
            }
            
        except Exception as e:
            logger.error(f"Error in database analysis workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "database_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }


class WikipediaAnalysisWorkflow(BaseWorkflow):
    """Workflow specifically for Wikipedia data analysis tasks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_wikipedia_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_wikipedia_prompt(self) -> ChatPromptTemplate:
        """Create Wikipedia analysis prompt"""
        system_message = """You are a Wikipedia data analysis expert.
Your task is to analyze Wikipedia pages and extract structured data for analysis.

Focus on:
1. Wikipedia table extraction and parsing
2. Data cleaning for Wikipedia-specific formatting
3. Statistical analysis of extracted data
4. Correlation and regression analysis
5. Data visualization with base64 encoding
6. Handling Wikipedia citations and footnotes
"""
        
        human_message = """
Wikipedia Analysis Task: {task_description}
Wikipedia URL: {wikipedia_url}
Target Tables/Data: {target_data}
Analysis Questions: {analysis_questions}
Visualization Requirements: {visualization_requirements}

Provide a complete solution including:
1. Python code to scrape Wikipedia tables
2. Data cleaning for footnotes and formatting
3. Statistical analysis code
4. Visualization code with base64 encoding
5. Answers to specific questions
6. Performance optimizations

Ensure the solution handles Wikipedia's specific formatting challenges.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Wikipedia analysis workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                wikipedia_url=input_data.get("wikipedia_url", ""),
                target_data=input_data.get("target_data", ""),
                analysis_questions=json.dumps(input_data.get("analysis_questions", []), indent=2),
                visualization_requirements=input_data.get("visualization_requirements", "")
            )
            
            return {
                "analysis_plan": result,
                "workflow_type": "wikipedia_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "source_url": input_data.get("wikipedia_url", "")
            }
            
        except Exception as e:
            logger.error(f"Error in Wikipedia analysis workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "wikipedia_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }


class LegalDataAnalysisWorkflow(BaseWorkflow):
    """Workflow for legal/court data analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_legal_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_legal_prompt(self) -> ChatPromptTemplate:
        """Create legal data analysis prompt"""
        system_message = """You are an expert in legal data analysis and Indian High Court systems.
Your task is to analyze legal datasets, particularly court judgments and case data.

Focus on:
1. Legal terminology and court procedures
2. Case disposal patterns and timelines
3. Judicial efficiency metrics
4. Court workload analysis
5. Legal trend analysis
6. Case outcome predictions
7. Statistical analysis of legal proceedings
"""
        
        human_message = """
Task: {task_description}

Legal Dataset Information: {dataset_info}

Court System Context: {court_context}

Analysis Parameters: {parameters}

Please provide a comprehensive analysis plan for this legal data task, including:
- Data preprocessing specific to legal documents
- Relevant legal metrics and KPIs
- Court system considerations
- Timeline and case flow analysis
- Visualization recommendations for legal data
- Insights relevant to judicial administration
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute legal data analysis workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                dataset_info=json.dumps(input_data.get("dataset_info", {}), indent=2),
                court_context=input_data.get("court_context", "Indian High Courts"),
                parameters=json.dumps(input_data.get("parameters", {}), indent=2)
            )
            
            return {
                "analysis_plan": result,
                "workflow_type": "legal_data_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "domain": "legal",
                "court_system": input_data.get("court_context", "Indian High Courts")
            }
            
        except Exception as e:
            logger.error(f"Error in legal data analysis workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "legal_data_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }


class WikipediaScrapingWorkflow(BaseWorkflow):
    """Enhanced workflow specifically for Wikipedia scraping and analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_wikipedia_scraping_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_wikipedia_scraping_prompt(self) -> ChatPromptTemplate:
        """Create Wikipedia scraping analysis prompt"""
        system_message = """You are an expert in web scraping and Wikipedia data extraction.
Your task is to analyze Wikipedia pages and extract structured data from tables and content.

Focus on:
1. Table structure identification and extraction
2. Data cleaning and standardization
3. Handling Wikipedia-specific formatting
4. Statistical analysis of extracted data
5. Data visualization recommendations
6. Correlation and trend analysis
"""
        
        human_message = """
Task: {task_description}

Wikipedia URL: {wikipedia_url}

Target Data: {target_data_description}

Analysis Goals: {analysis_goals}

Please provide a comprehensive plan for:
- Scraping strategy for the Wikipedia page
- Table extraction and data cleaning methods
- Statistical analysis approach
- Visualization recommendations
- Specific code snippets for data extraction
- Expected data format and structure
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Wikipedia scraping workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                wikipedia_url=input_data.get("wikipedia_url", ""),
                target_data_description=input_data.get("target_data_description", ""),
                analysis_goals=input_data.get("analysis_goals", "")
            )
            
            return {
                "scraping_plan": result,
                "workflow_type": "wikipedia_scraping",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "source": "wikipedia",
                "url": input_data.get("wikipedia_url", "")
            }
            
        except Exception as e:
            logger.error(f"Error in Wikipedia scraping workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "wikipedia_scraping",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }


class StatisticalAnalysisWorkflow(BaseWorkflow):
    """Workflow for statistical analysis including correlation and regression"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_statistical_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_statistical_prompt(self) -> ChatPromptTemplate:
        """Create statistical analysis prompt"""
        system_message = """You are an expert statistician and data analyst.
Your task is to perform comprehensive statistical analysis including correlation, regression, and trend analysis.

Focus on:
1. Descriptive statistics and data summarization
2. Correlation analysis and interpretation
3. Regression modeling and validation
4. Statistical significance testing
5. Trend analysis and forecasting
6. Data visualization for statistical insights
"""
        
        human_message = """
Task: {task_description}

Dataset Description: {dataset_description}

Variables of Interest: {variables}

Statistical Methods Required: {methods}

Please provide:
- Statistical analysis approach
- Correlation analysis plan
- Regression modeling strategy
- Visualization recommendations
- Code snippets for analysis
- Interpretation guidelines
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                dataset_description=input_data.get("dataset_description", ""),
                variables=json.dumps(input_data.get("variables", []), indent=2),
                methods=input_data.get("statistical_methods", "correlation, regression")
            )
            
            return {
                "statistical_plan": result,
                "workflow_type": "statistical_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "methods": input_data.get("statistical_methods", "correlation, regression")
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "statistical_analysis",
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
            "data_visualization": DataVisualizationWorkflow(),
            "web_scraping": WebScrapingWorkflow(),
            "database_analysis": DatabaseAnalysisWorkflow(),
            "wikipedia_analysis": WikipediaAnalysisWorkflow(),
            "legal_data_analysis": LegalDataAnalysisWorkflow(),
            "wikipedia_scraping": WikipediaScrapingWorkflow(),
            "statistical_analysis": StatisticalAnalysisWorkflow()
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
                "data_visualization": "Visualization recommendations and code generation",
                "web_scraping": "Web scraping and data extraction from websites",
                "database_analysis": "SQL analysis using DuckDB for large datasets",
                "wikipedia_analysis": "Specialized Wikipedia data extraction and analysis",
                "legal_data_analysis": "Analysis of legal/court data and judgments",
                "wikipedia_scraping": "Enhanced Wikipedia scraping with table extraction",
                "statistical_analysis": "Statistical analysis including correlation and regression"
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
                "Integration with multiple LLM providers",
                "Legal data analysis capabilities",
                "Wikipedia specialized scraping",
                "Statistical analysis and visualization"
            ]
        }
