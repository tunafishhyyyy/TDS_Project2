"""
Specific workflow implementations for data analysis tasks
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import json
from datetime import datetime
import logging
from chains.base import BaseWorkflow, WorkflowOrchestrator
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

logger = logging.getLogger(__name__)

class ExploratoryDataAnalysisWorkflow(BaseWorkflow):
    """Workflow for Exploratory Data Analysis (EDA)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_eda_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
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


class DataAnalysisWorkflow(BaseWorkflow):
    """Generalized workflow for data analysis tasks"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a data analyst. Analyze the provided data and answer the questions."),
            ("human", "Questions: {questions}\nFiles: {files}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing DataAnalysisWorkflow")
        try:
            result = self.chain.run(questions=input_data.get("task_description", ""), files=input_data.get("files", []))
            return {"analysis_result": result, "workflow_type": "data_analysis", "status": "completed", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error in DataAnalysisWorkflow: {e}")
            return {"error": str(e), "workflow_type": "data_analysis", "status": "error", "timestamp": datetime.now().isoformat()}

class ImageAnalysisWorkflow(BaseWorkflow):
    """Workflow for image analysis tasks"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in image analysis. Analyze the provided image and answer the questions."),
            ("human", "Questions: {questions}\nImage: {image_file}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing ImageAnalysisWorkflow")
        try:
            result = self.chain.run(questions=input_data.get("task_description", ""), image_file=input_data.get("image_file", ""))
            return {"image_analysis_result": result, "workflow_type": "image_analysis", "status": "completed", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error in ImageAnalysisWorkflow: {e}")
            return {"error": str(e), "workflow_type": "image_analysis", "status": "error", "timestamp": datetime.now().isoformat()}

class CodeGenerationWorkflow(BaseWorkflow):
    """Workflow for Python code generation and execution"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Python expert specializing in data analysis code generation. 
            Generate clean, executable Python code that:
            1. Is syntactically correct and follows Python best practices
            2. Includes necessary imports at the top
            3. Has clear comments explaining each section
            4. Handles potential errors gracefully
            5. Returns meaningful results
            6. Uses common data analysis libraries (pandas, numpy, matplotlib, seaborn)
            
            Always return ONLY the Python code without any markdown formatting or explanation text."""),
            ("human", "Questions: {questions}\nGenerate Python code to: {task_description}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing CodeGenerationWorkflow")
        try:
            questions = input_data.get("questions", "")
            task_description = input_data.get("task_description", "")
            
            # Generate Python code
            code = self.chain.run(
                questions=questions,
                task_description=task_description
            )
            
            # Clean the code (remove markdown formatting if present)
            cleaned_code = self._clean_generated_code(code)
            
            # Try to validate and execute the code
            exec_result = self._safe_execute_code(cleaned_code, input_data)
            
            return {
                "generated_code": cleaned_code,
                "execution_result": exec_result,
                "code_validation": self._validate_python_syntax(cleaned_code),
                "workflow_type": "code_generation",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "questions_processed": questions
            }
        except Exception as e:
            logger.error(f"Error in CodeGenerationWorkflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "code_generation",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code by removing markdown formatting"""
        # Remove markdown code blocks
        import re
        # Remove ```python and ``` markers
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        # Remove any leading/trailing whitespace
        return code.strip()
    
    def _validate_python_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax without executing"""
        try:
            compile(code, '<string>', 'exec')
            return {"valid": True, "message": "Syntax is valid"}
        except SyntaxError as e:
            return {
                "valid": False, 
                "error": str(e),
                "line": e.lineno,
                "position": e.offset
            }
    
    def _safe_execute_code(self, code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute generated code with restricted environment"""
        try:
            # Create a restricted execution environment
            safe_globals = {
                '__builtins__': {
                    'len': len, 'str': str, 'int': int, 'float': float,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'range': range, 'enumerate': enumerate, 'zip': zip,
                    'print': print, 'type': type, 'isinstance': isinstance
                }
            }
            
            # Add common data science imports
            exec_locals = {}
            setup_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Create sample data if needed
sample_data = {
    'numbers': [1, 2, 3, 4, 5],
    'categories': ['A', 'B', 'C', 'D', 'E'],
    'values': [10, 20, 15, 25, 30]
}
df_sample = pd.DataFrame(sample_data)
"""
            
            exec(setup_code, safe_globals, exec_locals)
            exec(code, safe_globals, exec_locals)
            
            # Extract meaningful results
            results = {}
            for key, value in exec_locals.items():
                if not key.startswith('_') and key not in ['pd', 'np', 'plt', 'sns', 'datetime', 'json']:
                    try:
                        # Convert to serializable format
                        if hasattr(value, 'to_dict'):  # DataFrame
                            results[key] = str(value.head())
                        elif hasattr(value, 'tolist'):  # NumPy array
                            results[key] = str(value)
                        else:
                            results[key] = str(value)
                    except:
                        results[key] = f"<{type(value).__name__}>"
            
            return {
                "execution_status": "success",
                "variables_created": list(results.keys()),
                "results": results,
                "output_summary": f"Code executed successfully, created {len(results)} variables"
            }
            
        except Exception as e:
            return {
                "execution_status": "failed",
                "error": str(e),
                "error_type": type(e).__name__
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


class MultiStepWebScrapingWorkflow(BaseWorkflow):
    """Enhanced workflow for multi-step web scraping tasks with actual execution"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_multi_step_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_multi_step_prompt(self) -> ChatPromptTemplate:
        """Create multi-step web scraping prompt"""
        system_message = """You are a web scraping expert specializing in multi-step data analysis tasks.
Your task is to execute complete web scraping workflows including:
1. Web scraping and data extraction from any website
2. Data cleaning and preprocessing (handling various data formats)
3. Data analysis and visualization
4. Answering specific questions about the data

You must provide executable Python code that actually performs these tasks.
IMPORTANT: 
- Always inspect the actual data structure before processing
- Handle dynamic column names and various data formats
- Make the code generic enough to work with different types of data
- Include proper error handling for different scenarios
"""
        
        human_message = """
Multi-Step Task: {task_description}
Target URL: {url}
Data Requirements: {data_requirements}
Output Format: {output_format}
Special Instructions: {special_instructions}

Provide a complete solution that:
1. Scrapes the data from the specified URL using pandas read_html (preferred) or requests
2. Inspects the actual data structure and column names before processing
3. Cleans and processes the data (remove symbols, convert to numeric, handle various formats)
4. Creates visualizations as requested
5. Performs analysis to answer specific questions
6. Returns the final answers

IMPORTANT: 
- Generate executable Python code that actually performs these tasks, not just a plan
- Use pandas read_html() for web scraping when possible (it's more reliable)
- ALWAYS inspect the actual data structure first (print column names, data types, first few rows)
- Handle dynamic column names - NEVER assume specific column names like 'Country/Territory' exist
- Use data.columns[0] for the first column, data.columns[1] for second, etc.
- Make the code generic enough to work with different types of data (not just GDP data)
- Include all necessary imports and error handling
- Make sure the code can run without external dependencies like BeautifulSoup
- Print the final answers clearly
- Add debug prints to show what data is being processed
- Handle various data formats and structures automatically
- Always use dynamic column references instead of hardcoded column names
- Keep the code simple and avoid complex variable names that might be interpreted as template variables
- CRITICAL: After cleaning data, use data.select_dtypes(include=[np.number]).columns.tolist() to find numeric columns
- CRITICAL: Never assume data.columns[1] is a column name - it might be a value
- CRITICAL: Always verify column types before using them for analysis
- CRITICAL: For Wikipedia data, the main table is usually the one with the most rows
- CRITICAL: Always print table information to verify you're selecting the right table
- CRITICAL: Store final answers in variables so they can be captured in the response

Example approach:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Web scraping and data extraction
url = "your_url_here"
print(f"Scraping data from: {{url}}")

# Try different scraping methods
try:
    # Method 1: pandas read_html (works for tables)
    tables = pd.read_html(url)
    print(f"Found {{len(tables)}} tables on the page")
    
    # Inspect all tables to understand the data structure
    for i, table in enumerate(tables):
        print(f"\\nTable {{i}}:")
        print(f"  Shape: {{table.shape}}")
        print(f"  Columns: {{table.columns.tolist()}}")
        print(f"  Sample data:")
        print(table.head(3))
    
            # Select the most relevant table based on content
        # Look for the table with the most rows (usually the main data table)
        data = tables[0]  # Start with first table
        max_rows = 0
        best_table_idx = 0
        
        for i, table in enumerate(tables):
            print(f"Table {{i}}: {{table.shape[0]}} rows, {{table.shape[1]}} columns")
            if table.shape[0] > max_rows:
                max_rows = table.shape[0]
                best_table_idx = i
                data = table
        
        print(f"Selected table {{best_table_idx}} with {{data.shape[0]}} rows and {{data.shape[1]}} columns")
        print(f"This should be the main data table with the most rows")
    
except Exception as e:
    print(f"pandas read_html failed: {{e}}")
    # Method 2: requests + BeautifulSoup (if available)
    try:
        import requests
        from bs4 import BeautifulSoup
        
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find tables
        tables = soup.find_all('table')
        print(f"Found {{len(tables)}} tables using BeautifulSoup")
        
        # Convert to pandas DataFrame
        data = pd.read_html(str(tables[0]))[0]
        
    except Exception as e2:
        print(f"BeautifulSoup also failed: {{e2}}")
        raise Exception("Could not scrape data from the URL")

# Step 2: Data inspection and cleaning
print(f"\\nSelected data shape: {{data.shape}}")
print(f"Columns: {{data.columns.tolist()}}")
print("\\nFirst few rows:")
print(data.head())

# Clean the data structure
# Handle complex column structures (like MultiIndex columns)
if isinstance(data.columns, pd.MultiIndex):
    # Flatten MultiIndex columns
    data.columns = [col[1] if isinstance(col, tuple) and col[1] else str(col) for col in data.columns]
    print("\\nFlattened MultiIndex columns:")
    print(f"Columns: {{data.columns.tolist()}}")

# Check if first row contains headers
if data.iloc[0].dtype == 'object':
    # Use first row as headers
    data.columns = data.iloc[0]
    data = data[1:].reset_index(drop=True)
    print("\\nAfter setting headers:")
    print(f"Columns: {{data.columns.tolist()}}")

# Clean numeric columns (remove symbols, convert to numeric)
for col in data.columns:
    if data[col].dtype == 'object':
        # Try to convert to numeric, removing common symbols and footnotes
        cleaned = data[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('€', '').str.replace('£', '')
        # Remove footnote references like [1], [2], etc.
        cleaned = cleaned.str.replace(r'\\[\\d+\\]', '', regex=True)
        # Remove any remaining non-numeric characters except decimal points
        cleaned = cleaned.str.replace(r'[^\\d.]', '', regex=True)
        data[col] = pd.to_numeric(cleaned, errors='coerce')

print("\\nAfter cleaning:")
print(data.head())

# Step 3: Data analysis
# Find numeric columns for analysis - CRITICAL: Use select_dtypes to find numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
print(f"\\nNumeric columns available: {{numeric_cols}}")

if len(numeric_cols) > 0:
    # Use the first numeric column for analysis (usually the GDP column)
    analysis_col = numeric_cols[0]
    print(f"Using column '{{analysis_col}}' for analysis")
    
    # Verify the column exists and is numeric
    if analysis_col in data.columns:
        print(f"Column '{{analysis_col}}' found in data")
        
        # Remove rows with NaN values in the analysis column
        data_clean = data.dropna(subset=[analysis_col])
        print(f"After removing NaN values: {{data_clean.shape[0]}} rows")
        
        # Sort by the analysis column
        data_sorted = data_clean.sort_values(analysis_col, ascending=False)
        
        # Get top 10 items
        top_10 = data_sorted.head(10)
        print(f"\\nTop 10 by {{analysis_col}}:")
        print(top_10[[data.columns[0], analysis_col]])  # Show first column and analysis column
    
    # Step 4: Visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_10)), top_10[analysis_col])
    plt.xticks(range(len(top_10)), top_10[data.columns[0]], rotation=45)
    plt.title(f'Top 10 by {{analysis_col}}')
    plt.xlabel(data.columns[0])
    plt.ylabel(analysis_col)
    plt.tight_layout()
    plt.show()
    
    # Step 5: Answer specific questions
    if len(top_10) >= 5:
        # Use dynamic column names - don't assume specific names
        first_col = data.columns[0]  # First column (usually name/identifier)
        fifth_item = top_10.iloc[4][first_col]
        total_top_10 = top_10[analysis_col].sum()
        
        # Store answers in variables for capture
        answer_1 = fifth_item
        answer_2 = total_top_10
        analysis_results = top_10[[first_col, analysis_col]].to_dict('records')
        
        print(f"\\nANSWERS:")
        print(f"Item ranking 5th by {{analysis_col}}: {{fifth_item}}")
        print(f"Total {{analysis_col}} of top 10: {{total_top_10}}")
        
        # Also print the full top 10 for reference
        print(f"\\nFull top 10 list:")
        for i, (idx, row) in enumerate(top_10.iterrows()):
            print(f"{{i+1}}. {{row[first_col]}}: {{row[analysis_col]}}")
    else:
        print(f"\\nNot enough data for ranking analysis")
        answer_1 = "Not enough data"
        answer_2 = 0
        analysis_results = []
else:
    print("\\nNo numeric columns found for analysis")
    answer_1 = "No numeric data found"
    answer_2 = 0
    analysis_results = []
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-step web scraping workflow with actual execution"""
        try:
            logger.info(f"Starting multi-step web scraping workflow")
            
            # Extract URL from task description
            task_description = input_data.get("task_description", "")
            url = self._extract_url_from_task(task_description)
            
            # Generate the complete solution
            result = self.chain.run(
                task_description=task_description,
                url=url,
                data_requirements=input_data.get("data_requirements", "Extract table data and perform analysis"),
                output_format=input_data.get("output_format", "structured data with visualizations"),
                special_instructions=input_data.get("special_instructions", "Execute all steps and provide final answers")
            )
            
            # Execute the generated code
            execution_result = await self._execute_generated_code(result, input_data)
            
            logger.info(f"Multi-step web scraping workflow completed successfully")
            
            return {
                "scraping_plan": result,
                "execution_result": execution_result,
                "workflow_type": "multi_step_web_scraping",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "target_url": url,
                "steps_executed": [
                    "web_scraping",
                    "data_cleaning", 
                    "data_analysis",
                    "visualization",
                    "question_answering"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in multi-step web scraping workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "multi_step_web_scraping",
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_url_from_task(self, task_description: str) -> str:
        """Extract URL from task description"""
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, task_description)
        return urls[0] if urls else ""
    
    async def _execute_generated_code(self, generated_code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the generated Python code safely"""
        try:
            # Extract code blocks from the generated response
            code_blocks = self._extract_code_blocks(generated_code)
            
            if not code_blocks:
                return {"error": "No executable code found in response"}
            
            # Execute the code blocks
            execution_results = []
            for i, code_block in enumerate(code_blocks):
                try:
                    result = await self._safe_execute_code_block(code_block, input_data)
                    execution_results.append({
                        "block_index": i,
                        "status": "success",
                        "result": result
                    })
                except Exception as e:
                    execution_results.append({
                        "block_index": i,
                        "status": "error",
                        "error": str(e)
                    })
            
            return {
                "execution_results": execution_results,
                "total_blocks": len(code_blocks),
                "successful_blocks": len([r for r in execution_results if r["status"] == "success"])
            }
            
        except Exception as e:
            logger.error(f"Error executing generated code: {e}")
            return {"error": str(e)}
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from text"""
        import re
        code_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        return matches
    
    async def _safe_execute_code_block(self, code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute a code block"""
        try:
            # Create a safe execution environment with proper imports
            exec_globals = {}
            
            # Import required modules safely
            try:
                exec_globals['pd'] = pd
                exec_globals['requests'] = __import__('requests')
                exec_globals['matplotlib'] = __import__('matplotlib')
                exec_globals['plt'] = __import__('matplotlib.pyplot')
                exec_globals['json'] = json
                exec_globals['datetime'] = datetime
                exec_globals['logging'] = logging
                
                # Try to import BeautifulSoup, fallback if not available
                try:
                    exec_globals['BeautifulSoup'] = __import__('bs4').BeautifulSoup
                except ImportError:
                    logger.warning("BeautifulSoup not available, using alternative approach")
                    # Use pandas read_html as alternative
                    exec_globals['BeautifulSoup'] = None
                
                # Import additional useful modules
                try:
                    exec_globals['numpy'] = __import__('numpy')
                    exec_globals['np'] = exec_globals['numpy']
                except ImportError:
                    pass
                
            except ImportError as e:
                logger.error(f"Failed to import required module: {e}")
                return {
                    "status": "error",
                    "error": f"Missing dependency: {e}",
                    "code_attempted": code[:200] + "..." if len(code) > 200 else code
                }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Try to capture any output variables
            output_vars = {}
            
            # Capture common data variables
            common_vars = ['df', 'result', 'data', 'tables', 'gdp_data', 'gdp_column', 'analysis_data', 'cleaned_data', 'processed_data']
            for var_name in common_vars:
                if var_name in exec_globals:
                    var_value = exec_globals[var_name]
                    if hasattr(var_value, 'to_string'):
                        output_vars[var_name] = var_value.to_string()
                    elif hasattr(var_value, 'shape'):
                        output_vars[var_name] = f"DataFrame shape: {var_value.shape}, columns: {list(var_value.columns)}"
                    elif hasattr(var_value, '__len__') and len(var_value) > 0:
                        output_vars[var_name] = f"List/Array with {len(var_value)} items: {str(var_value)[:200]}..."
                    else:
                        output_vars[var_name] = str(var_value)
            
            # Capture ALL variables that might contain answers (generic approach)
            for var_name, var_value in exec_globals.items():
                if not var_name.startswith('_') and var_name not in ['pd', 'np', 'plt', 'requests', 'json', 'datetime', 'logging', 'BeautifulSoup']:
                    # Skip already captured variables
                    if var_name not in output_vars:
                        try:
                            # Capture any variable that might be an answer
                            if isinstance(var_value, (str, int, float, list, dict)):
                                output_vars[var_name] = str(var_value)
                            elif hasattr(var_value, 'to_string'):
                                output_vars[var_name] = var_value.to_string()
                            elif hasattr(var_value, 'shape'):
                                output_vars[var_name] = f"DataFrame shape: {var_value.shape}, columns: {list(var_value.columns)}"
                            else:
                                output_vars[var_name] = str(var_value)
                        except:
                            pass
            
            return {
                "status": "success",
                "output_variables": output_vars,
                "code_executed": code[:200] + "..." if len(code) > 200 else code
            }
            
        except Exception as e:
            logger.error(f"Error executing code block: {e}")
            return {
                "status": "error",
                "error": str(e),
                "code_attempted": code[:200] + "..." if len(code) > 200 else code
            }


class AdvancedWorkflowOrchestrator(WorkflowOrchestrator):
    """Enhanced orchestrator with domain-specific workflows"""
    
    def __init__(self):
        super().__init__()
        # Initialize LLM for workflow detection
        try:
            from langchain_openai import ChatOpenAI
            from config import OPENAI_API_KEY, DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS
            self.llm = ChatOpenAI(
                model=DEFAULT_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                api_key=OPENAI_API_KEY
            )
        except Exception as e:
            logger.warning(f"Could not initialize LLM for workflow detection: {e}")
            self.llm = None
        
        # Add specialized workflows including multi-modal support
        self.workflows.update({
            "data_analysis": DataAnalysisWorkflow(),
            "image_analysis": ImageAnalysisWorkflow(),
            "text_analysis": DataAnalysisWorkflow(),
            "code_generation": CodeGenerationWorkflow(),
            "exploratory_data_analysis": ExploratoryDataAnalysisWorkflow(),
            "predictive_modeling": PredictiveModelingWorkflow(),
            "data_visualization": DataVisualizationWorkflow(),
            "web_scraping": WebScrapingWorkflow(),
            "multi_step_web_scraping": MultiStepWebScrapingWorkflow(),
            "database_analysis": DatabaseAnalysisWorkflow(),
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
                "image_analysis": "Image processing, computer vision, and image-based analysis", 
                "text_analysis": "Natural language processing and text analytics",
                "code_generation": "Generate Python code for data analysis tasks",
                "exploratory_data_analysis": "Comprehensive EDA planning and execution",
                "predictive_modeling": "Machine learning model development guidance",
                "data_visualization": "Visualization recommendations and code generation",
                "web_scraping": "Web scraping and data extraction from websites",
                "database_analysis": "SQL analysis using DuckDB for large datasets",
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
                "Statistical analysis and visualization",
                "Multi-modal analysis (text, image, code)",
                "Synchronous processing",
                "Multiple file upload support",
                "LLM-based workflow detection"
            ]
        }

# Import new web scraping steps
from .web_scraping_steps import (
    ScrapeTableStep,
    InspectTableStep,
    CleanDataStep,
    AnalyzeDataStep,
    VisualizeStep,
    AnswerQuestionsStep,
)

# --- Step Registry (updated to include new web scraping steps) ---
STEP_REGISTRY = {
    'scrape_table': ScrapeTableStep,
    'inspect_table': InspectTableStep,
    'clean_data': CleanDataStep,
    'analyze_data': AnalyzeDataStep,
    'visualize': VisualizeStep,
    'answer': AnswerQuestionsStep,
}

# --- Orchestrator (usage example) ---
def run_web_scraping_workflow(url: str, top_n: int = 10) -> dict:
    """
    Example usage of the new web scraping step classes in a workflow.
    """
    # Step plan (could be generated by LLM)
    plan = [
        {'step': 'scrape_table', 'url': url},
        {'step': 'inspect_table'},
        {'step': 'clean_data'},
        {'step': 'analyze_data', 'top_n': top_n},
        {'step': 'visualize'},
        {'step': 'answer'},
    ]
    data = {}
    for step_cfg in plan:
        step_name = step_cfg['step']
        params = {k: v for k, v in step_cfg.items() if k != 'step'}
        step_cls = STEP_REGISTRY[step_name]
        step = step_cls()
        step_input = {**data, **params}
        data = step.run(step_input)
    return data

def detect_steps_from_prompt(user_request: str, llm=None) -> list:
    """
    Use an LLM to generate a step plan from a user request.
    """
    prompt = f"""
You are an expert workflow planner for data analysis and web scraping tasks.
Given a user request, break it down into a sequence of high-level, reusable steps.
Each step should have a type (e.g., scrape_table, inspect_table, clean_data, analyze_data, visualize, answer) and relevant parameters.
Output the plan as a JSON list, where each item is a step with its parameters.
Do not generate code, only the step plan.

User Request:
{user_request}

Output Format Example:
[
  {{"step": "scrape_table", "url": "https://example.com/table"}},
  {{"step": "inspect_table"}},
  {{"step": "clean_data"}},
  {{"step": "analyze_data", "top_n": 10}},
  {{"step": "visualize"}},
  {{"step": "answer", "questions": [
    "Which country ranks 5th by GDP?",
    "What is the total GDP of the top 10 countries?"
  ]}}
]

Now, generate the step plan for the following user request.
"""
    if llm is not None:
        response = llm(prompt)
        import json, re
        try:
            plan = json.loads(response)
            return plan
        except Exception:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
    else:
        # Fallback: simple hardcoded plan for demo
        return [
            {"step": "scrape_table", "url": "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"},
            {"step": "inspect_table"},
            {"step": "clean_data"},
            {"step": "analyze_data", "top_n": 10},
            {"step": "visualize"},
            {"step": "answer", "questions": [
                "Which country ranks 5th by GDP?",
                "What is the total GDP of the top 10 countries?"
            ]}
        ]


def run_llm_planned_workflow(user_request: str, llm=None) -> dict:
    """
    Use the LLM to generate a step plan from the user request, then execute the plan using the modular step orchestrator.
    """
    plan = detect_steps_from_prompt(user_request, llm=llm)
    data = {}
    for step_cfg in plan:
        step_name = step_cfg['step']
        params = {k: v for k, v in step_cfg.items() if k != 'step'}
        step_cls = STEP_REGISTRY[step_name]
        step = step_cls()
        step_input = {**data, **params}
        data = step.run(step_input)
    return data

# --- Usage Example ---
# Suppose you want to run the workflow for the content of questions.txt:
#
# with open('Project2/questions.txt', 'r') as f:
#     user_request = f.read()
# result = run_llm_planned_workflow(user_request, llm=my_llm)
# print(result)
