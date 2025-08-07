"""
Specific workflow implementations for data analysis tasks
"""

from typing import Dict, Any, List
import pandas as pd
import json
from datetime import datetime
import logging
import re
from chains.base import BaseWorkflow, WorkflowOrchestrator
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from utils.prompts import (
    EDA_SYSTEM_PROMPT,
    EDA_HUMAN_PROMPT,
    DATA_ANALYSIS_SYSTEM_PROMPT,
    DATA_ANALYSIS_HUMAN_PROMPT,
    IMAGE_ANALYSIS_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_HUMAN_PROMPT,
    CODE_WORKFLOW_SYSTEM_PROMPT,
    CODE_WORKFLOW_HUMAN_PROMPT,
    STATISTICAL_SYSTEM_PROMPT,
    STATISTICAL_HUMAN_PROMPT,
    MULTI_STEP_SYSTEM_PROMPT,
    MULTI_STEP_HUMAN_PROMPT,
    PREDICTIVE_MODELING_SYSTEM_PROMPT,
    PREDICTIVE_MODELING_HUMAN_PROMPT,
    DATA_VISUALIZATION_SYSTEM_PROMPT,
    DATA_VISUALIZATION_HUMAN_PROMPT,
    WEB_SCRAPING_SYSTEM_PROMPT,
    WEB_SCRAPING_HUMAN_PROMPT,
    DATABASE_ANALYSIS_SYSTEM_PROMPT,
    DATABASE_ANALYSIS_HUMAN_PROMPT,
)
# Import new web scraping steps
from .web_scraping_steps import (
    DetectDataFormatStep,
    ScrapeTableStep,
    InspectTableStep,
    CleanDataStep,
    AnalyzeDataStep,
    VisualizeStep,
    AnswerQuestionsStep,
)

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
                parameters=json.dumps(input_data.get("parameters", {}), indent=2),
            )

            return {
                "eda_plan": result,
                "workflow_type": "exploratory_data_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "dataset_summary": dataset_info,
            }

        except Exception as e:
            logger.error(f"Error in EDA workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "exploratory_data_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    def _create_eda_prompt(self) -> ChatPromptTemplate:
        """Create EDA-specific prompt"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", EDA_SYSTEM_PROMPT),
                ("human", EDA_HUMAN_PROMPT),
            ]
        )


class DataAnalysisWorkflow(BaseWorkflow):
    """Generalized workflow for data analysis tasks"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", DATA_ANALYSIS_SYSTEM_PROMPT),
                ("human", DATA_ANALYSIS_HUMAN_PROMPT),
            ]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing DataAnalysisWorkflow")
        try:
            result = self.chain.run(questions=input_data.get("task_description", ""), files=input_data.get("files", []))
            return {
                "analysis_result": result,
                "workflow_type": "data_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error in DataAnalysisWorkflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "data_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class ImageAnalysisWorkflow(BaseWorkflow):
    """Workflow for image analysis tasks"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", IMAGE_ANALYSIS_SYSTEM_PROMPT),
                ("human", IMAGE_ANALYSIS_HUMAN_PROMPT),
            ]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing ImageAnalysisWorkflow")
        try:
            result = self.chain.run(
                questions=input_data.get("task_description", ""), image_file=input_data.get("image_file", "")
            )
            return {
                "image_analysis_result": result,
                "workflow_type": "image_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error in ImageAnalysisWorkflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "image_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class CodeGenerationWorkflow(BaseWorkflow):
    """Workflow for Python code generation and execution"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", CODE_WORKFLOW_SYSTEM_PROMPT),
            ("human", CODE_WORKFLOW_HUMAN_PROMPT),
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing CodeGenerationWorkflow")
        try:
            questions = input_data.get("questions", "")
            task_description = input_data.get("task_description", "")

            # Generate Python code
            code = self.chain.run(questions=questions, task_description=task_description)

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
                "questions_processed": questions,
            }
        except Exception as e:
            logger.error(f"Error in CodeGenerationWorkflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "code_generation",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code by removing markdown formatting"""
        # Remove markdown code blocks
        import re

        # Remove ```python and ``` markers
        code = re.sub(r"```python\n?", "", code)
        code = re.sub(r"```\n?", "", code)
        # Remove any leading/trailing whitespace
        return code.strip()

    def _validate_python_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax without executing"""
        try:
            compile(code, "<string>", "exec")
            return {"valid": True, "message": "Syntax is valid"}
        except SyntaxError as e:
            return {"valid": False, "error": str(e), "line": e.lineno, "position": e.offset}

    def _safe_execute_code(self, code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute generated code with restricted environment"""
        try:
            # Create a restricted execution environment
            safe_globals = {
                "__builtins__": {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "print": print,
                    "type": type,
                    "isinstance": isinstance,
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
                if not key.startswith("_") and key not in ["pd", "np", "plt", "sns", "datetime", "json"]:
                    try:
                        # Convert to serializable format
                        if hasattr(value, "to_dict"):  # DataFrame
                            results[key] = str(value.head())
                        elif hasattr(value, "tolist"):  # NumPy array
                            results[key] = str(value)
                        else:
                            results[key] = str(value)
                    except Exception:
                        results[key] = f"<{type(value).__name__}>"

            return {
                "execution_status": "success",
                "variables_created": list(results.keys()),
                "results": results,
                "output_summary": f"Code executed successfully, created {len(results)} variables",
            }

        except Exception as e:
            return {"execution_status": "failed", "error": str(e), "error_type": type(e).__name__}


class PredictiveModelingWorkflow(BaseWorkflow):
    """Workflow for predictive modeling tasks"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_modeling_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_modeling_prompt(self) -> ChatPromptTemplate:
        """Create predictive modeling prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", PREDICTIVE_MODELING_SYSTEM_PROMPT),
            ("human", PREDICTIVE_MODELING_HUMAN_PROMPT)
        ])

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute predictive modeling workflow"""
        try:
            result = self.chain.run(
                problem_statement=input_data.get("problem_statement", ""),
                target_variable=input_data.get("target_variable", ""),
                dataset_characteristics=json.dumps(input_data.get("dataset_characteristics", {}), indent=2),
                business_requirements=input_data.get("business_requirements", ""),
                performance_requirements=input_data.get("performance_requirements", ""),
            )

            return {
                "modeling_plan": result,
                "workflow_type": "predictive_modeling",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "problem_type": self._identify_problem_type(input_data),
            }

        except Exception as e:
            logger.error(f"Error in predictive modeling workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "predictive_modeling",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
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
        return ChatPromptTemplate.from_messages([
            ("system", DATA_VISUALIZATION_SYSTEM_PROMPT),
            ("human", DATA_VISUALIZATION_HUMAN_PROMPT)
        ])

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization workflow"""
        try:
            result = self.chain.run(
                data_description=input_data.get("data_description", ""),
                variables=json.dumps(input_data.get("variables", []), indent=2),
                analysis_goals=input_data.get("analysis_goals", ""),
                target_audience=input_data.get("target_audience", "technical team"),
                platform=input_data.get("platform", "Python (matplotlib/seaborn)"),
            )

            return {
                "visualization_plan": result,
                "workflow_type": "data_visualization",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "recommended_tools": ["matplotlib", "seaborn", "plotly"],
            }

        except Exception as e:
            logger.error(f"Error in visualization workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "data_visualization",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class WebScrapingWorkflow(BaseWorkflow):
    """Workflow for web scraping and data extraction"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_scraping_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_scraping_prompt(self) -> ChatPromptTemplate:
        """Create web scraping prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", WEB_SCRAPING_SYSTEM_PROMPT),
            ("human", WEB_SCRAPING_HUMAN_PROMPT)
        ])

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web scraping workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                url=input_data.get("url", ""),
                data_requirements=input_data.get("data_requirements", ""),
                output_format=input_data.get("output_format", "structured data"),
                special_instructions=input_data.get("special_instructions", ""),
            )

            return {
                "scraping_plan": result,
                "workflow_type": "web_scraping",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "target_url": input_data.get("url", ""),
            }

        except Exception as e:
            logger.error(f"Error in web scraping workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "web_scraping",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class DatabaseAnalysisWorkflow(BaseWorkflow):
    """Workflow for database analysis using DuckDB and SQL"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_database_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_database_prompt(self) -> ChatPromptTemplate:
        """Create database analysis prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", DATABASE_ANALYSIS_SYSTEM_PROMPT),
            ("human", DATABASE_ANALYSIS_HUMAN_PROMPT)
        ])

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database analysis workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                database_info=json.dumps(input_data.get("database_info", {}), indent=2),
                schema_info=json.dumps(input_data.get("schema_info", {}), indent=2),
                analysis_goals=input_data.get("analysis_goals", ""),
                performance_requirements=input_data.get("performance_requirements", ""),
            )

            return {
                "analysis_plan": result,
                "workflow_type": "database_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "database_type": "DuckDB",
            }

        except Exception as e:
            logger.error(f"Error in database analysis workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "database_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class StatisticalAnalysisWorkflow(BaseWorkflow):
    """Workflow for statistical analysis including correlation and regression"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_statistical_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_statistical_prompt(self) -> ChatPromptTemplate:
        """Create statistical analysis prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", STATISTICAL_SYSTEM_PROMPT),
            ("human", STATISTICAL_HUMAN_PROMPT)
        ])

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis workflow"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                dataset_description=input_data.get("dataset_description", ""),
                variables=json.dumps(input_data.get("variables", []), indent=2),
                methods=input_data.get("statistical_methods", "correlation, regression"),
            )

            return {
                "statistical_plan": result,
                "workflow_type": "statistical_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "methods": input_data.get("statistical_methods", "correlation, regression"),
            }

        except Exception as e:
            logger.error(f"Error in statistical analysis workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "statistical_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class MultiStepWebScrapingWorkflow(BaseWorkflow):
    """Enhanced workflow for multi-step web scraping tasks with actual execution"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_multi_step_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_multi_step_prompt(self) -> ChatPromptTemplate:
        """Create multi-step web scraping prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", MULTI_STEP_SYSTEM_PROMPT),
            ("human", MULTI_STEP_HUMAN_PROMPT)
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
                special_instructions=input_data.get(
                    "special_instructions", "Execute all steps and provide final answers"
                ),
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
                    "question_answering",
                ],
            }

        except Exception as e:
            logger.error(f"Error in multi-step web scraping workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "multi_step_web_scraping",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    def _extract_url_from_task(self, task_description: str) -> str:
        """Extract URL from task description"""
        import re

        url_pattern = r"https?://[^\s]+"
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
                    execution_results.append({"block_index": i, "status": "success", "result": result})
                except Exception as e:
                    execution_results.append({"block_index": i, "status": "error", "error": str(e)})

            return {
                "execution_results": execution_results,
                "total_blocks": len(code_blocks),
                "successful_blocks": len([r for r in execution_results if r["status"] == "success"]),
            }

        except Exception as e:
            logger.error(f"Error executing generated code: {e}")
            return {"error": str(e)}

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from text"""
        import re

        code_pattern = r"```python\s*\n(.*?)\n```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        return matches

    async def _safe_execute_code_block(self, code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute a code block"""
        try:
            # Create a safe execution environment with proper imports
            exec_globals = {}

            # Import required modules safely
            try:
                exec_globals["pd"] = pd
                exec_globals["requests"] = __import__("requests")
                exec_globals["matplotlib"] = __import__("matplotlib")
                exec_globals["plt"] = __import__("matplotlib.pyplot")
                exec_globals["json"] = json
                exec_globals["datetime"] = datetime
                exec_globals["logging"] = logging

                # Try to import BeautifulSoup, fallback if not available
                try:
                    exec_globals["BeautifulSoup"] = __import__("bs4").BeautifulSoup
                except ImportError:
                    logger.warning("BeautifulSoup not available, using alternative approach")
                    # Use pandas read_html as alternative
                    exec_globals["BeautifulSoup"] = None

                # Import additional useful modules
                try:
                    exec_globals["numpy"] = __import__("numpy")
                    exec_globals["np"] = exec_globals["numpy"]
                except ImportError:
                    pass

            except ImportError as e:
                logger.error(f"Failed to import required module: {e}")
                return {
                    "status": "error",
                    "error": f"Missing dependency: {e}",
                    "code_attempted": code[:200] + "..." if len(code) > 200 else code,
                }

            # Execute the code
            exec(code, exec_globals)

            # Try to capture any output variables
            output_vars = {}

            # Capture common data variables
            common_vars = [
                "df",
                "result",
                "data",
                "tables",
                "gdp_data",
                "gdp_column",
                "analysis_data",
                "cleaned_data",
                "processed_data",
            ]
            for var_name in common_vars:
                if var_name in exec_globals:
                    var_value = exec_globals[var_name]
                    if hasattr(var_value, "to_string"):
                        output_vars[var_name] = var_value.to_string()
                    elif hasattr(var_value, "shape"):
                        output_vars[var_name] = (
                            f"DataFrame shape: {var_value.shape}, columns: {list(var_value.columns)}"
                        )
                    elif hasattr(var_value, "__len__") and len(var_value) > 0:
                        output_vars[var_name] = f"List/Array with {len(var_value)} items: {str(var_value)[:200]}..."
                    else:
                        output_vars[var_name] = str(var_value)

            # Capture ALL variables that might contain answers (generic approach)
            for var_name, var_value in exec_globals.items():
                if not var_name.startswith("_") and var_name not in [
                    "pd",
                    "np",
                    "plt",
                    "requests",
                    "json",
                    "datetime",
                    "logging",
                    "BeautifulSoup",
                ]:
                    # Skip already captured variables
                    if var_name not in output_vars:
                        try:
                            # Capture any variable that might be an answer
                            if isinstance(var_value, (str, int, float, list, dict)):
                                output_vars[var_name] = str(var_value)
                            elif hasattr(var_value, "to_string"):
                                output_vars[var_name] = var_value.to_string()
                            elif hasattr(var_value, "shape"):
                                output_vars[var_name] = (
                                    f"DataFrame shape: {var_value.shape}, columns: {list(var_value.columns)}"
                                )
                            else:
                                output_vars[var_name] = str(var_value)
                        except Exception:
                            pass

            return {
                "status": "success",
                "output_variables": output_vars,
                "code_executed": code[:200] + "..." if len(code) > 200 else code,
            }

        except Exception as e:
            logger.error(f"Error executing code block: {e}")
            return {
                "status": "error",
                "error": str(e),
                "code_attempted": code[:200] + "..." if len(code) > 200 else code,
            }


class ModularWebScrapingWorkflow(BaseWorkflow):
    """Fallback workflow using modular step-based approach when LLM is not available"""

    def __init__(self, **kwargs):
        # Don't call super().__init__ since we don't need LLM for this approach
        pass

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute modular web scraping workflow using step classes"""
        try:
            logger.info("Executing ModularWebScrapingWorkflow with enhanced format detection")

            # Import the step classes from web_scraping_steps
            try:
                from .web_scraping_steps import (
                    DetectDataFormatStep,
                    ScrapeTableStep,
                    InspectTableStep,
                    CleanDataStep,
                    AnalyzeDataStep,
                    VisualizeStep,
                    AnswerQuestionsStep,
                )
            except ImportError:
                logger.error("Could not import web_scraping_steps module")
                return {
                    "error": "Web scraping steps module not found",
                    "workflow_type": "multi_step_web_scraping",
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                }

            # Extract URL from task description
            task_description = input_data.get("task_description", "")
            url = self._extract_url_from_task(task_description)

            if not url:
                return {
                    "error": "No URL found in task description",
                    "workflow_type": "multi_step_web_scraping",
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                }

            # Execute the step-based workflow
            execution_log = []
            data = {"task_description": task_description}  # Pass task description to all steps

            try:
                # Step 0: Detect data format (NEW)
                step0 = DetectDataFormatStep()
                step0_input = {"url": url, "task_description": task_description}
                step0_result = step0.run(step0_input)
                data.update(step0_result)
                execution_log.append("✓ Data format detection completed")

                # Step 1: Enhanced data extraction
                step1 = ScrapeTableStep()
                step1_input = {**data, "url": url, "task_description": task_description}
                step1_result = step1.run(step1_input)
                data.update(step1_result)
                execution_log.append("✓ Data extraction completed")

                # Step 2: Inspect table
                step2 = InspectTableStep()
                step2_result = step2.run(data)
                data.update(step2_result)
                execution_log.append("✓ Table inspection completed")

                # Step 3: Clean data
                step3 = CleanDataStep()
                step3_result = step3.run(data)
                data.update(step3_result)
                execution_log.append("✓ Data cleaning completed")

                # Step 4: Analyze data
                step4 = AnalyzeDataStep()
                step4_input = {**data, "top_n": 20}  # Increased to handle more data types
                step4_result = step4.run(step4_input)
                data.update(step4_result)
                execution_log.append("✓ Data analysis completed")

                # Step 5: Visualize (auto-detect chart type from task)
                step5 = VisualizeStep()
                step5_input = {**data, "return_base64": True}
                step5_result = step5.run(step5_input)
                data.update(step5_result)
                execution_log.append("✓ Visualization completed")

                # Step 6: Answer questions
                step6 = AnswerQuestionsStep()
                step6_result = step6.run(data)
                data.update(step6_result)
                execution_log.append("✓ Question answering completed")

                return {
                    "workflow_type": "multi_step_web_scraping",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "target_url": url,
                    "execution_log": execution_log,
                    "results": data.get("answers", {}),
                    "plot_path": data.get("plot_path"),
                    "plot_base64": data.get("plot_base64"),
                    "chart_type": data.get("chart_type"),
                    "image_size_bytes": data.get("image_size_bytes"),
                    "message": "Workflow completed using step-based approach",
                    "fallback_mode": True,
                }

            except Exception as e:
                logger.error(f"Error in step execution: {e}")
                return {
                    "error": f"Step execution failed: {str(e)}",
                    "workflow_type": "multi_step_web_scraping",
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "execution_log": execution_log,
                    "target_url": url,
                }

        except Exception as e:
            logger.error(f"Error in modular web scraping workflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "multi_step_web_scraping",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    def _extract_url_from_task(self, task_description: str) -> str:
        """Extract URL from task description"""
        import re

        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, task_description)
        return urls[0] if urls else ""


class AdvancedWorkflowOrchestrator(WorkflowOrchestrator):
    """Enhanced orchestrator with domain-specific workflows"""

    def __init__(self):
        super().__init__()
        # Initialize LLM for workflow detection
        try:
            from config import get_chat_model, OPENAI_API_KEY, GEMINI_API_KEY

            # Prefer OpenAI if available, otherwise fallback to Gemini
            if OPENAI_API_KEY:
                self.llm = get_chat_model(provider="openai")
                logger.info("LLM initialized successfully using OpenAI.")
            elif GEMINI_API_KEY:
                self.llm = get_chat_model(provider="gemini")
                logger.info("LLM initialized successfully using Gemini as a fallback.")
            else:
                # If no key is available, we cannot proceed with this orchestrator
                raise ValueError("No OpenAI or Gemini API key found. Cannot initialize LLM for AdvancedWorkflowOrchestrator.")

        except Exception as e:
            logger.error(f"Critical error initializing LLM for AdvancedWorkflowOrchestrator: {e}")
            # Re-raise the exception to be caught by the main application logic
            raise e

        # Add specialized workflows including multi-modal support
        # Only initialize workflows that require LLM if LLM is available
        self.workflows.update(
            {
                "data_analysis": DataAnalysisWorkflow(llm=self.llm) if self.llm else None,
                "image_analysis": ImageAnalysisWorkflow(llm=self.llm) if self.llm else None,
                "text_analysis": DataAnalysisWorkflow(llm=self.llm) if self.llm else None,
                "code_generation": CodeGenerationWorkflow(llm=self.llm) if self.llm else None,
                "exploratory_data_analysis": ExploratoryDataAnalysisWorkflow(llm=self.llm) if self.llm else None,
                "predictive_modeling": PredictiveModelingWorkflow(llm=self.llm) if self.llm else None,
                "data_visualization": DataVisualizationWorkflow(llm=self.llm) if self.llm else None,
                "web_scraping": WebScrapingWorkflow(llm=self.llm) if self.llm else None,
                "multi_step_web_scraping": ModularWebScrapingWorkflow(),
                "database_analysis": DatabaseAnalysisWorkflow(llm=self.llm) if self.llm else None,
                "statistical_analysis": StatisticalAnalysisWorkflow(llm=self.llm) if self.llm else None,
            }
        )

        # Remove None workflows
        self.workflows = {k: v for k, v in self.workflows.items() if v is not None}

    async def execute_complete_analysis_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete data analysis pipeline"""

        # Define the analysis pipeline steps
        pipeline_steps = [
            {
                "workflow_type": "exploratory_data_analysis",
                "input_data": {
                    "dataset_info": input_data.get("dataset_info", {}),
                    "business_context": input_data.get("business_context", ""),
                    "parameters": input_data.get("eda_parameters", {}),
                },
            },
            {
                "workflow_type": "data_visualization",
                "input_data": {
                    "data_description": input_data.get("data_description", ""),
                    "variables": input_data.get("variables", []),
                    "analysis_goals": "Exploratory data analysis and pattern discovery",
                    "target_audience": input_data.get("target_audience", "technical team"),
                },
            },
        ]

        # Add predictive modeling if specified
        if input_data.get("include_modeling", False):
            pipeline_steps.append(
                {
                    "workflow_type": "predictive_modeling",
                    "input_data": {
                        "problem_statement": input_data.get("problem_statement", ""),
                        "target_variable": input_data.get("target_variable", ""),
                        "dataset_characteristics": input_data.get("dataset_info", {}),
                        "business_requirements": input_data.get("business_requirements", ""),
                        "performance_requirements": input_data.get("performance_requirements", ""),
                    },
                }
            )

        # Add report generation
        pipeline_steps.append(
            {
                "workflow_type": "report_generation",
                "input_data": {
                    "analysis_results": "Will be populated from previous steps",
                    "data_summary": json.dumps(input_data.get("dataset_info", {})),
                    "key_findings": "Will be extracted from analysis",
                    "audience": input_data.get("target_audience", "technical team"),
                },
            }
        )

        # Execute the pipeline
        result = await self.execute_multi_step_workflow(pipeline_steps)

        return {
            "pipeline_result": result,
            "pipeline_type": "complete_analysis",
            "timestamp": datetime.now().isoformat(),
            "input_summary": {
                "dataset_info": input_data.get("dataset_info", {}),
                "include_modeling": input_data.get("include_modeling", False),
                "target_audience": input_data.get("target_audience", "technical team"),
            },
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
                "statistical_analysis": "Statistical analysis including correlation and regression",
            },
            "pipeline_capabilities": ["complete_analysis_pipeline", "multi_step_workflow"],
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
                "LLM-based workflow detection",
            ],
        }


# --- Step Registry (updated to include new web scraping steps) ---
STEP_REGISTRY = {
    "detect_format": DetectDataFormatStep,
    "scrape_table": ScrapeTableStep,
    "inspect_table": InspectTableStep,
    "clean_data": CleanDataStep,
    "analyze_data": AnalyzeDataStep,
    "visualize": VisualizeStep,
    "answer": AnswerQuestionsStep,
}


# --- Orchestrator (usage example) ---
def run_web_scraping_workflow(url: str, top_n: int = 10) -> dict:
    """
    Example usage of the new web scraping step classes in a workflow.
    """
    # Step plan (could be generated by LLM)
    plan = [
        {"step": "scrape_table", "url": url},
        {"step": "inspect_table"},
        {"step": "clean_data"},
        {"step": "analyze_data", "top_n": top_n},
        {"step": "visualize"},
        {"step": "answer"},
    ]
    data = {}
    for step_cfg in plan:
        step_name = step_cfg["step"]
        params = {k: v for k, v in step_cfg.items() if k != "step"}
        step_cls = STEP_REGISTRY[step_name]
        step = step_cls()
        step_input = {**data, **params}
        data = step.run(step_input)
    return data


def detect_steps_from_prompt(user_request: str, llm=None) -> list:
    """
    Use an LLM to generate a step plan from a user request.
    """
    from utils.prompts import DETECT_STEPS_PROMPT
    prompt = DETECT_STEPS_PROMPT.format(user_request=user_request)
    if llm is not None:
        response = llm(prompt)
        import json
        import re

        try:
            plan = json.loads(response)
            return plan
        except Exception:
            match = re.search(r"\[.*\]", response, re.DOTALL)
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
            {
                "step": "answer",
                "questions": ["Which country ranks 5th by GDP?", "What is the total GDP of the top 10 countries?"],
            },
        ]


def run_llm_planned_workflow(user_request: str, llm=None) -> dict:
    """
    Use the LLM to generate a step plan from the user request, then execute the plan
    using the modular step orchestrator.
    """
    plan = detect_steps_from_prompt(user_request, llm=llm)
    data = {}
    for step_cfg in plan:
        step_name = step_cfg["step"]
        params = {k: v for k, v in step_cfg.items() if k != "step"}
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
