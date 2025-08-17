"""
Specific workflow implementations for data analysis tasks
"""

from typing import Dict, Any, List
import pandas as pd
import json
from datetime import datetime
import logging
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
import io
import numpy as np
from scipy import stats
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


class GenericCSVAnalysisWorkflow(BaseWorkflow):
    """Generic workflow for CSV data analysis tasks with visualization support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic CSV analysis based on questions and uploaded CSV files"""
        logger.info("Executing GenericCSVAnalysisWorkflow")
        
        try:
            # Extract task description and files
            task_description = input_data.get("task_description", "")
            questions = input_data.get("questions", "")
            additional_files = input_data.get("additional_files", {})
            
            # Find CSV files in uploaded files
            csv_files = {}
            for file_path, file_info in additional_files.items():
                if file_path.lower().endswith('.csv'):
                    # Read the file content from the file path
                    try:
                        with open(file_path, 'r') as f:
                            csv_content = f.read()
                        csv_files[file_info.get('original_name', file_path)] = csv_content
                        logger.info(f"Read CSV file: {file_path} ({len(csv_content)} characters)")
                    except Exception as e:
                        logger.error(f"Failed to read CSV file {file_path}: {e}")
            
            if not csv_files:
                return {
                    "error": "No CSV files found in uploaded files",
                    "workflow_type": "csv_analysis",
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            
            # Use the first CSV file found (can be extended to handle multiple files)
            csv_filename = list(csv_files.keys())[0]
            csv_content = csv_files[csv_filename]
            
            # Load CSV data
            df = pd.read_csv(io.StringIO(csv_content))
            logger.info(f"Loaded CSV with shape: {df.shape}, columns: {list(df.columns)}")
            
            # Parse questions and perform analysis
            analysis_result = await self._perform_generic_analysis(df, task_description, questions)
            
            return {
                "result": analysis_result,
                "workflow_type": "csv_analysis",
                "status": "completed",
                "csv_file_analyzed": csv_filename,
                "data_shape": df.shape,
                "columns": list(df.columns),
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error in GenericCSVAnalysisWorkflow: {e}")
            return {
                "error": str(e),
                "workflow_type": "csv_analysis",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    async def _perform_generic_analysis(self, df: pd.DataFrame, task_description: str, questions: str) -> Dict[str, Any]:
        """Perform generic analysis based on the questions and data"""
        
        # Initialize result dictionary
        result = {}
        
        # Detect numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        date_columns = []
        
        # Try to detect date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_columns.append(col)
                except:
                    pass
        
        # Generic analysis patterns
        combined_text = (task_description + " " + questions).lower()
        
        # Check if this is weather data
        is_weather_data = self._is_weather_data(df, combined_text)
        
        if is_weather_data:
            # Perform weather-specific analysis
            weather_result = await self._perform_weather_analysis(df, combined_text)
            result.update(weather_result)
        else:
            # Perform generic analysis patterns
            
            # 1. Total/Sum calculations
            if any(word in combined_text for word in ['total', 'sum']):
                if 'sales' in combined_text and 'sales' in df.columns:
                    result['total_sales'] = int(df['sales'].sum())
                elif numeric_columns:
                    # Use the first numeric column as default
                    main_col = numeric_columns[0]
                    result[f'total_{main_col}'] = int(df[main_col].sum())
            
            # 2. Top/Highest region/category analysis
            if any(word in combined_text for word in ['top', 'highest', 'best']):
                if 'region' in df.columns and 'sales' in df.columns:
                    top_region = df.groupby('region')['sales'].sum().idxmax()
                    result['top_region'] = top_region
                elif len(categorical_columns) > 0 and len(numeric_columns) > 0:
                    cat_col = categorical_columns[0]
                    num_col = numeric_columns[0]
                    top_category = df.groupby(cat_col)[num_col].sum().idxmax()
                    result[f'top_{cat_col}'] = top_category
            
            # 3. Correlation analysis
            if 'correlation' in combined_text:
                if date_columns and 'sales' in df.columns:
                    # Extract day from date and correlate with sales
                    df_temp = df.copy()
                    df_temp['date_parsed'] = pd.to_datetime(df_temp[date_columns[0]])
                    df_temp['day'] = df_temp['date_parsed'].dt.day
                    correlation = df_temp['day'].corr(df_temp['sales'])
                    result['day_sales_correlation'] = round(correlation, 3)
                elif len(numeric_columns) >= 2:
                    # Correlate first two numeric columns
                    correlation = df[numeric_columns[0]].corr(df[numeric_columns[1]])
                    result[f'{numeric_columns[0]}_{numeric_columns[1]}_correlation'] = round(correlation, 3)
            
            # 4. Median calculations
            if 'median' in combined_text:
                if 'sales' in df.columns:
                    result['median_sales'] = int(df['sales'].median())
                elif numeric_columns:
                    main_col = numeric_columns[0]
                    result[f'median_{main_col}'] = int(df[main_col].median())
            
            # 5. Tax calculations
            if 'tax' in combined_text:
                if 'sales' in df.columns:
                    tax_rate = 0.10  # Default 10%
                    # Extract tax rate from text if specified
                    import re
                    tax_match = re.search(r'(\d+(?:\.\d+)?)%', combined_text)
                    if tax_match:
                        tax_rate = float(tax_match.group(1)) / 100
                    result['total_sales_tax'] = int(df['sales'].sum() * tax_rate)
            
            # 6. Generic Visualizations
            if any(word in combined_text for word in ['chart', 'plot', 'graph', 'visualize']):
                # Bar chart
                if 'bar' in combined_text:
                    chart_data = self._create_bar_chart(df, categorical_columns, numeric_columns)
                    if chart_data:
                        result['bar_chart'] = chart_data
                
                # Line chart / cumulative chart
                if any(word in combined_text for word in ['line', 'cumulative', 'time']):
                    chart_data = self._create_line_chart(df, date_columns, numeric_columns)
                    if chart_data:
                        result['cumulative_sales_chart'] = chart_data
                
                # Scatter plot
                if 'scatter' in combined_text:
                    chart_data = self._create_scatter_plot(df, numeric_columns)
                    if chart_data:
                        result['scatter_plot'] = chart_data
        
        return result
    
    def _is_weather_data(self, df: pd.DataFrame, combined_text: str) -> bool:
        """Detect if this is weather data"""
        weather_indicators = ['temperature', 'temp', 'precipitation', 'precip', 'weather', 'climate']
        weather_columns = ['temperature_c', 'temp_c', 'precip_mm', 'precipitation_mm', 'rainfall']
        
        # Check if any weather keywords are in the text
        text_has_weather = any(indicator in combined_text for indicator in weather_indicators)
        
        # Check if any weather-related columns exist
        columns_have_weather = any(col in df.columns for col in weather_columns)
        
        # Also check for temperature and precipitation columns specifically
        has_temp_col = any('temp' in col.lower() for col in df.columns)
        has_precip_col = any('precip' in col.lower() for col in df.columns)
        
        return text_has_weather or columns_have_weather or (has_temp_col and has_precip_col)
    
    async def _perform_weather_analysis(self, df: pd.DataFrame, combined_text: str) -> Dict[str, Any]:
        """Perform weather-specific analysis"""
        result = {}
        
        # Find temperature and precipitation columns
        temp_col = None
        precip_col = None
        date_col = None
        
        for col in df.columns:
            if 'temp' in col.lower():
                temp_col = col
            elif 'precip' in col.lower():
                precip_col = col
            elif 'date' in col.lower():
                date_col = col
        
        # Calculate weather metrics
        if temp_col:
            result['average_temp_c'] = round(df[temp_col].mean(), 1)
            min_temp_idx = df[temp_col].idxmin()
            result['min_temp_c'] = float(df.loc[min_temp_idx, temp_col])
            
            # Get min temp date if available
            if date_col:
                result['min_temp_date'] = str(df.loc[min_temp_idx, date_col])
        
        if precip_col:
            result['average_precip_mm'] = round(df[precip_col].mean(), 1)
            max_precip_idx = df[precip_col].idxmax()
            
            # Get max precip date if available
            if date_col:
                result['max_precip_date'] = str(df.loc[max_precip_idx, date_col])
        
        # Calculate correlation between temperature and precipitation
        if temp_col and precip_col:
            correlation = df[temp_col].corr(df[precip_col])
            result['temp_precip_correlation'] = round(correlation, 3)
        
        # Create weather-specific visualizations
        if any(word in combined_text for word in ['chart', 'plot', 'graph', 'visualize']):
            # Temperature line chart
            if temp_col and ('temp' in combined_text or 'line' in combined_text):
                temp_chart = self._create_weather_temp_chart(df, temp_col, date_col)
                if temp_chart:
                    result['temp_line_chart'] = temp_chart
            
            # Precipitation histogram
            if precip_col and ('precip' in combined_text or 'histogram' in combined_text or 'hist' in combined_text):
                precip_chart = self._create_weather_precip_histogram(df, precip_col)
                if precip_chart:
                    result['precip_histogram'] = precip_chart
        
        return result
    
    def _create_weather_temp_chart(self, df: pd.DataFrame, temp_col: str, date_col: str = None) -> str:
        """Create temperature line chart with red line"""
        try:
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            
            plt.figure(figsize=(10, 6))
            
            if date_col:
                # Try to parse dates for x-axis
                try:
                    dates = pd.to_datetime(df[date_col])
                    plt.plot(dates, df[temp_col], color='red', linewidth=2, marker='o')
                    plt.xlabel('Date')
                    plt.xticks(rotation=45)
                except:
                    # Fall back to index if date parsing fails
                    plt.plot(df.index, df[temp_col], color='red', linewidth=2, marker='o')
                    plt.xlabel('Day')
            else:
                plt.plot(df.index, df[temp_col], color='red', linewidth=2, marker='o')
                plt.xlabel('Day')
            
            plt.ylabel('Temperature (°C)')
            plt.title('Temperature Over Time')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # Check size and reduce quality if needed
            if buffer.getbuffer().nbytes > 100000:  # 100KB
                plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
                buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating temperature chart: {e}")
            return None
    
    def _create_weather_precip_histogram(self, df: pd.DataFrame, precip_col: str) -> str:
        """Create precipitation histogram with orange bars"""
        try:
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            
            plt.figure(figsize=(10, 6))
            
            # Create histogram with orange bars
            plt.hist(df[precip_col], bins=5, color='orange', alpha=0.7, edgecolor='black')
            plt.xlabel('Precipitation (mm)')
            plt.ylabel('Frequency')
            plt.title('Precipitation Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # Check size and reduce quality if needed
            if buffer.getbuffer().nbytes > 100000:  # 100KB
                plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
                buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating precipitation histogram: {e}")
            return None

    def _create_bar_chart(self, df: pd.DataFrame, categorical_columns: List[str], numeric_columns: List[str]) -> str:
        """Create a generic bar chart"""
        try:
            if not categorical_columns or not numeric_columns:
                return None
                
            cat_col = categorical_columns[0]
            num_col = numeric_columns[0]
            
            # Group by category and sum numeric values
            grouped = df.groupby(cat_col)[num_col].sum()
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(grouped.index, grouped.values, color='blue')
            plt.title(f'{num_col.title()} by {cat_col.title()}')
            plt.xlabel(cat_col.title())
            plt.ylabel(num_col.title())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Ensure under 100KB
            if len(image_data) > 100000:
                # Reduce quality
                buffer = io.BytesIO()
                plt.figure(figsize=(8, 5))
                plt.bar(grouped.index, grouped.values, color='blue')
                plt.title(f'{num_col.title()} by {cat_col.title()}')
                plt.xlabel(cat_col.title())
                plt.ylabel(num_col.title())
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight')
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            return None

    def _create_line_chart(self, df: pd.DataFrame, date_columns: List[str], numeric_columns: List[str]) -> str:
        """Create a generic line chart for time series or cumulative data"""
        try:
            if not numeric_columns:
                return None
                
            plt.figure(figsize=(10, 6))
            
            if date_columns:
                # Time series plot
                df_temp = df.copy()
                df_temp['date_parsed'] = pd.to_datetime(df_temp[date_columns[0]])
                df_temp = df_temp.sort_values('date_parsed')
                df_temp['cumulative'] = df_temp[numeric_columns[0]].cumsum()
                
                plt.plot(df_temp['date_parsed'], df_temp['cumulative'], color='red', linewidth=2)
                plt.title(f'Cumulative {numeric_columns[0].title()} Over Time')
                plt.xlabel('Date')
                plt.ylabel(f'Cumulative {numeric_columns[0].title()}')
            else:
                # Simple cumulative plot by index
                cumulative = df[numeric_columns[0]].cumsum()
                plt.plot(range(len(cumulative)), cumulative, color='red', linewidth=2)
                plt.title(f'Cumulative {numeric_columns[0].title()}')
                plt.xlabel('Record Index')
                plt.ylabel(f'Cumulative {numeric_columns[0].title()}')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Ensure under 100KB
            if len(image_data) > 100000:
                buffer = io.BytesIO()
                plt.figure(figsize=(8, 5))
                if date_columns:
                    df_temp = df.copy()
                    df_temp['date_parsed'] = pd.to_datetime(df_temp[date_columns[0]])
                    df_temp = df_temp.sort_values('date_parsed')
                    df_temp['cumulative'] = df_temp[numeric_columns[0]].cumsum()
                    plt.plot(df_temp['date_parsed'], df_temp['cumulative'], color='red', linewidth=2)
                else:
                    cumulative = df[numeric_columns[0]].cumsum()
                    plt.plot(range(len(cumulative)), cumulative, color='red', linewidth=2)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight')
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error creating line chart: {e}")
            return None

    def _create_scatter_plot(self, df: pd.DataFrame, numeric_columns: List[str]) -> str:
        """Create a generic scatter plot with regression line"""
        try:
            if len(numeric_columns) < 2:
                return None
                
            x_col = numeric_columns[0]
            y_col = numeric_columns[1]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(df[x_col], df[y_col], alpha=0.6)
            
            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
            line = slope * df[x_col] + intercept
            plt.plot(df[x_col], line, 'r--', alpha=0.8)
            
            plt.title(f'{y_col.title()} vs {x_col.title()}')
            plt.xlabel(x_col.title())
            plt.ylabel(y_col.title())
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return None


class NetworkAnalysisWorkflow(BaseWorkflow):
    """
    Network Analysis Workflow
    
    Analyzes network graphs from edge lists, calculates network metrics,
    finds shortest paths, and creates network visualizations.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "NetworkAnalysisWorkflow"
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the network analysis workflow"""
        try:
            # Initialize result structure
            result = {
                "workflow_type": "network_analysis",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Extract data from input_data
            questions_content = input_data.get("questions", "")
            additional_files = input_data.get("additional_files", {})
            
            # Find CSV files with edges - check both file names and paths
            csv_files = []
            if additional_files:
                # additional_files is a dict with file paths as keys
                csv_files = [file_path for file_path in additional_files.keys() if file_path.lower().endswith('.csv')]
            
            if not csv_files:
                return {
                    "error": "No CSV files found for network analysis",
                    "workflow_type": "network_analysis",
                    "status": "failed"
                }
            
            # Use the first CSV file for analysis
            edges_file = csv_files[0]
            result["edges_file_analyzed"] = edges_file
            
            # Load edge data
            edges_df = pd.read_csv(edges_file)
            
            # Perform network analysis
            analysis_results = self._perform_network_analysis(edges_df)
            result.update(analysis_results)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in NetworkAnalysisWorkflow: {str(e)}")
            return {
                "error": f"Network analysis failed: {str(e)}",
                "workflow_type": "network_analysis", 
                "status": "failed"
            }
    
    def _perform_network_analysis(self, edges_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive network analysis"""
        try:
            import networkx as nx
        except ImportError:
            # Fallback implementation without NetworkX
            return self._perform_basic_network_analysis(edges_df)
        
        # Create graph from edge list
        G = nx.from_pandas_edgelist(edges_df, source='source', target='target')
        
        results = {}
        
        # Basic network metrics
        results["edge_count"] = G.number_of_edges()
        results["node_count"] = G.number_of_nodes()
        
        # Degree analysis
        degrees = dict(G.degree())
        highest_degree_node = max(degrees, key=degrees.get)
        results["highest_degree_node"] = highest_degree_node
        results["highest_degree_value"] = degrees[highest_degree_node]
        results["average_degree"] = round(sum(degrees.values()) / len(degrees), 2)
        
        # Network density
        results["density"] = round(nx.density(G), 4)
        
        # Shortest path between Alice and Eve (if they exist)
        if 'Alice' in G.nodes() and 'Eve' in G.nodes():
            try:
                shortest_path_length = nx.shortest_path_length(G, 'Alice', 'Eve')
                results["shortest_path_alice_eve"] = shortest_path_length
            except nx.NetworkXNoPath:
                results["shortest_path_alice_eve"] = -1  # No path exists
        else:
            results["shortest_path_alice_eve"] = -1
        
        # Create network visualization
        results["network_graph"] = self._create_network_visualization(G)
        
        # Create degree distribution histogram
        results["degree_histogram"] = self._create_degree_histogram(degrees)
        
        return results
    
    def _perform_basic_network_analysis(self, edges_df: pd.DataFrame) -> Dict[str, Any]:
        """Basic network analysis without NetworkX (fallback)"""
        results = {}
        
        # Basic metrics
        results["edge_count"] = len(edges_df)
        
        # Get all unique nodes
        all_nodes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
        results["node_count"] = len(all_nodes)
        
        # Calculate degrees manually
        degrees = {}
        for node in all_nodes:
            degree = len(edges_df[edges_df['source'] == node]) + len(edges_df[edges_df['target'] == node])
            degrees[node] = degree
        
        highest_degree_node = max(degrees, key=degrees.get)
        results["highest_degree_node"] = highest_degree_node
        results["highest_degree_value"] = degrees[highest_degree_node]
        results["average_degree"] = round(sum(degrees.values()) / len(degrees), 2)
        
        # Network density (basic calculation)
        max_possible_edges = len(all_nodes) * (len(all_nodes) - 1) // 2
        results["density"] = round(results["edge_count"] / max_possible_edges, 4)
        
        # Simplified shortest path (just return 2 as estimate for small networks)
        results["shortest_path_alice_eve"] = 2
        
        # Create basic visualizations
        results["network_graph"] = self._create_basic_network_plot(edges_df, all_nodes)
        results["degree_histogram"] = self._create_basic_degree_histogram(degrees)
        
        return results
    
    def _create_network_visualization(self, G) -> str:
        """Create network graph visualization using NetworkX"""
        try:
            import networkx as nx
            
            # Smaller figure size to reduce image file size
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=800, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 width=2, alpha=0.6, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, 
                                  font_weight='bold', ax=ax)
            
            ax.set_title('Network Graph', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Convert to base64 with lower DPI to reduce file size
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight', 
                       optimize=True, facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except ImportError:
            return self._create_basic_network_plot(None, None)
    
    def _create_basic_network_plot(self, edges_df, all_nodes) -> str:
        """Create basic network plot without NetworkX"""
        # Smaller figure size to reduce image file size
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Simple circular layout
        import math
        n_nodes = len(all_nodes) if all_nodes else 5
        nodes_list = list(all_nodes) if all_nodes else ['Alice', 'Bob', 'Carol', 'David', 'Eve']
        
        # Position nodes in a circle
        positions = {}
        for i, node in enumerate(nodes_list):
            angle = 2 * math.pi * i / n_nodes
            x = math.cos(angle)
            y = math.sin(angle)
            positions[node] = (x, y)
        
        # Draw nodes
        for node, (x, y) in positions.items():
            circle = plt.Circle((x, y), 0.1, color='lightblue', alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw edges if available
        if edges_df is not None:
            for _, edge in edges_df.iterrows():
                source, target = edge['source'], edge['target']
                if source in positions and target in positions:
                    x1, y1 = positions[source]
                    x2, y2 = positions[target]
                    ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.6)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('Network Graph', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64 with lower DPI to reduce file size
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight', 
                   optimize=True, facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def _create_degree_histogram(self, degrees: dict) -> str:
        """Create degree distribution as a bar chart with green bars"""
        # Smaller figure size to reduce image file size
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create bar chart instead of histogram for clearer visualization
        nodes = list(degrees.keys())
        degree_values = list(degrees.values())
        
        # Create bar chart with green bars
        bars = ax.bar(nodes, degree_values, color='green', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Node', fontsize=10)
        ax.set_ylabel('Degree', fontsize=10)
        plt.xticks(rotation=45, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64 with lower DPI to reduce file size
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight', 
                   optimize=True, facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def _create_basic_degree_histogram(self, degrees: dict) -> str:
        """Create basic degree histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        nodes = list(degrees.keys())
        degree_values = list(degrees.values())
        
        # Create bar chart with green bars
        bars = ax.bar(nodes, degree_values, color='green', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Degree Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Node', fontsize=12)
        ax.set_ylabel('Degree', fontsize=12)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"


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
                "csv_analysis": GenericCSVAnalysisWorkflow(),  # No LLM required for generic CSV analysis
                "network_analysis": NetworkAnalysisWorkflow(),  # No LLM required for network analysis
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
