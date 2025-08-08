from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, List

import sys
import logging
import os
from utils.prompts import WORKFLOW_DETECTION_SYSTEM_PROMPT, WORKFLOW_DETECTION_HUMAN_PROMPT
from utils.constants import (
    VALID_WORKFLOWS, API_TITLE, API_DESCRIPTION, API_VERSION, API_FEATURES,
    API_ENDPOINTS, STATUS_OPERATIONAL, STATUS_HEALTHY, STATUS_AVAILABLE,
    STATUS_UNAVAILABLE, LOG_FORMAT, LOG_FILE, STATIC_DIRECTORY, STATIC_NAME,
    DEFAULT_WORKFLOW, DEFAULT_PRIORITY, DEFAULT_TARGET_AUDIENCE, DEFAULT_PIPELINE_TYPE,
    DEFAULT_OUTPUT_REQUIREMENTS, SCRAPING_KEYWORDS, MULTI_STEP_KEYWORDS,
    IMAGE_KEYWORDS, TEXT_KEYWORDS, LEGAL_KEYWORDS, STATS_KEYWORDS, DB_KEYWORDS,
    VIZ_KEYWORDS, EDA_KEYWORDS, ML_KEYWORDS, CODE_KEYWORDS, WEB_KEYWORDS,
    DATA_TYPE_FINANCIAL, DATA_TYPE_RANKING, DATABASE_TYPE_SQL, FILE_FORMAT_PARQUET,
    CHART_TYPE_SCATTER, OUTPUT_FORMAT_BASE64, MAX_FILE_SIZE,
    CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_TEXT,
    PLOT_CHART_KEYWORDS,
    FORMAT_KEYWORDS, KEY_INCLUDE_VISUALIZATIONS, KEY_VISUALIZATION_FORMAT,
    KEY_MAX_SIZE, KEY_FORMAT, VISUALIZATION_FORMAT_BASE64, MAX_SIZE_BYTES,
    FINANCIAL_DETECTION_KEYWORDS, RANKING_DETECTION_KEYWORDS, DATABASE_DETECTION_KEYWORDS,
    CHART_TYPE_KEYWORDS, REGRESSION_KEYWORDS, BASE64_KEYWORDS, URL_PATTERN, S3_PATH_PATTERN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chains"))
# Global objects
try:
    # Attempt to initialize the advanced orchestrator
    from chains.workflows import AdvancedWorkflowOrchestrator

    # Import enhanced workflow components
    from chains.iterative_reasoning import create_iterative_reasoning_workflow
    from chains.logging_and_benchmarking import workflow_logger, accuracy_benchmark, create_test_benchmark_suite

    orchestrator = AdvancedWorkflowOrchestrator()
    logger.info("AdvancedWorkflowOrchestrator initialized successfully.")

    # Initialize logging and benchmarking
    logger.info("Workflow logging and benchmarking systems initialized.")

    # Create test benchmark suite if it doesn't exist
    try:
        create_test_benchmark_suite()
        logger.info("Test benchmark suite created / verified.")
    except Exception as e:
        logger.warning(f"Could not create test benchmark suite: {e}")

except Exception as e:
    logger.error(f"Could not import or initialize workflows: {e}")
    # Create a minimal, self-contained orchestrator that doesn't depend on LangChain
    try:
        # Local minimal web-scraping workflow using step classes directly
        class MinimalWebScrapingWorkflow:
            async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    from chains.web_scraping_steps import (
                        DetectDataFormatStep,
                        ScrapeTableStep,
                        InspectTableStep,
                        CleanDataStep,
                        AnalyzeDataStep,
                        VisualizeStep,
                        AnswerQuestionsStep,
                    )

                    task_description = input_data.get("task_description", "")
                    # Extract URL from task description
                    import re
                    
                    # First try to extract from Markdown links [text](url)
                    markdown_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
                    markdown_matches = re.findall(markdown_pattern, task_description)
                    if markdown_matches:
                        url = markdown_matches[0][1]  # Return the URL part
                    else:
                        # Fallback to regular URL extraction, but exclude common punctuation
                        urls = re.findall(r"https?://[^\s\)\]\},]+", task_description)
                        url = urls[0] if urls else input_data.get("url")
                    if not url:
                        return {
                            "workflow_type": "multi_step_web_scraping",
                            "status": "error",
                            "error": "No URL found in task description",
                        }

                    data: Dict[str, Any] = {"task_description": task_description, "url": url}
                    log: List[str] = []

                    # Step 0: Detect format
                    step0 = DetectDataFormatStep()
                    data.update(step0.run({"url": url, "task_description": task_description}))
                    log.append("✓ Data format detection completed")

                    # Step 1: Scrape
                    step1 = ScrapeTableStep()
                    data.update(step1.run({**data, "url": url, "task_description": task_description}))
                    log.append("✓ Data extraction completed")

                    # Step 2: Inspect
                    step2 = InspectTableStep()
                    data.update(step2.run(data))
                    log.append("✓ Table inspection completed")

                    # Step 3: Clean
                    step3 = CleanDataStep()
                    data.update(step3.run(data))
                    log.append("✓ Data cleaning completed")

                    # Step 4: Analyze
                    step4 = AnalyzeDataStep()
                    data.update(step4.run({**data, "top_n": 20}))
                    log.append("✓ Data analysis completed")

                    # Step 5: Visualize
                    step5 = VisualizeStep()
                    data.update(step5.run({**data, "return_base64": True}))
                    log.append("✓ Visualization completed")

                    # Step 6: Answer questions
                    step6 = AnswerQuestionsStep()
                    data.update(step6.run(data))
                    log.append("✓ Question answering completed")

                    return {
                        "workflow_type": "multi_step_web_scraping",
                        "status": "completed",
                        "target_url": url,
                        "execution_log": log,
                        "results": data.get("answers", {}),
                        "plot_base64": data.get("plot_base64"),
                        "chart_type": data.get("chart_type"),
                        "image_size_bytes": data.get("image_size_bytes"),
                        "timestamp": datetime.now().isoformat(),
                        "fallback_mode": True,
                    }
                except Exception as e2:  # pragma: no cover
                    logger.error(f"Minimal web scraping workflow failed: {e2}")
                    return {
                        "workflow_type": "multi_step_web_scraping",
                        "status": "error",
                        "error": str(e2),
                    }

        class MinimalOrchestrator:
            def __init__(self):
                self.workflows: Dict[str, Any] = {
                    "multi_step_web_scraping": MinimalWebScrapingWorkflow()
                }

            async def execute_workflow(self, workflow_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
                if workflow_type not in self.workflows:
                    return {
                        "error": f"Workflow {workflow_type} not found",
                        "available_workflows": list(self.workflows.keys()),
                    }
                workflow = self.workflows[workflow_type]
                return await workflow.execute(input_data)

        orchestrator = MinimalOrchestrator()
        logger.info("Created minimal orchestrator with local multi_step_web_scraping workflow")

        # Define a no-op workflow logger so API does not crash when advanced logging isn't available
        class _NoOpWorkflowLogger:
            def start_workflow(self, *args, **kwargs):
                return None

            def complete_workflow(self, *args, **kwargs):
                return None

            def start_step(self, *args, **kwargs):
                return None
        workflow_logger = _NoOpWorkflowLogger()
    except Exception as e2:
        logger.error(f"Could not create minimal orchestrator: {e2}")
        orchestrator = None

        class _NoOpWorkflowLogger:
            def start_workflow(self, *args, **kwargs):
                return None

            def complete_workflow(self, *args, **kwargs):
                return None

            def start_step(self, *args, **kwargs):
                return None

        workflow_logger = _NoOpWorkflowLogger()

app = FastAPI(  # FastAPI app instance
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)
app.mount(f"/{STATIC_NAME}", StaticFiles(directory=STATIC_DIRECTORY), name=STATIC_NAME)  # Mount static files


# Health check and info endpoints
@app.get("/")
async def root():  # Root endpoint with API info
    """Root endpoint with API information"""
    return {
        "message": f"{API_TITLE} v{API_VERSION}",
        "description": API_DESCRIPTION,
        "features": API_FEATURES,
        "endpoints": API_ENDPOINTS,
        "status": STATUS_OPERATIONAL,
    }


@app.get("/health")
async def health_check():  # Health check endpoint
    """Health check endpoint"""
    orchestrator_status = STATUS_AVAILABLE if orchestrator else STATUS_UNAVAILABLE

    return {
        "status": STATUS_HEALTHY,
        "timestamp": datetime.now().isoformat(),
        "orchestrator": orchestrator_status,
        "workflows_available": (len(orchestrator.workflows) if orchestrator else 0),
        "version": API_VERSION,
    }


# Pydantic models for request / response
class TaskRequest(BaseModel):  # Model for analysis task requests
    """Model for analysis task requests"""

    task_description: str = Field(..., description="Description of the analysis task")
    workflow_type: Optional[str] = Field(DEFAULT_WORKFLOW, description="Type of workflow to execute")
    data_source: Optional[str] = Field(None, description="Optional data source information")
    dataset_info: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Dataset characteristics and metadata"
    )
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters for the task")
    priority: Optional[str] = Field(DEFAULT_PRIORITY, description="Task priority: low, normal, high")
    include_modeling: Optional[bool] = Field(False, description="Include predictive modeling in analysis")
    target_audience: Optional[str] = Field(DEFAULT_TARGET_AUDIENCE, description="Target audience for reports")


class WorkflowRequest(BaseModel):  # Model for specific workflow requests
    """Model for specific workflow requests"""

    workflow_type: str = Field(..., description="Type of workflow to execute")
    input_data: Dict[str, Any] = Field(..., description="Input data for the workflow")


class MultiStepWorkflowRequest(BaseModel):
    # Model for multi - step workflow requests
    """Model for multi - step workflow requests"""
    steps: List[Dict[str, Any]] = Field(..., description="List of workflow steps to execute")
    pipeline_type: Optional[str] = Field(DEFAULT_PIPELINE_TYPE, description="Type of pipeline")


class TaskResponse(BaseModel):  # Model for task response
    """Model for task response"""

    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    task_details: Dict[str, Any] = Field(..., description="Details of the submitted task")
    created_at: str = Field(..., description="Task creation timestamp")
    workflow_result: Optional[Dict[str, Any]] = Field(None, description="LangChain workflow execution result")


def extract_output_requirements(
    task_description: str,
) -> Dict[str, Any]:  # Extract output requirements from task description
    """Extract specific output requirements from task description"""
    requirements = DEFAULT_OUTPUT_REQUIREMENTS.copy()

    task_lower = task_description.lower()

    # Check for visualization requirements
    if any(keyword in task_lower for keyword in PLOT_CHART_KEYWORDS):
        requirements[KEY_INCLUDE_VISUALIZATIONS] = True
        requirements[KEY_VISUALIZATION_FORMAT] = VISUALIZATION_FORMAT_BASE64
        requirements[KEY_MAX_SIZE] = MAX_SIZE_BYTES

    # Check for specific format requirements
    if any(keyword in task_lower for keyword in FORMAT_KEYWORDS):
        if "json" in task_lower:
            requirements[KEY_FORMAT] = CONTENT_TYPE_JSON
        elif "csv" in task_lower:
            requirements[KEY_FORMAT] = CONTENT_TYPE_CSV
        elif "table" in task_lower:
            requirements[KEY_FORMAT] = "table"

    return requirements


async def detect_workflow_type_llm(
    task_description: str, default_workflow: str = DEFAULT_WORKFLOW
) -> str:  # LLM - based workflow type detection
    """
    Use LLM prompting to determine the workflow type based on the
    input task description
    """
    if not task_description:
        return default_workflow

    logger.info(f"Detecting workflow type for task: {task_description[:100]}...")

    try:
        if orchestrator and orchestrator.llm:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains import LLMChain

            workflow_detection_prompt = ChatPromptTemplate.from_messages([
                ("system", WORKFLOW_DETECTION_SYSTEM_PROMPT),
                ("human", WORKFLOW_DETECTION_HUMAN_PROMPT),
            ])

            chain = LLMChain(llm=orchestrator.llm, prompt=workflow_detection_prompt)
            result = chain.run(task_description=task_description)

            # Clean and validate the result
            detected_workflow = result.strip().lower()

            # List of valid workflows (generalized)
            if detected_workflow in VALID_WORKFLOWS:
                logger.info(f"LLM detected workflow type: {detected_workflow}")
                return detected_workflow
            else:
                logger.warning(f"LLM returned invalid workflow: {detected_workflow}, " f"using fallback")
                return detect_workflow_type_fallback(task_description, default_workflow)

        else:
            logger.warning("LLM not available, using fallback workflow detection")
            return detect_workflow_type_fallback(task_description, default_workflow)

    except Exception as e:
        logger.error(f"Error in LLM workflow detection: {e}")
        return detect_workflow_type_fallback(task_description, default_workflow)


def detect_workflow_type_fallback(
    task_description: str, default_workflow: str = DEFAULT_WORKFLOW
) -> str:  # Fallback keyword - based workflow detection
    """
    Fallback keyword - based workflow detection when LLM is not available
    """
    if not task_description:
        return default_workflow

    task_lower = task_description.lower()

    # Web scraping patterns - PRIORITIZE BEFORE IMAGE ANALYSIS
    if any(keyword in task_lower for keyword in SCRAPING_KEYWORDS):
        # Check if it involves multiple steps
        # (cleaning, analysis, visualization, questions)
        if any(keyword in task_lower for keyword in MULTI_STEP_KEYWORDS):
            return "multi_step_web_scraping"
        else:
            return "multi_step_web_scraping"  # Image analysis patterns

    if any(keyword in task_lower for keyword in IMAGE_KEYWORDS):
        return "image_analysis"

    # Text analysis patterns
    if any(keyword in task_lower for keyword in TEXT_KEYWORDS):
        return "text_analysis"

    # Legal / Court data patterns - map to general data analysis
    if any(keyword in task_lower for keyword in LEGAL_KEYWORDS):
        return "data_analysis"

    # Statistical analysis patterns
    if any(keyword in task_lower for keyword in STATS_KEYWORDS):
        return "statistical_analysis"

    # Database analysis patterns
    if any(keyword in task_lower for keyword in DB_KEYWORDS):
        return "database_analysis"

    # Data visualization patterns
    if any(keyword in task_lower for keyword in VIZ_KEYWORDS):
        return "data_visualization"

    # Exploratory data analysis patterns
    if any(keyword in task_lower for keyword in EDA_KEYWORDS):
        return "exploratory_data_analysis"

    # Predictive modeling patterns
    if any(keyword in task_lower for keyword in ML_KEYWORDS):
        return "predictive_modeling"

    # Code generation patterns
    if any(keyword in task_lower for keyword in CODE_KEYWORDS):
        return "code_generation"

    # Generic web scraping patterns
    if any(keyword in task_lower for keyword in WEB_KEYWORDS):
        return "multi_step_web_scraping"

    return default_workflow


def prepare_workflow_parameters(
    task_description: str, workflow_type: str, file_content: str = None
) -> Dict[str, Any]:  # Prepare parameters for workflow execution
    """
    Prepare specific parameters based on workflow type and task description
    """
    params = {}
    task_lower = task_description.lower() if task_description else ""

    # Generic URL extraction
    if "http" in task_lower:
        import re

        urls = re.findall(URL_PATTERN, task_description)
        params["target_urls"] = urls

    # Generic data type detection
    if any(kw in task_lower for kw in FINANCIAL_DETECTION_KEYWORDS):
        params["data_type"] = DATA_TYPE_FINANCIAL
    elif any(kw in task_lower for kw in RANKING_DETECTION_KEYWORDS):
        params["data_type"] = DATA_TYPE_RANKING

    # Database parameters (generic)
    if "s3://" in task_lower:
        import re

        s3_paths = re.findall(S3_PATH_PATTERN, task_description)
        params["s3_paths"] = s3_paths
    if any(kw in task_lower for kw in DATABASE_DETECTION_KEYWORDS):
        params["database_type"] = DATABASE_TYPE_SQL
    if "parquet" in task_lower:
        params["file_format"] = FILE_FORMAT_PARQUET

    # Visualization parameters (generic)
    if any(kw in task_lower for kw in CHART_TYPE_KEYWORDS):
        params["chart_type"] = CHART_TYPE_SCATTER
    if any(kw in task_lower for kw in REGRESSION_KEYWORDS):
        params["include_regression"] = True
    if any(kw in task_lower for kw in BASE64_KEYWORDS):
        params["output_format"] = OUTPUT_FORMAT_BASE64
        params["max_size"] = MAX_FILE_SIZE  # 100KB limit

    # File content analysis
    if file_content:
        params["file_content_length"] = len(file_content)
        content_stripped = file_content.strip()
        if content_stripped.startswith(("{", "[")):
            params["content_type"] = CONTENT_TYPE_JSON
        elif "\t" in file_content or "," in file_content:
            params["content_type"] = CONTENT_TYPE_CSV
        else:
            params["content_type"] = CONTENT_TYPE_TEXT

    return params


@app.post("/api/")
async def analyze_data(
    questions_txt: UploadFile = File(..., description="Required questions.txt file"),
    files: List[UploadFile] = File(default=[], description="Optional additional files"),
    enable_iterative_reasoning: bool = False,
    enable_logging: bool = True,
):
    """
    Enhanced main endpoint with architectural improvements including:
    - Iterative reasoning with self - check passes
    - Comprehensive logging at each step
    - Strong data validation layer
    - Modular and composable workflow system

    - **questions_txt**: Required questions.txt file containing the questions
    - **files**: Optional additional files (images, CSV, JSON, etc.)
    - **enable_iterative_reasoning**: Use iterative reasoning workflow for higher accuracy
    - **enable_logging**: Enable detailed step - by - step logging
    """
    try:  # Enhanced main API endpoint for data analysis
        task_id = str(uuid.uuid4())
        logger.info(f"Starting enhanced synchronous task {task_id}")

        # Initialize workflow logging if enabled
        workflow_execution = None
        if enable_logging:
            workflow_execution = workflow_logger.start_workflow(
                task_id, "enhanced_analysis", {"enable_iterative_reasoning": enable_iterative_reasoning}
            )

        # Process required questions.txt file
        if not (questions_txt.filename.lower().endswith(".txt") or "question" in questions_txt.filename.lower()):
            raise HTTPException(
                status_code=400,
                detail=(
                    "questions.txt file is required and must be named "
                    "appropriately (must contain 'question' in filename)"
                ),
            )

        questions_content = await questions_txt.read()
        questions_text = questions_content.decode("utf-8")
        logger.info(f"Processed questions.txt with {len(questions_text)} characters")

        # Process additional files with enhanced validation
        processed_files = {}
        file_contents = {}

        for file in files:
            if file.filename:
                content = await file.read()
                try:
                    file_text = content.decode("utf-8")
                    file_contents[file.filename] = file_text
                    logger.info(f"Processed text file: {file.filename}")
                except UnicodeDecodeError:
                    # Handle binary files (images, etc.)
                    file_contents[file.filename] = f"Binary file: {file.filename} ({len(content)} bytes)"
                    logger.info(f"Processed binary file: {file.filename} ({len(content)} bytes)")

                processed_files[file.filename] = {
                    "content_type": file.content_type,
                    "size": len(content),
                    "is_text": file.filename.endswith((".txt", ".csv", ".json", ".md")),
                }

        # Use questions as task description (content of questions.txt)
        task_description = questions_text

        # Intelligent workflow type detection using LLM
        detected_workflow = await detect_workflow_type_llm(task_description, "multi_step_web_scraping")
        logger.info(f"Detected workflow: {detected_workflow}")
        logger.info(f"Task description: {task_description[:200]}...")

        # Prepare enhanced workflow input with validation requirements
        workflow_input = {
            "task_description": task_description,
            "questions": questions_text,
            "additional_files": file_contents,
            "processed_files_info": processed_files,
            "workflow_type": detected_workflow,
            "parameters": prepare_workflow_parameters(task_description, detected_workflow, questions_text),
            "output_requirements": extract_output_requirements(task_description),
            "enable_validation": True,  # Enable strong data validation layer
            "enable_self_check": enable_iterative_reasoning,  # Enable self - check passes
            "enable_logging": enable_logging,
        }

        logger.info(f"Enhanced workflow input prepared with {len(workflow_input)} keys")
        logger.info(f"Additional files: {list(file_contents.keys())}")

        # Choose workflow execution method based on configuration
        if enable_iterative_reasoning:
            logger.info(f"Using iterative reasoning workflow for task {task_id}")
            result = await execute_iterative_workflow(workflow_input, task_id)
        else:
            logger.info(f"Using standard workflow for task {task_id}")
            result = await execute_workflow_sync(detected_workflow, workflow_input, task_id)

        # Complete workflow logging
        if workflow_execution and enable_logging:
            workflow_logger.complete_workflow(result, "completed")

        logger.info(f"Task {task_id} completed successfully")
        logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        return {
            "task_id": task_id,
            "status": "completed",
            "workflow_type": detected_workflow,
            "result": result,
            "processing_info": {
                "questions_file": questions_txt.filename,
                "additional_files": list(processed_files.keys()),
                "workflow_auto_detected": True,
                "processing_time": "synchronous",
                "iterative_reasoning_enabled": enable_iterative_reasoning,
                "logging_enabled": enable_logging,
                "enhanced_features": [
                    "data_validation",
                    "modular_steps",
                    "comprehensive_logging"
                ]
            },
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing enhanced request: {str(e)}")
        if workflow_execution and enable_logging:
            workflow_logger.complete_workflow({}, "failed", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


async def execute_iterative_workflow(workflow_input: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Execute enhanced iterative reasoning workflow"""
    try:
        # Create iterative reasoning workflow
        iterative_workflow = create_iterative_reasoning_workflow(
            llm=orchestrator.llm if orchestrator else None,
            max_iterations=3,
            confidence_threshold=0.8
        )

        logger.info(f"Executing iterative reasoning workflow for task {task_id}")
        result = await iterative_workflow.execute(workflow_input)

        return result

    except Exception as e:
        logger.error(f"Iterative workflow execution failed for task {task_id}: {e}")
        # Fallback to standard workflow
        return await execute_workflow_sync("data_analysis", workflow_input, task_id)


@app.post("/api/benchmark")
async def run_accuracy_benchmark(
    suite_name: str = "test_suite",
    workflow_type: str = "data_analysis"
):
    """
    Run accuracy benchmarks to measure workflow performance

    - **suite_name**: Name of the benchmark suite to run
    - **workflow_type**: Type of workflow to benchmark
    """
    try:
        logger.info(f"Starting accuracy benchmark for {workflow_type} on suite {suite_name}")

        # Import workflow class
        from chains.workflows import DataAnalysisWorkflow

        # Run benchmark
        benchmark_results = accuracy_benchmark.run_accuracy_benchmark(
            DataAnalysisWorkflow,
            suite_name,
            {"llm": orchestrator.llm if orchestrator else None}
        )

        return {
            "benchmark_results": benchmark_results,
            "suite_name": suite_name,
            "workflow_type": workflow_type,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark execution failed: {str(e)}"
        )


@app.get("/api/workflow-capabilities")
async def get_workflow_capabilities():
    """Get information about available workflows and their capabilities"""
    try:
        if orchestrator:
            capabilities = orchestrator.get_workflow_capabilities()
        else:
            capabilities = {
                "error": "Orchestrator not available",
                "available_workflows": [],
                "enhanced_features": []
            }

        # Add information about architectural enhancements
        capabilities["architectural_enhancements"] = {
            "generalized_data_analysis": "Unified workflow for multiple data source types",
            "modular_composable_system": "Independent, reusable step functions",
            "data_validation_layer": "Schema checks, outlier detection, type enforcement",
            "iterative_reasoning": "Self - check passes with LLM validation",
            "logging_benchmarking": "Comprehensive step - by - step logging and accuracy tracking",
            "extensible_workflows": "Plugin pattern for new data sources"
        }

        return capabilities

    except Exception as e:
        logger.error(f"Error getting workflow capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting capabilities: {str(e)}"
        )


async def execute_workflow_sync(
    workflow_type: str, workflow_input: Dict[str, Any], task_id: str
) -> Dict[str, Any]:  # Execute workflow synchronously
    """Execute workflow synchronously with enhanced error handling"""
    try:
        if orchestrator is None:
            logger.warning("No orchestrator available, cannot execute workflows")
            return {
                "workflow_type": workflow_type,
                "status": "completed_fallback",
                "message": "Orchestrator not available, using fallback response",
                "task_analysis": (
                    f"Detected workflow: {workflow_type} for questions: "
                    f"{workflow_input.get('questions', '')[:100]}..."
                ),
                "recommendations": [
                    "Check workflow initialization",
                    "Install required dependencies",
                    "Configure OpenAI API key",
                ],
                "parameters_prepared": workflow_input.get("parameters", {}),
                "files_processed": list(workflow_input.get("additional_files", {}).keys()),
            }
        else:
            logger.info(f"Executing workflow {workflow_type} with orchestrator")
            logger.info(f"Available workflows: {list(orchestrator.workflows.keys())}")

            if workflow_type not in orchestrator.workflows:
                logger.warning(
                    f"Workflow {workflow_type} not found, available: " f"{list(orchestrator.workflows.keys())}"
                )
                return {
                    "workflow_type": workflow_type,
                    "status": "error",
                    "message": f"Workflow {workflow_type} not found",
                    "available_workflows": list(orchestrator.workflows.keys()),
                }

            result = await orchestrator.execute_workflow(workflow_type, workflow_input)
            logger.info(f"Workflow {workflow_type} executed successfully for " f"task {task_id}")
            logger.info(f"Result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"Result keys: {list(result.keys())}")
            return result
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_type}: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e
