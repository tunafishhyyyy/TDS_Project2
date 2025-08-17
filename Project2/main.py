from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uuid
import asyncio
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
    CHART_TYPE_KEYWORDS, REGRESSION_KEYWORDS, BASE64_KEYWORDS, URL_PATTERN, S3_PATH_PATTERN,
    CSV_ANALYSIS_KEYWORDS, NETWORK_ANALYSIS_KEYWORDS
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

    orchestrator = AdvancedWorkflowOrchestrator()
    logger.info("AdvancedWorkflowOrchestrator initialized successfully.")
except Exception as e:
    logger.error(f"Could not import or initialize workflows: {e}")
    # Try to create a minimal orchestrator with just the fallback workflow
    try:
        from chains.workflows import ModularWebScrapingWorkflow
        from chains.base import WorkflowOrchestrator

        class MinimalOrchestrator(WorkflowOrchestrator):
            def __init__(self):
                super().__init__()  # This was missing
                self.llm = None
                self.workflows = {"multi_step_web_scraping": ModularWebScrapingWorkflow()}

        orchestrator = MinimalOrchestrator()
        logger.info("Created minimal orchestrator with fallback workflows")
    except Exception as e2:
        logger.error(f"Could not create minimal orchestrator: {e2}")
        orchestrator = None

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


# Pydantic models for request/response
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
    # Model for multi-step workflow requests
    """Model for multi-step workflow requests"""
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
) -> str:  # LLM-based workflow type detection
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
                return detect_workflow_type_fallback(task_description, default_workflow, {})

        else:
            logger.warning("LLM not available, using fallback workflow detection")
            return detect_workflow_type_fallback(task_description, default_workflow, {})

    except Exception as e:
        logger.error(f"Error in LLM workflow detection: {e}")
        return detect_workflow_type_fallback(task_description, default_workflow, {})


def detect_workflow_type_fallback(
    task_description: str, default_workflow: str = DEFAULT_WORKFLOW, additional_files: Dict[str, Any] = None
) -> str:  # Fallback keyword-based workflow detection
    """
    Fallback keyword-based workflow detection when LLM is not available
    """
    if not task_description:
        return default_workflow

    task_lower = task_description.lower()
    
    # Check if CSV files are uploaded - strong indicator for CSV analysis
    if additional_files:
        csv_files = [f for f in additional_files.keys() if f.lower().endswith('.csv')]
        if csv_files:
            # Check if it's network analysis (edges.csv or network-related keywords)
            network_file_check = any('edge' in f.lower() for f in csv_files)
            network_keyword_check = any(keyword in task_lower for keyword in NETWORK_ANALYSIS_KEYWORDS)
            if network_file_check or network_keyword_check:
                return "network_analysis"
            
            # If CSV files are present and task mentions analysis keywords, use CSV analysis
            analysis_keywords = ['analyze', 'analysis', 'total', 'sales', 'median', 'correlation', 'chart', 'plot']
            if any(keyword in task_lower for keyword in analysis_keywords):
                return "csv_analysis"

    # DuckDB/database detection (priority)
    duckdb_keywords = ["duckdb", "parquet", "s3://", "bucket", "sql", "select count(*)", "read_parquet"]
    if any(keyword in task_lower for keyword in duckdb_keywords):
        return "database_analysis"

    # Database analysis patterns
    if any(keyword in task_lower for keyword in DB_KEYWORDS):
        return "database_analysis"

    # Network analysis patterns
    if any(keyword in task_lower for keyword in NETWORK_ANALYSIS_KEYWORDS):
        return "network_analysis"

    # CSV analysis patterns (check before general data analysis)
    if any(keyword in task_lower for keyword in CSV_ANALYSIS_KEYWORDS):
        return "csv_analysis"

    # Web scraping patterns
    if any(keyword in task_lower for keyword in SCRAPING_KEYWORDS):
        if any(keyword in task_lower for keyword in MULTI_STEP_KEYWORDS):
            return "multi_step_web_scraping"
        else:
            return "multi_step_web_scraping"

    if any(keyword in task_lower for keyword in IMAGE_KEYWORDS):
        return "image_analysis"
    if any(keyword in task_lower for keyword in TEXT_KEYWORDS):
        return "text_analysis"
    if any(keyword in task_lower for keyword in LEGAL_KEYWORDS):
        return "data_analysis"
    if any(keyword in task_lower for keyword in STATS_KEYWORDS):
        return "statistical_analysis"
    if any(keyword in task_lower for keyword in VIZ_KEYWORDS):
        return "data_visualization"
    if any(keyword in task_lower for keyword in EDA_KEYWORDS):
        return "exploratory_data_analysis"
    if any(keyword in task_lower for keyword in ML_KEYWORDS):
        return "predictive_modeling"
    if any(keyword in task_lower for keyword in CODE_KEYWORDS):
        return "code_generation"
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

            # Save questions.txt to /tmp
            params["content_type"] = CONTENT_TYPE_JSON
        elif "\t" in file_content or "," in file_content:
            params["content_type"] = CONTENT_TYPE_CSV
        else:
            params["content_type"] = CONTENT_TYPE_TEXT

    return params


@app.post("/api/")
async def analyze_data(
    questions_txt: UploadFile = File(..., alias="questions.txt", description="Required questions.txt file (must contain 'question' in filename)"),
    files: List[UploadFile] = File(default=[], description="Optional additional files (CSV, images, etc.)"),
):
    """
    Main endpoint that accepts multiple file uploads with required questions.txt.
    All processing is synchronous and returns results immediately.

    - **questions_txt**: Required questions.txt file containing the questions
      (must contain 'question' in filename)
    - **files**: Optional additional files (images, CSV, JSON, etc.)
    """
    try:  # Main API endpoint for data analysis
        task_id = str(uuid.uuid4())
        logger.info(f"Starting synchronous task {task_id}")

        # Process required questions.txt file

        # Validate required questions.txt file
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

        # Save questions.txt to /tmp with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        questions_save_name = f"{os.path.splitext(questions_txt.filename)[0]}_{timestamp_str}{os.path.splitext(questions_txt.filename)[1]}"
        tmp_questions_path = os.path.join("/tmp", questions_save_name)
        try:
            with open(tmp_questions_path, "wb") as f:
                f.write(questions_content)
            logger.info(f"Saved questions.txt to {tmp_questions_path}")
        except Exception as e:
            logger.error(f"Failed to save questions.txt to /tmp: {e}")

        # Process additional files (CSV, images, etc.)
        processed_files = {}
        file_contents = {}

        for file in files:
            if file.filename:
                content = await file.read()
                # Handle CSV files
                if file.filename.lower().endswith(".csv"):
                    try:
                        file_text = content.decode("utf-8")
                        file_contents[file.filename] = file_text
                        logger.info(f"Processed CSV file: {file.filename}")
                    except UnicodeDecodeError:
                        file_contents[file.filename] = f"Binary CSV file: {file.filename} ({len(content)} bytes)"
                        logger.info(f"Processed binary CSV file: {file.filename} ({len(content)} bytes)")
                # Handle image files
                elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_contents[file.filename] = content  # Store raw bytes for images
                    logger.info(f"Processed image file: {file.filename} ({len(content)} bytes)")
                # Handle other text files
                else:
                    try:
                        file_text = content.decode("utf-8")
                        file_contents[file.filename] = file_text
                        logger.info(f"Processed text file: {file.filename}")
                    except UnicodeDecodeError:
                        file_contents[file.filename] = f"Binary file: {file.filename} ({len(content)} bytes)"
                        logger.info(f"Processed binary file: {file.filename} ({len(content)} bytes)")

                processed_files[file.filename] = {
                    "content_type": file.content_type,
                    "size": len(content),
                    "is_text": file.filename.endswith((".txt", ".csv", ".json", ".md")),
                    "is_image": file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')),
                }

                # Save each uploaded file to /tmp with timestamp
                file_save_name = f"{os.path.splitext(file.filename)[0]}_{timestamp_str}{os.path.splitext(file.filename)[1]}"
                tmp_file_path = os.path.join("/tmp", file_save_name)
                try:
                    with open(tmp_file_path, "wb") as f:
                        f.write(content)
                    logger.info(f"Saved {file.filename} to {tmp_file_path}")
                    
                    # Store the actual saved file path in file_contents for workflows
                    # Remove original filename entry and add the actual file path
                    original_content = file_contents.get(file.filename, content)
                    if file.filename in file_contents:
                        del file_contents[file.filename]  # Remove entry with original filename
                    file_contents[tmp_file_path] = original_content
                    
                except Exception as e:
                    logger.error(f"Failed to save {file.filename} to /tmp: {e}")

        # Use questions as task description (content of questions.txt)
        task_description = questions_text

        # Intelligent workflow type detection using LLM (enhanced with file info)
        detected_workflow = await detect_workflow_type_llm(task_description, "multi_step_web_scraping")
        
        # If LLM detection fails or returns generic result, try fallback with file info
        if detected_workflow in ["data_analysis", "multi_step_web_scraping"] and file_contents:
            fallback_workflow = detect_workflow_type_fallback(task_description, detected_workflow, file_contents)
            if fallback_workflow != detected_workflow:
                detected_workflow = fallback_workflow
                logger.info(f"Enhanced detection changed workflow to: {detected_workflow}")
        logger.info(f"Detected workflow: {detected_workflow}")
        logger.info(f"Task description: {task_description[:200]}...")

        # Prepare enhanced workflow input
        workflow_input = {
            "task_description": task_description,
            "questions": questions_text,
            "additional_files": file_contents,
            "processed_files_info": processed_files,
            "workflow_type": detected_workflow,
            "parameters": prepare_workflow_parameters(task_description, detected_workflow, questions_text),
            "output_requirements": extract_output_requirements(task_description),
        }

        logger.info(f"Workflow input prepared with {len(workflow_input)} keys")
        logger.info(f"Additional files: {list(file_contents.keys())}")

        # Execute workflow synchronously (always within 3 minutes)
        logger.info(f"Processing task {task_id} synchronously with " f"workflow: {detected_workflow}")

        try:
            logger.info(f"Starting workflow execution for {detected_workflow}")
            result = await asyncio.wait_for(
                execute_workflow_sync(detected_workflow, workflow_input, task_id), timeout=180  # 3 minutes
            )

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
                },
                "timestamp": datetime.now().isoformat(),
            }

        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out after 3 minutes")
            raise HTTPException(
                status_code=408,
                detail=("Request timed out after 3 minutes. Please simplify " "your request or try again."),
            )
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


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
