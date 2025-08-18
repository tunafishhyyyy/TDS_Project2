from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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

async def analyze_data(
    request: Request,
    questions_txt: UploadFile = File(..., alias="questions.txt")
):ETECTION_HUMAN_PROMPT
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
    task_description: str, default_workflow: str = "multi_step_web_scraping", additional_files: dict = None
) -> str:  # Fallback workflow detection using keywords
    """
    Fallback keyword-based workflow detection enhanced with file information
    """
    if not task_description:
        return default_workflow
    
    logger.info(f"Starting fallback workflow detection for task: {task_description[:200]}...")
    logger.info(f"Additional files provided: {list(additional_files.keys()) if additional_files else 'None'}")

    task_lower = task_description.lower()

    # Enhanced CSV file detection
    if additional_files:
        csv_files = [f for f in additional_files.keys() if f.lower().endswith('.csv')]
        logger.info(f"Found CSV files: {csv_files}")
        
        if csv_files:
            # Check if it's network analysis based on specific file patterns
            network_file_check = any('edge' in f.lower() or 'node' in f.lower() or 'graph' in f.lower() for f in csv_files)
            logger.info(f"Network file check: {network_file_check}")
            
            # If CSV files are present and task mentions analysis keywords, prioritize CSV analysis
            analysis_keywords = ['analyze', 'analysis', 'total', 'sales', 'median', 'correlation', 'chart', 'plot']
            analysis_keyword_match = any(keyword in task_lower for keyword in analysis_keywords)
            logger.info(f"CSV analysis keyword match: {analysis_keyword_match}")
            
            # Only use network analysis if specific network files OR (network keywords AND no general analysis keywords)
            network_keyword_check = any(keyword in task_lower for keyword in NETWORK_ANALYSIS_KEYWORDS)
            logger.info(f"Network analysis checks - file: {network_file_check}, keywords: {network_keyword_check}")
            
            if network_file_check or (network_keyword_check and not analysis_keyword_match):
                logger.info("Detected network analysis workflow")
                return "network_analysis"
            
            if analysis_keyword_match:
                logger.info("Detected CSV analysis workflow")
                return "csv_analysis"
    
    # Check for web scraping keywords
    web_scraping_keywords = ['scrape', 'extract', 'wikipedia', 'website', 'url', 'http', 'web', 'crawl', 'parse']
    if any(keyword in task_lower for keyword in web_scraping_keywords):
        logger.info("Detected web scraping workflow")
        return "multi_step_web_scraping"
    
    # Default fallback
    logger.info(f"No specific workflow detected, using default: {default_workflow}")
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


def save_questions_file(filename: str, content: bytes) -> str:
    """Save questions file to temporary directory"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = filename.replace(".txt", "")
    temp_filename = f"{base_name}_{timestamp}.txt"
    temp_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_path, "wb") as f:
        f.write(content)
    
    return temp_path

def save_csv_file(filename: str, content: bytes) -> str:
    """Save CSV file to temporary directory"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = filename.replace(".csv", "")
    temp_filename = f"{base_name}_{timestamp}.csv"
    temp_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_path, "wb") as f:
        f.write(content)
    
    return temp_path

def save_other_file(filename: str, content: bytes) -> str:
    """Save other file types to temporary directory"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name, ext = os.path.splitext(filename)
    temp_filename = f"{name}_{timestamp}{ext}"
    temp_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_path, "wb") as f:
        f.write(content)
    
    return temp_path


def detect_workflow_type(task_description: str, additional_files: dict = None) -> str:
    """Detect workflow type using LLM and fallback methods"""
    logger.info("Starting LLM-based workflow detection...")
    logger.info(f"Additional files provided to detect_workflow_type: {list(additional_files.keys()) if additional_files else 'None'}")
    
    # First try LLM detection
    try:
        import asyncio
        if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
            # We're in an async context, but need to call sync function
            loop = asyncio.get_event_loop()
            llm_result = loop.run_until_complete(detect_workflow_type_llm(task_description))
        else:
            # Not in async context
            llm_result = asyncio.run(detect_workflow_type_llm(task_description))
        logger.info(f"LLM detected workflow type: {llm_result}")
        logger.info(f"LLM detected workflow: {llm_result}")
        
        # Apply fallback detection with file info
        logger.info("Applying fallback workflow detection with file info...")
        enhanced_result = detect_workflow_type_fallback(task_description, llm_result, additional_files or {})
        logger.info(f"Enhanced detection changed workflow to: {enhanced_result}")
        logger.info(f"Final detected workflow: {enhanced_result}")
        
        return enhanced_result
        
    except Exception as e:
        logger.warning(f"LLM workflow detection failed: {e}, using fallback")
        fallback_result = detect_workflow_type_fallback(task_description, "data_analysis", additional_files or {})
        logger.info(f"Fallback detected workflow: {fallback_result}")
        return fallback_result

def prepare_workflow_input(
    task_description: str,
    questions: str,
    additional_files: Dict[str, Any],
    workflow_type: str,
    file_content: str
) -> Dict[str, Any]:
    """Prepare input for workflow execution"""
    
    # Extract basic parameters
    parameters = {
        "file_content_length": len(file_content) if file_content else 0,
        "content_type": "text/plain",
        "output_format": "base64_data_uri",
        "max_size": 100000
    }
    
    workflow_input = {
        "task_description": task_description,
        "questions": questions,
        "additional_files": additional_files,
        "processed_files_info": {},
        "workflow_type": workflow_type,
        "parameters": parameters,
        "output_requirements": {
            "format": "json",
            "include_charts": True,
            "max_chart_size": 100000  # 100KB
        }
    }
    
    return workflow_input


@app.post("/api/")
async def analyze_data(
    request: Request,
    questions_txt: UploadFile = File(..., alias="questions.txt")
):
    """Main API endpoint for data analysis workflows."""
    task_id = str(uuid.uuid4())
    logger.info(f"Starting synchronous task {task_id}")
    
    try:
        # Log ALL request information
        logger.info(f"=== FULL REQUEST DEBUG INFO ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request URL: {request.url}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Content-Type: {request.headers.get('content-type', 'Not set')}")
        
        # Log form data if available
        try:
            form = await request.form()
            logger.info(f"Form keys: {list(form.keys())}")
            for key, value in form.items():
                if hasattr(value, 'filename'):
                    logger.info(f"Form field '{key}': FILE - filename='{value.filename}', content_type='{getattr(value, 'content_type', 'unknown')}'")
                else:
                    logger.info(f"Form field '{key}': TEXT - value='{str(value)[:200]}{'...' if len(str(value)) > 200 else ''}'")
        except Exception as e:
            logger.warning(f"Could not parse form data: {e}")
        
        # Log FastAPI parsed parameters
        logger.info(f"FastAPI questions_txt: filename='{questions_txt.filename if questions_txt else 'None'}', content_type='{questions_txt.content_type if questions_txt else 'None'}'")
        
        logger.info(f"=== END REQUEST DEBUG INFO ===")
        
        logger.info(f"Received questions_txt filename: {questions_txt.filename}")
        # Extract additional files from form data (flexible field names)
        form_data = await request.form()
        additional_files_from_form = []
        
        logger.info(f"Extracting additional files from form data...")
        for field_name, field_value in form_data.items():
            if field_name != "questions.txt" and hasattr(field_value, 'filename'):
                # This is a file field
                logger.info(f"Found additional file: {field_value.filename}, Content-Type: {getattr(field_value, 'content_type', 'unknown')}")
                additional_files_from_form.append(field_value)
        
        logger.info(f"Received {len(additional_files_from_form)} additional files: {[f.filename for f in additional_files_from_form]}")
        
        # Read questions.txt content
        questions_content = await questions_txt.read()
        logger.info(f"Processed questions.txt with {len(questions_content)} characters")
        
        # Save questions.txt to /tmp
        questions_file_path = save_questions_file(questions_txt.filename, questions_content)
        logger.info(f"Saved questions.txt to {questions_file_path}")
        
        # Process additional files
        additional_files = {}
        logger.info(f"Processing {len(additional_files_from_form)} additional files...")
        for file in additional_files_from_form:
            logger.info(f"Processing file: {file.filename}, Content-Type: {file.content_type}")
            file_content = await file.read()
            logger.info(f"Read {len(file_content)} bytes from {file.filename}")
            
            if file.filename.endswith('.csv'):
                logger.info(f"Processing CSV file: {file.filename}")
                # Save CSV file to temporary directory
                csv_file_path = save_csv_file(file.filename, file_content)
                additional_files[csv_file_path] = {"type": "csv", "original_name": file.filename}
                logger.info(f"Processed CSV file: {file.filename}")
                logger.info(f"Saved {file.filename} to {csv_file_path}")
                
                # Verify file was saved
                if os.path.exists(csv_file_path):
                    file_size = os.path.getsize(csv_file_path)
                    logger.info(f"Verified: {csv_file_path} exists with size {file_size} bytes")
                else:
                    logger.error(f"Error: {csv_file_path} was not saved properly")
            else:
                # Save other file types to temporary directory
                other_file_path = save_other_file(file.filename, file_content)
                additional_files[other_file_path] = {"type": "other", "original_name": file.filename}
        
        # Use questions content as task description
        task_description = questions_content.decode('utf-8')
        logger.info(f"Using task description from questions.txt: {task_description[:200]}...")
        
        # Detect workflow type
        logger.info("Starting LLM-based workflow detection...")
        workflow_type = detect_workflow_type(task_description, additional_files)
        logger.info(f"Detected workflow type: {workflow_type}")
        logger.info(f"Task description length: {len(task_description)} chars")
        logger.info(f"Available files: {list(additional_files.keys())}")
        
        # Prepare workflow input
        workflow_input = prepare_workflow_input(
            task_description=task_description,
            questions=task_description,
            additional_files=additional_files,
            workflow_type=workflow_type,
            file_content=task_description
        )
        logger.info(f"Workflow input prepared with {len(workflow_input)} keys")
        logger.info(f"Additional files in workflow_input: {list(workflow_input.get('additional_files', {}).keys())}")
        logger.info(f"Workflow parameters: {workflow_input.get('parameters', {})}")
        
        # Execute workflow synchronously
        logger.info(f"Processing task {task_id} synchronously with workflow: {workflow_type}")
        result = await execute_workflow_sync(workflow_type, workflow_input, task_id)
        
        # Add processing information to result
        if isinstance(result, dict):
            result.update({
                "processing_info": {
                    "questions_file": questions_txt.filename,
                    "additional_files": [f.filename for f in additional_files_from_form],
                    "workflow_auto_detected": workflow_type
                }
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "workflow_type": "error"
            }
        )


async def execute_workflow_sync(
    workflow_type: str, workflow_input: Dict[str, Any], task_id: str
) -> Dict[str, Any]:  # Execute workflow synchronously
    """Execute workflow synchronously with enhanced error handling"""
    try:
        logger.info(f"[{task_id}] Executing workflow {workflow_type}")
        logger.info(f"[{task_id}] Workflow input keys: {list(workflow_input.keys())}")
        logger.info(f"[{task_id}] Additional files: {list(workflow_input.get('additional_files', {}).keys())}")
        
        if orchestrator is None:
            logger.warning(f"[{task_id}] No orchestrator available, cannot execute workflows")
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
            logger.info(f"[{task_id}] Executing workflow {workflow_type} with orchestrator")
            logger.info(f"[{task_id}] Available workflows: {list(orchestrator.workflows.keys())}")

            if workflow_type not in orchestrator.workflows:
                logger.warning(
                    f"[{task_id}] Workflow {workflow_type} not found, available: {list(orchestrator.workflows.keys())}"
                )
                return {
                    "workflow_type": workflow_type,
                    "status": "error",
                    "message": f"Workflow {workflow_type} not found",
                    "available_workflows": list(orchestrator.workflows.keys()),
                }

            logger.info(f"[{task_id}] Calling orchestrator.execute_workflow...")
            result = await orchestrator.execute_workflow(workflow_type, workflow_input)
            logger.info(f"[{task_id}] Workflow {workflow_type} executed successfully")
            logger.info(f"[{task_id}] Result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"[{task_id}] Result keys: {list(result.keys())}")
                if "analysis_result" in result:
                    logger.info(f"[{task_id}] Analysis result preview: {str(result['analysis_result'])[:200]}...")
            return result
    except Exception as e:
        logger.error(f"[{task_id}] Error executing workflow {workflow_type}: {e}")
        logger.error(f"[{task_id}] Exception type: {type(e)}")
        import traceback

        logger.error(f"[{task_id}] Traceback: {traceback.format_exc()}")
        raise e
