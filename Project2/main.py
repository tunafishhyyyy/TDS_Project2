from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import json
import io
from typing import Optional, Union
import uuid
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import sys
import os
import re
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import API_VERSION, TIMEOUT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chains')
)
try:
    from chains.workflows import AdvancedWorkflowOrchestrator
    orchestrator = AdvancedWorkflowOrchestrator()
    logger.info("Successfully initialized AdvancedWorkflowOrchestrator")
    logger.info(
        f"Available workflows: {list(orchestrator.workflows.keys())}"
    )
except Exception as e:
    logger.error(f"Could not import or initialize workflows: {e}")
    # Try to create a minimal orchestrator with just the fallback workflow
    try:
        from chains.workflows import ModularWebScrapingWorkflow
        from chains.base import WorkflowOrchestrator
        
        class MinimalOrchestrator(WorkflowOrchestrator):
            def __init__(self):
                super().__init__()
                self.llm = None
                self.workflows = {
                    "multi_step_web_scraping": ModularWebScrapingWorkflow()
                }
        
        orchestrator = MinimalOrchestrator()
        logger.info("Created minimal orchestrator with fallback workflows")
    except Exception as e2:
        logger.error(f"Could not create minimal orchestrator: {e2}")
        orchestrator = None

app = FastAPI(  # FastAPI app instance
    title="Data Analysis API",
    description=(
        "An API that uses LLMs to source, prepare, analyze, and "
        "visualize any data with multi-modal support."
    ),
    version="2.0.0"
)
app.mount(
    "/static", StaticFiles(directory="."), name="static"
)  # Mount static files

# Health check and info endpoints
@app.get("/")
async def root():  # Root endpoint with API info
    """Root endpoint with API information"""
    return {
        "message": "Data Analysis API v2.0",
        "description": "Multi-modal data analysis with LLM-powered workflows",
        "features": [
            "Multiple file upload support",
            "Synchronous processing",
            "LLM-based workflow detection",
            "Multi-modal analysis (text, image, code)",
            "12+ specialized workflows"
        ],
        "endpoints": {
            "main": "/api/ (POST - multiple files with required "
                   "questions.txt)",
            "health": "/health (GET)",
            "docs": "/docs (GET - Swagger UI)"
        },
        "status": "operational"
    }

@app.get("/health")
async def health_check():  # Health check endpoint
    """Health check endpoint"""
    orchestrator_status = "available" if orchestrator else "unavailable"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "orchestrator": orchestrator_status,
        "workflows_available": (
            len(orchestrator.workflows) if orchestrator else 0
        ),
        "version": "2.0.0"
    }
# Pydantic models for request/response
class TaskRequest(BaseModel):  # Model for analysis task requests
    """Model for analysis task requests"""
    task_description: str = Field(
        ..., description="Description of the analysis task"
    )
    workflow_type: Optional[str] = Field(
        "data_analysis", description="Type of workflow to execute"
    )
    data_source: Optional[str] = Field(
        None, description="Optional data source information"
    )
    dataset_info: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Dataset characteristics and metadata"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional parameters for the task"
    )
    priority: Optional[str] = Field(
        "normal", description="Task priority: low, normal, high"
    )
    include_modeling: Optional[bool] = Field(
        False, description="Include predictive modeling in analysis"
    )
    target_audience: Optional[str] = Field(
        "technical team", description="Target audience for reports"
    )

class WorkflowRequest(BaseModel):  # Model for specific workflow requests
    """Model for specific workflow requests"""
    workflow_type: str = Field(
        ..., description="Type of workflow to execute"
    )
    input_data: Dict[str, Any] = Field(
        ..., description="Input data for the workflow"
    )

class MultiStepWorkflowRequest(BaseModel):
    # Model for multi-step workflow requests
    """Model for multi-step workflow requests"""
    steps: List[Dict[str, Any]] = Field(
        ..., description="List of workflow steps to execute"
    )
    pipeline_type: Optional[str] = Field(
        "custom", description="Type of pipeline"
    )

class TaskResponse(BaseModel):  # Model for task response
    """Model for task response"""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    task_details: Dict[str, Any] = Field(
        ..., description="Details of the submitted task"
    )
    created_at: str = Field(..., description="Task creation timestamp")
    workflow_result: Optional[Dict[str, Any]] = Field(
        None, description="LangChain workflow execution result"
    )


def extract_output_requirements(
    task_description: str
) -> Dict[str, Any]:  # Extract output requirements from task description
    """Extract specific output requirements from task description"""
    requirements = {
        "format": "json",
        "include_visualizations": False,
        "response_time_limit": "3 minutes"
    }
    
    task_lower = task_description.lower()
    
    # Check for visualization requirements
    if any(
        keyword in task_lower
        for keyword in [
            'plot', 'chart', 'graph', 'visualization', 'base64', 'data uri'
        ]
    ):
        requirements["include_visualizations"] = True
        requirements["visualization_format"] = "base64_data_uri"
        requirements["max_size"] = "100000 bytes"
    
    # Check for specific format requirements
    if "json" in task_lower:
        requirements["format"] = "json"
    elif "csv" in task_lower:
        requirements["format"] = "csv"
    elif "table" in task_lower:
        requirements["format"] = "table"
    
    return requirements

async def detect_workflow_type_llm(
    task_description: str, 
    default_workflow: str = "data_analysis"
) -> str:  # LLM-based workflow type detection
    """
    Use LLM prompting to determine the workflow type based on the 
    input task description
    """
    if not task_description:
        return default_workflow
    
    logger.info(
        f"Detecting workflow type for task: {task_description[:100]}..."
    )
    
    try:
        if orchestrator and orchestrator.llm:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains import LLMChain
            
            workflow_detection_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert workflow classifier for data analysis tasks. 
                Analyze the task description and classify it into one of these workflow types:
                
                - data_analysis: General data analysis and recommendations
                - image_analysis: Image processing, computer vision
                - code_generation: Generate Python code for analysis
                - exploratory_data_analysis: Comprehensive EDA planning
                - predictive_modeling: Machine learning model development
                - data_visualization: Creating charts, graphs, visualizations
                - web_scraping: Extract data from websites
                - multi_step_web_scraping: Multi-step web scraping with analysis
                - database_analysis: SQL analysis using databases
                - statistical_analysis: Statistical analysis, correlation, regression
                - text_analysis: Natural language processing and text analytics
                
                IMPORTANT: If the task involves web scraping AND multiple steps 
                (scraping, cleaning, analysis, visualization, answering questions), 
                use 'multi_step_web_scraping'. If it's just basic web scraping 
                without complex analysis, use 'web_scraping'.
                
                Return ONLY the workflow type name, nothing else."""),
                ("human", "Task: {task_description}")
            ])
            
            chain = LLMChain(llm=orchestrator.llm, prompt=workflow_detection_prompt)
            result = chain.run(task_description=task_description)
            
            # Clean and validate the result
            detected_workflow = result.strip().lower()
            
            # List of valid workflows (generalized)
            valid_workflows = [
                "data_analysis", "image_analysis", "code_generation", 
                "exploratory_data_analysis", "predictive_modeling", 
                "data_visualization", "web_scraping", 
                "multi_step_web_scraping", "database_analysis", 
                "statistical_analysis", "text_analysis"
            ]
            
            if detected_workflow in valid_workflows:
                logger.info(f"LLM detected workflow type: {detected_workflow}")
                return detected_workflow
            else:
                logger.warning(
                    f"LLM returned invalid workflow: {detected_workflow}, "
                    f"using fallback"
                )
                return detect_workflow_type_fallback(
                    task_description, default_workflow
                )
                
        else:
            logger.warning(
                "LLM not available, using fallback workflow detection"
            )
            return detect_workflow_type_fallback(
                task_description, default_workflow
            )
            
    except Exception as e:
        logger.error(f"Error in LLM workflow detection: {e}")
        return detect_workflow_type_fallback(task_description, default_workflow)

def detect_workflow_type_fallback(
    task_description: str, 
    default_workflow: str = "data_analysis"
) -> str:  # Fallback keyword-based workflow detection
    """
    Fallback keyword-based workflow detection when LLM is not available
    """
    if not task_description:
        return default_workflow
    
    task_lower = task_description.lower()
    
    # Web scraping patterns - PRIORITIZE BEFORE IMAGE ANALYSIS
    scraping_keywords = [
        'scrape', 'extract', 'data from', 'table from', 'list of',
        'website', 'url', 'html', 'web page', 'site'
    ]
    if any(keyword in task_lower for keyword in scraping_keywords):
        # Check if it involves multiple steps 
        # (cleaning, analysis, visualization, questions)
        multi_step_keywords = [
            'clean', 'plot', 'top', 'rank', 'total', 'answer',
            'question', 'extract', 'analyze', 'visualization'
        ]
        if any(keyword in task_lower for keyword in multi_step_keywords):
            return "multi_step_web_scraping"
        else:
            return "multi_step_web_scraping"    # Image analysis patterns
    image_keywords = [
        'image', 'photo', 'picture', 'visual', 'png', 'jpg', 'jpeg',
        'computer vision', 'image processing'
    ]
    if any(keyword in task_lower for keyword in image_keywords):
        return "image_analysis"
    
    # Text analysis patterns  
    text_keywords = [
        'text analysis', 'nlp', 'sentiment', 'language', 'document',
        'natural language processing', 'text mining'
    ]
    if any(keyword in task_lower for keyword in text_keywords):
        return "text_analysis"
    
    # Legal/Court data patterns - map to general data analysis
    legal_keywords = [
        'court', 'judgment', 'legal', 'case', 'disposal', 'judge',
        'cnr', 'ecourts', 'law', 'litigation'
    ]
    if any(keyword in task_lower for keyword in legal_keywords):
        return "data_analysis"
    
    # Statistical analysis patterns
    stats_keywords = [
        'correlation', 'regression', 'statistical', 'trend', 'slope',
        'analysis', 'statistics', 'hypothesis'
    ]
    if any(keyword in task_lower for keyword in stats_keywords):
        return "statistical_analysis"
    
    # Database analysis patterns
    db_keywords = [
        'sql', 'duckdb', 'database', 'query', 'parquet', 's3://',
        'mysql', 'postgresql', 'sqlite'
    ]
    if any(keyword in task_lower for keyword in db_keywords):
        return "database_analysis"
    
    # Data visualization patterns
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualization', 'scatterplot',
        'base64', 'data uri', 'histogram', 'bar chart'
    ]
    if any(keyword in task_lower for keyword in viz_keywords):
        return "data_visualization"
    
    # Exploratory data analysis patterns
    eda_keywords = [
        'explore', 'eda', 'exploratory', 'distribution', 'summary',
        'describe', 'overview'
    ]
    if any(keyword in task_lower for keyword in eda_keywords):
        return "exploratory_data_analysis"
    
    # Predictive modeling patterns
    ml_keywords = [
        'predict', 'model', 'machine learning', 'ml', 'forecast',
        'classification', 'clustering', 'neural network'
    ]
    if any(keyword in task_lower for keyword in ml_keywords):
        return "predictive_modeling"
    
    # Code generation patterns
    code_keywords = [
        'generate code', 'python code', 'script', 'function',
        'programming', 'code'
    ]
    if any(keyword in task_lower for keyword in code_keywords):
        return "code_generation"
    
    # Generic web scraping patterns
    web_keywords = [
        'scrape', 'extract', 'web', 'html', 'website'
    ]
    if any(keyword in task_lower for keyword in web_keywords):
        return "multi_step_web_scraping"
    
    return default_workflow

def prepare_workflow_parameters(
    task_description: str, 
    workflow_type: str, 
    file_content: str = None
) -> Dict[str, Any]:  # Prepare parameters for workflow execution
    """
    Prepare specific parameters based on workflow type and task description
    """
    params = {}
    task_lower = task_description.lower() if task_description else ""
    
    # Generic URL extraction
    if "http" in task_lower:
        import re
        urls = re.findall(r'https?://[^\s]+', task_description)
        params["target_urls"] = urls
    
    # Generic data type detection
    if any(kw in task_lower for kw in ["gross", "revenue", "profit"]):
        params["data_type"] = "financial"
    elif any(kw in task_lower for kw in ["rank", "position", "top"]):
        params["data_type"] = "ranking"
    
    # Database parameters (generic)
    if "s3://" in task_lower:
        import re
        s3_paths = re.findall(r's3://[^\s]+', task_description)
        params["s3_paths"] = s3_paths
    if any(kw in task_lower for kw in ["duckdb", "sql", "database"]):
        params["database_type"] = "sql_based"
    if "parquet" in task_lower:
        params["file_format"] = "parquet"
    
    # Visualization parameters (generic)
    if "scatterplot" in task_lower or "scatter" in task_lower:
        params["chart_type"] = "scatter"
    if "regression" in task_lower:
        params["include_regression"] = True
    if "base64" in task_lower or "data uri" in task_lower:
        params["output_format"] = "base64_uri"
        params["max_size"] = 100000  # 100KB limit
    
    # File content analysis
    if file_content:
        params["file_content_length"] = len(file_content)
        content_stripped = file_content.strip()
        if content_stripped.startswith(('{', '[')):
            params["content_type"] = "json"
        elif '\t' in file_content or ',' in file_content:
            params["content_type"] = "csv"
        else:
            params["content_type"] = "text"
    
    return params

def extract_output_requirements(task_description: str) -> Dict[str, Any]:
    """Extract output format requirements from task description"""
    """
    Extract specific output format requirements from task description
    """
    requirements = {}
    task_lower = task_description.lower() if task_description else ""
    
    # Time constraints
    if "3 minutes" in task_lower or "within 3" in task_lower:
        requirements["time_limit"] = 180  # seconds
    
    # Format requirements
    if "json" in task_lower:
        requirements["format"] = "json"
    if "base64" in task_lower:
        requirements["encoding"] = "base64"
    if "data uri" in task_lower:
        requirements["uri_format"] = True
    if "under 100,000" in task_lower or "100,000 bytes" in task_lower:
        requirements["size_limit"] = 100000
    
    # Response structure
    if "array" in task_lower:
        requirements["structure"] = "array"
    elif "object" in task_lower:
        requirements["structure"] = "object"
    
    return requirements

@app.post("/api/")
async def analyze_data(
    questions_txt: UploadFile = File(
        ..., description="Required questions.txt file"
    ),
    files: List[UploadFile] = File(
        default=[], description="Optional additional files"
    )
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
        if not (
            questions_txt.filename.lower().endswith('.txt') or 
            'question' in questions_txt.filename.lower()
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "questions.txt file is required and must be named "
                    "appropriately (must contain 'question' in filename)"
                )
            )
        
        questions_content = await questions_txt.read()
        questions_text = questions_content.decode('utf-8')
        logger.info(f"Processed questions.txt with {len(questions_text)} characters")
        
        # Process additional files
        processed_files = {}
        file_contents = {}
        
        for file in files:
            if file.filename:
                content = await file.read()
                try:
                    file_text = content.decode('utf-8')
                    file_contents[file.filename] = file_text
                    logger.info(f"Processed text file: {file.filename}")
                except UnicodeDecodeError:
                    # Handle binary files (images, etc.)
                    file_contents[file.filename] = (
                        f"Binary file: {file.filename} ({len(content)} bytes)"
                    )
                    logger.info(
                        f"Processed binary file: {file.filename} "
                        f"({len(content)} bytes)"
                    )
                
                processed_files[file.filename] = {
                    "content_type": file.content_type,
                    "size": len(content),
                    "is_text": file.filename.endswith((
                        '.txt', '.csv', '.json', '.md'
                    ))
                }
        
        # Use questions as task description (content of questions.txt)
        task_description = questions_text
        
        # Intelligent workflow type detection using LLM
        detected_workflow = await detect_workflow_type_llm(
            task_description, "multi_step_web_scraping"
        )
        logger.info(f"Detected workflow: {detected_workflow}")
        logger.info(f"Task description: {task_description[:200]}...")
        
        # Prepare enhanced workflow input
        workflow_input = {
            "task_description": task_description,
            "questions": questions_text,
            "additional_files": file_contents,
            "processed_files_info": processed_files,
            "workflow_type": detected_workflow,
            "parameters": prepare_workflow_parameters(
                task_description, detected_workflow, questions_text
            ),
            "output_requirements": extract_output_requirements(
                task_description
            )
        }
        
        logger.info(f"Workflow input prepared with {len(workflow_input)} keys")
        logger.info(f"Additional files: {list(file_contents.keys())}")
        
        # Execute workflow synchronously (always within 3 minutes)
        logger.info(
            f"Processing task {task_id} synchronously with "
            f"workflow: {detected_workflow}"
        )
        
        try:
            logger.info(f"Starting workflow execution for {detected_workflow}")
            result = await asyncio.wait_for(
                execute_workflow_sync(detected_workflow, workflow_input, task_id),
                timeout=180  # 3 minutes
            )
            
            logger.info(f"Task {task_id} completed successfully")
            logger.info(
                f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
            )
            
            return {
                "task_id": task_id,
                "status": "completed",
                "workflow_type": detected_workflow,
                "result": result,
                "processing_info": {
                    "questions_file": questions_txt.filename,
                    "additional_files": list(processed_files.keys()),
                    "workflow_auto_detected": True,
                    "processing_time": "synchronous"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out after 3 minutes")
            raise HTTPException(
                status_code=408, 
                detail=(
                    "Request timed out after 3 minutes. Please simplify "
                    "your request or try again."
                )
            )
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Processing failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

async def execute_workflow_sync(
    workflow_type: str, 
    workflow_input: Dict[str, Any], 
    task_id: str
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
                    "Configure OpenAI API key"
                ],
                "parameters_prepared": workflow_input.get("parameters", {}),
                "files_processed": list(workflow_input.get("additional_files", {}).keys())
            }
        else:
            logger.info(f"Executing workflow {workflow_type} with orchestrator")
            logger.info(f"Available workflows: {list(orchestrator.workflows.keys())}")
            
            if workflow_type not in orchestrator.workflows:
                logger.warning(
                    f"Workflow {workflow_type} not found, available: "
                    f"{list(orchestrator.workflows.keys())}"
                )
                return {
                    "workflow_type": workflow_type,
                    "status": "error",
                    "message": f"Workflow {workflow_type} not found",
                    "available_workflows": list(orchestrator.workflows.keys())
                }
            
            result = await orchestrator.execute_workflow(
                workflow_type, workflow_input
            )
            logger.info(
                f"Workflow {workflow_type} executed successfully for "
                f"task {task_id}"
            )
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


