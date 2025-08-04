from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import API_VERSION, TIMEOUT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chains'))
try:
    from workflows import AdvancedWorkflowOrchestrator
    orchestrator = AdvancedWorkflowOrchestrator()
except ImportError as e:
    print(f"Warning: Could not import workflows: {e}")
    orchestrator = None

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Data Analysis API",
    description="An API that uses LLMs to source, prepare, analyze, and visualize any data.",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="."), name="static")
# Pydantic models for request/response
class TaskRequest(BaseModel):
    """Model for analysis task requests"""
    task_description: str = Field(..., description="Description of the analysis task")
    workflow_type: Optional[str] = Field("data_analysis", description="Type of workflow to execute")
    data_source: Optional[str] = Field(None, description="Optional data source information")
    dataset_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Dataset characteristics and metadata")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters for the task")
    priority: Optional[str] = Field("normal", description="Task priority: low, normal, high")
    include_modeling: Optional[bool] = Field(False, description="Include predictive modeling in analysis")
    target_audience: Optional[str] = Field("technical team", description="Target audience for reports")

class WorkflowRequest(BaseModel):
    """Model for specific workflow requests"""
    workflow_type: str = Field(..., description="Type of workflow to execute")
    input_data: Dict[str, Any] = Field(..., description="Input data for the workflow")

class MultiStepWorkflowRequest(BaseModel):
    """Model for multi-step workflow requests"""
    steps: List[Dict[str, Any]] = Field(..., description="List of workflow steps to execute")
    pipeline_type: Optional[str] = Field("custom", description="Type of pipeline")

class TaskResponse(BaseModel):
    """Model for task response"""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    task_details: Dict[str, Any] = Field(..., description="Details of the submitted task")
    created_at: str = Field(..., description="Task creation timestamp")
    workflow_result: Optional[Dict[str, Any]] = Field(None, description="LangChain workflow execution result")

# In-memory storage for background tasks (in production, use a proper database)
tasks_storage = {}

def detect_workflow_type(task_description: str, default_workflow: str = "data_analysis") -> str:
    """Intelligent workflow type detection based on task description"""
    if not task_description:
        return default_workflow
    
    task_lower = task_description.lower()
    
    # Legal/Court data patterns
    if any(keyword in task_lower for keyword in ['court', 'judgment', 'legal', 'case', 'disposal', 'judge', 'cnr', 'ecourts']):
        return "legal_data_analysis"
    
    # Wikipedia patterns
    if any(keyword in task_lower for keyword in ['wikipedia', 'wiki', 'scrape', 'list of', 'table from']):
        return "wikipedia_scraping"
    
    # Statistical analysis patterns
    if any(keyword in task_lower for keyword in ['correlation', 'regression', 'statistical', 'trend', 'slope']):
        return "statistical_analysis"
    
    # Database analysis patterns
    if any(keyword in task_lower for keyword in ['sql', 'duckdb', 'database', 'query', 'parquet', 's3://']):
        return "database_analysis"
    
    # Data visualization patterns
    if any(keyword in task_lower for keyword in ['plot', 'chart', 'graph', 'visualization', 'scatterplot', 'base64', 'data uri']):
        return "data_visualization"
    
    # Exploratory data analysis patterns
    if any(keyword in task_lower for keyword in ['explore', 'eda', 'exploratory', 'distribution', 'summary']):
        return "exploratory_data_analysis"
    
    # Predictive modeling patterns
    if any(keyword in task_lower for keyword in ['predict', 'model', 'machine learning', 'ml', 'forecast']):
        return "predictive_modeling"
    
    # Code generation patterns
    if any(keyword in task_lower for keyword in ['generate code', 'python code', 'script', 'function']):
        return "code_generation"
    
    # Web scraping patterns
    if any(keyword in task_lower for keyword in ['scrape', 'extract', 'web', 'html', 'website']):
        return "web_scraping"
    
    return default_workflow

def prepare_workflow_parameters(task_description: str, workflow_type: str, file_content: str = None) -> Dict[str, Any]:
    """Prepare workflow-specific parameters based on task and content"""
    parameters = {}
    
    if workflow_type == "legal_data_analysis":
        parameters.update({
            "court_context": "Indian High Courts",
            "dataset_info": {
                "type": "legal_judgments",
                "source": "ecourts.gov.in",
                "columns": ["court_code", "title", "description", "judge", "pdf_link", "cnr", "date_of_registration", "decision_date", "disposal_nature"]
            }
        })
    
    elif workflow_type == "wikipedia_scraping":
        # Extract Wikipedia URL if present
        url_match = re.search(r'https?://[^\s]+wikipedia[^\s]+', task_description)
        if url_match:
            parameters["wikipedia_url"] = url_match.group()
        
        parameters.update({
            "target_data_description": "table data extraction",
            "analysis_goals": "statistical analysis and visualization"
        })
    
    elif workflow_type == "statistical_analysis":
        parameters.update({
            "statistical_methods": "correlation, regression, trend analysis",
            "variables": ["rank", "peak", "revenue", "year"]
        })
    
    elif workflow_type == "database_analysis":
        parameters.update({
            "database_type": "DuckDB",
            "data_source": "S3/Parquet files",
            "schema_info": {
                "court_data": ["court_code", "title", "judge", "date_of_registration", "decision_date"]
            }
        })
    
    return parameters

def extract_output_requirements(task_description: str) -> Dict[str, Any]:
    """Extract specific output requirements from task description"""
    requirements = {
        "format": "json",
        "include_visualizations": False,
        "response_time_limit": "3 minutes"
    }
    
    task_lower = task_description.lower()
    
    # Check for visualization requirements
    if any(keyword in task_lower for keyword in ['plot', 'chart', 'graph', 'visualization', 'base64', 'data uri']):
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

def detect_workflow_type(task_description: str, default_workflow: str = "data_analysis") -> str:
    """
    Intelligently detect workflow type based on task description keywords
    """
    if not task_description:
        return default_workflow
    
    task_lower = task_description.lower()
    
    # Wikipedia-related tasks (Example 1: highest grossing films)
    if any(word in task_lower for word in ["wikipedia", "wiki", "scrape", "highest grossing", "films", "extract from web"]):
        return "wikipedia_analysis"
    
    # Database/SQL analysis tasks (Example 2: Indian High Court dataset)
    if any(word in task_lower for word in ["sql", "duckdb", "database", "query", "table", "parquet", "s3://", "high court", "judgments", "metadata"]):
        return "database_analysis"
    
    # General web scraping tasks
    if any(word in task_lower for word in ["scrape", "scraping", "web scraping", "extract data", "crawl", "html"]):
        return "web_scraping"
    
    # Tasks requiring visualization (scatterplot, regression line, base64 encoding)
    if any(word in task_lower for word in ["scatterplot", "regression line", "base64", "data uri", "plot", "chart", "graph", "visualize", "visualization", "dashboard"]):
        return "data_visualization"
    
    # Exploratory Data Analysis
    if any(word in task_lower for word in ["eda", "exploratory", "explore", "summary statistics", "data exploration", "correlation", "rank", "peak"]):
        return "exploratory_data_analysis"
    
    # Predictive modeling
    if any(word in task_lower for word in ["predict", "model", "machine learning", "ml", "forecast", "classification", "regression", "slope"]):
        return "predictive_modeling"
    
    # Code generation
    if any(word in task_lower for word in ["code", "script", "python", "generate code", "programming"]):
        return "code_generation"
    
    # Report generation
    if any(word in task_lower for word in ["report", "summary", "document", "presentation"]):
        return "report_generation"
    
    return default_workflow

def prepare_workflow_parameters(task_description: str, workflow_type: str, file_content: str = None) -> Dict[str, Any]:
    """
    Prepare specific parameters based on workflow type and task description
    """
    params = {}
    task_lower = task_description.lower() if task_description else ""
    
    if workflow_type == "wikipedia_analysis":
        # Extract Wikipedia URLs or topics
        if "wikipedia.org" in task_lower:
            import re
            urls = re.findall(r'https?://[^\s]+wikipedia[^\s]+', task_description)
            params["target_urls"] = urls
        if "highest grossing" in task_lower:
            params["data_type"] = "movie_revenue"
            params["table_indicators"] = ["rank", "peak", "film", "gross"]
    
    elif workflow_type == "database_analysis":
        # DuckDB and S3 parameters
        if "s3://" in task_lower:
            import re
            s3_paths = re.findall(r's3://[^\s]+', task_description)
            params["s3_paths"] = s3_paths
        if "duckdb" in task_lower:
            params["database_type"] = "duckdb"
        if "parquet" in task_lower:
            params["file_format"] = "parquet"
        if "high court" in task_lower:
            params["domain"] = "legal_judgments"
            params["table_schema"] = {
                "court_code": "VARCHAR",
                "title": "VARCHAR", 
                "judge": "VARCHAR",
                "decision_date": "DATE",
                "disposal_nature": "VARCHAR"
            }
    
    elif workflow_type == "data_visualization":
        # Visualization parameters
        if "scatterplot" in task_lower:
            params["chart_type"] = "scatter"
        if "regression line" in task_lower:
            params["include_regression"] = True
        if "base64" in task_lower or "data uri" in task_lower:
            params["output_format"] = "base64_uri"
            params["max_size"] = 100000  # 100KB limit
        if "dotted red" in task_lower:
            params["line_style"] = {"color": "red", "style": "dotted"}
    
    # Add file content analysis
    if file_content:
        params["file_content_length"] = len(file_content)
        if file_content.strip().startswith('{') or file_content.strip().startswith('['):
            params["content_type"] = "json"
        elif '\t' in file_content or ',' in file_content:
            params["content_type"] = "csv"
        else:
            params["content_type"] = "text"
    
    return params

def extract_output_requirements(task_description: str) -> Dict[str, Any]:
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
    file: Optional[UploadFile] = File(None),
    task_description: Optional[str] = Form(None),
    workflow_type: Optional[str] = Form("data_analysis"),
    business_context: Optional[str] = Form(None)
):
    """
    Main endpoint that accepts either file upload or direct task description.
    
    - **file**: Optional file upload (txt, csv, json, etc.)
    - **task_description**: Description of the analysis task
    - **workflow_type**: Type of workflow to execute
    - **business_context**: Additional business context
    """
    try:
        task_id = str(uuid.uuid4())
        # Handle file upload
        file_content = None
        if file:
            content = await file.read()
            file_content = content.decode('utf-8')
            if not task_description:
                task_description = f"Analyze the data from file: {file.filename}"
        # Validate input
        if not task_description and not file_content:
            raise HTTPException(
                status_code=400,
                detail="Either task_description or file must be provided"
            )
        # Intelligent workflow type detection based on task description
        detected_workflow = detect_workflow_type(task_description, workflow_type)
        
        # Prepare enhanced workflow input for LangChain orchestrator
        workflow_input = {
            "task_description": task_description,
            "file_name": file.filename if file else None,
            "file_content": file_content,
            "workflow_type": detected_workflow,
            "business_context": business_context,
            "parameters": prepare_workflow_parameters(task_description, detected_workflow, file_content),
            "output_requirements": extract_output_requirements(task_description)
        }
        # Store initial task
        tasks_storage[task_id] = {
            "task_id": task_id,
            "task_description": task_description,
            "workflow_type": detected_workflow,
            "original_workflow_type": workflow_type,
            "business_context": business_context,
            "file_name": file.filename if file else None,
            "file_content": file_content,
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "workflow_detected": detected_workflow != workflow_type
        }
        
        # Run LangChain workflow in background
        async def run_workflow():
            try:
                if orchestrator is None:
                    # Fallback when LangChain is not available
                    result = {
                        "workflow_type": detected_workflow,
                        "status": "completed_fallback",
                        "message": "LangChain orchestrator not available, using fallback response",
                        "task_analysis": f"Detected workflow: {detected_workflow} for task: {task_description}",
                        "recommendations": ["Set up LangChain integration", "Install required dependencies", "Configure OpenAI API key"],
                        "parameters_prepared": workflow_input.get("parameters", {}),
                        "output_requirements": workflow_input.get("output_requirements", {})
                    }
                else:
                    result = await orchestrator.execute_workflow(detected_workflow, workflow_input)
                
                tasks_storage[task_id]["result"] = result
                tasks_storage[task_id]["status"] = "completed"
                tasks_storage[task_id]["completed_at"] = datetime.now().isoformat()
            except Exception as e:
                tasks_storage[task_id]["status"] = "failed"
                tasks_storage[task_id]["error"] = str(e)
        
        asyncio.create_task(run_workflow())
        
        return {
            "message": "Task received and processing started",
            "task_id": task_id,
            "status": "processing",
            "workflow_type": detected_workflow,
            "workflow_auto_detected": detected_workflow != workflow_type,
            "endpoint_info": {
                "status_check": f"/api/tasks/{task_id}/status",
                "estimated_time": "3 minutes"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the status of a specific task"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks_storage[task_id]

@app.post("/api/analyze")
async def analyze_task_json(task: dict):
    """
    Legacy endpoint for JSON-based requests
    """
    task_id = str(uuid.uuid4())
    
    task_data = {
        "task_id": task_id,
        "task_description": task.get("task_description", ""),
        "workflow_type": task.get("workflow_type", "data_analysis"),
        "business_context": task.get("business_context", ""),
        "file_name": None,
        "file_content": None,
        "status": "received",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "original_task": task
    }
    
    tasks_storage[task_id] = task_data
    asyncio.create_task(process_analysis_task(task_id))
    
    return {
        "message": "Task received", 
        "task_id": task_id,
        "task": task,
        "status": "processing"
    }

async def process_analysis_task(task_id: str):
    """Background task processor (simulated)"""
    try:
        # Update status to processing
        tasks_storage[task_id]["status"] = "processing"
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Simulate analysis result
        task_data = tasks_storage[task_id]
        
        result = {
            "analysis_type": task_data["workflow_type"],
            "summary": f"Analysis completed for: {task_data['task_description']}",
            "insights": [
                "Data quality assessment completed",
                "Statistical analysis performed",
                "Visualizations generated"
            ],
            "recommendations": [
                "Consider additional data sources",
                "Implement data validation checks",
                "Schedule regular analysis updates"
            ]
        }
        
        if task_data["file_content"]:
            result["file_analysis"] = {
                "file_name": task_data["file_name"],
                "content_preview": task_data["file_content"][:200] + "..." if len(task_data["file_content"]) > 200 else task_data["file_content"],
                "content_length": len(task_data["file_content"]),
                "content_type": "text"
            }
        
        # Update task with result
        tasks_storage[task_id]["status"] = "completed"
        tasks_storage[task_id]["result"] = result
        tasks_storage[task_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        tasks_storage[task_id]["status"] = "failed"
        tasks_storage[task_id]["error"] = str(e)
