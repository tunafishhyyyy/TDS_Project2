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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import API_VERSION, TIMEOUT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chains'))
try:
    from workflows import AdvancedWorkflowOrchestrator
    orchestrator = AdvancedWorkflowOrchestrator()
except ImportError:
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
            
            # If no task description provided, use file content as task
            if not task_description:
                task_description = f"Analyze the data from file: {file.filename}"
        
        # Validate input
        if not task_description and not file_content:
            raise HTTPException(
                status_code=400, 
                detail="Either task_description or file must be provided"
            )
        
        # Create task data
        task_data = {
            "task_id": task_id,
            "task_description": task_description,
            "workflow_type": workflow_type,
            "business_context": business_context,
            "file_name": file.filename if file else None,
            "file_content": file_content,
            "status": "received",
            "created_at": datetime.now().isoformat(),
            "result": None
        }
        
        # Store task
        tasks_storage[task_id] = task_data
        
        # Start background processing (simulate)
        asyncio.create_task(process_analysis_task(task_id))
        
        return {
            "message": "Task received and processing started",
            "task_id": task_id,
            "status": "processing",
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
