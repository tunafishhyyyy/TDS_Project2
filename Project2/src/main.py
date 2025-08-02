from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Union, Dict, Any, Optional, List
from datetime import datetime
import json
import uuid
import asyncio

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_VERSION, TIMEOUT

# Import LangChain workflows
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chains'))
from workflows import AdvancedWorkflowOrchestrator

app = FastAPI(
    title="Data Analysis API with LangChain",
    description="API for processing data analysis tasks using LangChain workflows",
    version=API_VERSION
)

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
    created_at: datetime = Field(..., description="Task creation timestamp")
    workflow_result: Optional[Dict[str, Any]] = Field(None, description="LangChain workflow execution result")

# In-memory storage for demonstration (replace with database in production)
task_storage = {}

# Initialize LangChain workflow orchestrator
orchestrator = AdvancedWorkflowOrchestrator()

async def execute_workflow_task(task_id: str, workflow_type: str, input_data: Dict[str, Any]):
    """Background task to execute LangChain workflow"""
    try:
        # Update task status to processing
        if task_id in task_storage:
            task_storage[task_id]["status"] = "processing"
            task_storage[task_id]["updated_at"] = datetime.now()
        
        # Execute the workflow
        result = await orchestrator.execute_workflow(workflow_type, input_data)
        
        # Update task with results
        if task_id in task_storage:
            task_storage[task_id]["status"] = "completed"
            task_storage[task_id]["workflow_result"] = result
            task_storage[task_id]["updated_at"] = datetime.now()
    
    except Exception as e:
        # Update task with error
        if task_id in task_storage:
            task_storage[task_id]["status"] = "failed"
            task_storage[task_id]["error"] = str(e)
            task_storage[task_id]["updated_at"] = datetime.now()

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Data Analysis API with LangChain",
        "version": API_VERSION,
        "endpoints": {
            "analyze": "/api/analyze",
            "workflow": "/api/workflow",
            "pipeline": "/api/pipeline",
            "complete_analysis": "/api/analyze/complete",
            "status": "/api/tasks/{task_id}/status",
            "capabilities": "/api/capabilities",
            "health": "/health"
        },
        "langchain_integration": True
    }

@app.post("/api/analyze", response_model=TaskResponse)
async def analyze_task(request: Union[TaskRequest, Dict[str, Any]], background_tasks: BackgroundTasks):
    """
    Accept analysis tasks and execute using LangChain workflows
    Supports multiple input formats and asynchronous processing
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Handle different input types
        if isinstance(request, str):
            # Plain text input
            task_data = {
                "task_description": request,
                "workflow_type": "data_analysis",
                "data_source": None,
                "dataset_info": {},
                "parameters": {},
                "priority": "normal",
                "include_modeling": False,
                "target_audience": "technical team"
            }
        elif isinstance(request, dict):
            # Dictionary input - extract or set defaults
            task_data = {
                "task_description": request.get("task_description", str(request)),
                "workflow_type": request.get("workflow_type", "data_analysis"),
                "data_source": request.get("data_source"),
                "dataset_info": request.get("dataset_info", {}),
                "parameters": request.get("parameters", {}),
                "priority": request.get("priority", "normal"),
                "include_modeling": request.get("include_modeling", False),
                "target_audience": request.get("target_audience", "technical team")
            }
        else:
            # TaskRequest model
            task_data = request.dict()
        
        # Validate required fields
        if not task_data.get("task_description"):
            raise HTTPException(
                status_code=400, 
                detail="task_description is required"
            )
        
        # Store task (in production, use a proper database)
        task_storage[task_id] = {
            **task_data,
            "task_id": task_id,
            "status": "queued",
            "created_at": current_time,
            "updated_at": current_time,
            "workflow_result": None
        }
        
        # Prepare workflow input data
        workflow_input = {
            "task_description": task_data["task_description"],
            "data_context": task_data.get("data_source", "No specific data source provided"),
            "parameters": task_data.get("parameters", {}),
            "dataset_info": task_data.get("dataset_info", {}),
            "business_context": task_data.get("task_description", ""),
            "include_modeling": task_data.get("include_modeling", False),
            "target_audience": task_data.get("target_audience", "technical team")
        }
        
        # Execute workflow in background
        workflow_type = task_data.get("workflow_type", "data_analysis")
        background_tasks.add_task(
            execute_workflow_task, 
            task_id, 
            workflow_type, 
            workflow_input
        )
        
        # Create response
        response = TaskResponse(
            task_id=task_id,
            status="queued",
            message=f"Task successfully queued for {workflow_type} workflow execution",
            task_details=task_data,
            created_at=current_time,
            workflow_result=None
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the status of a specific task including workflow results"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "task_description": task["task_description"],
        "workflow_type": task.get("workflow_type", "data_analysis")
    }
    
    # Include workflow results if available
    if task.get("workflow_result"):
        response["workflow_result"] = task["workflow_result"]
    
    # Include error if failed
    if task.get("error"):
        response["error"] = task["error"]
    
    return response

@app.post("/api/workflow")
async def execute_specific_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Execute a specific LangChain workflow"""
    try:
        task_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Validate workflow type
        capabilities = orchestrator.get_workflow_capabilities()
        if request.workflow_type not in capabilities["available_workflows"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow type. Available: {capabilities['available_workflows']}"
            )
        
        # Store task
        task_storage[task_id] = {
            "task_id": task_id,
            "workflow_type": request.workflow_type,
            "input_data": request.input_data,
            "status": "queued",
            "created_at": current_time,
            "updated_at": current_time,
            "workflow_result": None
        }
        
        # Execute workflow in background
        background_tasks.add_task(
            execute_workflow_task,
            task_id,
            request.workflow_type,
            request.input_data
        )
        
        return {
            "task_id": task_id,
            "status": "queued",
            "workflow_type": request.workflow_type,
            "message": f"Workflow {request.workflow_type} queued for execution",
            "created_at": current_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing workflow: {str(e)}")

@app.post("/api/pipeline")
async def execute_multi_step_pipeline(request: MultiStepWorkflowRequest, background_tasks: BackgroundTasks):
    """Execute a multi-step workflow pipeline"""
    try:
        task_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Store task
        task_storage[task_id] = {
            "task_id": task_id,
            "pipeline_type": request.pipeline_type,
            "steps": request.steps,
            "status": "queued",
            "created_at": current_time,
            "updated_at": current_time,
            "workflow_result": None
        }
        
        # Execute pipeline in background
        async def execute_pipeline_task(task_id: str, steps: List[Dict[str, Any]]):
            try:
                if task_id in task_storage:
                    task_storage[task_id]["status"] = "processing"
                    task_storage[task_id]["updated_at"] = datetime.now()
                
                result = await orchestrator.execute_multi_step_workflow(steps)
                
                if task_id in task_storage:
                    task_storage[task_id]["status"] = "completed"
                    task_storage[task_id]["workflow_result"] = result
                    task_storage[task_id]["updated_at"] = datetime.now()
            
            except Exception as e:
                if task_id in task_storage:
                    task_storage[task_id]["status"] = "failed"
                    task_storage[task_id]["error"] = str(e)
                    task_storage[task_id]["updated_at"] = datetime.now()
        
        background_tasks.add_task(execute_pipeline_task, task_id, request.steps)
        
        return {
            "task_id": task_id,
            "status": "queued",
            "pipeline_type": request.pipeline_type,
            "total_steps": len(request.steps),
            "message": "Multi-step pipeline queued for execution",
            "created_at": current_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing pipeline: {str(e)}")

@app.post("/api/analyze/complete")
async def execute_complete_analysis(request: TaskRequest, background_tasks: BackgroundTasks):
    """Execute a complete data analysis pipeline"""
    try:
        task_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Store task
        task_storage[task_id] = {
            "task_id": task_id,
            "pipeline_type": "complete_analysis",
            "input_data": request.dict(),
            "status": "queued",
            "created_at": current_time,
            "updated_at": current_time,
            "workflow_result": None
        }
        
        # Execute complete analysis pipeline in background
        async def execute_complete_analysis_task(task_id: str, input_data: Dict[str, Any]):
            try:
                if task_id in task_storage:
                    task_storage[task_id]["status"] = "processing"
                    task_storage[task_id]["updated_at"] = datetime.now()
                
                result = await orchestrator.execute_complete_analysis_pipeline(input_data)
                
                if task_id in task_storage:
                    task_storage[task_id]["status"] = "completed"
                    task_storage[task_id]["workflow_result"] = result
                    task_storage[task_id]["updated_at"] = datetime.now()
            
            except Exception as e:
                if task_id in task_storage:
                    task_storage[task_id]["status"] = "failed"
                    task_storage[task_id]["error"] = str(e)
                    task_storage[task_id]["updated_at"] = datetime.now()
        
        background_tasks.add_task(execute_complete_analysis_task, task_id, request.dict())
        
        return {
            "task_id": task_id,
            "status": "queued",
            "pipeline_type": "complete_analysis",
            "message": "Complete analysis pipeline queued for execution",
            "created_at": current_time,
            "estimated_duration": "2-5 minutes"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing complete analysis: {str(e)}")

@app.get("/api/capabilities")
async def get_capabilities():
    """Get information about available workflows and capabilities"""
    try:
        capabilities = orchestrator.get_workflow_capabilities()
        return {
            **capabilities,
            "api_version": API_VERSION,
            "langchain_integration": True,
            "execution_modes": ["synchronous", "asynchronous"],
            "supported_input_formats": ["JSON", "plain text", "structured data"]
        }
    except Exception as e:
        return {
            "error": f"Error retrieving capabilities: {str(e)}",
            "basic_capabilities": ["data_analysis", "code_generation", "report_generation"]
        }

@app.get("/api/tasks")
async def list_tasks():
    """List all tasks"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "created_at": task["created_at"],
                "task_description": task["task_description"][:100] + "..." if len(task["task_description"]) > 100 else task["task_description"]
            }
            for task_id, task in task_storage.items()
        ],
        "total_tasks": len(task_storage)
    }

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a specific task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    deleted_task = task_storage.pop(task_id)
    return {
        "message": f"Task {task_id} deleted successfully",
        "deleted_task": {
            "task_id": task_id,
            "task_description": deleted_task["task_description"]
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": API_VERSION,
        "active_tasks": len(task_storage)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
