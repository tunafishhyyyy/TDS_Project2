# LangChain Integration Guide

This document explains how LangChain is integrated into the Data Analysis API for orchestrating LLM workflows.

## Overview

LangChain is a framework for developing applications powered by language models. In this project, it's used to:

1. **Orchestrate complex data analysis workflows**
2. **Generate specialized analysis plans and code**
3. **Create comprehensive reports**
4. **Handle multi-step analysis pipelines**
5. **Provide context-aware recommendations**

## Architecture

### Core Components

#### 1. Base Workflow Classes (`chains/base.py`)

- **BaseWorkflow**: Abstract base class for all workflows
- **DataAnalysisChain**: General data analysis using LangChain's Chain class
- **CodeGenerationChain**: Python code generation for data analysis
- **ReportGenerationChain**: Comprehensive report generation
- **WorkflowOrchestrator**: Manages and executes multiple workflows

#### 2. Specialized Workflows (`chains/workflows.py`)

- **ExploratoryDataAnalysisWorkflow**: EDA planning and execution
- **PredictiveModelingWorkflow**: ML model development guidance
- **DataVisualizationWorkflow**: Visualization recommendations
- **AdvancedWorkflowOrchestrator**: Enhanced orchestrator with domain-specific workflows

### Key Features

#### Memory Management

```python
# Conversation memory across interactions
self.memory = ConversationBufferWindowMemory(
    k=10,  # Keep last 10 interactions
    return_messages=True
)
```

#### Multi-Step Workflows

```python
# Execute workflows in sequence with context passing
pipeline_steps = [
    {"workflow_type": "exploratory_data_analysis", "input_data": {...}},
    {"workflow_type": "data_visualization", "input_data": {...}},
    {"workflow_type": "report_generation", "input_data": {...}}
]
result = await orchestrator.execute_multi_step_workflow(pipeline_steps)
```

#### Error Handling

```python
# Robust error handling with fallbacks
try:
    result = await workflow.execute(input_data)
except Exception as e:
    logger.error(f"Workflow error: {e}")
    return {"error": str(e), "status": "failed"}
```

## Required Libraries

### Core LangChain Libraries

```bash
pip install langchain
pip install langchain-openai       # OpenAI integration
pip install langchain-community    # Community components
pip install langchain-core         # Core abstractions
pip install langsmith             # Monitoring and tracing
```

### Supporting Libraries

```bash
pip install openai                # OpenAI API client
pip install tiktoken             # Token counting
pip install python-dotenv        # Environment variable management
pip install faiss-cpu           # Vector similarity search
pip install chromadb            # Vector database
pip install jinja2              # Template processing
```

## Environment Setup

### 1. Create `.env` file

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

### 2. Configuration (`config.py`)

```python
# Model settings
DEFAULT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# Vector store settings
VECTOR_STORE_TYPE = "chromadb"
EMBEDDING_MODEL = "text-embedding-ada-002"
```

## Usage Examples

### 1. Basic Data Analysis

```python
# API Request
POST /api/analyze
{
    "task_description": "Analyze customer churn data",
    "workflow_type": "data_analysis",
    "dataset_info": {
        "description": "Customer data with demographics and usage",
        "columns": ["age", "tenure", "monthly_charges", "churn"],
        "sample_size": 7043
    }
}
```

### 2. Exploratory Data Analysis

```python
# API Request  
POST /api/workflow
{
    "workflow_type": "exploratory_data_analysis",
    "input_data": {
        "dataset_info": {
            "description": "Sales performance dataset",
            "columns": ["date", "product", "sales_amount", "region"],
            "sample_size": 10000
        },
        "business_context": "Quarterly sales analysis"
    }
}
```

### 3. Multi-Step Pipeline

```python
# API Request
POST /api/pipeline
{
    "pipeline_type": "custom_analysis",
    "steps": [
        {
            "workflow_type": "exploratory_data_analysis",
            "input_data": {...}
        },
        {
            "workflow_type": "data_visualization", 
            "input_data": {...}
        },
        {
            "workflow_type": "report_generation",
            "input_data": {...}
        }
    ]
}
```

### 4. Complete Analysis Pipeline

```python
# API Request
POST /api/analyze/complete
{
    "task_description": "Comprehensive customer behavior analysis",
    "dataset_info": {...},
    "include_modeling": true,
    "target_audience": "business stakeholders"
}
```

## Workflow Types

### 1. Data Analysis (`data_analysis`)

- **Purpose**: General data analysis and recommendations
- **Input**: Task description, data context, parameters
- **Output**: Analysis plan, recommendations, metadata
- **Use Case**: Initial analysis planning

### 2. Code Generation (`code_generation`)

- **Purpose**: Generate Python code for data analysis
- **Input**: Task description, data context, required libraries
- **Output**: Python code with explanations
- **Use Case**: Automated code creation

### 3. Report Generation (`report_generation`)

- **Purpose**: Create comprehensive analysis reports
- **Input**: Analysis results, data summary, key findings
- **Output**: Structured markdown report
- **Use Case**: Executive summaries and documentation

### 4. Exploratory Data Analysis (`exploratory_data_analysis`)

- **Purpose**: Comprehensive EDA planning
- **Input**: Dataset information, business context
- **Output**: Detailed EDA plan with code snippets
- **Use Case**: Data exploration strategy

### 5. Predictive Modeling (`predictive_modeling`)

- **Purpose**: ML model development guidance
- **Input**: Problem statement, dataset characteristics
- **Output**: Modeling approach and implementation plan
- **Use Case**: Machine learning project planning

### 6. Data Visualization (`data_visualization`)

- **Purpose**: Visualization recommendations
- **Input**: Data description, analysis goals
- **Output**: Chart recommendations with code
- **Use Case**: Creating effective visualizations

## Advanced Features

### Prompt Engineering

```python
# Structured prompts for consistent outputs
system_message = """You are a data analysis expert. Your task is to:
1. Understand the analysis request
2. Provide structured analysis approach
3. Suggest appropriate data processing steps
4. Recommend visualizations and insights
5. Identify potential issues or limitations"""

human_message = """
Analysis Request: {task_description}
Data Context: {data_context}
Parameters: {parameters}

Please provide a comprehensive analysis plan...
"""
```

### Context Management

```python
# Context passing between workflow steps
for step in steps:
    # Inject context from previous steps
    if context:
        input_data.update(context)
    
    step_result = await execute_workflow(step)
    
    # Update context for next step
    if "analysis_result" in step_result:
        context["previous_analysis"] = step_result["analysis_result"]
```

### Execution History

```python
# Track execution history for debugging and analysis
self.execution_history.append({
    "workflow_type": workflow_type,
    "timestamp": datetime.now().isoformat(),
    "input_data": input_data,
    "result": result
})
```

## Monitoring and Debugging

### 1. LangSmith Integration

```python
# Enable tracing for monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

### 2. Logging

```python
# Comprehensive logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Initialized {self.__class__.__name__} with model {self.model_name}")
logger.error(f"Error in workflow: {e}")
```

### 3. Status Tracking

```python
# Real-time status updates
GET /api/tasks/{task_id}/status

# Response includes workflow results
{
    "task_id": "uuid",
    "status": "completed|processing|failed",
    "workflow_result": {...},
    "error": "error message if failed"
}
```

## Best Practices

### 1. Error Handling

- Always wrap LLM calls in try-catch blocks
- Provide meaningful error messages
- Implement fallback strategies

### 2. Token Management

- Monitor token usage with tiktoken
- Implement token limits and warnings
- Use appropriate model sizes for tasks

### 3. Memory Optimization

- Use ConversationBufferWindowMemory for bounded memory
- Clear memory for unrelated tasks
- Store only essential context

### 4. Performance

- Use background tasks for long-running workflows
- Implement caching for repeated requests
- Monitor execution times

### 5. Security

- Never expose API keys in code
- Use environment variables for configuration
- Validate and sanitize all inputs

## Testing

### Run Basic Tests

```bash
python test_api.py
```

### Run LangChain Tests

```bash
python test_langchain_api.py
```

### Test Specific Workflows

```python
# Test individual workflows
from chains.workflows import AdvancedWorkflowOrchestrator

orchestrator = AdvancedWorkflowOrchestrator()
result = await orchestrator.execute_workflow("data_analysis", input_data)
```

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Check `.env` file exists and contains `OPENAI_API_KEY`
   - Verify environment variable loading in config.py

2. **"Import errors for chains"**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify module structure

3. **"Workflow timeouts"**
   - Increase timeout values in config
   - Check OpenAI API limits and quotas
   - Monitor network connectivity

4. **"Memory errors with large contexts"**
   - Reduce conversation window size
   - Implement context truncation
   - Use summary-based memory

### Performance Optimization

1. **Model Selection**
   - Use gpt-3.5-turbo for most tasks
   - Reserve gpt-4 for complex analysis
   - Consider fine-tuned models for specialized tasks

2. **Prompt Optimization**
   - Keep prompts concise but specific
   - Use structured output formats
   - Test and iterate on prompt effectiveness

3. **Caching**
   - Implement response caching for repeated queries
   - Cache embedding calculations
   - Store processed results for reuse

## Future Enhancements

1. **Additional LLM Providers**
   - Anthropic Claude integration
   - Google Vertex AI support
   - Azure OpenAI Service

2. **Advanced Memory Systems**
   - Long-term memory with vector storage
   - Semantic memory retrieval
   - Conversation summarization

3. **Specialized Workflows**
   - Time series analysis
   - Natural language processing
   - Computer vision integration

4. **Production Features**
   - Rate limiting and quota management
   - Workflow versioning
   - A/B testing capabilities
   - Performance analytics

This integration provides a robust foundation for AI-powered data analysis workflows using LangChain's powerful orchestration capabilities.
