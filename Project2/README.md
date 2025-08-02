# Data Analysis API with LangChain

A FastAPI-based REST API that uses LangChain to orchestrate LLM workflows for data analysis tasks.

## Quick Start

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**

```bash
cp .env.template .env
# Edit .env and add your OpenAI API key
```

3. **Start the server:**

```bash
uvicorn src.main:app --reload
```

4. **Test the API:**

```bash
python test_api.py              # Basic tests
python test_langchain_api.py    # LangChain workflow tests
```

## Available Workflows

- **Data Analysis**: General analysis and recommendations
- **Code Generation**: Python code for data analysis tasks  
- **Report Generation**: Comprehensive analysis reports
- **Exploratory Data Analysis**: EDA planning and execution
- **Predictive Modeling**: ML model development guidance
- **Data Visualization**: Visualization recommendations

## API Endpoints

- `POST /api/analyze` - Submit analysis tasks
- `POST /api/workflow` - Execute specific workflows
- `POST /api/pipeline` - Multi-step workflow pipelines
- `POST /api/analyze/complete` - Complete analysis pipeline
- `GET /api/tasks/{id}/status` - Check task status
- `GET /api/capabilities` - Available workflows and features

## Example Usage

```python
import requests

# Basic analysis
response = requests.post("http://localhost:8000/api/analyze", json={
    "task_description": "Analyze customer churn data",
    "workflow_type": "data_analysis",
    "dataset_info": {
        "description": "Customer data with demographics",
        "columns": ["age", "tenure", "charges", "churn"],
        "sample_size": 7043
    }
})

task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/api/tasks/{task_id}/status")
print(status.json())
```

## Configuration

Required environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `LANGCHAIN_TRACING_V2` - Enable LangSmith tracing (optional)
- `LANGCHAIN_API_KEY` - LangSmith API key (optional)

See `LANGCHAIN_GUIDE.md` for detailed documentation.
