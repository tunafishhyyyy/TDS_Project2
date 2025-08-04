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

3. **Start the server (development):**

```bash
uvicorn main:app --reload
```

4. **Build and run with Docker:**

```bash
bash run_docker.sh
# Or manually:
docker build -t data-analysis-api .
docker run -d --name data-analysis-api-container -p 8000:80 --env-file .env data-analysis-api
```

5. **Test the API:**

```bash
python test_api.py              # Basic tests
python test_file_upload_api.py  # File upload tests
python test_langchain_api.py    # LangChain workflow tests
```

6. **Test in browser:**
- Open `test_upload.html` in your browser for a user-friendly file upload and workflow interface.

## Available Workflows

- **Data Analysis**: General analysis and recommendations
- **Code Generation**: Python code for data analysis tasks  
- **Report Generation**: Comprehensive analysis reports
- **Exploratory Data Analysis**: EDA planning and execution
- **Predictive Modeling**: ML model development guidance
- **Data Visualization**: Visualization recommendations

## API Endpoints

- `POST /api/` - Submit analysis tasks (file upload or form data)
- `POST /api/analyze` - Legacy JSON endpoint
- `POST /api/workflow` - Execute specific workflows
- `POST /api/pipeline` - Multi-step workflow pipelines
- `POST /api/analyze/complete` - Complete analysis pipeline
- `GET /api/tasks/{id}/status` - Check task status
- `GET /api/capabilities` - Available workflows and features

## Example Usage

### Python (basic analysis)
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

### Curl (file upload)
```bash
curl "http://localhost:8000/api/" -F "file=@question.txt" -F "workflow_type=data_analysis"
```

## Configuration

Required environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `LANGCHAIN_TRACING_V2` - Enable LangSmith tracing (optional)
- `LANGCHAIN_API_KEY` - LangSmith API key (optional)

See `LANGCHAIN_GUIDE.md` for detailed documentation.
