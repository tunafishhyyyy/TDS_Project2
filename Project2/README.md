# Data Analysis API with LangChain

A FastAPI-based REST API that uses LangChain to orchestrate LLM workflows for data analysis tasks.

## VM Setup & Installation (Linux)

Follow these steps to set up a fresh VM and run the project:

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install git, Docker, and vi editor
sudo apt-get install -y git docker.io vim

# (Optional) Install Python if you want to run locally
sudo apt-get install -y python3 python3-pip

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Add your user to the docker group (optional, for non-root usage)
sudo usermod -aG docker $USER
# You may need to log out and log back in for group changes to take effect

# Clone the public repository
# Replace <your-repo-url> with the actual repo URL
git clone https://github.com/tunafishhyyyy/TDS_Project2.git
cd TDS_Project2/Project2

# Copy and edit environment variables
cp .env.template .env
vim .env  # Add your OpenAI API key and other secrets

# Build and run the Docker container
bash run_docker.sh

# The API will be available at http://localhost:8000/
```

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

- Open `http://84.247.184.189:8000/static/test_upload.html` in your browser for a user-friendly file upload and workflow interface.

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
curl "http://84.247.184.189:8000/api/" -F "file=@question.txt" -F "workflow_type=data_analysis"
```

## Configuration

Required environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `LANGCHAIN_TRACING_V2` - Enable LangSmith tracing (optional)
- `LANGCHAIN_API_KEY` - LangSmith API key (optional)

See `LANGCHAIN_GUIDE.md` for detailed documentation.
