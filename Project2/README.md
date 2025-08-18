# TDS Data Analysis API - Advanced Multi-Modal Intelligence Platform

A FastAPI-based REST API that uses LangChain to orchestrate sophisticated LLM workflows for comprehensive data analysis tasks with multi-modal support.

## 🏗️ System Architecture

The TDS Data Analysis API is built on a modern, scalable architecture:

- **Agent-Orchestrated Intelligence**: LangChain-powered agents route tasks to specialized workflows
- **Multi-Modal Processing**: Unified handling of text, images, PDFs, CSV, and web data  
- **Fault-Tolerant LLM Management**: Automatic failover across multiple API keys and models
- **Sandboxed Execution**: Secure Python environment with resource constraints and timeout protection
- **Real-Time Analytics**: Synchronous processing with comprehensive result formatting
- **Enterprise Monitoring**: Advanced diagnostics, health checks, and performance metrics

### Core Components

- **`chains/main_app.py`**: Main FastAPI application with LLM orchestration
- **`chains/workflows.py`**: 12+ specialized analysis workflows
- **`chains/web_scraping_steps.py`**: Modular 6-step web scraping pipeline
- **`chains/base.py`**: Abstract workflow classes and LangChain integration
- **`config.py`**: Environment and model configuration management
- **`utils/`**: Enhanced utilities for LLM management and image optimization

## 🚀 New Features (v2.0)

- **Multiple File Upload**: Required `questions.txt` + optional additional files (CSV, images, etc.)
- **Synchronous Processing**: Get results immediately (≤3 minutes)
- **LLM-Based Workflow Detection**: Intelligent workflow classification using AI
- **Multi-Modal Analysis**: Support for text, images, PDFs, and code generation
- **Enhanced Logging**: Comprehensive logging throughout the execution flow
- **10+ Generalized Workflows**: Including image analysis, text analysis, and code generation
- **OCR Support**: Text extraction from images using Tesseract
- **PDF Processing**: Table and text extraction from PDF files

## 📋 Requirements

The API now enforces that:

- `questions.txt` file is **ALWAYS** required and must contain the analysis questions
- Zero or more additional files can be uploaded (images, CSV, JSON, PDFs, etc.)
- All processing is synchronous (≤3 minutes)
- All generated code is executable Python with proper error handling

## 🛠️ Available Workflows

1. **data_analysis** - General data analysis and recommendations
2. **image_analysis** - Image processing and computer vision with OCR
3. **text_analysis** - Natural language processing and text analytics
4. **code_generation** - Generate executable Python code
5. **exploratory_data_analysis** - Comprehensive EDA planning
6. **predictive_modeling** - Machine learning model development
7. **data_visualization** - Chart and graph generation
8. **statistical_analysis** - Statistical analysis and correlations
9. **web_scraping** - Web data extraction
10. **database_analysis** - SQL and DuckDB analysis

## 📁 Supported File Types

- **Text**: `.txt` (required questions file)
- **Data**: `.csv`, `.xlsx`, `.xls`, `.json`, `.parquet`
- **Images**: `.png`, `.jpg`, `.jpeg` (with OCR text extraction)
- **Documents**: `.pdf` (with table and text extraction)

## 🧪 Comprehensive Test Cases

### 📊 CSV Data Analysis Tests

#### Test 1: Sales Data Analysis
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_sales_questions.txt" \
  -F "files=@sample-sales.csv" \
  --max-time 120
```
**Questions File**: `test_sales_questions.txt`
**Data**: Sample sales data with regions and amounts
**Tests**: Total sales, top regions, averages, visualizations

#### Test 2: Weather Data Analysis
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_weather_questions.txt" \
  -F "files=@sample-weather.csv" \
  --max-time 120
```
**Questions File**: `test_weather_questions.txt`
**Data**: Weather data with temperatures and conditions
**Tests**: Temperature trends, correlations, seasonal analysis

#### Test 3: Network Data Analysis
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_network_questions.txt" \
  -F "files=@network.csv" \
  --max-time 120
```
**Questions File**: `test_network_questions.txt`
**Data**: Network topology and connection data
**Tests**: Network analysis, centrality measures, graph metrics

### 🌐 Web Scraping Tests

#### Test 4: Wikipedia Data Extraction
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_wikipedia.txt" \
  --max-time 300
```
**Questions File**: `test_wikipedia.txt`
**Target**: Wikipedia tables and data
**Tests**: Web scraping, table extraction, data analysis

#### Test 5: IMDB Data Scraping
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_imdb_questions.txt" \
  --max-time 300
```
**Questions File**: `test_imdb_questions.txt`
**Target**: Movie ratings and information
**Tests**: Web scraping, movie analysis, ratings processing

### 📄 PDF Processing Tests

#### Test 6: PDF Table Extraction
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_pdf_questions.txt" \
  -F "files=@test_table.pdf" \
  --max-time 120
```
**Questions File**: `test_pdf_questions.txt`
**Data**: PDF with employee table data
**Tests**: PDF table extraction, employee analysis, salary calculations

#### Test 7: PDF Text Processing
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_text_pdf_questions.txt" \
  -F "files=@test_text.pdf" \
  --max-time 120
```
**Questions File**: `test_text_pdf_questions.txt**
**Data**: PDF with text-based financial report
**Tests**: PDF text extraction, financial data analysis

### 🖼️ Image Processing Tests

#### Test 8: Simple Text Image OCR
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_simple_text_questions.txt" \
  -F "files=@test_simple_text.png" \
  --max-time 120
```
**Questions File**: `test_simple_text_questions.txt`
**Data**: Image with sales text data
**Tests**: OCR text extraction, sales analysis
**Status**: ✅ **FULLY WORKING**

#### Test 9: Employee Table Image
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_employee_table_questions.txt" \
  -F "files=@test_employee_table.png" \
  --max-time 120
```
**Questions File**: `test_employee_table_questions.txt`
**Data**: Image with employee table
**Tests**: Table OCR extraction, employee analysis
**Status**: ✅ **FULLY WORKING**

#### Test 10: Sales Chart Analysis
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_sales_chart_questions.txt" \
  -F "files=@test_sales_chart.png" \
  --max-time 120
```
**Questions File**: `test_sales_chart_questions.txt`
**Data**: Sales bar chart image
**Tests**: Chart OCR, sales data extraction
**Status**: ⚠️ **PARTIAL** (Limited chart text extraction)

#### Test 11: Correlation Plot Analysis
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_correlation_questions.txt" \
  -F "files=@test_correlation_plot.png" \
  --max-time 120
```
**Questions File**: `test_correlation_questions.txt`
**Data**: Scatter plot with correlation data
**Tests**: Plot analysis, correlation interpretation
**Status**: ⚠️ **PARTIAL** (Limited visual analysis)

#### Test 12: Mixed Content Image
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_mixed_content_questions.txt" \
  -F "files=@test_mixed_content.png" \
  --max-time 120
```
**Questions File**: `test_mixed_content_questions.txt`
**Data**: Chart with text summary panel
**Tests**: Complex layout OCR, mixed data analysis
**Status**: ⚠️ **PARTIAL** (Complex layout challenges)

### 🏛️ Complex Data Tests

#### Test 13: High Court Analysis (Large Dataset)
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@high_court_quesion.txt" \
  --max-time 600
```
**Questions File**: `high_court_quesion.txt`
**Data**: S3-hosted large legal dataset
**Tests**: Large data processing, legal analysis
**Status**: ⏳ **TIMEOUT ISSUES** (Dataset too large)

### 🧪 Test Automation Scripts

#### Run All Image Tests
```bash
chmod +x test_image_processing.sh
./test_image_processing.sh
```

#### Run Comprehensive Test Suite
```bash
chmod +x test_examples.sh
./test_examples.sh
```

## 📋 Test File Structure

### Created Test Files
- **Questions Files**: `test_*_questions.txt` - Contains analysis questions
- **Data Files**: `sample-*.csv`, `*.json` - Sample datasets  
- **Image Files**: `test_*.png` - Generated test images with various content
- **PDF Files**: `test_*.pdf` - Sample PDF documents
- **Scripts**: `test_*.sh` - Automated test execution scripts

### Test Categories
1. **Data Analysis**: CSV/JSON data processing and analysis
2. **Web Scraping**: External data extraction from websites
3. **PDF Processing**: Document text and table extraction  
4. **Image OCR**: Text extraction from images using Tesseract
5. **Chart Analysis**: Visual data interpretation (limited)
6. **Complex Datasets**: Large-scale data processing

### Dependencies for Testing
```bash
# Core dependencies
pip install fastapi uvicorn pandas numpy matplotlib seaborn

# PDF processing
pip install PyPDF2 tabula-py
sudo apt install -y default-jre  # Required for tabula-py

# Image OCR
pip install pytesseract opencv-python
sudo apt install -y tesseract-ocr

# LLM and LangChain
pip install langchain langchain-google-genai google-generativeai
```

### Test Results Summary
- ✅ **CSV Data Analysis**: All tests working
- ✅ **PDF Processing**: Table and text extraction working
- ✅ **Simple Image OCR**: Text extraction working
- ⚠️ **Complex Image Analysis**: Limited visual interpretation
- ⚠️ **Large Datasets**: Timeout issues with very large data
- ✅ **Web Scraping**: Basic scraping functionality working

## ⚡ Quick Test Commands

### Most Reliable Tests (Recommended)
```bash
# Simple image OCR (✅ Works perfectly)
curl -X POST "http://localhost:8001/api/" -F "questions.txt=@test_simple_text_questions.txt" -F "files=@test_simple_text.png" --max-time 60

# Employee table extraction (✅ Works perfectly)
curl -X POST "http://localhost:8001/api/" -F "questions.txt=@test_employee_table_questions.txt" -F "files=@test_employee_table.png" --max-time 60

# PDF table processing (✅ Works perfectly)
curl -X POST "http://localhost:8001/api/" -F "questions.txt=@test_pdf_questions.txt" -F "files=@test_table.pdf" --max-time 60

# CSV data analysis (✅ Works perfectly)
curl -X POST "http://localhost:8001/api/" -F "questions.txt=@test_sales_questions.txt" -F "files=@sample-sales.csv" --max-time 60
```

### Generate Test Files
```bash
# Create test images and PDFs
python3 -c "
# (The image/PDF generation code from our tests)
"
```

## 🌐 API Endpoints

### Main Endpoint

```bash
POST /api/
```

**Required**: `questions.txt` file containing analysis questions
**Optional**: Additional files (images, CSV, JSON, PDFs, etc.)

**Enhanced Server (Port 8001)**:
```bash
curl -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@questions.txt" \
  -F "files=@data.csv" \
  -F "files=@image.png"
```

**Standard Server (Port 8000)**:
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions_txt=@questions.txt" \
  -F "files=@data.csv" \
  -F "files=@image.png"
```

### Health Check

```bash
GET /health
GET /summary  # Enhanced diagnostics
```

## 🔧 Server Management

### Enhanced Server (Port 8001)
```bash
# Start/Stop/Status
./server_8001.sh start
./server_8001.sh stop  
./server_8001.sh status

# Direct execution
python chains/main_app.py
# Or using the launcher script
python run_main_server.py
```

### Standard Server (Port 8000)
```bash
# Docker deployment
bash run_docker.sh

# Local development
uvicorn main:app --reload
```

### Docker Deployment
```bash
# Enhanced system with main_app.py
bash run_main_docker.sh

# Standard system
bash run_docker.sh
```

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
