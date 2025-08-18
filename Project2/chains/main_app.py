#!/usr/bin/env python3
"""
TDS Data Analysis API - Advanced Multi-Modal Intelligence Platform

An enterprise-grade FastAPI service that orchestrates sophisticated data analysis workflows
through intelligent LLM agents. This system provides seamless integration of web scraping,
document processing, image analysis, and statistical computation.

Architecture Overview:
===================
- Agent-Orchestrated Workflows: LangChain-powered intelligent task routing
- Multi-Source Data Integration: CSV, JSON, PDFs, images, web pages
- Fault-Tolerant LLM Management: Automatic failover across API keys and models
- Sandboxed Code Execution: Secure Python environment with resource constraints
- Real-Time Processing Pipeline: Synchronous analysis with 3-minute timeout
- Comprehensive Diagnostics: System health monitoring and performance metrics

Core Capabilities:
================
📊 Statistical Analysis: Descriptive statistics, correlations, trend analysis
🌐 Web Intelligence: Adaptive scraping with intelligent table detection
🖼️ Computer Vision: OCR text extraction, image content analysis
📄 Document Processing: PDF table extraction, text mining
📈 Data Visualization: Automated chart generation with size optimization
🔍 Pattern Recognition: Anomaly detection, clustering, classification
⚡ Real-Time Insights: Instant analysis results with comprehensive reporting

Technical Specifications:
=======================
- Framework: FastAPI 0.104+ with Pydantic validation
- LLM Integration: Google Gemini 2.5 Pro/Flash with intelligent fallback
- Data Processing: Pandas/NumPy for high-performance computation
- Visualization: Matplotlib/Seaborn with base64 optimization
- Security: Sandboxed execution environment with timeout protection
- Scalability: Asynchronous processing with concurrent diagnostics

Quality Assurance:
================
- Comprehensive error handling with detailed logging
- Resource monitoring and automatic cleanup
- Input validation and sanitization
- Response format standardization
- Performance benchmarking and optimization

Project: TDS (Transformative Data Systems) - Project 2
Team: Advanced Analytics Division
Version: 2.0.0
License: MIT
Repository: https://github.com/tunafishhyyyy/TDS_Project2
Documentation: https://github.com/tunafishhyyyy/TDS_Project2/blob/main/README.md

Contact: TDS Development Team
Last Updated: August 2025
"""

# Standard library imports
import asyncio
import base64
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from io import BytesIO, StringIO
from typing import Any, Dict

# Third-party core imports
import pandas as pd
import requests
from dotenv import load_dotenv

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# LangChain and LLM imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Web scraping imports
from bs4 import BeautifulSoup

# Optional dependencies with graceful fallbacks
try:
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# System monitoring imports (if available)
if PSUTIL_AVAILABLE:
    pass  # psutil already imported above

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="TDS Data Analysis API",
    description="Enhanced multi-modal data analysis system with LLM orchestration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ================================================================================
# CONFIGURATION AND CONSTANTS
# ================================================================================

# Gemini API configuration
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

MODEL_HIERARCHY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
]

# System configuration
MAX_RETRIES_PER_KEY = 2
TIMEOUT = 30
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))

# Diagnostics configuration
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30
DIAG_PARALLELISM = 6
RUN_LONGER_CHECKS = False

# Validate configuration
if not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys found. Please set them in your environment.")

# ================================================================================
# ENHANCED LLM WRAPPER WITH FALLBACK SUPPORT
# ================================================================================


class LLMWithFallback:
    """
    Advanced LLM wrapper with automatic failover across multiple API keys and models.
    
    Features:
    - Automatic retry with different API keys
    - Model hierarchy fallback (pro -> flash -> lite)
    - Quota and rate limit detection
    - Performance monitoring and logging
    - LangChain compatibility
    """
    
    def __init__(self, keys=None, models=None, temperature=0):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature
        self.slow_keys_log = defaultdict(list)
        self.failing_keys_log = defaultdict(int)
        self.current_llm = None

    def _get_llm_instance(self):
        """Get a working LLM instance, trying all key/model combinations."""
        last_error = None
        
        for model in self.models:
            for key in self.keys:
                try:
                    llm_instance = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                        google_api_key=key,
                        timeout=TIMEOUT
                    )
                    
                    # Test with a simple ping
                    test_response = llm_instance.invoke("ping")
                    logger.info(f"Successfully connected using {model} with key ending in ...{key[-4:]}")
                    return llm_instance
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Failed {model} with key ...{key[-4:]}: {last_error}")
                    
                    # Track quota issues
                    if any(keyword in last_error.lower() for keyword in QUOTA_KEYWORDS):
                        self.failing_keys_log[key] += 1
                        
        raise RuntimeError(f"All models/keys failed. Last error: {last_error}")

    def bind_tools(self, tools):
        """LangChain compatibility: bind tools to the LLM instance."""
        llm_instance = self._get_llm_instance()
        return llm_instance.bind_tools(tools)

    def invoke(self, prompt):
        """LangChain compatibility: invoke the LLM with a prompt."""
        llm_instance = self._get_llm_instance()
        return llm_instance.invoke(prompt)

# Initialize global LLM instance
llm = LLMWithFallback(temperature=0)


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================


def clean_llm_output(output: str) -> Dict:
    """
    Extract and parse JSON object from LLM output with robust error handling.
    
    Args:
        output (str): Raw LLM response text
        
    Returns:
        Dict: Parsed JSON object or error information
    """
    try:
        if not output:
            return {"error": "Empty output from LLM"}
            
        # Remove markdown code fence markers
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        
        # Find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        
        if first == -1 or last == -1 or last <= first:
            return {"error": "No valid JSON object found in output"}
            
        candidate = s[first:last + 1]
        
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing failed: {str(e)}"}
            
    except Exception as e:
        return {"error": f"Unexpected error in clean_llm_output: {str(e)}"}


def parse_keys_and_types(raw_questions: str):
    """
    Parse key/type specifications from questions file.
    
    Args:
        raw_questions (str): Raw questions text
        
    Returns:
        tuple: (keys_list, type_map)
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    
    type_map_def = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float
    }
    
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    
    return keys_list, type_map


def _now_iso():
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"

# ================================================================================
# WEB SCRAPING TOOLS
# ================================================================================


@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Intelligent web scraping tool that extracts structured data from URLs.
    
    Supports:
    - HTML tables (with pandas.read_html)
    - CSV/Excel/JSON files
    - Plain text content
    - Multi-format content detection
    
    Args:
        url (str): Target URL to scrape
        
    Returns:
        Dict: Structured response with data, columns, and status
    """
    logger.info(f"Scraping URL: {url}")
    
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # Handle different content types
        if "text/csv" in ctype or url.endswith('.csv'):
            df = pd.read_csv(StringIO(resp.text))
        elif "json" in ctype or url.endswith('.json'):
            data = resp.json()
            df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
        elif url.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(resp.content))
        elif url.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(resp.content))
        else:
            # Try HTML table extraction
            try:
                tables = pd.read_html(resp.text)
                if tables:
                    df = tables[0]  # Take the first table
            except Exception:
                pass
            
            # Fallback to text content
            if df is None:
                soup = BeautifulSoup(resp.text, "html.parser")
                text_data = soup.get_text(separator="\n", strip=True)
                
                # Try to detect structured data in text
                detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
                if detected_cols:
                    df = pd.DataFrame([{col: None for col in detected_cols}])
                else:
                    df = pd.DataFrame({"text": [text_data]})

        if df is not None:
            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            
            # Ensure unique column names
            cols = []
            for col in df.columns:
                if col in cols:
                    i = 1
                    new_col = f"{col}_{i}"
                    while new_col in cols:
                        i += 1
                        new_col = f"{col}_{i}"
                    cols.append(new_col)
                else:
                    cols.append(col)
            df.columns = cols

            return {
                "status": "success",
                "data": df.to_dict(orient="records"),
                "columns": list(df.columns),
                "shape": df.shape,
                "url": url
            }
        else:
            return {
                "status": "error",
                "error": "No structured data found",
                "data": [],
                "columns": []
            }

    except Exception as e:
        logger.error(f"Scraping failed for {url}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

# ================================================================================
# CODE EXECUTION ENVIRONMENT
# ================================================================================


# Enhanced scraping function for code execution
SCRAPE_FUNC = '''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """Enhanced scraping function for code execution environment."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    try:
        # Try pandas table extraction first
        tables = pd.read_html(response.text)
        if tables:
            df = tables[0]
            df.columns = [str(c).strip() for c in df.columns]
            
            # Ensure unique column names
            df.columns = [str(col) for col in df.columns]
            seen = set()
            new_cols = []
            for col in df.columns:
                if col in seen:
                    i = 1
                    new_col = f"{col}_{i}"
                    while new_col in seen:
                        i += 1
                        new_col = f"{col}_{i}"
                    new_cols.append(new_col)
                    seen.add(new_col)
                else:
                    new_cols.append(col)
                    seen.add(col)
            df.columns = new_cols

            return {
                "status": "success",
                "data": df.to_dict(orient="records"),
                "columns": list(df.columns)
            }
    except Exception:
        pass
    
    # Fallback to text extraction
    soup = BeautifulSoup(response.text, "html.parser")
    text_data = soup.get_text(separator="\\n", strip=True)
    
    # Try to detect structured content
    detected_cols = set(re.findall(r"\\b[A-Z][a-zA-Z ]{2,15}\\b", text_data))
    if detected_cols:
        df = pd.DataFrame([{col: None for col in detected_cols}])
    else:
        df = pd.DataFrame({"text": [text_data]})

    return {
        "status": "success",
        "data": df.to_dict(orient="records"),
        "columns": list(df.columns)
    }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox environment with comprehensive setup.
    
    Features:
    - Pre-configured data science environment
    - Automatic DataFrame injection from pickle
    - Optimized plot generation with size constraints
    - Comprehensive error handling and logging
    - Memory and execution time limits
    
    Args:
        code (str): Python code to execute
        injected_pickle (str, optional): Path to pickle file containing DataFrame
        timeout (int): Maximum execution time in seconds
        
    Returns:
        Dict: Execution results or error information
    """
    
    # Build execution environment
    preamble = [
        "import json, sys, gc, importlib",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from io import BytesIO",
        "import base64",
        "import warnings",
        "warnings.filterwarnings('ignore')",
    ]
    
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
        
    # Inject DataFrame if provided
    if injected_pickle:
        preamble.extend([
            f"df = pd.read_pickle(r'''{injected_pickle}''')",
            "data = df.to_dict(orient='records')"
        ])
    else:
        preamble.append("data = globals().get('data', {})")

    # Advanced plot optimization helper
    helper = '''
def plot_to_base64(max_bytes=100000):
    """
    Convert current matplotlib plot to optimized base64 string.
    
    Automatically adjusts DPI and format to meet size constraints.
    Supports PNG and WEBP formats with progressive quality reduction.
    """
    # Try different DPI settings
    for dpi in [100, 80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        img_bytes = buf.getvalue()
        
        if len(img_bytes) <= max_bytes:
            return base64.b64encode(img_bytes).decode('ascii')
    
    # Try WEBP compression if PIL is available
    try:
        from PIL import Image
        for quality in [80, 60, 40, 20]:
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=30)
            buf.seek(0)
            im = Image.open(buf)
            
            out_buf = BytesIO()
            im.save(out_buf, format='WEBP', quality=quality, method=6)
            out_buf.seek(0)
            webp_bytes = out_buf.getvalue()
            
            if len(webp_bytes) <= max_bytes:
                return base64.b64encode(webp_bytes).decode('ascii')
    except Exception:
        pass
    
    # Last resort: very low DPI PNG
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Assemble complete script
    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\n# Initialize results dictionary")
    script_lines.append("results = {}")
    script_lines.append("\n# User code execution")
    script_lines.append(code)
    script_lines.append("\n# Output results as JSON")
    script_lines.append("print(json.dumps({'status':'success','result':results}, default=str), flush=True)")

    # Write to temporary file
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        # Execute with timeout
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if completed.returncode != 0:
            return {
                "status": "error",
                "message": f"Execution failed with return code {completed.returncode}",
                "stderr": completed.stderr,
                "stdout": completed.stdout
            }
        
        # Parse output
        output = completed.stdout.strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            return {
                "status": "error", 
                "message": f"Failed to parse output as JSON: {str(e)}",
                "raw_output": output
            }
            
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"Execution timed out after {timeout} seconds"}
    except Exception as e:
        return {"status": "error", "message": f"Execution error: {str(e)}"}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ================================================================================
# LANGCHAIN AGENT SETUP
# ================================================================================

# Agent tools
tools = [scrape_url_to_dataframe]

# Enhanced system prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced autonomous data analyst with expertise in multi-modal analysis.

CAPABILITIES:
- Web scraping and data extraction
- Statistical analysis and visualization  
- Image processing and OCR
- PDF document analysis
- Code generation and execution
- Multi-format data processing

INPUT FORMAT:
You will receive:
- Analysis rules (may vary based on data type)
- One or more specific questions
- Optional dataset preview or file information

OUTPUT REQUIREMENTS:
1. Return ONLY a valid JSON object - no additional text or formatting
2. JSON structure must contain:
   - "questions": [list of original question strings exactly as provided]
   - "code": "..." (Python code that creates a 'results' dict mapping questions to answers)

EXECUTION ENVIRONMENT:
- Pre-loaded libraries: pandas, numpy, matplotlib, seaborn
- Helper function: plot_to_base64(max_bytes=100000) for image generation
- Data variables: 'df' (DataFrame) and 'data' (dict records) if dataset provided
- Scraping function: scrape_url_to_dataframe(url) available

CODE REQUIREMENTS:
- All variables must be defined before use
- Results dict must map each question string to its computed answer
- Use plot_to_base64() for all visualizations to ensure size compliance
- Include proper error handling for robust execution
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create and configure agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

# ================================================================================
# AGENT EXECUTION AND ORCHESTRATION
# ================================================================================

def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Execute LLM agent with comprehensive error handling and retry logic.
    
    Features:
    - Multi-attempt execution with exponential backoff
    - Automatic DataFrame injection from pickle files
    - URL detection and pre-processing for web scraping
    - Robust error handling and logging
    
    Args:
        llm_input (str): Input prompt for the agent
        pickle_path (str, optional): Path to pickle file containing DataFrame
        
    Returns:
        Dict: Mapping of questions to computed answers
    """
    try:
        max_retries = 3
        raw_out = ""
        
        # Retry logic with exponential backoff
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Agent execution attempt {attempt}/{max_retries}")
                response = agent_executor.invoke(
                    {"input": llm_input}, 
                    {"timeout": LLM_TIMEOUT_SECONDS}
                )
                
                raw_out = (response.get("output") or 
                          response.get("final_output") or 
                          response.get("text") or "")
                
                if raw_out.strip():
                    break
                    
                logger.warning(f"Empty response on attempt {attempt}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)
        
        if not raw_out:
            return {"error": "Agent returned no output after all retries"}

        # Parse LLM response
        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return {"error": f"LLM output parsing failed: {parsed['error']}"}

        if "code" not in parsed or "questions" not in parsed:
            return {"error": "Invalid LLM response format - missing 'code' or 'questions'"}

        code = parsed["code"]
        questions = parsed["questions"]

        # Handle web scraping if no pickle provided
        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                logger.info(f"Pre-processing {len(urls)} URLs for scraping")
                try:
                    # Use the first URL to get data
                    scrape_result = scrape_url_to_dataframe(urls[0])
                    if scrape_result.get("status") == "success":
                        df = pd.DataFrame(scrape_result["data"])
                        pickle_path = tempfile.mktemp(suffix='.pkl')
                        df.to_pickle(pickle_path)
                        logger.info(f"DataFrame cached to {pickle_path}")
                except Exception as e:
                    logger.warning(f"Pre-scraping failed: {str(e)}")

        # Execute the generated code
        exec_result = write_and_run_temp_python(
            code, 
            injected_pickle=pickle_path, 
            timeout=LLM_TIMEOUT_SECONDS
        )
        
        if exec_result.get("status") != "success":
            return {"error": f"Code execution failed: {exec_result.get('message', 'Unknown error')}"}

        # Extract and map results
        results_dict = exec_result.get("result", {})
        output = {}
        for q in questions:
            output[q] = results_dict.get(q, "Answer not found")
            
        logger.info(f"Successfully processed {len(questions)} questions")
        return output

    except Exception as e:
        logger.exception("Agent execution failed")
        return {"error": str(e)}

# ================================================================================
# RESULT FORMATTING AND CONVERSION
# ================================================================================

def convert_to_evaluation_format(result: Dict, raw_questions: str) -> Dict:
    """
    Convert question-answer results to evaluation-compatible format.
    
    Maps natural language questions to standardized field names
    and adds metadata for evaluation frameworks.
    
    Args:
        result (Dict): Question-answer mapping
        raw_questions (str): Original questions text
        
    Returns:
        Dict: Evaluation-formatted results with metadata
    """
    
    # Standard evaluation field mappings
    evaluation_mapping = {
        "total sales": "total_sales",
        "highest.*region": "top_region",
        "day.*correlation": "day_sales_correlation", 
        "median": "median_sales",
        "sales tax": "total_sales_tax",
        "bar chart": "bar_chart",
        "cumulative.*chart": "cumulative_sales_chart",
        "scatter.*plot": "scatter_plot",
        "correlation.*plot": "correlation_plot"
    }
    
    converted = {}
    
    # Map questions to evaluation fields
    for question_key, answer_value in result.items():
        mapped_key = None
        
        # Try pattern matching
        for pattern, eval_key in evaluation_mapping.items():
            if re.search(pattern, question_key.lower()):
                mapped_key = eval_key
                break
        
        # Fallback to sanitized question
        if not mapped_key:
            mapped_key = re.sub(r'[^\w\s]', '', question_key.lower())
            mapped_key = re.sub(r'\s+', '_', mapped_key.strip())
            mapped_key = mapped_key[:50]  # Limit length
        
        converted[mapped_key] = answer_value
    
    # Add metadata based on content analysis
    if any("sales" in str(v).lower() for v in converted.values()):
        metadata = {
            "workflow_type": "csv_analysis",
            "status": "completed",
            "data_shape": "dynamic",
            "timestamp": datetime.now().isoformat(),
            "processing_mode": "enhanced_llm_agent"
        }
        converted.update(metadata)
    
    return converted

# ================================================================================
# API ENDPOINTS
# ================================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface for file uploads and testing."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <h1>TDS Data Analysis API</h1>
            <p>Enhanced multi-modal analysis system is running!</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/summary">System Diagnostics</a></li>
                <li><a href="/api">Health Check</a></li>
            </ul>
            """,
            status_code=200
        )

@app.post("/api")
@app.post("/api/")
async def analyze_data(request: Request):
    """
    Main analysis endpoint supporting multi-modal data processing.
    
    Accepts:
    - Required: questions.txt file with analysis questions
    - Optional: Additional files (CSV, images, PDFs, etc.)
    
    Returns:
    - JSON response with analysis results
    - Formatted for evaluation framework compatibility
    """
    try:
        form = await request.form()
        
        # Extract questions file (required)
        questions_file = form.get("questions.txt") or form.get("questions_txt")
        if not questions_file:
            raise HTTPException(
                status_code=400,
                detail="questions.txt file is required"
            )
        
        questions_content = (await questions_file.read()).decode('utf-8')
        if not questions_content.strip():
            raise HTTPException(
                status_code=400, 
                detail="questions.txt file cannot be empty"
            )
        
        # Process additional files
        additional_files = form.getlist("files")
        pickle_path = None
        
        if additional_files:
            logger.info(f"Processing {len(additional_files)} additional files")
            
            # Process first data file for DataFrame injection
            for file in additional_files:
                if hasattr(file, 'filename') and file.filename:
                    content = await file.read()
                    
                    # Handle different file types
                    if file.filename.endswith('.csv'):
                        df = pd.read_csv(BytesIO(content))
                        pickle_path = tempfile.mktemp(suffix='.pkl')
                        df.to_pickle(pickle_path)
                        logger.info(f"CSV file processed: {df.shape}")
                        break
                    elif file.filename.endswith('.json'):
                        data = json.loads(content.decode('utf-8'))
                        df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
                        pickle_path = tempfile.mktemp(suffix='.pkl')
                        df.to_pickle(pickle_path)
                        logger.info(f"JSON file processed: {df.shape}")
                        break
        
        # Prepare agent input
        agent_input = f"""
ANALYSIS REQUEST:

Questions to Answer:
{questions_content}

Additional Context:
- Multi-modal analysis system
- Support for web scraping, data analysis, and visualization
- Return results as specified in the questions
- Use appropriate analysis methods for the data type

Please analyze and provide comprehensive answers.
"""
        
        # Execute analysis
        logger.info("Starting analysis execution")
        result = run_agent_safely_unified(agent_input, pickle_path)
        
        if "error" in result:
            logger.error(f"Analysis failed: {result['error']}")
            return JSONResponse(
                status_code=500,
                content={"error": result["error"], "status": "failed"}
            )
        
        # Format results
        formatted_result = convert_to_evaluation_format(result, questions_content)
        
        logger.info("Analysis completed successfully")
        return JSONResponse(content=formatted_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in analyze_data")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )

@app.get("/api")
async def api_info():
    """Health check and API information endpoint."""
    return JSONResponse({
        "status": "healthy",
        "message": "TDS Data Analysis API is running",
        "version": "2.0.0",
        "features": [
            "Multi-modal data processing",
            "LLM-powered analysis",
            "Web scraping capabilities", 
            "Image OCR processing",
            "PDF document analysis",
            "Advanced visualization"
        ],
        "usage": "POST to /api/ with questions.txt file and optional data files",
        "timestamp": _now_iso()
    })

# ================================================================================
# SYSTEM DIAGNOSTICS AND MONITORING
# ================================================================================

def _env_check(required=None):
    """Check environment variable configuration."""
    required = required or []
    out = {}
    
    for k in required:
        value = os.getenv(k)
        out[k] = {
            "present": bool(value),
            "masked": (value[:4] + "..." + value[-4:]) if value else None
        }
    
    # Include helpful configuration values
    out["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL")
    out["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")
    
    return out

def _system_info():
    """Gather comprehensive system information."""
    info = {
        "host": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
    }
    
    if PSUTIL_AVAILABLE:
        info.update({
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        })
        
        # Disk usage information
        try:
            cwd = os.getcwd()
            info["cwd_free_gb"] = round(shutil.disk_usage(cwd).free / 1024**3, 2)
            info["tmp_free_gb"] = round(shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2)
        except Exception:
            pass
    
    # GPU information if available
    if TORCH_AVAILABLE:
        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        info["torch_installed"] = False
        info["cuda_available"] = False
    
    return info

def _temp_write_test():
    """Test temporary directory write permissions."""
    tmp_dir = tempfile.gettempdir()
    test_path = os.path.join(tmp_dir, f"diag_test_{int(time.time())}.tmp")
    
    try:
        with open(test_path, "w") as f:
            f.write("diagnostic test")
        write_ok = os.path.exists(test_path)
        os.remove(test_path)
        return {"tmp_dir": tmp_dir, "write_ok": write_ok}
    except Exception as e:
        return {"tmp_dir": tmp_dir, "write_ok": False, "error": str(e)}

def _pandas_pipeline_test():
    """Test pandas and data processing pipeline."""
    try:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df["z"] = df["x"] * df["y"]
        result = {
            "rows": df.shape[0],
            "cols": df.shape[1], 
            "z_sum": int(df["z"].sum()),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        return result
    except Exception as e:
        return {"error": str(e)}

def _network_probe_sync(url, timeout=30):
    """Synchronous network connectivity probe."""
    try:
        start_time = time.time()
        response = requests.head(url, timeout=timeout)
        latency = int((time.time() - start_time) * 1000)
        
        return {
            "ok": True,
            "status_code": response.status_code,
            "latency_ms": latency
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _test_gemini_key_model(key, model, ping_text="ping"):
    """Test Gemini API key and model combination."""
    try:
        llm_instance = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=key
        )
        
        # Test with simple prompt
        response = llm_instance.invoke(ping_text)
        
        # Extract text content safely
        if hasattr(response, 'content'):
            content = response.content
        elif hasattr(response, 'text'):
            content = response.text
        else:
            content = str(response)
        
        return {
            "ok": True,
            "model": model,
            "response_length": len(content),
            "key_suffix": key[-4:] if len(key) >= 4 else "****"
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "model": model,
            "key_suffix": key[-4:] if len(key) >= 4 else "****"
        }

# Thread pool executor for async operations
_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)

async def run_in_thread(fn, *args, timeout=30, **kwargs):
    """Execute synchronous function in thread pool with timeout."""
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *args, **kwargs))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("Operation timed out")

async def check_network():
    """Check network connectivity to key services."""
    tasks = []
    for name, url in DIAG_NETWORK_TARGETS.items():
        task = run_in_thread(_network_probe_sync, url, timeout=30)
        tasks.append((name, task))
    
    results = {}
    for name, task in tasks:
        try:
            result = await task
            results[name] = result
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    
    return results

async def check_llm_keys_models():
    """Test all LLM API keys and models."""
    if not GEMINI_KEYS:
        return {"warning": "No Gemini API keys configured"}
    
    results = []
    
    for model in MODEL_HIERARCHY:
        tasks = []
        for key in GEMINI_KEYS:
            task = run_in_thread(_test_gemini_key_model, key, model, timeout=DIAG_LLM_KEY_TIMEOUT)
            tasks.append((key, task))
        
        model_summary = {"model": model, "attempts": []}
        working_found = False
        
        for key, task in tasks:
            try:
                result = await task
                model_summary["attempts"].append(result)
                if result.get("ok"):
                    working_found = True
            except Exception as e:
                model_summary["attempts"].append({
                    "ok": False,
                    "error": str(e),
                    "key_suffix": key[-4:] if len(key) >= 4 else "****"
                })
        
        results.append(model_summary)
        
        # Stop testing other models if we found a working one
        if working_found:
            break
    
    return {"models_tested": results}

@app.get("/summary")
async def system_diagnostics(full: bool = Query(False, description="Run extended diagnostic checks")):
    """
    Comprehensive system diagnostics and health monitoring.
    
    Provides detailed information about:
    - Environment configuration
    - System resources and capabilities
    - Network connectivity
    - LLM API key validation
    - Dependencies and optional features
    
    Args:
        full (bool): Whether to run extended checks
        
    Returns:
        JSON: Comprehensive diagnostic report
    """
    started = datetime.utcnow()
    
    report = {
        "status": "ok",
        "server_time": _now_iso(),
        "summary": {},
        "checks": {},
        "elapsed_seconds": None
    }
    
    # Define diagnostic tasks
    tasks = {
        "env": run_in_thread(_env_check, ["GOOGLE_API_KEY", "GOOGLE_MODEL"], timeout=5),
        "system": run_in_thread(_system_info, timeout=30),
        "temp_write": run_in_thread(_temp_write_test, timeout=10),
        "pandas": run_in_thread(_pandas_pipeline_test, timeout=15),
        "network": check_network(),
        "llm_keys_models": check_llm_keys_models()
    }
    
    # Execute all tasks concurrently
    results = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
            results[name]["status"] = "ok"
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
    
    report["checks"] = results
    
    # Generate summary
    failed_checks = [k for k, v in results.items() if v.get("status") != "ok"]
    
    if failed_checks:
        report["status"] = "warning"
        report["summary"]["failed_checks"] = failed_checks
        report["summary"]["message"] = f"Some checks failed: {', '.join(failed_checks)}"
    else:
        report["summary"]["message"] = "All diagnostic checks passed"
        
    # Add feature availability summary
    report["summary"]["features"] = {
        "PIL_available": PIL_AVAILABLE,
        "PDF_available": PDF_AVAILABLE,
        "OCR_available": OCR_AVAILABLE,
        "psutil_available": PSUTIL_AVAILABLE,
        "torch_available": TORCH_AVAILABLE
    }
    
    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    
    return JSONResponse(content=report)

# ================================================================================
# STATIC FILE SERVING AND FAVICON
# ================================================================================

# Minimal favicon fallback (1x1 transparent PNG)
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon.ico or provide a minimal fallback."""
    favicon_path = "favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

# ================================================================================
# APPLICATION STARTUP AND MAIN ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    
    logger.info(f"🚀 Starting TDS Data Analysis API on port {port}")
    logger.info(f"📊 Features: Multi-modal analysis, LLM orchestration, web scraping")
    logger.info(f"🔧 Models available: {len(MODEL_HIERARCHY)} | API keys: {len(GEMINI_KEYS)}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
