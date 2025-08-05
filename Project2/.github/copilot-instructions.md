# Copilot Instructions for TDS Project2 Data Analysis API

## Project Overview
This is a **multi-modal data analysis API** using FastAPI + LangChain that orchestrates LLM workflows for comprehensive data analysis. The system runs **exclusively in Docker containers** and specializes in **generic web scraping** with adaptive data processing.

## Critical Architecture Understanding

### 🏗️ Core Components
- **`main.py`**: FastAPI app with orchestrator initialization and fallback handling
- **`chains/workflows.py`**: 12+ specialized workflows (web scraping, data analysis, image analysis, etc.)
- **`chains/web_scraping_steps.py`**: Modular 6-step generic web scraping pipeline
- **`chains/base.py`**: Abstract workflow classes and LangChain component setup
- **`config.py`**: Environment and model configuration management

### 🔄 Data Flow Architecture
1. **File Upload** → `POST /api/` (requires `questions.txt` + optional files)
2. **Workflow Detection** → LLM-based classification in `main.py:detect_workflow_type()`
3. **Orchestration** → `AdvancedWorkflowOrchestrator` routes to appropriate workflow
4. **Execution** → Step-based processing with fallback to `ModularWebScrapingWorkflow`
5. **Response** → Synchronous JSON with results (≤3 minutes timeout)

## 🚨 Critical Constraints & Rules

### Docker-Only Execution
```bash
# NEVER run Python commands directly - everything runs in Docker
bash run_docker.sh  # Primary deployment method
docker build -t data-analysis-api . && docker run --env-file .env -p 8000:80 data-analysis-api
```

### Generic Web Scraping Requirements
- **Table selection MUST use LLM prompting** (as noted in `prompt.txt`)
- All scraping logic must work for **any table**, not specific use cases
- Support: Wikipedia, IMDB, Trading Economics, Worldometers, ESPN, etc.
- Auto-detect data types: financial, health, sports, economic, entertainment

### File Upload Patterns
```python
# ALWAYS required: questions.txt file
# Optional: CSV, images, JSON files
curl "http://localhost:8000/api/" \
  -F "questions_txt=@questions.txt" \
  -F "files=@data.csv" \
  -F "files=@image.png"
```

## 🔧 Development Patterns

### Adding New Workflows
1. **Inherit from `BaseWorkflow`** in `chains/workflows.py`
2. **Implement `async execute()`** method
3. **Register in `AdvancedWorkflowOrchestrator.workflows`** dict
4. **Add to workflow detection logic** in `main.py:detect_workflow_type()`

### Web Scraping Step Pattern
```python
# Follow 6-step modular approach in web_scraping_steps.py:
# 1. ScrapeTableStep - pandas.read_html() with intelligent table selection
# 2. InspectTableStep - handle MultiIndex, detect headers
# 3. CleanDataStep - remove symbols, convert to numeric
# 4. AnalyzeDataStep - dynamic column selection, filter summary rows
# 5. VisualizeStep - auto-detect chart types, base64 encoding
# 6. AnswerQuestionsStep - domain-aware question handling
```

### LLM Integration Points
```python
# Use ChatPromptTemplate for structured prompts
system_message = """You are a [domain] expert..."""
human_message = """Task: {task_description}..."""
prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])
```

## 🎯 Key Implementation Guidelines

### Generic Data Processing
- **ALWAYS use `data.select_dtypes(include=[np.number])` to find numeric columns**
- **Dynamic column references**: `data.columns[0]`, `data.columns[1]` (never hardcode names)
- **Multi-domain support**: Auto-detect financial, health, sports, economic data types
- **Robust cleaning**: Handle currency symbols, footnotes, scale factors (billion/million)

### Visualization Requirements
- **Auto-detect chart types** from task descriptions
- **Scatter plots MUST include dotted red regression lines**
- **Base64 encoding under 100KB** for API responses
- **Smart column pairing** for correlations (Rank vs Peak, Cases vs Deaths, etc.)

### Error Handling & Fallbacks
```python
# Orchestrator initialization with fallback
try:
    orchestrator = AdvancedWorkflowOrchestrator()
except Exception:
    orchestrator = MinimalOrchestrator()  # Falls back to ModularWebScrapingWorkflow
```

## 🚀 LLM Prompting Opportunities (Priority Areas for Improvement)

### 🔥 CRITICAL: Replace Hardcoded Logic with LLM Prompting

#### 1. Table Selection Logic (ScrapeTableStep)
**Current**: Hardcoded scoring algorithm with predefined indicators
**Should be**: LLM-powered table evaluation
```python
# REPLACE THIS in web_scraping_steps.py lines 34-98:
# Hardcoded scoring with content_indicators and data_indicators lists
# WITH LLM prompting approach:

system_prompt = """You are an expert web scraping assistant. Given multiple HTML tables, 
select the most relevant table for data analysis based on the task description."""

human_prompt = """
Task: {task_description}
Available tables (showing first 3 rows of each):
{table_previews}

Which table (0-{max_index}) contains the most relevant data for this analysis?
Respond with just the table index number and brief reasoning.
"""
```

#### 2. Header Detection (InspectTableStep)
**Current**: Hardcoded header_indicators list with specific keywords
**Should be**: LLM-based header pattern recognition
```python
# REPLACE header_indicators in lines 143-155 with LLM evaluation:
prompt = """Analyze these table rows and determine if the first row contains headers:
{table_sample}
Task context: {task_description}
Is the first row headers? Respond: YES/NO with reasoning."""
```

#### 3. Data Cleaning Strategy (CleanDataStep)
**Current**: Fixed symbol removal patterns and scale detection
**Should be**: LLM-guided cleaning decisions
```python
# ENHANCE cleaning logic with LLM guidance:
prompt = """Analyze this data column and suggest cleaning approach:
Column name: {column_name}
Sample values: {sample_values}
Context: {task_description}
What symbols/patterns should be removed? What scale factors apply?"""
```

#### 4. Column Selection for Analysis (AnalyzeDataStep)
**Current**: Hardcoded filtering of summary_keywords and column exclusion rules
**Should be**: LLM-powered column relevance scoring
```python
# REPLACE hardcoded filtering in lines 308-320 with:
prompt = """Given these numeric columns for analysis:
{column_descriptions}
Task: {task_description}
Which column is most relevant for the primary analysis? Exclude summary/total columns."""
```

#### 5. Chart Type Detection (VisualizeStep)
**Current**: Basic keyword matching for chart type selection
**Should be**: LLM-based visualization strategy
```python
# ENHANCE _auto_detect_chart_type() with sophisticated LLM analysis:
prompt = """Recommend the best chart type for this data analysis:
Task: {task_description}
Data characteristics: {data_summary}
Available chart types: bar, scatter, histogram, time_series
Best chart type and reasoning:"""
```

#### 6. Domain Detection
**Current**: Mixed hardcoded indicators across steps
**Should be**: Centralized LLM-based domain classification
```python
# ADD new domain detection utility:
prompt = """Classify the data domain:
Task: {task_description}
Sample data: {data_preview}
Domain options: financial, health, sports, entertainment, economic, other
Primary domain:"""
```

## 🧪 Testing & Validation

### Local Development (Limited)
```bash
# Basic syntax validation only
python -m py_compile main.py
python -m py_compile chains/web_scraping_steps.py
```

### Docker Testing
```bash
# Test API endpoints
curl "http://localhost:8000/health"
curl "http://localhost:8000/api/" -F "questions_txt=@test_questions.txt"
```

## 📁 Project Structure Essentials
```
Project2/
├── main.py                    # FastAPI app with orchestrator
├── config.py                  # Environment configuration
├── chains/
│   ├── base.py               # Abstract workflow classes
│   ├── workflows.py          # 12+ specialized workflows
│   └── web_scraping_steps.py # 6-step generic scraping pipeline
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
└── run_docker.sh           # Primary deployment script
```

## 🎪 Multi-Modal Capabilities
- **Web scraping**: Generic table extraction and analysis
- **Image analysis**: Computer vision workflows
- **Text analysis**: NLP and text processing
- **Code generation**: Executable Python code with validation
- **Data visualization**: Multiple chart types with base64 encoding

## ⚠️ Common Pitfalls to Avoid
1. **Never assume specific column names** - always use dynamic detection
2. **Don't hardcode domain logic** - use LLM prompting for adaptability
3. **Avoid direct command execution** - everything must work in Docker
4. **Don't create test files** unless explicitly requested
5. **Always handle MultiIndex columns** in web scraping
6. **Ensure base64 images are under 100KB** for API compliance

## 🔄 LLM Prompting Implementation Pattern
```python
# Standard pattern for replacing hardcoded logic:
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

def llm_enhanced_function(data, task_description, llm_model):
    system_message = "You are an expert in [domain]..."
    human_message = "Analyze: {input_data}\nTask: {task}\nProvide: {expected_output}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])
    
    chain = prompt | llm_model | StrOutputParser()
    result = chain.invoke({
        "input_data": data,
        "task": task_description,
        "expected_output": "specific format requirements"
    })
    
    return parse_llm_response(result)
```

## 🎯 Implementation Priority Order
1. **Table Selection** (ScrapeTableStep) - Most critical for accuracy
2. **Domain Detection** - Central to all other decisions  
3. **Column Selection** (AnalyzeDataStep) - Affects analysis quality
4. **Chart Type Detection** (VisualizeStep) - Improves presentation
5. **Header Detection** (InspectTableStep) - Data structure parsing
6. **Cleaning Strategy** (CleanDataStep) - Data quality enhancement
