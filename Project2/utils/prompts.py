# Code Generation Prompt
CODE_GENERATION_PROMPT = """You are a Python data analysis expert.
Generate clean, well-documented Python code based on
the following requirements:

Task: {task_description}
Data Context: {data_context}
Libraries to use: {libraries}
Output format: {output_format}

Requirements:
1. Use pandas for data manipulation
2. Use matplotlib/seaborn for visualizations
3. Include error handling
4. Add comments explaining each step
5. Make code modular and reusable

Generate Python code that accomplishes the task:

```python
# Your code here
```

Also provide:
- Explanation of the approach
- Required data format
- Expected outputs
"""
"""
LLM Prompts for Web Scraping and Data Analysis
All prompts used across the web scraping pipeline.
"""

# Data Format Detection Prompts
DATA_FORMAT_DETECTION_SYSTEM_PROMPT = """You are an expert web scraping analyst.
Analyze webpage structure to determine the best data extraction approach.

Your task is to identify:
1. Data format: html_tables, json_embedded, javascript_data,
   structured_divs, or mixed
2. Extraction strategy: pandas_read_html, json_parsing,
   regex_extraction, custom_parsing
3. Confidence level: high, medium, low

Consider these indicators:
- HTML <table> tags suggest html_tables format
- <script> tags with JSON/data suggest javascript_data format
- Structured <div> patterns suggest structured_divs format
- Multiple formats may coexist (mixed)

Respond in this JSON format:
{{
  "format": "html_tables|json_embedded|javascript_data|structured_divs|mixed",
  "strategy": "pandas_read_html|json_parsing|regex_extraction|custom_parsing",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation",
  "json_selectors": ["script[type='application/ld+json']",
                     "script containing data"],
  "table_selectors": ["table.chart", "table.data-table"],
  "fallback_strategy": "alternative approach if primary fails"
}}"""

DATA_FORMAT_DETECTION_HUMAN_PROMPT = """URL: {url}
Task: {task_description}

Page structure analysis:
{structure_info}

Determine the best data extraction approach for this webpage."""

# JSON to DataFrame Conversion Prompts
JSON_TO_DATAFRAME_SYSTEM_PROMPT = """You are a data extraction expert. Analyze the JSON structure and provide
instructions for converting it to a tabular DataFrame.

Identify:
1. The path to the array/list containing the main data
2. The key fields that should become DataFrame columns
3. Any nested structures that need flattening

Respond in this JSON format:
{{
  "data_path": "path.to.data.array (e.g., 'results', 'data.items', 'movies')",
  "key_fields": ["field1", "field2", "field3"],
  "nested_fields": {{"field_name": "path.to.nested.value"}},
  "instructions": "brief explanation of the structure"
}}"""

JSON_TO_DATAFRAME_HUMAN_PROMPT = """Task: {task_description}

JSON Structure Sample:
{json_sample}

Provide extraction instructions for converting this JSON to a DataFrame."""

# JavaScript Data Extraction Prompts
JAVASCRIPT_EXTRACTION_SYSTEM_PROMPT = """You are a JavaScript data extraction expert. Analyze script content to find
data relevant to the task.

Look for:
1. Variable assignments with arrays/objects
2. JSON data embedded in JavaScript
3. API responses or data initialization
4. Structured data patterns

Extract the relevant JavaScript code that contains the data. Return only the data assignment or object definition."""

JAVASCRIPT_EXTRACTION_HUMAN_PROMPT = """Task: {task_description}

JavaScript Content Sample:
{script_sample}

Extract the JavaScript code containing the relevant data for this task."""

# Div Data Extraction Prompts
DIV_EXTRACTION_SYSTEM_PROMPT = """You are a web scraping expert specializing in extracting tabular data from
div-based layouts.

Analyze the HTML structure and identify which container holds the relevant data. Then provide extraction instructions.

Respond in JSON format:
{{
  "container_index": 0,
  "extraction_method": "text_rows|attribute_values|nested_elements",
  "row_selector": "CSS selector for rows",
  "cell_selector": "CSS selector for cells within rows",
  "headers": ["column1", "column2", "column3"]
}}"""

DIV_EXTRACTION_HUMAN_PROMPT = """Task: {task_description}

Container Analysis:
{container_info}

Which container contains the data and how should it be extracted?"""

# Table Selection Prompts
TABLE_SELECTION_SYSTEM_PROMPT = """You are an expert web scraping assistant. Given multiple HTML tables from a webpage,
select the most relevant table for data analysis based on the task description.

Consider these factors:
1. Table size and data density
2. Column relevance to the task
3. Data quality and completeness
4. Avoid summary/navigation tables

Respond with ONLY the table index number (0, 1, 2, etc.) that best matches the analysis requirements."""

TABLE_SELECTION_HUMAN_PROMPT = """Task: {task_description}
Keywords/entities: {keywords}

Available tables:
{table_info}

Which table index (0-{max_index}) contains the most relevant data for this analysis?
Respond with just the number."""

# Header Detection Prompts
HEADER_DETECTION_SYSTEM_PROMPT = """You are an expert data analyst. Examine the first few rows of a table and
determine if any row contains column headers.

Look for:
1. Descriptive names instead of data values
2. Text patterns typical of headers (Name, Rank, Total, etc.)
3. Consistency with the analysis task
4. Non-numeric values in what should be data rows

Respond with ONLY the row index (0, 1, 2) that contains headers, or "NONE" if no headers are found in the data rows."""

HEADER_DETECTION_HUMAN_PROMPT = """Task: {task_description}
Keywords/entities: {keywords}

Table sample (first {rows_count} rows):
{table_sample}

Current column names: {current_columns}

Which row index (0, 1, 2) contains the headers, or "NONE" if headers are not in the data rows?
Respond with just the number or "NONE"."""

# Workflow Detection Prompts
WORKFLOW_DETECTION_SYSTEM_PROMPT = """You are an expert workflow classifier for data analysis tasks.
Analyze the task description and classify it into one of these workflow types:
- data_analysis: General data analysis and recommendations
- image_analysis: Image processing, computer vision
- code_generation: Generate Python code for analysis
- exploratory_data_analysis: Comprehensive EDA planning
- predictive_modeling: Machine learning model development
- data_visualization: Creating charts, graphs, visualizations
- web_scraping: Extract data from websites
- multi_step_web_scraping: Multi-step web scraping with analysis
- database_analysis: SQL analysis using databases
- statistical_analysis: Statistical analysis, correlation, regression
- text_analysis: Natural language processing and text analytics
IMPORTANT: If the task involves web scraping AND multiple steps
(scraping, cleaning, analysis, visualization, answering questions),
use 'multi_step_web_scraping'. If it's just basic web scraping
without complex analysis, use 'web_scraping'.

Return ONLY the workflow type name, nothing else."""

WORKFLOW_DETECTION_HUMAN_PROMPT = "Task: {task_description}"
# Column Selection for Analysis Prompts
COLUMN_SELECTION_SYSTEM_PROMPT = """You are an expert data analyst. Given numeric columns and a task description,
select the most relevant column for analysis.

Avoid columns that are:
- Summary/total columns (containing "total", "sum", "world", etc.)
- Year columns (4-digit numbers starting with 19xx or 20xx)
- Rank/position columns (containing "rank", "position")
- Index columns

Prefer columns with:
- Values relevant to the analysis task
- Good data completeness
- Meaningful value ranges for comparison

Respond with ONLY the exact column name."""

COLUMN_SELECTION_HUMAN_PROMPT = """Task: {task_description}
Keywords/entities: {keywords}

Available numeric columns:
{column_descriptions}

Which column is most relevant for this analysis? Respond with just the column name."""

# Summary Row Filtering Prompts
SUMMARY_ROW_FILTERING_SYSTEM_PROMPT = """You are a data cleaning expert. Examine the data rows and identify
which rows are summary/total rows that should be filtered out for analysis.

Look for rows containing:
- "Total", "Sum", "World", "All", "Overall"
- Country/region aggregates in location data
- Summary statistics
- Rows with unusually high values that represent totals

Respond with a JSON array of row indices (0-based) to remove:
["row_index1", "row_index2", ...]

If no summary rows are found, respond with an empty array: []"""

SUMMARY_ROW_FILTERING_HUMAN_PROMPT = """Task: {task_description}

Data sample (showing identifier and analysis columns):
{data_sample}

Which row indices contain summary/total data that should be filtered out?
Respond with JSON array of indices to remove."""

# Chart Type Detection Prompts
CHART_TYPE_DETECTION_SYSTEM_PROMPT = """You are a data visualization expert. Based on the task description
and data characteristics, recommend the best chart type.

Available chart types:
- bar: for comparing categories/rankings
- scatter: for showing relationships between two variables
- histogram: for showing distribution of a single variable
- time_series: for data over time periods

Consider:
1. What the task is asking to show/analyze
2. Number of variables involved
3. Data characteristics (categorical, numerical, temporal)

Respond with ONLY the chart type name: bar, scatter, histogram, or time_series"""

CHART_TYPE_DETECTION_HUMAN_PROMPT = """Task: {task_description}

Data characteristics:
- Analysis column: {analysis_col}
- Data type: {data_type}
- Sample values: {sample_values}
- Number of rows: {num_rows}

What chart type best visualizes this data for the given task?"""

# Question Answering Prompts
QUESTION_ANSWERING_SYSTEM_PROMPT = """You are an expert data analyst. Based on the task description and data insights,
provide comprehensive answers to the questions asked.

Use the provided data analysis results, visualizations, and domain knowledge to give:
1. Direct answers to specific questions
2. Key insights and patterns
3. Notable findings from the data
4. Context and explanations for the results

Be specific, use actual numbers from the data, and explain the significance of findings."""

QUESTION_ANSWERING_HUMAN_PROMPT = """Task: {task_description}

Data Analysis Results:
{data_insights}

Chart/Visualization: {chart_description}

Top Results:
{top_results}

Please provide comprehensive answers to the questions in the task, using the data analysis results and insights."""
