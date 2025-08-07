# Code Generation Prompt
CODE_GENERATION_PROMPT = """You are a Python data analysis expert.
Generate clean, well-documented Python code based on the following requirements:

Task: {task_description}
Data Context: {data_context}
Libraries to use: {libraries}
Output format: {output_format}

Requirements:
0. Use `requests.get` with appropriate headers to fetch HTML content before parsing.
1. Use pandas for data manipulation
2. Use matplotlib/seaborn for visualizations
3. Include error handling
4. Add comments explaining each step
5. Make code modular and reusable

Generate Python code that accomplishes the task:

```python
# Your code here
```
"""

# CodeGenerationWorkflow prompts for workflows.py
CODE_WORKFLOW_SYSTEM_PROMPT = """You are a Python expert specializing in data analysis code generation.
Generate clean, executable Python code that:
1. Is syntactically correct and follows Python best practices
2. Includes necessary imports at the top
3. Has clear comments explaining each section
4. Handles potential errors gracefully
5. Returns meaningful results
6. Uses common data analysis libraries (pandas, numpy, matplotlib, seaborn)

Always return ONLY the Python code without any markdown formatting or explanation text.
"""

CODE_WORKFLOW_HUMAN_PROMPT = """Questions: {questions}
Generate Python code to: {task_description}
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
TABLE_SELECTION_SYSTEM_PROMPT = """You are an expert web scraping assistant specialized in identifying the primary data table from multiple HTML tables extracted from a webpage.

CRITICAL: Your selection determines the accuracy of the entire analysis. Choose the table that contains the MAIN dataset for analysis.

Selection criteria (in order of importance):
1. **Content relevance**: Table contains data directly related to the task (movies, countries, sports, etc.)
2. **Data completeness**: Table has substantial rows with meaningful data (avoid navigation, summary, or stub tables)
3. **Column structure**: Table has relevant columns for analysis (ranks, names, numerical values)
4. **Data density**: Prioritize tables with more complete data over sparse tables
5. **Size appropriateness**: Large enough for analysis (typically 20+ rows) but not just repetitive formatting

AVOID:
- Navigation tables (menus, links)
- Summary/total tables (single rows with totals)
- Reference/citation tables 
- Tables with mostly empty cells
- Formatting/layout tables

Respond with ONLY the table index number (0, 1, 2, etc.) of the PRIMARY data table."""

TABLE_SELECTION_HUMAN_PROMPT = """Analysis Task: {task_description}
Relevant Keywords: {keywords}

AVAILABLE TABLES WITH PREVIEWS:
{table_info}

INSTRUCTIONS:
Examine each table's shape, columns, and sample data carefully.
Identify which table contains the PRIMARY dataset for "{task_description}".
Focus on tables with substantial data rows and relevant columns.

Respond with ONLY the table index number (0-{max_index}) that contains the main analysis data."""

# Header Detection Prompts
HEADER_DETECTION_SYSTEM_PROMPT = """You are an expert data analyst specializing in table structure analysis.

Your task: Determine if any of the first few data rows contains column headers that should replace the current column names.

HEADER IDENTIFICATION CRITERIA:
1. **Descriptive names**: Row contains descriptive text like "Rank", "Film", "Year", "Gross" rather than data values
2. **Data type mismatch**: Row has text values in what appear to be numeric columns  
3. **Pattern consistency**: Values look like column names throughout the row
4. **Context relevance**: Header names match the analysis context

IMPORTANT: Only identify rows that contain BETTER headers than the current column names.
If current columns are already descriptive, respond "NONE".

Respond with ONLY:
- The row index (0, 1, 2) if that row contains proper headers
- "NONE" if no headers found in the sample rows"""

HEADER_DETECTION_HUMAN_PROMPT = """Analysis Context: {task_description}
Keywords: {keywords}

TABLE STRUCTURE ANALYSIS:
Current column names: {current_columns}

Sample data (first {rows_count} rows):
{table_sample}

QUESTION: Do any of these {rows_count} sample rows contain better column headers than the current column names?

Examine each row:
- Row 0: Does this look like headers rather than data?
- Row 1: Does this look like headers rather than data?
- Row 2: Does this look like headers rather than data?

Respond with ONLY:
- The row number (0, 1, or 2) if that row has proper headers
- "NONE" if no sample rows contain headers"""

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
Respond with JSON array of indices to remove: ["row_index1", "row_index2", ...]"""

# ===== CHART TYPE DETECTION PROMPTS =====
CHART_TYPE_DETECTION_SYSTEM_PROMPT = """You are a visualization expert. Recommend the best chart type for data analysis.
Available chart types: bar, scatter, histogram, time_series
Provide just the chart type name and brief reasoning."""

CHART_TYPE_DETECTION_HUMAN_PROMPT = """Task: {task_description}
Data characteristics:
- Numeric columns: {numeric_cols}
- Text columns: {text_cols}
- Data shape: {data_shape}
Keywords: {keywords}

Which chart type (bar/scatter/histogram/time_series) is most appropriate and why?"""

# ===== QUESTION ANSWERING PROMPTS =====
QUESTION_ANSWERING_SYSTEM_PROMPT = """You are an expert data analyst providing insights based on processed data
analysis results. Your task is to answer questions with PRECISION using the data insights that have already been
analyzed.

CRITICAL INSTRUCTIONS:
1. You are receiving PROCESSED DATA INSIGHTS from a comprehensive web scraping and analysis pipeline
2. The data has already been scraped, cleaned, and analyzed - you don't need to scrape anything
3. Pay special attention to YEARS, DATES, and NUMERICAL VALUES in the processed results
4. For temporal questions (e.g., "before 2000"), check year information in the analysis results
5. For ranking questions, use the processed data rankings provided
6. VERIFY your answers against the analysis results and top results provided
7. If data seems inconsistent in the results, note this in your response
8. Base all answers on the PROCESSED ANALYSIS RESULTS provided to you

RESPONSE FORMAT: Valid JSON with clear question-answer mapping.
ACCURACY IS PARAMOUNT - Use the processed data insights provided."""

QUESTION_ANSWERING_HUMAN_PROMPT = """Task: {task_description}

IMPORTANT: The following is REAL DATA from the analysis. Use this data to answer questions accurately:

Data Analysis Results:
{data_insights}

Chart/Visualization: {chart_description}

Top Results (VERIFY against this data):
{top_results}

INSTRUCTIONS:
1. Examine the sample data rows carefully
2. Look for year information in the data
3. Cross-reference rankings and values
4. Answer each question based on the ACTUAL DATA provided
5. If you cannot find sufficient data to answer a question accurately, state so

Please provide accurate answers in JSON format, showing your reasoning based on the data provided."""

# Code Generation Prompt
CODE_GENERATION_PROMPT = """You are a Python data analysis expert.
Generate clean, well-documented Python code based on the following requirements:

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
"""

# CodeGenerationWorkflow prompts for workflows.py
CODE_WORKFLOW_SYSTEM_PROMPT = """You are a Python expert specializing in data analysis code generation.
Generate clean, executable Python code that:
1. Is syntactically correct and follows Python best practices
2. Includes necessary imports at the top
3. Has clear comments explaining each section
4. Handles potential errors gracefully
5. Returns meaningful results
6. Uses common data analysis libraries (pandas, numpy, matplotlib, seaborn)

Always return ONLY the Python code without any markdown formatting or explanation text.
"""

CODE_WORKFLOW_HUMAN_PROMPT = """Questions: {questions}
Generate Python code to: {task_description}
"""
QUESTION_ANSWERING_HUMAN_PROMPT = """Task: {task_description}

Data Analysis Results:
{data_insights}

Chart/Visualization: {chart_description}

Top Results:
{top_results}

Please provide comprehensive answers to the questions in the task, using the data analysis results and insights."""

# EDA Prompts
EDA_SYSTEM_PROMPT = """You are an expert data scientist specializing in Exploratory Data Analysis (EDA).
Your task is to provide a comprehensive EDA plan and insights based on the provided dataset information.

Focus on:
1. Data quality assessment
2. Distribution analysis
3. Correlation analysis
4. Outlier detection
5. Missing value analysis
6. Feature engineering opportunities
7. Visualization recommendations
"""

EDA_HUMAN_PROMPT = """Dataset Information:
- Description: {dataset_description}
- Columns: {columns_info}
- Data Types: {data_types}
- Sample Size: {sample_size}
- Business Context: {business_context}

Additional Parameters: {parameters}

Provide a structured EDA plan including:
1. Initial data inspection steps
2. Statistical summaries to compute
3. Visualizations to create
4. Data quality checks
5. Feature relationships to explore
6. Potential data issues to investigate
7. Python code snippets for key analyses

Format your response with clear sections and actionable recommendations.
"""

# Data Analysis Prompts
DATA_ANALYSIS_SYSTEM_PROMPT = "You are a data analyst. Analyze the provided data and answer the questions."
DATA_ANALYSIS_HUMAN_PROMPT = "Questions: {questions}\nFiles: {files}"

# Image Analysis Prompts
IMAGE_ANALYSIS_SYSTEM_PROMPT = "You are an expert in image analysis. Analyze the provided image and answer " \
    "the questions."
IMAGE_ANALYSIS_HUMAN_PROMPT = "Questions: {questions}\nImage: {image_file}"

# Multi-step Web Scraping Prompts
MULTI_STEP_SYSTEM_PROMPT = """You are a web scraping expert specializing in multi-step data analysis tasks.
Your task is to execute complete web scraping workflows including:
1. Web scraping and data extraction from any website
2. Data cleaning and preprocessing (handling various data formats)
3. Data analysis and visualization
4. Answering specific questions about the data

You must provide executable Python code that actually performs these tasks.
IMPORTANT:
- Always inspect the actual data structure before processing
- Handle dynamic column names and various data formats
- Make the code generic enough to work with different types of data
- Include proper error handling for different scenarios
"""

MULTI_STEP_HUMAN_PROMPT = """Multi-Step Task: {task_description}
Target URL: {url}
Data Requirements: {data_requirements}
Output Format: {output_format}
Special Instructions: {special_instructions}

Provide a complete solution that:
1. Scrapes the data from the specified URL using pandas read_html (preferred) or requests
2. Inspects the actual data structure and column names before processing
3. Cleans and processes the data (remove symbols, convert to numeric, handle various formats)
4. Creates visualizations as requested
5. Performs analysis to answer specific questions
6. Returns the final answers

IMPORTANT:
- Generate executable Python code that actually performs these tasks, not just a plan
- Use pandas read_html() for web scraping when possible (it's more reliable)
- ALWAYS inspect the actual data structure first (print column names, data types, first few rows)
- Handle dynamic column names - NEVER assume specific column names like 'Country/Territory' exist
- Use data.columns[0] for the first column, data.columns[1] for second, etc.
- Make the code generic enough to work with different types of data (not just GDP data)
- Include all necessary imports and error handling
- Make sure the code can run without external dependencies like BeautifulSoup
- Print the final answers clearly
- Add debug prints to show what data is being processed
- Handle various data formats and structures automatically
- Always use dynamic column references instead of hardcoded column names
- Keep the code simple and avoid complex variable names that might be interpreted as template variables
- CRITICAL: After cleaning data, use data.select_dtypes(include=[np.number]).columns.tolist() to find numeric columns
- CRITICAL: Never assume data.columns[1] is a column name - it might be a value
- CRITICAL: Always verify column types before using them for analysis
- CRITICAL: For Wikipedia data, the main table is usually the one with the most rows
- CRITICAL: Always print table information to verify you're selecting the right table
- CRITICAL: Store final answers in variables so they can be captured in the response
"""

# Statistical Analysis Prompts
STATISTICAL_SYSTEM_PROMPT = """You are an expert statistician and data analyst.
Your task is to perform comprehensive statistical analysis including correlation, regression, and trend analysis.

Focus on:
1. Descriptive statistics and data summarization
2. Correlation analysis and interpretation
3. Regression modeling and validation
4. Statistical significance testing
5. Trend analysis and forecasting
6. Data visualization for statistical insights
"""

STATISTICAL_HUMAN_PROMPT = """Task: {task_description}

Dataset Description: {dataset_description}

Variables of Interest: {variables}

Statistical Methods Required: {methods}

Please provide:
- Statistical analysis approach
- Correlation analysis plan
- Regression modeling strategy
- Visualization recommendations
- Code snippets for analysis
- Interpretation guidelines
"""

# Predictive Modeling Prompts
PREDICTIVE_MODELING_SYSTEM_PROMPT = """You are a machine learning expert specializing in predictive modeling.
Your task is to design an appropriate modeling approach based on the problem description and data characteristics.

Consider:
1. Problem type (classification, regression, clustering, etc.)
2. Data characteristics and quality
3. Model selection and evaluation metrics
4. Feature engineering requirements
5. Cross-validation strategy
6. Model interpretability needs
7. Production deployment considerations
"""

PREDICTIVE_MODELING_HUMAN_PROMPT = """Problem Statement: {problem_statement}
Target Variable: {target_variable}
Dataset Characteristics: {dataset_characteristics}
Business Requirements: {business_requirements}
Performance Requirements: {performance_requirements}

Provide a comprehensive modeling approach including:
1. Problem formulation
2. Recommended algorithms
3. Feature engineering strategy
4. Model evaluation approach
5. Cross-validation strategy
6. Performance metrics
7. Implementation roadmap
8. Potential challenges and mitigation strategies

Include Python code examples using scikit-learn and other relevant libraries.
"""

# Data Visualization Prompts
DATA_VISUALIZATION_SYSTEM_PROMPT = """You are a data visualization expert specializing in creating effective
and insightful charts and graphs.
Your task is to recommend appropriate visualizations based on the data characteristics and analysis goals.

Consider:
1. Data types (categorical, numerical, temporal)
2. Number of variables and relationships
3. Target audience
4. Story to tell with the data
5. Interactive vs static requirements
6. Best practices for clarity and aesthetics
"""

DATA_VISUALIZATION_HUMAN_PROMPT = """Data Description: {data_description}
Variables: {variables}
Analysis Goals: {analysis_goals}
Target Audience: {target_audience}
Platform/Tools: {platform}

Recommend appropriate visualizations including:
1. Chart types for each analysis goal
2. Layout and design considerations
3. Interactive features (if applicable)
4. Color schemes and styling
5. Python code using matplotlib, seaborn, or plotly
6. Dashboard structure (if multiple charts)

Provide detailed rationale for each recommendation.
"""

# Web Scraping Prompts
WEB_SCRAPING_SYSTEM_PROMPT = """You are a web scraping expert specializing in data extraction from websites.
Your task is to provide Python code and analysis for web scraping tasks.

Focus on:
1. URL analysis and data structure identification
2. HTML parsing strategies using BeautifulSoup/Selenium
3. Data cleaning and transformation
4. Handling pagination and dynamic content
5. Error handling and rate limiting
6. Data validation and quality checks
"""

WEB_SCRAPING_HUMAN_PROMPT = """Scraping Task: {task_description}
Target URL: {url}
Data Requirements: {data_requirements}
Output Format: {output_format}
Special Instructions: {special_instructions}

Provide a complete solution including:
1. Python code for scraping the data
2. Data cleaning and processing steps
3. Analysis of the extracted data
4. Visualization code if requested
5. Error handling strategies
6. Expected output format

Format your response with clear code blocks and explanations.
"""

# Database Analysis Prompts
DATABASE_ANALYSIS_SYSTEM_PROMPT = """You are a database analysis expert specializing in SQL queries and data analysis.
Your task is to provide SQL code and analysis strategies for complex datasets.

Focus on:
1. SQL query optimization for large datasets
2. DuckDB-specific features and functions
3. Data aggregation and statistical analysis
4. Performance optimization strategies
5. Cloud storage integration (S3, etc.)
6. Data visualization and reporting
"""

DATABASE_ANALYSIS_HUMAN_PROMPT = """Analysis Task: {task_description}
Database/Dataset: {database_info}
Data Schema: {schema_info}
Analysis Goals: {analysis_goals}
Performance Requirements: {performance_requirements}

Provide a comprehensive solution including:
1. SQL queries for data analysis
2. Performance optimization strategies
3. Data processing pipeline
4. Statistical analysis methods
5. Visualization recommendations
6. Expected insights and outputs

Use DuckDB syntax and best practices for cloud data access.
"""

# Workflow Step Detection Prompt
DETECT_STEPS_PROMPT = '''
You are an expert workflow planner for data analysis and web scraping tasks.
Given a user request, break it down into a sequence of high-level, reusable steps.
Each step should have a type (e.g., scrape_table, inspect_table, clean_data,
analyze_data, visualize, answer)
and relevant parameters.
Output the plan as a JSON list, where each item is a step with its parameters.
Do not generate code, only the step plan.

User Request:
{user_request}

Output Format Example:
[
  {"step": "scrape_table", "url": "https://example.com/table"},
  {"step": "inspect_table"},
  {"step": "clean_data"},
  {"step": "analyze_data", "top_n": 10},
  {"step": "visualize"},
  {"step": "answer", "questions": [
    "Which country ranks 5th by GDP?",
    "What is the total GDP of the top 10 countries?"
  ]}
]

Now, generate the step plan for the following user request.
'''
