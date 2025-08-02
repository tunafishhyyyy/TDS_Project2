# Data Analyst Agent Project

> [Project Brief](https://tds.s-anand.net/#/project-data-analyst-agent?id=project-data-analyst-agent)

## Overview

Deploy a data analyst agent: an API that uses LLMs to source, prepare, analyze, and visualize any data.

Your application exposes an API endpoint (e.g., `https://app.example.com/api/`).

The endpoint must accept a POST request with a data analysis task description in the body. For example:

```bash
curl "https://app.example.com/api/" -F "@question.txt"
```

The answers must be sent within 3 minutes in the format requested.

---

## Task Breakdown

1. **API Design & Setup**
   - Design a REST API endpoint to accept POST requests with analysis tasks.
   - Ensure the endpoint can process plain text or JSON task descriptions.
   - Set up hosting (cloud or local).

2. **LLM Integration**
   - Integrate with an LLM (e.g., OpenAI, Azure, or open-source) to interpret and execute analysis tasks.

3. **Data Sourcing**
   - Implement logic to fetch data from web sources (e.g., Wikipedia) or cloud storage (e.g., S3 buckets).

4. **Data Preparation & Analysis**
   - Parse, clean, and transform data as required by the task.
   - Perform statistical analysis, regression, and correlation as needed.

5. **Visualization**
   - Generate plots (e.g., scatterplots, regression lines) and encode as base64 data URIs.

6. **Response Formatting**
   - Format answers as JSON arrays or objects, as specified in the task.
   - Ensure responses are returned within 3 minutes.

7. **Testing & Validation**
   - Test with sample questions and datasets.
   - Validate output format and timing.

---

## Sample Questions

These are examples of `question.txt` that will be sent (actual questions will be secret):

### Example 1: Wikipedia Data

Scrape the list of highest grossing films from Wikipedia:
<https://en.wikipedia.org/wiki/List_of_highest-grossing_films>

**Questions:**

1. How many $2 bn movies were released before 2020?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it. Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.

---

### Example 2: Indian High Court Judgement Dataset

The Indian high court judgement dataset contains judgements from the Indian High Courts, downloaded from [ecourts website](https://judgments.ecourts.gov.in/). It contains judgments of 25 high courts, along with raw metadata (as .json) and structured metadata (as .parquet).

- 25 high courts
- ~16M judgments
- ~1TB of data

**Structure of the data in the bucket:**

- `data/pdf/year=2025/court=xyz/bench=xyz/judgment1.pdf,judgment2.pdf`
- `metadata/json/year=2025/court=xyz/bench=xyz/judgment1.json,judgment2.json`
- `metadata/parquet/year=2025/court=xyz/bench=xyz/metadata.parquet`
- `metadata/tar/year=2025/court=xyz/bench=xyz/metadata.tar.gz`
- `data/tar/year=2025/court=xyz/bench=xyz/pdfs.tar`

**DuckDB Query Example:**

```sql
INSTALL httpfs; LOAD httpfs;
INSTALL parquet; LOAD parquet;

SELECT COUNT(*) FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1');
```

**Columns in the Data:**

| Column                 | Type    | Description                    |
|------------------------|---------|--------------------------------|
| `court_code`           | VARCHAR | Court identifier (e.g., 33~10) |
| `title`                | VARCHAR | Case title and parties         |
| `description`          | VARCHAR | Case description               |
| `judge`                | VARCHAR | Presiding judge(s)             |
| `pdf_link`             | VARCHAR | Link to judgment PDF           |
| `cnr`                  | VARCHAR | Case Number Register           |
| `date_of_registration` | VARCHAR | Registration date              |
| `decision_date`        | DATE    | Date of judgment               |
| `disposal_nature`      | VARCHAR | Case outcome                   |
| `court`                | VARCHAR | Court name                     |
| `raw_html`             | VARCHAR | Original HTML content          |
| `bench`                | VARCHAR | Bench identifier               |
| `year`                 | BIGINT  | Year partition                 |

**Sample Row:**

```json
{
  "court_code": "33~10",
  "title": "CRL MP(MD)/4399/2023 of Vinoth Vs The Inspector of Police",
  "description": "No.4399 of 2023 BEFORE THE MADURAI BENCH OF MADRAS HIGH COURT ( Criminal Jurisdiction ) Thursday, ...",
  "judge": "HONOURABLE  MR JUSTICE G.K. ILANTHIRAIYAN",
  "pdf_link": "court/cnrorders/mdubench/orders/HCMD010287762023_1_2023-03-16.pdf",
  "cnr": "HCMD010287762023",
  "date_of_registration": "14-03-2023",
  "decision_date": "2023-03-16",
  "disposal_nature": "DISMISSED",
  "court": "33_10",
  "raw_html": "<button type='button' role='link'..",
  "bench": "mdubench",
  "year": 2023
}
```

**Sample Questions:**

```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}
```

---

## Project Scope & Expectations

This is a broad and ambitious requirement. The examiner is essentially asking for a general-purpose data analysis agent that can:

- **Understand natural language analysis tasks**
- **Source data** from arbitrary locations (web, cloud, etc.)
- **Prepare, analyze, and visualize data**
- **Return results** in a specified format, quickly and reliably

This is a challenging open-ended problem, more like building a flexible data science assistant or a mini-AutoML/AutoAnalysis platform powered by LLMs. The expectation is not to cover "everything," but to demonstrate:

- Robust API design
- Integration with LLMs for task understanding
- Ability to fetch and process real-world data
- Dynamic code execution (e.g., running SQL, Python, or visualization code)
- Handling diverse data and question types
- Producing correct, well-formatted, and timely responses

A strong solution would focus on modularity, extensibility, and clear handling of a few representative task types, showing how the system could be extended to more. The examiner is likely looking for your approach, architecture, and ability to handle ambiguity, not a perfect "do-everything" agent.

---

## Recommended Tools & Frameworks

**API & Backend**

- FastAPI (Python) or Flask: For quick REST API development
- Uvicorn or Gunicorn: For serving FastAPI/Flask apps

**LLM Integration**

- OpenAI API (for GPT-3.5/4) or Azure OpenAI
- LangChain: For orchestrating LLM workflows and tool use
- Hugging Face Transformers: For open-source LLMs

**Data Sourcing & Processing**

- Requests, httpx, or aiohttp: For web scraping and HTTP requests
- BeautifulSoup, lxml: For HTML parsing
- Pandas: For data manipulation and analysis
- DuckDB: For SQL queries on local/cloud data
- pyarrow: For working with Parquet files

**Visualization**

- Matplotlib or Seaborn: For generating plots
- io.BytesIO + base64: For encoding images as data URIs

**Cloud & Storage**

- AWS S3 SDK (boto3): For accessing S3 buckets
- Google Cloud Storage SDK

**Testing & Validation**

- Pytest: For automated testing
- Postman or curl: For API testing

**Deployment**

- Docker: For containerization
- Render, Heroku, or AWS Lambda: For quick deployment

These tools will help you build, test, and deploy a modular, scalable, and robust data analyst agent efficiently.

---

## High-Level Design with LangChain

### 1. Input Layer

- **API Endpoint**: A REST API (e.g., built with FastAPI) to accept POST requests containing the task description.
- **Input Format**: Plain text or JSON describing the data analysis task.

---

### 2. Task Orchestration with LangChain

- **LangChain Workflow**:
  1. **Task Parsing**: Use LangChain to parse the input task description using an LLM (e.g., OpenAI GPT-4).
  2. **Tool Selection**: Dynamically select tools (e.g., web scraper, SQL executor, or visualization generator) based on the task.
  3. **Execution Chain**: Chain together the required steps (e.g., data retrieval → analysis → visualization).

---

### 3. Tool Integration

- **Data Retrieval**:
  - Use LangChain’s `RequestsWrapper` for web scraping.
  - Integrate with cloud storage APIs (e.g., AWS S3) for structured data.
- **Data Analysis**:
  - Use Python libraries (e.g., Pandas, DuckDB) for statistical analysis and transformations.
  - Execute SQL queries dynamically using LangChain’s `SQLDatabaseChain`.
- **Visualization**:
  - Generate plots using Matplotlib or Seaborn.
  - Encode plots as base64 data URIs for API responses.

---

### 4. Response Generation

- **Formatting**:
  - Use LangChain’s `OutputParser` to format results as JSON or other requested formats.
- **Validation**:
  - Ensure responses are well-structured and returned within the 3-minute SLA.

---

### 5. Modular Components

- **Chains**:
  - **Data Retrieval Chain**: Fetch data from web or cloud.
  - **Analysis Chain**: Perform statistical or machine learning tasks.
  - **Visualization Chain**: Generate and encode visual outputs.
- **Agents**:
  - Use LangChain’s `AgentExecutor` to dynamically decide which chain or tool to invoke based on the task.

---

### 6. Deployment

- **Containerization**: Use Docker to package the application.
- **Hosting**: Deploy on platforms like AWS, Heroku, or Render.
- **Scaling**: Use serverless functions (e.g., AWS Lambda) for handling multiple requests.
