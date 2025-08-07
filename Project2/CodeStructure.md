# CodeStructure.md

This document explains the main classes, methods, and their roles in the following files:
- `main.py`
- `chains/workflows.py`
- `chains/base.py`
- `chains/web_scraping_steps.py`

---

## main.py

**Purpose:**
- FastAPI application entry point, orchestrates workflow selection and API endpoints.

**Key Classes & Methods:**
- `TaskRequest`, `WorkflowRequest`, `MultiStepWorkflowRequest`, `TaskResponse`: Pydantic models for request/response validation.
- `extract_output_requirements(task_description)`: Extracts output requirements (e.g., visualization, format) from the task description.
- `detect_workflow_type_llm(task_description, default_workflow)`: Uses LLM to classify workflow type from the task description.
- `detect_workflow_type_fallback(task_description, default_workflow)`: Fallback keyword-based workflow detection.
- `prepare_workflow_parameters(task_description, workflow_type, file_content)`: Prepares workflow parameters based on the task and workflow type.
- FastAPI endpoints: `/`, `/health`, `/api/` for root, health check, and main API.
- Orchestrator initialization: Tries to initialize `AdvancedWorkflowOrchestrator`, falls back to `MinimalOrchestrator` if needed.

---

## chains/workflows.py

**Purpose:**
- Implements specialized workflows for data analysis, web scraping, image analysis, code generation, etc.

**Key Classes & Methods:**
- `ExploratoryDataAnalysisWorkflow`, `DataAnalysisWorkflow`, `ImageAnalysisWorkflow`, `CodeGenerationWorkflow`, `PredictiveModelingWorkflow`, `DataVisualizationWorkflow`, `WebScrapingWorkflow`, `DatabaseAnalysisWorkflow`, `StatisticalAnalysisWorkflow`, `MultiStepWebScrapingWorkflow`, `ModularWebScrapingWorkflow`: Each inherits from `BaseWorkflow` and implements an `async execute()` method for its domain.
- `AdvancedWorkflowOrchestrator`: Enhanced orchestrator that registers and manages all workflows.
- `STEP_REGISTRY`: Maps step names to web scraping step classes.
- Utility functions: `run_web_scraping_workflow`, `detect_steps_from_prompt`, `run_llm_planned_workflow` for orchestrating modular step-based workflows.

---

## chains/base.py

**Purpose:**
- Defines base classes and utilities for LangChain-powered workflows.

**Key Classes & Methods:**
- `BaseWorkflow`: Abstract base class for all workflows. Handles LLM/model initialization, memory, and requires `async execute()` implementation.
- `DataAnalysisChain`, `CodeGenerationChain`, `ReportGenerationChain`: Specialized chains for data analysis, code generation, and report creation. Each sets up its own prompt template and chain.
- `WorkflowOrchestrator`: Manages multiple workflows, provides execution history, and multi-step workflow execution.

---

## chains/web_scraping_steps.py

**Purpose:**
- Implements a modular, step-based web scraping and data analysis pipeline.

**Key Classes & Methods:**
- `DetectDataFormatStep`: Step 0. Uses LLM to detect data format and extraction strategy from HTML.
- `ScrapeTableStep`: Step 1. Extracts data using the detected format, supports HTML tables, JSON, JS, divs, etc.
- `InspectTableStep`: Step 2. Inspects and preprocesses tables, handles headers and MultiIndex columns.
- `CleanDataStep`: Step 3. Cleans data (symbols, footnotes, numeric conversion).
- `AnalyzeDataStep`: Step 4. Selects relevant columns, analyzes data, filters summary rows.
- `VisualizeStep`: Step 5. Auto-detects chart type, visualizes data, encodes images.
- `AnswerQuestionsStep`: Step 6. Answers domain-specific questions using cleaned/analyzed data.
- Each step class has a `run(input_data)` method and private helper methods for LLM-powered logic and fallbacks.

---

**Note:**
- All workflows and steps are designed to be generic, domain-adaptive, and leverage LLM prompting for key decisions (table selection, header detection, cleaning, column selection, chart type, domain detection).
- The orchestrator ensures robust error handling and fallback mechanisms.
