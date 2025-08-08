"""
Generalized DataAnalysisWorkflow that replaces separate hardcoded pipelines.
This workflow can handle multiple data source types and dynamically compose analysis steps.
"""

import logging
from typing import Dict, Any, List, Optional
from chains.base import BaseWorkflow
from .data_analysis_steps import (
    DataIngestionStep,
    DataValidationStep,
    DataPreprocessingStep,
    AnalysisStep,
    VisualizationStep,
    OutputStep
)

logger = logging.getLogger(__name__)


class DataAnalysisWorkflow(BaseWorkflow):
    """
    Generalized data analysis workflow that can handle multiple data source types
    and dynamically compose analysis steps based on requirements.

    This replaces separate WikipediaScrapingWorkflow, LegalDataAnalysisWorkflow, etc.
    with a unified, flexible approach.
    """

    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.steps = {}
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize all available analysis steps"""
        self.steps = {
            'ingestion': DataIngestionStep(llm=self.llm),
            'validation': DataValidationStep(llm=self.llm),
            'preprocessing': DataPreprocessingStep(llm=self.llm),
            'analysis': AnalysisStep(llm=self.llm),
            'visualization': VisualizationStep(llm=self.llm),
            'output': OutputStep(llm=self.llm)
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data analysis workflow dynamically based on input requirements
        """
        try:
            # Extract key parameters
            task_description = input_data.get('task_description', '')
            data_source_type = self._detect_data_source_type(input_data)
            analysis_requirements = self._extract_analysis_requirements(task_description)

            logger.info(f"Executing DataAnalysisWorkflow for {data_source_type} data source")

            # Step 1: Data Ingestion
            ingestion_input = {
                'source_type': data_source_type,
                'task_description': task_description,
                **input_data
            }

            ingestion_result = self.steps['ingestion'].execute(ingestion_input)
            data = ingestion_result['data']

            # Step 2: Data Validation (with strong validation layer)
            validation_input = {
                'data': data,
                'task_description': task_description
            }

            validation_result = self.steps['validation'].execute(validation_input)
            validated_data = validation_result['cleaned_data']

            # Step 3: Data Preprocessing
            preprocessing_config = self._determine_preprocessing_config(
                validated_data, task_description, analysis_requirements
            )

            preprocessing_input = {
                'data': validated_data,
                'task_description': task_description,
                'preprocessing_config': preprocessing_config
            }

            preprocessing_result = self.steps['preprocessing'].execute(preprocessing_input)
            processed_data = preprocessing_result['data']

            # Step 4: Analysis (LLM / ML - based with iterative reasoning)
            analysis_input = {
                'data': processed_data,
                'task_description': task_description,
                'analysis_config': analysis_requirements
            }

            analysis_result = self.steps['analysis'].execute(analysis_input)

            # Step 5: Self - check and iterative reasoning (if enabled)
            if analysis_requirements.get('enable_self_check', True):
                analysis_result = self._perform_self_check(
                    processed_data, analysis_result, task_description
                )

            # Step 6: Visualization
            visualization_input = {
                'data': processed_data,
                'task_description': task_description,
                'analysis_results': analysis_result,
                'visualization_config': analysis_requirements.get('visualization', {})
            }

            visualization_result = self.steps['visualization'].execute(visualization_input)

            # Step 7: Output Formatting
            output_input = {
                'results': {
                    'ingestion': ingestion_result,
                    'validation': validation_result,
                    'preprocessing': preprocessing_result,
                    'analysis': analysis_result,
                    'visualization': visualization_result
                },
                'task_description': task_description,
                'output_config': analysis_requirements.get('output', {})
            }

            final_result = self.steps['output'].execute(output_input)

            # Add workflow metadata
            final_result['workflow_metadata'] = {
                'workflow_type': 'DataAnalysisWorkflow',
                'data_source_type': data_source_type,
                'steps_executed': list(self.steps.keys()),
                'analysis_requirements': analysis_requirements,
                'validation_issues': validation_result.get('issues_found', []),
                'data_shape_final': processed_data.shape if hasattr(processed_data, 'shape') else None
            }

            return final_result

        except Exception as e:
            logger.error(f"DataAnalysisWorkflow execution failed: {e}")
            return {
                'error': str(e),
                'workflow_type': 'DataAnalysisWorkflow',
                'status': 'failed'
            }

    def _detect_data_source_type(self, input_data: Dict[str, Any]) -> str:
        """
        Detect the data source type from input parameters
        """
        # Check for explicit source type
        if 'source_type' in input_data:
            return input_data['source_type']

        # Check for URL (web scraping)
        if 'url' in input_data:
            return 'web_scraping'

        # Check for file uploads
        if 'files' in input_data:
            files = input_data['files']
            if files:
                first_file = files[0] if isinstance(files, list) else files
                filename = getattr(first_file, 'filename', str(first_file))

                if filename.endswith('.csv'):
                    return 'csv'
                elif filename.endswith(('.xls', '.xlsx')):
                    return 'excel'
                elif filename.endswith('.json'):
                    return 'json'

        # Check for data directly provided
        if 'data' in input_data:
            return 'direct'

        # Default fallback
        return 'web_scraping'

    def _extract_analysis_requirements(self, task_description: str) -> Dict[str, Any]:
        """
        Extract analysis requirements from task description using LLM
        """
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema import StrOutputParser
        import json

        system_prompt = """You are an expert at analyzing data analysis requirements.
        Based on the task description, extract specific analysis requirements and preferences."""

        human_prompt = """
        Task: {task_description}

        Please analyze this task and extract the following requirements:
        1. Analysis type preference (descriptive, predictive, comparative, etc.)
        2. Visualization needs (charts, plots, heatmaps, etc.)
        3. Statistical rigor level (basic, intermediate, advanced)
        4. Output format preferences (report, dashboard, raw data, etc.)
        5. Special preprocessing needs (normalization, encoding, etc.)
        6. Performance requirements (speed vs accuracy)

        Respond in JSON format with specific configuration options.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            llm_response = chain.invoke({
                'task_description': task_description
            })

            requirements = json.loads(llm_response)

            # Set defaults for missing fields
            default_requirements = {
                'analysis_type': 'descriptive',
                'enable_self_check': True,
                'visualization': {'enabled': True, 'max_charts': 3},
                'output': {'format': 'comprehensive_report'},
                'preprocessing': {'handle_missing': True, 'normalize': False, 'encode_categorical': True},
                'statistical_rigor': 'intermediate',
                'performance': 'balanced'
            }

            # Merge with defaults
            for key, default_value in default_requirements.items():
                if key not in requirements:
                    requirements[key] = default_value

            return requirements

        except Exception as e:
            logger.error(f"Failed to extract analysis requirements: {e}")

            # Return default requirements
            return {
                'analysis_type': 'descriptive',
                'enable_self_check': True,
                'visualization': {'enabled': True, 'max_charts': 3},
                'output': {'format': 'comprehensive_report'},
                'preprocessing': {'handle_missing': True, 'normalize': False, 'encode_categorical': True},
                'statistical_rigor': 'intermediate',
                'performance': 'balanced'
            }

    def _determine_preprocessing_config(self, data, task_description: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine preprocessing configuration based on data characteristics and requirements
        """
        import pandas as pd

        config = requirements.get('preprocessing', {})

        # Auto - detect needs based on data
        if hasattr(data, 'dtypes'):
            # Check if normalization is needed
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 1:
                # Check if scales are very different
                scales = []
                for col in numeric_cols:
                    if data[col].std() != 0:
                        scales.append(data[col].max() - data[col].min())

                if scales and max(scales) / min(scales) > 100:
                    config['normalize'] = True

            # Check for categorical encoding needs
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                config['encode_categorical'] = True

        return config

    def _perform_self_check(self, data, analysis_result: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """
        Perform self - check validation of analysis results using LLM
        """
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema import StrOutputParser
        import json

        system_prompt = """You are a data analysis validator. Review the analysis results
        and check for consistency, accuracy, and relevance to the original task."""

        human_prompt = """
        Original task: {task_description}

        Data characteristics:
        - Shape: {data_shape}
        - Columns: {columns}

        Analysis results to validate:
        {analysis_results}

        Please check:
        1. Do the results make sense given the data?
        2. Are the findings relevant to the original task?
        3. Are there any obvious errors or inconsistencies?
        4. What additional analysis might be needed?

        Respond with validation feedback and suggestions for improvement.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            data_shape = data.shape if hasattr(data, 'shape') else 'Unknown'
            columns = list(data.columns) if hasattr(data, 'columns') else 'Unknown'

            validation_feedback = chain.invoke({
                'task_description': task_description,
                'data_shape': data_shape,
                'columns': columns,
                'analysis_results': json.dumps(analysis_result, default=str, indent=2)
            })

            # Add validation feedback to results
            analysis_result['self_check'] = {
                'validation_performed': True,
                'feedback': validation_feedback,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Self - check validation failed: {e}")
            analysis_result['self_check'] = {
                'validation_performed': False,
                'error': str(e)
            }

        return analysis_result


class ComposableWorkflowBuilder:
    """
    Builder class for creating customized data analysis workflows
    with specific step configurations and compositions.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.workflow_steps = []
        self.step_configs = {}

    def add_step(self, step_name: str, step_class, config: Dict[str, Any] = None):
        """Add a step to the workflow composition"""
        self.workflow_steps.append(step_name)
        self.step_configs[step_name] = {
            'class': step_class,
            'config': config or {}
        }
        return self

    def build(self) -> 'ComposableDataWorkflow':
        """Build the composed workflow"""
        return ComposableDataWorkflow(
            steps=self.workflow_steps,
            step_configs=self.step_configs,
            llm=self.llm
        )


class ComposableDataWorkflow(BaseWorkflow):
    """
    A composable workflow that allows dynamic step composition at runtime
    """

    def __init__(self, steps: List[str], step_configs: Dict[str, Any], llm=None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.workflow_steps = steps
        self.step_configs = step_configs
        self.initialized_steps = {}
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize the configured steps"""
        for step_name in self.workflow_steps:
            step_config = self.step_configs[step_name]
            step_class = step_config['class']
            step_instance_config = step_config['config']

            self.initialized_steps[step_name] = step_class(
                llm=self.llm,
                **step_instance_config
            )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the composed workflow"""
        current_data = input_data
        execution_log = []

        try:
            for step_name in self.workflow_steps:
                step_instance = self.initialized_steps[step_name]

                logger.info(f"Executing step: {step_name}")

                step_result = step_instance.execute(current_data)

                execution_log.append({
                    'step_name': step_name,
                    'status': 'completed',
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                })

                # Update current_data for next step
                # Each step should preserve important data and add new results
                if 'data' in step_result:
                    current_data['data'] = step_result['data']

                current_data.update(step_result)

            return {
                'final_result': current_data,
                'execution_log': execution_log,
                'workflow_type': 'ComposableDataWorkflow',
                'steps_executed': self.workflow_steps
            }

        except Exception as e:
            logger.error(f"Composable workflow execution failed at step {step_name}: {e}")

            execution_log.append({
                'step_name': step_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': __import__('datetime').datetime.now().isoformat()
            })

            return {
                'error': str(e),
                'execution_log': execution_log,
                'workflow_type': 'ComposableDataWorkflow',
                'failed_at_step': step_name
            }


# Factory function for creating workflow instances
def create_data_analysis_workflow(
    source_type: str = 'auto',
    analysis_type: str = 'auto',
    custom_steps: Optional[List[str]] = None,
    llm=None
) -> BaseWorkflow:
    """
    Factory function to create appropriate data analysis workflow instances

    Args:
        source_type: Type of data source ('web_scraping', 'csv', 'json', etc.)
        analysis_type: Type of analysis ('descriptive', 'predictive', etc.)
        custom_steps: Custom list of steps to execute
        llm: Language model instance

    Returns:
        Configured workflow instance
    """

    if custom_steps:
        # Build a custom composable workflow
        builder = ComposableWorkflowBuilder(llm=llm)

        step_mapping = {
            'ingestion': DataIngestionStep,
            'validation': DataValidationStep,
            'preprocessing': DataPreprocessingStep,
            'analysis': AnalysisStep,
            'visualization': VisualizationStep,
            'output': OutputStep
        }

        for step_name in custom_steps:
            if step_name in step_mapping:
                builder.add_step(step_name, step_mapping[step_name])

        return builder.build()

    else:
        # Return the standard generalized workflow
        return DataAnalysisWorkflow(llm=llm)
