"""
Modular and composable data analysis steps that can be used across different workflows.
Each step is independent, reusable, and follows a consistent interface.
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from config import get_chat_model

logger = logging.getLogger(__name__)

class BaseDataAnalysisStep(ABC):
    """Base class for all data analysis steps"""

    """Base class for all data analysis steps"""

    def __init__(self, llm=None, **kwargs):
        self.llm = llm or get_chat_model()
        self.config = kwargs

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step with input data and return results"""
        pass

    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that required keys exist in input data"""
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        return True
class DataIngestionStep(BaseDataAnalysisStep):
    """Generic data ingestion step that handles multiple data source types"""

class DataIngestionStep(BaseDataAnalysisStep):
    """Generic data ingestion step that handles multiple data source types"""

    SUPPORTED_SOURCES = {
        'csv': '_load_csv',
        'json': '_load_json',
        'excel': '_load_excel',
        'api': '_load_api',
        'database': '_load_database',
        'web_scraping': '_load_web_scraping'
    }

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_input(input_data, ['source_type'])

        source_type = input_data['source_type']

        if source_type not in self.SUPPORTED_SOURCES:
            raise ValueError(
                f"Unsupported source type: {source_type}. "
                f"Supported types: {list(self.SUPPORTED_SOURCES.keys())}"
            )

        # Get the appropriate loader method
        loader_method = getattr(self, self.SUPPORTED_SOURCES[source_type])

        try:
            data = loader_method(input_data)

            return {
                'data': data,
                'source_type': source_type,
                'metadata': {
                    'shape': data.shape if hasattr(data, 'shape') else None,
                    'columns': list(data.columns) if hasattr(data, 'columns') else None,
                    'ingestion_timestamp': datetime.now().isoformat(),
                    'source_config': input_data.get('source_config', {})
                }
            }

        except Exception as e:
            logger.error(f"Data ingestion failed for {source_type}: {e}")
            raise

    def _load_csv(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from CSV file"""
        file_path = input_data.get('file_path') or input_data.get('data')
        config = input_data.get('source_config', {})

        return pd.read_csv(file_path, **config)

    def _load_json(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from JSON file"""
        file_path = input_data.get('file_path') or input_data.get('data')
        config = input_data.get('source_config', {})

        return pd.read_json(file_path, **config)

    def _load_excel(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from Excel file"""
        file_path = input_data.get('file_path') or input_data.get('data')
        config = input_data.get('source_config', {})

        return pd.read_excel(file_path, **config)

    def _load_api(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from API endpoint"""
        # Implementation would depend on API specifics
        raise NotImplementedError("API loading not yet implemented")

    def _load_database(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from database"""
        # Implementation would use database connectors
        raise NotImplementedError("Database loading not yet implemented")

    def _load_web_scraping(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Load data from web scraping"""
        # This would delegate to web scraping workflow
        from .web_scraping_steps import ScrapeTableStep
        scraper = ScrapeTableStep()
        result = scraper.run(input_data)
class DataValidationStep(BaseDataAnalysisStep):
    """Strong data validation layer with schema checks, outlier detection, and type enforcement"""


class DataValidationStep(BaseDataAnalysisStep):
    """Strong data validation layer with schema checks, outlier detection, and type enforcement"""

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_input(input_data, ['data'])

        data = input_data['data']
        task_description = input_data.get('task_description', '')

        validation_results = {
            'data': data,
            'validation_report': {},
            'cleaned_data': None,
            'issues_found': []
        }

        # Perform various validation checks
        validation_results = self._check_data_quality(data, validation_results)
        validation_results = self._detect_outliers(data, validation_results)
        validation_results = self._enforce_data_types(data, validation_results, task_description)
        validation_results = self._check_completeness(data, validation_results)

        # Create cleaned version of data
        validation_results['cleaned_data'] = self._apply_cleaning(data, validation_results)

        return validation_results

    def _check_data_quality(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check basic data quality metrics"""
        quality_metrics = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'duplicate_rows': data.duplicated().sum(),
            'missing_values_per_column': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }

        results['validation_report']['quality_metrics'] = quality_metrics

        # Add issues based on quality checks
        if quality_metrics['duplicate_rows'] > 0:
            results['issues_found'].append(f"Found {quality_metrics['duplicate_rows']} duplicate rows")

        high_missing_cols = [
            col for col, missing in quality_metrics['missing_values_per_column'].items()
            if missing > len(data) * 0.5
        ]
        if high_missing_cols:
            results['issues_found'].append(f"Columns with >50% missing values: {high_missing_cols}")

        return results

    def _detect_outliers(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in numerical columns using IQR method"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}

        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }

        results['validation_report']['outliers'] = outlier_info

        high_outlier_cols = [col for col, info in outlier_info.items() if info['percentage'] > 5]
        if high_outlier_cols:
            results['issues_found'].append(f"Columns with >5% outliers: {high_outlier_cols}")

        return results

    def _enforce_data_types(self, data: pd.DataFrame, results: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Use LLM to suggest appropriate data types based on task description"""

        system_prompt = """You are a data validation expert. Analyze the data columns and task description
        to suggest appropriate data types and identify potential type conversion issues."""

        human_prompt = """
        Task: {task_description}

        Data columns and current types:
        {column_info}

        Sample data (first 5 rows):
        {sample_data}

        Please suggest:
        1. Optimal data types for each column
        2. Any columns that need type conversion
        3. Potential issues with current data types

        Respond in JSON format with suggestions.
        """

        # Prepare column information
        column_info = {col: str(dtype) for col, dtype in data.dtypes.items()}
        sample_data = data.head().to_dict()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            llm_response = chain.invoke({
                'task_description': task_description,
                'column_info': json.dumps(column_info, indent=2),
                'sample_data': json.dumps(sample_data, indent=2)
            })

            results['validation_report']['type_suggestions'] = llm_response

        except Exception as e:
            logger.error(f"LLM type suggestion failed: {e}")
            results['issues_found'].append("Could not get LLM type suggestions")

        return results

    def _check_completeness(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check data completeness and suggest handling strategies"""
        completeness_report = {
            'missing_value_patterns': {},
            'completeness_by_column': {},
            'suggested_actions': []
        }

        # Check completeness by column
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            completeness_pct = (1 - missing_count / len(data)) * 100
            completeness_report['completeness_by_column'][col] = {
                'missing_count': missing_count,
                'completeness_percentage': completeness_pct
            }

            # Suggest actions based on completeness
            if completeness_pct < 70:
                completeness_report['suggested_actions'].append(
                    f"Consider dropping column '{col}' (only {completeness_pct:.1f}% complete)"
                )
            elif completeness_pct < 95:
                completeness_report['suggested_actions'].append(
                    f"Consider imputing missing values in column '{col}' ({completeness_pct:.1f}% complete)"
                )

        results['validation_report']['completeness'] = completeness_report

        return results

    def _apply_cleaning(self, data: pd.DataFrame, results: Dict[str, Any]) -> pd.DataFrame:
        """Apply basic cleaning based on validation results"""
        cleaned_data = data.copy()

        # Remove duplicates if found
        if results['validation_report']['quality_metrics']['duplicate_rows'] > 0:
            cleaned_data = cleaned_data.drop_duplicates()
            logger.info(f"Removed {results['validation_report']['quality_metrics']['duplicate_rows']} duplicate rows")

        # Basic missing value handling (can be enhanced with LLM guidance)
        # For now, just drop columns that are >90% missing
        cols_to_drop = []
        for col, completeness_info in results['validation_report']['completeness']['completeness_by_column'].items():
            if completeness_info['completeness_percentage'] < 10:
                cols_to_drop.append(col)

        if cols_to_drop:
            cleaned_data = cleaned_data.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns with <10% completeness: {cols_to_drop}")
class DataPreprocessingStep(BaseDataAnalysisStep):
    """Generic data preprocessing step with cleaning and normalization"""



class DataPreprocessingStep(BaseDataAnalysisStep):
    """Generic data preprocessing step with cleaning and normalization"""

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_input(input_data, ['data'])

        data = input_data['data']
        task_description = input_data.get('task_description', '')
        preprocessing_config = input_data.get('preprocessing_config', {})

        processed_data = data.copy()
        processing_log = []

        # Apply preprocessing operations based on configuration
        if preprocessing_config.get('handle_missing', True):
            processed_data, log_entry = self._handle_missing_values(processed_data, task_description)
            processing_log.append(log_entry)

        if preprocessing_config.get('normalize', False):
            processed_data, log_entry = self._normalize_data(processed_data)
            processing_log.append(log_entry)

        if preprocessing_config.get('encode_categorical', True):
            processed_data, log_entry = self._encode_categorical(processed_data)
            processing_log.append(log_entry)

        return {
            'data': processed_data,
            'original_data': data,
            'preprocessing_log': processing_log,
            'metadata': {
                'shape_before': data.shape,
                'shape_after': processed_data.shape,
                'columns_before': list(data.columns),
                'columns_after': list(processed_data.columns)
            }
        }

    def _handle_missing_values(self, data: pd.DataFrame, task_description: str) -> tuple:
        """Handle missing values using LLM - guided strategy"""

        system_prompt = """You are a data preprocessing expert. Recommend the best strategy
        for handling missing values based on the data characteristics and task requirements."""

        human_prompt = """
        Task: {task_description}

        Missing value summary:
        {missing_summary}

        Data types:
        {data_types}

        Recommend the best approach for each column with missing values:
        - drop: Remove rows / columns with missing values
        - mean_impute: Fill with mean (for numeric)
        - median_impute: Fill with median (for numeric)
        - mode_impute: Fill with mode (for categorical)
        - forward_fill: Use previous value
        - interpolate: Interpolate values
        - custom_value: Fill with specific value

        Respond in JSON format with column - specific recommendations.
        """

        # Get missing value summary
        missing_summary = data.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].to_dict()

        if not missing_summary:
            return data, "No missing values found"

        data_types = data.dtypes.to_dict()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            llm_response = chain.invoke({
                'task_description': task_description,
                'missing_summary': json.dumps(missing_summary, indent=2),
                'data_types': json.dumps({k: str(v) for k, v in data_types.items()}, indent=2)
            })

            # Apply basic missing value handling (simplified for now)
            processed_data = data.copy()

            # Simple strategy: fill numeric with median, categorical with mode
            for col in data.columns:
                if data[col].isnull().any():
                    if data[col].dtype in ['int64', 'float64']:
                        processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    else:
                        processed_data[col] = processed_data[col].fillna(processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else 'Unknown')

            log_entry = f"Applied missing value handling based on LLM recommendation: {llm_response[:200]}..."

            return processed_data, log_entry

        except Exception as e:
            logger.error(f"LLM - guided missing value handling failed: {e}")

            # Fallback to basic strategy
            processed_data = data.dropna()
            return processed_data, f"Applied fallback missing value handling (dropna): {e}"

    def _normalize_data(self, data: pd.DataFrame) -> tuple:
        """Normalize numerical columns"""
        processed_data = data.copy()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if processed_data[col].std() != 0:  # Avoid division by zero
                processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()

        return processed_data, f"Normalized {len(numeric_cols)} numerical columns"

    def _encode_categorical(self, data: pd.DataFrame) -> tuple:
        """Encode categorical variables"""
        processed_data = data.copy()
        categorical_cols = processed_data.select_dtypes(include=['object']).columns

        encoded_cols = []
        for col in categorical_cols:
            if processed_data[col].nunique() < 20:  # Only encode if not too many unique values
                # Use simple label encoding for now
                unique_values = processed_data[col].unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                processed_data[f"{col}_encoded"] = processed_data[col].map(encoding_map)
                encoded_cols.append(col)

        return processed_data, f"Encoded {len(encoded_cols)} categorical columns: {encoded_cols}"


class AnalysisStep(BaseDataAnalysisStep):
    """LLM / ML - based analysis step with flexible analysis types"""

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_input(input_data, ['data', 'task_description'])

        data = input_data['data']
        task_description = input_data['task_description']
        analysis_config = input_data.get('analysis_config', {})

        # Determine analysis type using LLM
        analysis_type = self._determine_analysis_type(data, task_description)

        # Perform the appropriate analysis
        analysis_results = self._perform_analysis(data, task_description, analysis_type, analysis_config)

        return {
            'analysis_results': analysis_results,
            'analysis_type': analysis_type,
            'data': data,
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_shape': data.shape,
                'analysis_config': analysis_config
            }
        }

    def _determine_analysis_type(self, data: pd.DataFrame, task_description: str) -> str:
        """Use LLM to determine the most appropriate analysis type"""

        system_prompt = """You are a data analysis expert. Based on the data characteristics
        and task description, determine the most appropriate type of analysis to perform."""

        human_prompt = """
        Task: {task_description}

        Data characteristics:
        - Shape: {data_shape}
        - Columns: {columns}
        - Numeric columns: {numeric_columns}
        - Categorical columns: {categorical_columns}

        Available analysis types:
        - descriptive: Basic statistical analysis and summaries
        - correlation: Relationship analysis between variables
        - comparative: Comparison between groups or categories
        - trend: Time series or sequential pattern analysis
        - predictive: Predictive modeling and forecasting
        - clustering: Grouping and segmentation analysis
        - statistical_test: Hypothesis testing and significance tests

        Which analysis type is most appropriate? Respond with just the analysis type name.
        """

        # Prepare data characteristics
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(data.select_dtypes(include=['object']).columns)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            analysis_type = chain.invoke({
                'task_description': task_description,
                'data_shape': data.shape,
                'columns': list(data.columns),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols
            })

            # Clean and validate the response
            analysis_type = analysis_type.strip().lower()
            valid_types = ['descriptive', 'correlation', 'comparative', 'trend', 'predictive', 'clustering', 'statistical_test']

            if analysis_type not in valid_types:
                logger.warning(f"LLM returned invalid analysis type: {analysis_type}. Using 'descriptive' as fallback.")
                analysis_type = 'descriptive'

            return analysis_type

        except Exception as e:
            logger.error(f"Analysis type determination failed: {e}")
            return 'descriptive'  # Fallback to descriptive analysis

    def _perform_analysis(self, data: pd.DataFrame, task_description: str, analysis_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the specified type of analysis"""

        analysis_methods = {
            'descriptive': self._descriptive_analysis,
            'correlation': self._correlation_analysis,
            'comparative': self._comparative_analysis,
            'trend': self._trend_analysis,
            'predictive': self._predictive_analysis,
            'clustering': self._clustering_analysis,
            'statistical_test': self._statistical_test_analysis
        }

        analysis_method = analysis_methods.get(analysis_type, self._descriptive_analysis)

        try:
            return analysis_method(data, task_description, config)
        except Exception as e:
            logger.error(f"Analysis execution failed for {analysis_type}: {e}")
            # Fallback to basic descriptive analysis
            return self._descriptive_analysis(data, task_description, config)

    def _descriptive_analysis(self, data: pd.DataFrame, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        results = {}

        # Basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results['numeric_summary'] = data[numeric_cols].describe().to_dict()

        # Summary for categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            results['categorical_summary'] = {}
            for col in categorical_cols:
                results['categorical_summary'][col] = {
                    'unique_count': data[col].nunique(),
                    'top_values': data[col].value_counts().head().to_dict()
                }

        # Overall dataset summary
        results['dataset_summary'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'missing_values': data.isnull().sum().sum()
        }

        return results

    def _correlation_analysis(self, data: pd.DataFrame, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation analysis"""
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            return {'error': 'Insufficient numeric columns for correlation analysis'}

        correlation_matrix = numeric_data.corr()

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': self._find_strong_correlations(correlation_matrix),
            'correlation_summary': f"Analyzed correlations between {numeric_data.shape[1]} numeric variables"
        }

    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations above threshold"""
        strong_corrs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) >= 0.8 else 'moderate'
                    })

        return strong_corrs

    def _comparative_analysis(self, data: pd.DataFrame, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis between groups"""
        # This is a simplified implementation
        # In a real scenario, this would use LLM to identify comparison groups
        return {'analysis_type': 'comparative', 'message': 'Comparative analysis implementation needed'}

    def _trend_analysis(self, data: pd.DataFrame, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform trend analysis"""
        # This is a simplified implementation
        # In a real scenario, this would identify time columns and perform trend analysis
        return {'analysis_type': 'trend', 'message': 'Trend analysis implementation needed'}

    def _predictive_analysis(self, data: pd.DataFrame, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform predictive analysis"""
        # This is a simplified implementation
        # In a real scenario, this would build ML models
        return {'analysis_type': 'predictive', 'message': 'Predictive analysis implementation needed'}

    def _clustering_analysis(self, data: pd.DataFrame, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering analysis"""
        # This is a simplified implementation
        # In a real scenario, this would perform clustering algorithms
        return {'analysis_type': 'clustering', 'message': 'Clustering analysis implementation needed'}

    def _statistical_test_analysis(self, data: pd.DataFrame, task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical test analysis"""
        # This is a simplified implementation
        # In a real scenario, this would perform appropriate statistical tests
        return {'analysis_type': 'statistical_test', 'message': 'Statistical test analysis implementation needed'}


class VisualizationStep(BaseDataAnalysisStep):
    """Visualization step with automatic chart type detection and generation"""

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_input(input_data, ['data', 'task_description'])

        data = input_data['data']
        task_description = input_data['task_description']
        viz_config = input_data.get('visualization_config', {})

        # Determine appropriate visualizations
        viz_recommendations = self._recommend_visualizations(data, task_description)

        # Generate visualizations
        visualizations = self._generate_visualizations(data, viz_recommendations, viz_config)

        return {
            'visualizations': visualizations,
            'recommendations': viz_recommendations,
            'data': data,
            'metadata': {
                'viz_timestamp': datetime.now().isoformat(),
                'data_shape': data.shape
            }
        }

    def _recommend_visualizations(self, data: pd.DataFrame, task_description: str) -> List[Dict[str, Any]]:
        """Use LLM to recommend appropriate visualizations"""

        system_prompt = """You are a data visualization expert. Recommend the most appropriate
        visualizations based on the data characteristics and analysis objectives."""

        human_prompt = """
        Task: {task_description}

        Data characteristics:
        - Shape: {data_shape}
        - Numeric columns: {numeric_columns}
        - Categorical columns: {categorical_columns}
        - Sample data: {sample_data}

        Available chart types:
        - bar: For categorical comparisons
        - scatter: For relationship between two numeric variables
        - line: For trends over time or ordered data
        - histogram: For distribution of numeric variable
        - box: For distribution comparison across categories
        - heatmap: For correlation matrices
        - pie: For part - to - whole relationships

        Recommend up to 3 most appropriate visualizations. For each, specify:
        - chart_type: One of the available types
        - x_column: Column for x - axis
        - y_column: Column for y - axis (if applicable)
        - purpose: Brief description of what the chart shows

        Respond in JSON format as a list of visualization recommendations.
        """

        # Prepare data info
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(data.select_dtypes(include=['object']).columns)
        sample_data = data.head(3).to_dict()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            llm_response = chain.invoke({
                'task_description': task_description,
                'data_shape': data.shape,
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'sample_data': json.dumps(sample_data, indent=2)
            })

            # Parse LLM response (simplified for now)
            recommendations = json.loads(llm_response)

            return recommendations if isinstance(recommendations, list) else [recommendations]

        except Exception as e:
            logger.error(f"Visualization recommendation failed: {e}")

            # Fallback recommendations
            fallback_recommendations = []

            if len(numeric_cols) >= 2:
                fallback_recommendations.append({
                    'chart_type': 'scatter',
                    'x_column': numeric_cols[0],
                    'y_column': numeric_cols[1],
                    'purpose': f'Relationship between {numeric_cols[0]} and {numeric_cols[1]}'
                })

            if len(numeric_cols) >= 1:
                fallback_recommendations.append({
                    'chart_type': 'histogram',
                    'x_column': numeric_cols[0],
                    'y_column': None,
                    'purpose': f'Distribution of {numeric_cols[0]}'
                })

            return fallback_recommendations

    def _generate_visualizations(self, data: pd.DataFrame, recommendations: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate the recommended visualizations"""
        visualizations = []

        for rec in recommendations:
            try:
                viz_result = self._create_single_visualization(data, rec, config)
                if viz_result:
                    visualizations.append(viz_result)
            except Exception as e:
                logger.error(f"Failed to create visualization {rec}: {e}")

        return visualizations

    def _create_single_visualization(self, data: pd.DataFrame, recommendation: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a single visualization based on recommendation"""
        # This is a simplified implementation
        # In a real scenario, this would generate actual plots using matplotlib / plotly

        chart_type = recommendation.get('chart_type')
        x_column = recommendation.get('x_column')
        y_column = recommendation.get('y_column')

        # Validate columns exist
        if x_column and x_column not in data.columns:
            logger.error(f"Column {x_column} not found in data")
            return None

        if y_column and y_column not in data.columns:
            logger.error(f"Column {y_column} not found in data")
            return None

        # For now, return metadata about the visualization that would be created
        viz_info = {
            'chart_type': chart_type,
            'x_column': x_column,
            'y_column': y_column,
            'purpose': recommendation.get('purpose', ''),
            'data_points': len(data),
        }

        return viz_info


class OutputStep(BaseDataAnalysisStep):
    """Final output step that formats and structures results"""

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_input(input_data, ['results'])

        results = input_data['results']
        task_description = input_data.get('task_description', '')
        output_config = input_data.get('output_config', {})

        # Format results using LLM
        formatted_output = self._format_results(results, task_description, output_config)

        return {
            'formatted_output': formatted_output,
            'raw_results': results,
            'metadata': {
                'output_timestamp': datetime.now().isoformat(),
                'output_format': output_config.get('format', 'json')
            }
        }

    def _format_results(self, results: Dict[str, Any], task_description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to format results in a user - friendly way"""

        system_prompt = """You are an expert at presenting data analysis results.
        Format the analysis results in a clear, structured, and insightful way for the user."""

        human_prompt = """
        Original task: {task_description}

        Analysis results: {results}

        Please format these results into a clear, professional report that includes:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Recommendations (if applicable)
        5. Limitations and Caveats

        Make the report accessible to both technical and non - technical audiences.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            formatted_report = chain.invoke({
                'task_description': task_description,
                'results': json.dumps(results, default=str, indent=2)
            })

            return {
                'report': formatted_report,
                'format': 'formatted_text',
                'raw_data': results
            }

        except Exception as e:
            logger.error(f"Result formatting failed: {e}")

            return {
                'report': 'Failed to format results with LLM',
                'format': 'raw',
                'raw_data': results
            }
