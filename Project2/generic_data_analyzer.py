"""
Generic Data Analysis Module - Replacement for domain-specific logic

This module provides a generalized approach to data analysis that can handle
any domain without hardcoded logic for specific use cases like movies, sports, etc.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GenericDataAnalyzer:
    """
    Generic data analyzer that replaces domain-specific hardcoded logic
    Uses LLM-based analysis to intelligently handle any data type
    """
    
    def __init__(self):
        self.llm = None
        
    def analyze_data_generically(
        self, 
        data: pd.DataFrame, 
        analysis_col: str, 
        name_col: str, 
        task_description: str,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Perform generic data analysis without domain-specific assumptions
        
        Args:
            data: The data to analyze
            analysis_col: Column to analyze (numeric values)
            name_col: Column with item names/identifiers
            task_description: Description of what user wants to analyze
            top_n: Number of top items to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        try:
            # Basic statistical analysis (universal)
            results.update(self._calculate_basic_stats(data, analysis_col))
            
            # Top N analysis (universal)
            top_items = self._get_top_n_items(data, analysis_col, name_col, top_n)
            results['top_items'] = top_items
            
            # Threshold analysis (adaptive based on data distribution)
            results.update(self._analyze_thresholds(data, analysis_col))
            
            # Temporal analysis if year/date columns exist
            temporal_results = self._analyze_temporal_patterns(
                data, analysis_col, name_col
            )
            if temporal_results:
                results.update(temporal_results)
            
            # LLM-based intelligent question answering
            llm_insights = self._get_llm_insights(
                data, analysis_col, name_col, task_description, top_items
            )
            results.update(llm_insights)
            
            logger.info(f"Generic analysis completed for {len(data)} records")
            
        except Exception as e:
            logger.error(f"Error in generic analysis: {e}")
            results['error'] = str(e)
            
        return results
    
    def _calculate_basic_stats(
        self, data: pd.DataFrame, analysis_col: str
    ) -> Dict[str, Any]:
        """Calculate universal statistical measures"""
        if analysis_col not in data.columns:
            return {}
            
        values = data[analysis_col].dropna()
        if len(values) == 0:
            return {}
        
        return {
            'total_count': len(data),
            'valid_count': len(values),
            'mean_value': float(values.mean()),
            'median_value': float(values.median()),
            'max_value': float(values.max()),
            'min_value': float(values.min()),
            'std_deviation': float(values.std()),
            'range': float(values.max() - values.min())
        }
    
    def _get_top_n_items(
        self, 
        data: pd.DataFrame, 
        analysis_col: str, 
        name_col: str, 
        n: int
    ) -> List[Dict[str, Any]]:
        """Get top N items by analysis column value"""
        if analysis_col not in data.columns or name_col not in data.columns:
            return []
        
        top_data = data.nlargest(n, analysis_col)
        
        top_items = []
        for i, (_, row) in enumerate(top_data.iterrows()):
            top_items.append({
                'rank': i + 1,
                'name': str(row[name_col]),
                'value': float(row[analysis_col]),
                'percentage_of_max': (
                    float(row[analysis_col]) / float(data[analysis_col].max()) * 100
                )
            })
        
        return top_items
    
    def _analyze_thresholds(
        self, data: pd.DataFrame, analysis_col: str
    ) -> Dict[str, Any]:
        """Analyze data using adaptive thresholds based on distribution"""
        if analysis_col not in data.columns:
            return {}
        
        values = data[analysis_col].dropna()
        if len(values) == 0:
            return {}
        
        # Calculate percentile-based thresholds
        percentiles = [50, 75, 90, 95, 99]
        threshold_analysis = {}
        
        for p in percentiles:
            threshold = np.percentile(values, p)
            count_above = len(values[values > threshold])
            threshold_analysis[f'above_{p}th_percentile'] = {
                'threshold': float(threshold),
                'count': int(count_above),
                'percentage': float(count_above / len(values) * 100)
            }
        
        return {'threshold_analysis': threshold_analysis}
    
    def _analyze_temporal_patterns(
        self, 
        data: pd.DataFrame, 
        analysis_col: str, 
        name_col: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze temporal patterns if date/year columns exist"""
        # Look for date/year columns
        date_columns = [
            col for col in data.columns 
            if any(keyword in col.lower() 
                   for keyword in ['year', 'date', 'time'])
        ]
        
        if not date_columns:
            return None
        
        temporal_results = {}
        
        for date_col in date_columns[:2]:  # Analyze first 2 date columns
            try:
                # Convert to numeric if it's a year column
                if 'year' in date_col.lower():
                    dates = pd.to_numeric(data[date_col], errors='coerce')
                else:
                    dates = pd.to_datetime(data[date_col], errors='coerce')
                
                # Analyze by decade or period
                if dates.notna().sum() > 0:
                    temporal_results[f'{date_col}_analysis'] = self._analyze_by_period(
                        data, analysis_col, dates, date_col
                    )
                    
            except Exception as e:
                logger.warning(f"Could not analyze temporal column {date_col}: {e}")
        
        return temporal_results if temporal_results else None
    
    def _analyze_by_period(
        self, 
        data: pd.DataFrame, 
        analysis_col: str, 
        dates: pd.Series, 
        date_col: str
    ) -> Dict[str, Any]:
        """Analyze data by time periods"""
        results = {}
        
        # For year columns, analyze by decade
        if 'year' in date_col.lower():
            # Filter valid years
            valid_mask = dates.notna() & (dates > 1900) & (dates < 2030)
            if valid_mask.sum() == 0:
                return results
            
            # Decade analysis
            decades = (dates[valid_mask] // 10) * 10
            decade_counts = decades.value_counts().sort_index()
            
            results['decade_distribution'] = {
                str(int(decade)): int(count) 
                for decade, count in decade_counts.items()
            }
            
            # Before/after analysis (dynamic threshold)
            median_year = dates[valid_mask].median()
            before_median = (dates < median_year).sum()
            after_median = (dates >= median_year).sum()
            
            results['temporal_split'] = {
                'median_year': float(median_year),
                'before_median': int(before_median),
                'after_median': int(after_median)
            }
        
        return results
    
    def _get_llm_insights(
        self, 
        data: pd.DataFrame, 
        analysis_col: str, 
        name_col: str,
        task_description: str,
        top_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to generate intelligent insights about the data"""
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Prepare data summary for LLM
            data_summary = {
                'total_records': len(data),
                'analysis_column': analysis_col,
                'name_column': name_col,
                'top_5_items': top_items[:5] if len(top_items) >= 5 else top_items,
                'data_range': {
                    'min': float(data[analysis_col].min()),
                    'max': float(data[analysis_col].max()),
                    'mean': float(data[analysis_col].mean())
                }
            }
            
            system_prompt = """You are an expert data analyst. Based on the provided data summary and task description, generate intelligent insights and answer likely questions about the data.

Focus on:
1. Key patterns and trends in the data
2. Notable outliers or extremes
3. Answers to likely analytical questions
4. Statistical insights that would be valuable

Return your analysis as a JSON object with specific insights and numerical answers."""

            human_prompt = """Task: {task_description}

Data Summary:
- Total records: {total_records}
- Analysis column: {analysis_column}
- Value range: {min_value:.2f} to {max_value:.2f}
- Average value: {mean_value:.2f}

Top 5 items:
{top_items}

Provide intelligent insights and likely question answers for this data."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "total_records": data_summary['total_records'],
                "analysis_column": analysis_col,
                "min_value": data_summary['data_range']['min'],
                "max_value": data_summary['data_range']['max'],
                "mean_value": data_summary['data_range']['mean'],
                "top_items": str(data_summary['top_5_items'])
            })
            
            # Try to parse as JSON, fallback to text summary
            try:
                import json
                llm_insights = json.loads(result)
                return {'llm_insights': llm_insights}
            except json.JSONDecodeError:
                # Extract key insights from text
                return {
                    'llm_insights': {
                        'summary': result[:500] + "..." if len(result) > 500 else result,
                        'analysis_method': 'text_based'
                    }
                }
                
        except Exception as e:
            logger.warning(f"LLM insights generation failed: {e}")
            return {'llm_insights': {'error': 'LLM analysis unavailable'}}


# Factory function to create analyzer
def create_generic_analyzer() -> GenericDataAnalyzer:
    """Factory function to create a generic data analyzer"""
    return GenericDataAnalyzer()


# Example usage function
def analyze_any_data(
    data: pd.DataFrame, 
    value_column: str, 
    identifier_column: str,
    task_description: str
) -> Dict[str, Any]:
    """
    Convenience function to analyze any tabular data generically
    
    Example:
        results = analyze_any_data(
            data=movie_df,
            value_column='gross_revenue',
            identifier_column='title',
            task_description='Analyze highest grossing movies'
        )
    """
    analyzer = create_generic_analyzer()
    return analyzer.analyze_data_generically(
        data, value_column, identifier_column, task_description
    )
