import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List
import math

def sanitize_for_json(obj):
    """
    Recursively sanitize dicts/lists/floats for JSON serialization.
    Converts NaN, inf, -inf to None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
import logging

logger = logging.getLogger(__name__)

class ScrapeTableStep:
    """
    Step 1: Generic web scraping and data extraction
    - Use pandas.read_html (preferred) for scraping tables
    - Print number of tables found
    - Select the most relevant table based on size and content
    - Print table info for verification
    - Robust error handling for various website types
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = input_data['url']
        try:
            tables = pd.read_html(url)
            print(f"Found {len(tables)} tables on the page")
            # Inspect all tables
            for i, table in enumerate(tables):
                print(f"\nTable {i}:")
                print(f"  Shape: {table.shape}")
                print(f"  Columns: {table.columns.tolist()}")
                print(f"  Sample data:")
                print(table.head(3))
            
            # LLM-powered table selection for generic web scraping
            best_table_idx = self._select_best_table_with_llm(tables, input_data.get('task_description', ''))
            data = tables[best_table_idx]
            
            print(f"LLM selected table {best_table_idx} with {data.shape[0]} rows and {data.shape[1]} columns")
            
        except Exception as e:
            print(f"Error scraping tables: {str(e)}")
            raise
        
        return sanitize_for_json({'data': data, 'url': url})
    
    def _select_best_table_with_llm(self, tables: List[pd.DataFrame], task_description: str = '') -> int:
        """
        Use LLM to intelligently select the most relevant table for analysis
        Based on task description and table previews
        """
        try:
            # Import LLM components here to avoid circular imports
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Create table previews for LLM analysis
            table_previews = []
            for i, table in enumerate(tables):
                # Get sample of first 3 rows for each table
                preview = {
                    'index': i,
                    'shape': f"{table.shape[0]} rows × {table.shape[1]} columns",
                    'columns': table.columns.tolist()[:10],  # Limit to first 10 columns
                    'sample_data': table.head(3).to_string(max_cols=10, max_rows=3)
                }
                table_previews.append(preview)
            
            # Create LLM prompt for table selection
            system_prompt = """You are an expert web scraping assistant. Given multiple HTML tables from a webpage, 
            select the most relevant table for data analysis based on the task description.
            
            Consider these factors:
            1. Table size and data density
            2. Column relevance to the task
            3. Data quality and completeness
            4. Avoid summary/navigation tables
            
            Respond with ONLY the table index number (0, 1, 2, etc.) that best matches the analysis requirements."""
            
            human_prompt = """Task: {task_description}

Available tables:
{table_info}

Which table index (0-{max_index}) contains the most relevant data for this analysis?
Respond with just the number."""
            
            # Format table information for LLM
            table_info_str = ""
            for preview in table_previews:
                table_info_str += f"\nTable {preview['index']}: {preview['shape']}\n"
                table_info_str += f"Columns: {preview['columns']}\n"
                table_info_str += f"Sample data:\n{preview['sample_data']}\n"
                table_info_str += "-" * 50 + "\n"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            # Get LLM model and create chain
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            # Invoke LLM with table data
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "table_info": table_info_str,
                "max_index": len(tables) - 1
            })
            
            # Parse LLM response to get table index
            try:
                selected_index = int(result.strip())
                if 0 <= selected_index < len(tables):
                    print(f"LLM selected table {selected_index} for task: {task_description}")
                    return selected_index
                else:
                    print(f"LLM returned invalid index {selected_index}, using fallback")
                    return self._fallback_table_selection(tables)
            except (ValueError, AttributeError):
                print(f"Could not parse LLM response: {result}, using fallback")
                return self._fallback_table_selection(tables)
                
        except Exception as e:
            print(f"Error in LLM table selection: {str(e)}, using fallback")
            return self._fallback_table_selection(tables)
    
    def _fallback_table_selection(self, tables: List[pd.DataFrame]) -> int:
        """
        Fallback method for table selection when LLM is unavailable
        Uses simple heuristics based on table size and content
        """
        if not tables:
            return 0
            
        best_idx = 0
        best_score = 0
        
        for i, table in enumerate(tables):
            score = 0
            
            # Prefer tables with reasonable size (10+ rows, 2+ columns)
            if table.shape[0] >= 10:
                score += min(table.shape[0] * 0.5, 50)
            if table.shape[1] >= 2:
                score += table.shape[1] * 5
                
            # Prefer tables with numeric data
            numeric_cols = table.select_dtypes(include=[np.number]).columns
            score += len(numeric_cols) * 10
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        print(f"Fallback selection: table {best_idx} (score: {best_score})")
        return best_idx

class InspectTableStep:
    """
    Step 2: Generic data inspection
    - Print shape, columns, and head
    - Handle MultiIndex columns (flatten if needed)
    - Check if first row contains headers and set as columns if so
    - Generic approach for various data types (movies, countries, sports, etc.)
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data'].copy()
        print(f"\nSelected data shape: {data.shape}")
        print(f"Original columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head())
        
        # Handle MultiIndex columns (common in Wikipedia and other tables)
        if isinstance(data.columns, pd.MultiIndex):
            print("\nDetected MultiIndex columns, flattening...")
            new_columns = []
            for col in data.columns:
                if isinstance(col, tuple):
                    # Take the most specific part of the column name
                    if col[1] and col[1] != col[0]:
                        new_columns.append(f"{col[0]}_{col[1]}")
                    else:
                        new_columns.append(str(col[0]))
                else:
                    new_columns.append(str(col))
            data.columns = new_columns
            print(f"Flattened columns: {data.columns.tolist()}")
        
        # LLM-powered header detection for generic web scraping
        first_row_is_header, header_row_idx = self._detect_headers_with_llm(data, input_data.get('task_description', ''))
        
        # Apply header detection results
        if first_row_is_header and header_row_idx is not None:
            if header_row_idx > 0:
                print(f"Found headers in row {header_row_idx}")
                data.columns = [str(val) for val in data.iloc[header_row_idx]]
                data = data[header_row_idx+1:].reset_index(drop=True)
            else:
                print("First row confirmed as headers, setting as column names...")
                data.columns = [str(val) for val in data.iloc[0]]
                data = data[1:].reset_index(drop=True)
            print(f"Updated columns: {data.columns.tolist()}")
        
        print("\nAfter column processing:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head(3))
        
        return sanitize_for_json({'data': data})
    
    def _detect_headers_with_llm(self, data: pd.DataFrame, task_description: str = '') -> tuple:
        """
        Use LLM to intelligently detect if any row contains headers
        Returns (is_header, row_index) tuple
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Check first 3 rows for potential headers
            rows_to_check = min(3, len(data))
            table_sample = data.head(rows_to_check).to_string(max_cols=10)
            
            system_prompt = """You are an expert data analyst. Examine the first few rows of a table and determine if any row contains column headers.
            
            Look for:
            1. Descriptive names instead of data values
            2. Text patterns typical of headers (Name, Rank, Total, etc.)
            3. Consistency with the analysis task
            4. Non-numeric values in what should be data rows
            
            Respond with ONLY the row index (0, 1, 2) that contains headers, or "NONE" if no headers are found in the data rows."""
            
            human_prompt = """Task: {task_description}

Table sample (first {rows_count} rows):
{table_sample}

Current column names: {current_columns}

Which row index (0, 1, 2) contains the headers, or "NONE" if headers are not in the data rows?
Respond with just the number or "NONE"."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "table_sample": table_sample,
                "rows_count": rows_to_check,
                "current_columns": data.columns.tolist()
            })
            
            # Parse LLM response
            result_clean = result.strip().upper()
            if result_clean == "NONE":
                print("LLM determined no headers in data rows")
                return False, None
            else:
                try:
                    header_row = int(result_clean)
                    if 0 <= header_row < rows_to_check:
                        print(f"LLM detected headers in row {header_row}")
                        return True, header_row
                    else:
                        print(f"LLM returned invalid row index: {result}, using fallback")
                        return self._fallback_header_detection(data)
                except ValueError:
                    print(f"Could not parse LLM response: {result}, using fallback")
                    return self._fallback_header_detection(data)
                    
        except Exception as e:
            print(f"Error in LLM header detection: {str(e)}, using fallback")
            return self._fallback_header_detection(data)
    
    def _fallback_header_detection(self, data: pd.DataFrame) -> tuple:
        """
        Fallback method for header detection when LLM is unavailable
        """
        # Check if column names are non-descriptive
        has_unnamed_cols = any(str(col).startswith('Unnamed') or str(col).isdigit() for col in data.columns)
        
        if has_unnamed_cols and len(data) > 0:
            # Check first row for header-like content
            first_row_values = data.iloc[0].astype(str).tolist()
            # Simple check for text that looks like headers
            header_like_count = sum(1 for val in first_row_values if 
                                  len(str(val)) > 1 and not str(val).replace('.', '').replace('-', '').isdigit())
            
            if header_like_count >= len(first_row_values) * 0.5:  # At least half look like headers
                print("Fallback detected likely headers in first row")
                return True, 0
        
        print("Fallback found no headers in data rows")
        return False, None

class CleanDataStep:
    """
    Step 3: Generic data cleaning
    - Remove symbols, footnotes, and convert to numeric
    - CRITICAL: Use select_dtypes to find numeric columns
    - Handle various data formats (currency, percentages, etc.)
    - Print after cleaning
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data'].copy()
        print("\n=== CLEANING DATA ===")
        print(f"Original data types:\n{data.dtypes}")
        
        # Clean each column
        for col in data.columns:
            if data[col].dtype == 'object':
                print(f"\nCleaning column: {col}")
                # Convert to string first
                cleaned = data[col].astype(str)
                
                # Remove common symbols and formatting (generic for various data types)
                cleaned = cleaned.str.replace('$', '', regex=False)  # Currency
                cleaned = cleaned.str.replace(',', '', regex=False)  # Thousands separator
                cleaned = cleaned.str.replace('€', '', regex=False)  # Euro
                cleaned = cleaned.str.replace('£', '', regex=False)  # Pound
                cleaned = cleaned.str.replace('¥', '', regex=False)  # Yen
                cleaned = cleaned.str.replace('%', '', regex=False)  # Percentage
                cleaned = cleaned.str.replace('₹', '', regex=False)  # Rupee
                cleaned = cleaned.str.replace('billion', '', regex=False)  # Scale indicators
                cleaned = cleaned.str.replace('million', '', regex=False)
                cleaned = cleaned.str.replace('trillion', '', regex=False)
                cleaned = cleaned.str.replace('bn', '', regex=False)
                cleaned = cleaned.str.replace('mn', '', regex=False)
                # Add more currency and number formats
                cleaned = cleaned.str.replace('B', '', regex=False)  # Billion abbreviation
                cleaned = cleaned.str.replace('M', '', regex=False)  # Million abbreviation
                cleaned = cleaned.str.replace('K', '', regex=False)  # Thousand abbreviation
                
                # Remove footnote references like [1], [n 1], etc.
                cleaned = cleaned.str.replace(r'\[.*?\]', '', regex=True)
                cleaned = cleaned.str.replace(r'\([^)]*\)', '', regex=True)  # Remove parentheses content
                
                # Remove any other non-numeric characters except decimal points and minus signs
                cleaned = cleaned.str.replace(r'[^\d.\-]', '', regex=True)
                
                # Handle empty strings and special cases
                cleaned = cleaned.replace('', np.nan)
                cleaned = cleaned.replace('nan', np.nan)
                cleaned = cleaned.replace('NaN', np.nan)
                cleaned = cleaned.replace('None', np.nan)
                cleaned = cleaned.replace('N/A', np.nan)
                cleaned = cleaned.replace('–', np.nan)  # En dash
                cleaned = cleaned.replace('—', np.nan)  # Em dash
                
                # Convert to numeric
                numeric_data = pd.to_numeric(cleaned, errors='coerce')
                
                # Only replace if we got some valid numbers (at least 5% of data should be numeric)
                valid_count = numeric_data.notna().sum()
                total_count = len(data)
                if valid_count > 0 and valid_count >= max(3, total_count * 0.05):  # Lower threshold for small datasets
                    print(f"  Converted {valid_count} values to numeric ({valid_count/total_count*100:.1f}%)")
                    data[col] = numeric_data
                    
                    # Handle scale factors (if column had billion/million indicators)
                    original_str = data[col].astype(str).str.lower()
                    if any('billion' in str(val) or 'bn' in str(val) for val in data[col].astype(str).iloc[:5]):
                        print(f"  Detected billion scale factor in {col}")
                        # Values are likely already in billions, don't multiply
                    elif any('million' in str(val) or 'mn' in str(val) for val in data[col].astype(str).iloc[:5]):
                        print(f"  Detected million scale factor in {col}, converting to billions")
                        data[col] = data[col] / 1000  # Convert millions to billions
                else:
                    print(f"  No valid numeric data found ({valid_count} values), keeping as text")
        
        print("\nAfter cleaning:")
        print(f"Data types:\n{data.dtypes}")
        print(data.head(3))
        
        # Find numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumeric columns found: {numeric_cols}")
        
        # Print some statistics about numeric columns
        for col in numeric_cols:
            valid_count = data[col].notna().sum()
            if valid_count > 0:
                print(f"  {col}: {valid_count} valid values, range: {data[col].min():.2f} to {data[col].max():.2f}")
        
        # Replace inf/-inf with nan, then nan with None for JSON
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.where(pd.notnull(data), None)
        return sanitize_for_json({'data': data, 'numeric_cols': numeric_cols})

class AnalyzeDataStep:
    """
    Step 4: Generic data analysis
    - Use most relevant numeric column for analysis
    - Remove NaNs, sort, get top N
    - Support various analysis types (rankings, filtering, etc.)
    - Print results in a generic format
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data']
        numeric_cols = input_data['numeric_cols']
        top_n = input_data.get('top_n', 10)
        
        print(f"\n=== ANALYZING DATA ===")
        print(f"Available numeric columns: {numeric_cols}")
        
        if not numeric_cols:
            print("ERROR: No numeric columns found for analysis")
            return {'top_n_df': pd.DataFrame(), 'analysis_col': None}
        
        # LLM-powered column selection for analysis
        best_col = self._select_analysis_column_with_llm(data, numeric_cols, input_data.get('task_description', ''))
        
        if not best_col:
            print("ERROR: No suitable column found for analysis")
            return {'top_n_df': pd.DataFrame(), 'analysis_col': None}
        
        print(f"Selected column '{best_col}' for analysis")
        
        # Clean and analyze - first filter out summary/total rows using LLM
        data_clean = data.dropna(subset=[best_col])
        data_clean = self._filter_summary_rows_with_llm(data_clean, input_data.get('task_description', ''))
        
        print(f"After removing NaN values and filtering summary rows: {data_clean.shape[0]} rows")
        
        if len(data_clean) == 0:
            print("ERROR: No valid data after cleaning and filtering")
            return {'top_n_df': pd.DataFrame(), 'analysis_col': best_col}
        
        # Sort by the analysis column (descending for most metrics like revenue, GDP, etc.)
        data_sorted = data_clean.sort_values(best_col, ascending=False)
        top_n_df = data_sorted.head(top_n)
        
        print(f"\nTop {len(top_n_df)} by {best_col}:")
        
        # Find the identifier column (usually first text column)
        text_cols = data.select_dtypes(include=['object']).columns.tolist()
        name_col = text_cols[0] if text_cols else data.columns[0]
        
        print(f"Using '{name_col}' as identifier column")
        
        # Display results
        for i, (idx, row) in enumerate(top_n_df.iterrows()):
            identifier = row[name_col]
            value = row[best_col]
            print(f"{i+1:2d}. {identifier}: {value:,.2f}")
        
        # Replace inf/-inf with nan, then nan with None for JSON
        top_n_df = top_n_df.replace([np.inf, -np.inf], np.nan)
        top_n_df = top_n_df.where(pd.notnull(top_n_df), None)
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
        data_clean = data_clean.where(pd.notnull(data_clean), None)
        return sanitize_for_json({'top_n_df': top_n_df, 'analysis_col': best_col, 'name_col': name_col, 'data_clean': data_clean})
    
    def _select_analysis_column_with_llm(self, data: pd.DataFrame, numeric_cols: List[str], task_description: str = '') -> str:
        """
        Use LLM to intelligently select the most relevant numeric column for analysis
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Create column descriptions for LLM
            column_info = []
            for col in numeric_cols:
                valid_count = data[col].notna().sum()
                if valid_count > 0:
                    sample_values = data[col].dropna().head(5).tolist()
                    value_range = f"{data[col].min():.2f} to {data[col].max():.2f}"
                    column_info.append(f"Column '{col}': {valid_count} valid values, range {value_range}, samples: {sample_values}")
            
            system_prompt = """You are an expert data analyst. Given numeric columns and a task description, 
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
            
            human_prompt = """Task: {task_description}

Available numeric columns:
{column_descriptions}

Which column is most relevant for this analysis? Respond with just the column name."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "column_descriptions": "\n".join(column_info)
            })
            
            # Parse LLM response
            selected_col = result.strip().strip("'\"")  # Remove quotes
            if selected_col in numeric_cols:
                print(f"LLM selected column: {selected_col}")
                return selected_col
            else:
                print(f"LLM returned invalid column: {selected_col}, using fallback")
                return self._fallback_column_selection(data, numeric_cols)
                
        except Exception as e:
            print(f"Error in LLM column selection: {str(e)}, using fallback")
            return self._fallback_column_selection(data, numeric_cols)
    
    def _fallback_column_selection(self, data: pd.DataFrame, numeric_cols: List[str]) -> str:
        """
        Fallback method for column selection when LLM is unavailable
        """
        best_col = None
        best_score = 0
        
        for col in numeric_cols:
            score = 0
            col_name = str(col).lower()
            
            # Avoid summary/total/rank columns
            if any(keyword in col_name for keyword in ['world', 'total', 'sum', 'rank', 'position']):
                continue
            
            # Avoid year columns
            if col_name.isdigit() or (len(col_name) == 4 and col_name.startswith('2')):
                continue
            
            # Score based on data completeness
            valid_count = data[col].notna().sum()
            score += valid_count
            
            # Score based on value range (prefer columns with variation)
            if valid_count > 0:
                value_range = data[col].max() - data[col].min()
                if value_range > 0:
                    score += 100
            
            if score > best_score:
                best_score = score
                best_col = col
        
        # If no column found, use first numeric column
        if not best_col and numeric_cols:
            best_col = numeric_cols[0]
        
        print(f"Fallback selected column: {best_col}")
        return best_col
    
    def _filter_summary_rows_with_llm(self, data: pd.DataFrame, task_description: str = '') -> pd.DataFrame:
        """
        Use LLM to identify and filter out summary/total rows
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Find text columns that might contain identifiers
            text_cols = data.select_dtypes(include=['object']).columns.tolist()
            if not text_cols or len(data) == 0:
                return data
                
            name_col = text_cols[0]
            sample_data = data.head(20)  # Look at first 20 rows
            
            system_prompt = """You are a data cleaning expert. Examine the data rows and identify which ones are summary/total/aggregate rows that should be filtered out for analysis.

Look for rows that contain:
- Total/sum values (like "Total", "World", "All Countries", "Overall")
- Summary statistics (like "Average", "Mean", "Median")
- Aggregate categories (like "Other", "Others", "Miscellaneous")
- Non-specific entries that aren't individual data points

Return a comma-separated list of the exact values from the name column that should be removed.
If no summary rows are found, respond with "NONE"."""

            human_prompt = """Task: {task_description}

Data sample (first 20 rows, name column '{name_col}'):
{data_sample}

Which values from the '{name_col}' column represent summary/total rows that should be filtered out?
Respond with exact values separated by commas, or "NONE" if no summary rows found."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "name_col": name_col,
                "data_sample": sample_data[[name_col]].to_string()
            })
            
            # Parse LLM response and filter rows
            result_clean = result.strip().upper()
            if result_clean != "NONE":
                rows_to_remove = [item.strip().strip('"\'') for item in result.split(',')]
                before_count = len(data)
                
                for row_value in rows_to_remove:
                    if row_value:  # Skip empty values
                        # Case-insensitive filtering
                        mask = ~data[name_col].astype(str).str.lower().str.contains(
                            row_value.lower(), na=False, regex=False
                        )
                        data = data[mask]
                
                after_count = len(data)
                if before_count != after_count:
                    print(f"LLM identified and filtered out {before_count - after_count} summary rows")
                    print(f"Removed values: {rows_to_remove}")
            else:
                print("LLM found no summary rows to filter")
            
            return data
            
        except Exception as e:
            print(f"Error in LLM summary row filtering: {str(e)}, using fallback")
            # Fallback to keyword-based approach
            if len(data) > 0:
                text_cols = data.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    name_col = text_cols[0]
                    print(f"Filtering summary rows using '{name_col}' column")
                    
                    # Generic summary keywords for various domains
                    summary_keywords = [
                        'world', 'total', 'sum', 'all', 'global', 'aggregate',
                        'overall', 'average', 'mean', 'median', 'other', 'others'
                    ]
                    before_count = len(data)
                    for keyword in summary_keywords:
                        data = data[~data[name_col].astype(str).str.lower().str.contains(keyword, na=False)]
                    
                    after_count = len(data)
                    if before_count != after_count:
                        print(f"Filtered out {before_count - after_count} summary rows")
            
            return data

class VisualizeStep:
    """
    Step 5: Enhanced generic visualization
    - Support multiple chart types (bar, scatter, histogram, time_series)
    - Auto-detect visualization type based on task requirements
    - Use dynamic column names
    - Return base64 encoded images when requested
    - Handle various data relationships (rank vs peak, total vs deaths, etc.)
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data.get('top_n_df')
        analysis_col = input_data.get('analysis_col')
        name_col = input_data.get('name_col')
        data_clean = input_data.get('data_clean')
        task_description = input_data.get('task_description', '')
        chart_type = input_data.get('chart_type', self._auto_detect_chart_type(task_description, data_clean))
        
        if top_n_df is None or top_n_df.empty or analysis_col is None:
            print("No data available for visualization.")
            return {'plot_path': None, 'plot_base64': None}
        
        try:
            import base64
            from io import BytesIO
            
            plt.figure(figsize=(12, 8))
            
            if chart_type == 'bar':
                # Standard bar chart for top N
                plt.bar(range(len(top_n_df)), top_n_df[analysis_col], color='skyblue', alpha=0.7)
                plt.xticks(range(len(top_n_df)), top_n_df[name_col], rotation=45, ha='right')
                plt.title(f'Top {len(top_n_df)} by {analysis_col}')
                plt.xlabel(name_col)
                plt.ylabel(analysis_col)
                
            elif chart_type == 'scatter':
                # Enhanced scatter plot with auto-detection of x/y columns
                numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
                x_col, y_col = self._select_scatter_columns(numeric_cols, analysis_col, task_description)
                
                if x_col and y_col and len(numeric_cols) >= 2:
                    clean_data = data_clean[[x_col, y_col]].dropna()
                    plt.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, s=50)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'{x_col} vs {y_col}')
                    
                    # Add regression line (dotted red line as specified in requirements)
                    if len(clean_data) > 1:
                        try:
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
                            line = slope * clean_data[x_col] + intercept
                            plt.plot(clean_data[x_col], line, 'r:', linewidth=2, alpha=0.8, 
                                   label=f'Regression Line (r²={r_value**2:.3f})')
                            plt.legend()
                        except ImportError:
                            # Fallback manual regression if scipy not available
                            z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                            p = np.poly1d(z)
                            plt.plot(clean_data[x_col], p(clean_data[x_col]), 'r:', linewidth=2, alpha=0.8, 
                                   label='Regression Line')
                            plt.legend()
                else:
                    # Fallback to bar chart if not enough numeric columns
                    chart_type = 'bar'
                    plt.bar(range(len(top_n_df)), top_n_df[analysis_col], color='skyblue', alpha=0.7)
                    plt.xticks(range(len(top_n_df)), top_n_df[name_col], rotation=45, ha='right')
                    plt.title(f'Top {len(top_n_df)} by {analysis_col}')
                    plt.xlabel(name_col)
                    plt.ylabel(analysis_col)
                    
            elif chart_type == 'histogram':
                # Enhanced histogram with better binning
                data_values = data_clean[analysis_col].dropna()
                if len(data_values) > 0:
                    bins = min(20, max(5, len(data_values) // 5))  # Dynamic bin sizing
                    plt.hist(data_values, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
                    plt.title(f'Distribution of {analysis_col}')
                    plt.xlabel(analysis_col)
                    plt.ylabel('Frequency')
                    # Add mean line
                    mean_val = data_values.mean()
                    plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                              label=f'Mean: {mean_val:.2f}')
                    plt.legend()
                    
            elif chart_type == 'time_series':
                # Enhanced time series with better date handling
                date_cols = self._find_date_columns(data_clean)
                
                if date_cols:
                    date_col = date_cols[0]
                    sorted_data = data_clean.sort_values(date_col)
                    plt.plot(sorted_data[date_col], sorted_data[analysis_col], 
                           marker='o', linewidth=2, markersize=6)
                    plt.title(f'{analysis_col} over {date_col}')
                    plt.xlabel(date_col)
                    plt.ylabel(analysis_col)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                else:
                    # Fallback to bar chart if no date column found
                    chart_type = 'bar'
                    plt.bar(range(len(top_n_df)), top_n_df[analysis_col], color='skyblue', alpha=0.7)
                    plt.xticks(range(len(top_n_df)), top_n_df[name_col], rotation=45, ha='right')
                    plt.title(f'Top {len(top_n_df)} by {analysis_col}')
                    plt.xlabel(name_col)
                    plt.ylabel(analysis_col)
            
            plt.tight_layout()
            
            # Always generate base64 for API responses
            plot_base64 = None
            buffer = BytesIO()
            # Optimize for size while maintaining quality
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            
            # Ensure base64 is under 100KB as required
            if len(plot_base64) > 100000:
                # Reduce quality if too large
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=60, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
            
            plt.show()
            
            print(f"Generated {chart_type} visualization for {analysis_col}")
            print(f"Base64 size: {len(plot_base64)} characters")
            
            return {
                'plot_path': 'generated', 
                'plot_base64': f"data:image/png;base64,{plot_base64}", 
                'chart_type': chart_type,
                'image_size_bytes': len(plot_base64)
            }
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return {'plot_path': None, 'plot_base64': None, 'error': str(e)}
    
    def _auto_detect_chart_type(self, task_description: str, data_clean) -> str:
        """Auto-detect the appropriate chart type based on task description using LLM"""
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Get column information for context
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = data_clean.select_dtypes(include=['object']).columns.tolist()
            
            system_prompt = """You are a data visualization expert. Based on the task description and data characteristics, 
            recommend the most appropriate chart type.

Available chart types:
- bar: For rankings, comparisons, top N items
- scatter: For correlations, relationships between two numeric variables
- histogram: For distributions of a single numeric variable
- time_series: For data over time periods

Consider:
1. What the task is asking for (correlation, distribution, ranking, time trends)
2. The type of analysis needed
3. The data structure and column types

Respond with ONLY one of: bar, scatter, histogram, time_series"""

            human_prompt = """Task: {task_description}

Data characteristics:
- Numeric columns: {numeric_cols}
- Text columns: {text_cols}
- Data shape: {data_shape}

What chart type is most appropriate? Respond with just: bar, scatter, histogram, or time_series"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "numeric_cols": numeric_cols[:5],  # Limit for context
                "text_cols": text_cols[:3],
                "data_shape": f"{data_clean.shape[0]} rows, {data_clean.shape[1]} columns"
            })
            
            chart_type = result.strip().lower()
            valid_types = ['bar', 'scatter', 'histogram', 'time_series']
            
            if chart_type in valid_types:
                print(f"LLM recommended chart type: {chart_type}")
                return chart_type
            else:
                print(f"LLM returned invalid chart type: {chart_type}, using fallback")
                return self._fallback_chart_detection(task_description)
                
        except Exception as e:
            print(f"Error in LLM chart type detection: {str(e)}, using fallback")
            return self._fallback_chart_detection(task_description)
    
    def _fallback_chart_detection(self, task_description: str) -> str:
        """Fallback chart type detection using keywords"""
        task_lower = task_description.lower()
        
        if 'scatterplot' in task_lower or 'scatter plot' in task_lower:
            return 'scatter'
        elif 'histogram' in task_lower:
            return 'histogram'
        elif 'time series' in task_lower or 'over time' in task_lower:
            return 'time_series'
        elif any(word in task_lower for word in ['correlation', 'vs', 'versus', 'relationship']):
            return 'scatter'
        else:
            return 'bar'
    
    def _select_scatter_columns(self, numeric_cols: List[str], analysis_col: str, task_description: str) -> tuple:
        """Select appropriate columns for scatter plot based on task description"""
        task_lower = task_description.lower()
        
        # Look for specific column relationships mentioned in task
        if 'rank' in task_lower and 'peak' in task_lower:
            # Find rank and peak columns
            rank_col = None
            peak_col = None
            for col in numeric_cols:
                col_lower = str(col).lower()
                if 'rank' in col_lower:
                    rank_col = col
                elif 'peak' in col_lower:
                    peak_col = col
            if rank_col and peak_col:
                return rank_col, peak_col
        
        elif 'cases' in task_lower and 'deaths' in task_lower:
            # Find cases and deaths columns
            cases_col = None
            deaths_col = None
            for col in numeric_cols:
                col_lower = str(col).lower()
                if 'cases' in col_lower or 'total' in col_lower:
                    cases_col = col
                elif 'deaths' in col_lower:
                    deaths_col = col
            if cases_col and deaths_col:
                return cases_col, deaths_col
        
        elif 'runs' in task_lower and 'average' in task_lower:
            # Find runs and average columns for cricket data
            runs_col = None
            avg_col = None
            for col in numeric_cols:
                col_lower = str(col).lower()
                if 'runs' in col_lower or 'total' in col_lower:
                    runs_col = col
                elif 'average' in col_lower or 'avg' in col_lower:
                    avg_col = col
            if runs_col and avg_col:
                return runs_col, avg_col
        
        # Default: use first two numeric columns, prioritizing analysis_col as y-axis
        if len(numeric_cols) >= 2:
            if analysis_col in numeric_cols:
                other_cols = [col for col in numeric_cols if col != analysis_col]
                return other_cols[0], analysis_col
            else:
                return numeric_cols[0], numeric_cols[1]
        
        return None, None
    
    def _find_date_columns(self, data_clean) -> List[str]:
        """Find columns that might contain date/time information"""
        date_cols = []
        for col in data_clean.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['year', 'date', 'time', 'month']):
                date_cols.append(col)
        return date_cols

class AnswerQuestionsStep:
    """
    Step 6: Enhanced generic question answering
    - Use dynamic column names
    - Support various types of questions across different domains
    - Handle financial, geographic, sports, health, and other data types
    - Store answers in variables for capture
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data.get('top_n_df')
        analysis_col = input_data.get('analysis_col')
        name_col = input_data.get('name_col')
        data_clean = input_data.get('data_clean')
        task_description = input_data.get('task_description', '')
        questions = input_data.get('questions', [])
        
        print(f"\n=== ANSWERING QUESTIONS ===")
        
        answers = {}
        
        if top_n_df is None or top_n_df.empty or analysis_col is None:
            print("ERROR: No data available for answering questions")
            answers = {
                'error': 'No data available for analysis',
                'status': 'failed'
            }
        else:
            print(f"Analyzing top {len(top_n_df)} entries by {analysis_col}")
            
            # Enhanced question answering based on data structure and domain
            
            # LLM-powered question interpretation and answering
            answers.update(self._answer_questions_with_llm(
                data_clean, top_n_df, analysis_col, name_col, task_description
            ))
            
            # Basic statistics and rankings
            if len(top_n_df) >= 5:
                rank_5_item = top_n_df.iloc[4][name_col]
                rank_5_value = top_n_df.iloc[4][analysis_col]
                answers['item_ranking_5th'] = rank_5_item
                answers['item_ranking_5th_value'] = rank_5_value
                print(f"Item ranking 5th: {rank_5_item} ({rank_5_value:,.2f})")
            else:
                answers['item_ranking_5th'] = f'Only {len(top_n_df)} items available'
                answers['item_ranking_5th_value'] = 0
                print(f"Not enough data - only {len(top_n_df)} items in dataset")
            
            # Total and average calculations
            total_top_n = top_n_df[analysis_col].sum()
            avg_top_n = top_n_df[analysis_col].mean()
            answers['total_top_n'] = total_top_n
            answers['average_top_n'] = avg_top_n
            print(f"Total {analysis_col} of top {len(top_n_df)}: {total_top_n:,.2f}")
            print(f"Average {analysis_col} of top {len(top_n_df)}: {avg_top_n:,.2f}")
            
            # Range analysis
            if len(data_clean) > 0:
                max_value = data_clean[analysis_col].max()
                min_value = data_clean[analysis_col].min()
                answers['max_value'] = max_value
                answers['min_value'] = min_value
                answers['range'] = max_value - min_value
                print(f"Range: {min_value:,.2f} to {max_value:,.2f}")
            
            # Correlation analysis if multiple numeric columns
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                correlations = {}
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        try:
                            corr = data_clean[[col1, col2]].corr().iloc[0, 1]
                            correlations[f"{col1}_vs_{col2}"] = corr
                            print(f"Correlation {col1} vs {col2}: {corr:.3f}")
                        except:
                            pass
                answers['correlations'] = correlations
            
            # Domain-specific question handling
            
            # 1. Financial/Revenue data (movies, GDP, etc.)
            if self._is_financial_data(analysis_col, task_description):
                self._answer_financial_questions(answers, data_clean, analysis_col, name_col, task_description)
            
            # 2. Health/Medical data (COVID, disease statistics)
            elif self._is_health_data(analysis_col, task_description):
                self._answer_health_questions(answers, data_clean, analysis_col, name_col, top_n_df)
            
            # 3. Sports data (cricket, football, etc.)
            elif self._is_sports_data(analysis_col, task_description):
                self._answer_sports_questions(answers, data_clean, analysis_col, name_col, top_n_df)
            
            # 4. Economic data (inflation, trading)
            elif self._is_economic_data(analysis_col, task_description):
                self._answer_economic_questions(answers, data_clean, analysis_col, name_col)
            
            # 5. Entertainment data (ratings, reviews)
            elif self._is_entertainment_data(analysis_col, task_description):
                self._answer_entertainment_questions(answers, data_clean, analysis_col, name_col, top_n_df)
            
            # Time-based analysis if year column exists
            year_cols = [col for col in data_clean.columns if 'year' in str(col).lower()]
            if year_cols:
                self._answer_temporal_questions(answers, data_clean, analysis_col, name_col, year_cols[0])
            
            # Full top N list for reference
            top_list = []
            for i, (idx, row) in enumerate(top_n_df.iterrows()):
                top_list.append({
                    'rank': i + 1,
                    'name': row[name_col],
                    'value': row[analysis_col]
                })
            answers['top_n_list'] = top_list
            
            # Summary metadata
            answers['summary'] = {
                'analysis_column': analysis_col,
                'name_column': name_col,
                'total_items_analyzed': len(top_n_df),
                'total_items_in_dataset': len(data_clean) if data_clean is not None else 0,
                'data_type': self._identify_data_type(analysis_col, task_description),
                'domain': self._identify_domain(task_description)
            }
            
            answers['status'] = 'success'
        
        print("\nFINAL ANSWERS:")
        for key, value in answers.items():
            if key not in ['top_n_list', 'correlations']:  # Don't print large objects
                print(f"  {key}: {value}")
        
        return sanitize_for_json({'answers': answers})
    
    def _answer_questions_with_llm(self, data_clean, top_n_df, analysis_col: str, name_col: str, task_description: str) -> dict:
        """Use LLM to intelligently interpret and answer questions from task description"""
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Extract key data insights for LLM context
            data_insights = {
                'total_rows': len(data_clean),
                'top_n_count': len(top_n_df),
                'analysis_column': analysis_col,
                'name_column': name_col,
                'max_value': float(data_clean[analysis_col].max()) if len(data_clean) > 0 else 0,
                'min_value': float(data_clean[analysis_col].min()) if len(data_clean) > 0 else 0,
                'average_value': float(data_clean[analysis_col].mean()) if len(data_clean) > 0 else 0,
            }
            
            # Get numeric columns for correlation analysis
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
            
            # Get year column if exists for temporal questions
            year_cols = [col for col in data_clean.columns if 'year' in str(col).lower()]
            
            # Prepare top N data for LLM
            if len(top_n_df) >= 5:
                top_5_items = []
                for i in range(min(5, len(top_n_df))):
                    top_5_items.append({
                        'rank': i + 1,
                        'name': str(top_n_df.iloc[i][name_col]),
                        'value': float(top_n_df.iloc[i][analysis_col])
                    })
            else:
                top_5_items = []
            
            system_prompt = """You are an expert data analyst. Based on the task description and data insights, 
            answer specific questions that might be asked about the data.

Common question types to address:
1. Ranking questions (Who/what ranks 5th? Which is earliest/latest?)
2. Threshold questions (How many above X value? Items before year Y?)
3. Statistical questions (What's the correlation? Average? Total?)
4. Domain-specific questions (death rates, averages above threshold, etc.)

Provide specific numeric answers and insights. Return answers as a JSON object with descriptive keys.
Focus on actionable insights that directly address the analysis goals."""

            human_prompt = """Task: {task_description}

Data insights:
- Total items: {total_rows}
- Analysis column: {analysis_column} (range: {min_value:.2f} to {max_value:.2f})
- Average value: {average_value:.2f}
- Top {top_n_count} items available
- Numeric columns: {numeric_cols}
- Year columns: {year_cols}

Top 5 items:
{top_5_data}

Based on this data and the task requirements, what are the key insights and answers to likely questions?
Respond with a JSON object containing relevant answers and insights."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "total_rows": data_insights['total_rows'],
                "analysis_column": analysis_col,
                "min_value": data_insights['min_value'],
                "max_value": data_insights['max_value'], 
                "average_value": data_insights['average_value'],
                "top_n_count": data_insights['top_n_count'],
                "numeric_cols": numeric_cols[:5],  # Limit for context
                "year_cols": year_cols,
                "top_5_data": str(top_5_items)
            })
            
            # Try to parse JSON response
            try:
                import json
                llm_answers = json.loads(result)
                print("LLM provided intelligent question answers")
                return llm_answers
            except json.JSONDecodeError:
                print("LLM response not in JSON format, extracting key insights")
                # Extract key insights from text response
                insights = {}
                if "5th" in result.lower() and len(top_5_items) >= 5:
                    insights['llm_fifth_item_insight'] = top_5_items[4]['name']
                if "total" in result.lower():
                    insights['llm_total_insight'] = sum(item['value'] for item in top_5_items)
                if "average" in result.lower():
                    insights['llm_average_insight'] = data_insights['average_value']
                
                insights['llm_analysis_summary'] = result[:200] + "..." if len(result) > 200 else result
                return insights
                
        except Exception as e:
            print(f"Error in LLM question answering: {str(e)}")
            return {'llm_error': str(e)}
    
    def _is_financial_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is financial/revenue data"""
        financial_keywords = ['gross', 'revenue', 'gdp', 'billion', 'million', 'box office', 'earnings']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in financial_keywords)
    
    def _is_health_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is health/medical data"""
        health_keywords = ['cases', 'deaths', 'covid', 'infection', 'recovery', 'mortality', 'disease']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in health_keywords)
    
    def _is_sports_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is sports data"""
        sports_keywords = ['runs', 'average', 'cricket', 'batsmen', 'matches', 'innings', 'wickets', 'goals', 'points']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in sports_keywords)
    
    def _is_economic_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is economic data"""
        economic_keywords = ['inflation', 'cpi', 'rate', 'economics', 'trading', 'price', 'index']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in economic_keywords)
    
    def _is_entertainment_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is entertainment data"""
        entertainment_keywords = ['rating', 'imdb', 'score', 'movie', 'film', 'review']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in entertainment_keywords)
    
    def _answer_financial_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, task_description: str):
        """Answer financial/revenue specific questions"""
        if 'billion' in str(analysis_col).lower() or 'billion' in task_description.lower():
            # Count items above certain thresholds
            above_1_5bn = len(data_clean[data_clean[analysis_col] > 1500])
            above_2bn = len(data_clean[data_clean[analysis_col] > 2000])
            answers['items_above_1_5_billion'] = above_1_5bn
            answers['items_above_2_billion'] = above_2bn
            print(f"Items above 1.5 billion: {above_1_5bn}")
            print(f"Items above 2 billion: {above_2bn}")
    
    def _answer_health_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, top_n_df):
        """Answer health/medical specific questions"""
        # Look for death rate calculations
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
        deaths_col = None
        cases_col = None
        
        for col in numeric_cols:
            col_lower = str(col).lower()
            if 'deaths' in col_lower:
                deaths_col = col
            elif 'cases' in col_lower or 'total' in col_lower:
                cases_col = col
        
        if deaths_col and cases_col:
            # Calculate death-to-case ratio
            data_clean['death_rate'] = (data_clean[deaths_col] / data_clean[cases_col] * 100).round(2)
            highest_death_rate_country = data_clean.loc[data_clean['death_rate'].idxmax(), name_col]
            highest_death_rate_value = data_clean['death_rate'].max()
            
            answers['highest_death_rate_country'] = highest_death_rate_country
            answers['highest_death_rate_value'] = highest_death_rate_value
            print(f"Highest death-to-case ratio: {highest_death_rate_country} ({highest_death_rate_value:.2f}%)")
            
            # Global average calculations
            global_death_rate = (data_clean[deaths_col].sum() / data_clean[cases_col].sum() * 100)
            answers['global_average_death_rate'] = global_death_rate
            print(f"Global average death rate: {global_death_rate:.2f}%")
    
    def _answer_sports_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, top_n_df):
        """Answer sports specific questions"""
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Look for runs and average columns
        runs_col = None
        avg_col = None
        
        for col in numeric_cols:
            col_lower = str(col).lower()
            if 'runs' in col_lower:
                runs_col = col
            elif 'average' in col_lower or 'avg' in col_lower:
                avg_col = col
        
        if runs_col and avg_col:
            # Find highest average among players with over 8000 runs
            high_runs_players = data_clean[data_clean[runs_col] > 8000]
            if len(high_runs_players) > 0:
                highest_avg_player = high_runs_players.loc[high_runs_players[avg_col].idxmax(), name_col]
                highest_avg_value = high_runs_players[avg_col].max()
                answers['highest_average_player_8000_runs'] = highest_avg_player
                answers['highest_average_value_8000_runs'] = highest_avg_value
                print(f"Highest average among 8000+ run players: {highest_avg_player} ({highest_avg_value:.2f})")
        
        # Count countries in top 10
        if 'country' in data_clean.columns:
            country_counts = top_n_df['country'].value_counts()
            most_common_country = country_counts.index[0]
            country_count = country_counts.iloc[0]
            answers['most_represented_country_top10'] = most_common_country
            answers['country_count_top10'] = country_count
            print(f"Most represented country in top 10: {most_common_country} ({country_count} players)")
    
    def _answer_economic_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str):
        """Answer economic data specific questions"""
        if len(data_clean) > 0:
            current_rate = data_clean[analysis_col].iloc[-1]  # Most recent
            highest_rate = data_clean[analysis_col].max()
            answers['current_inflation_rate'] = current_rate
            answers['highest_rate_period'] = highest_rate
            print(f"Current rate: {current_rate:.2f}")
            print(f"Highest rate: {highest_rate:.2f}")
    
    def _answer_entertainment_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, top_n_df):
        """Answer entertainment/rating specific questions"""
        if 'rating' in str(analysis_col).lower():
            avg_rating = data_clean[analysis_col].mean()
            answers['average_rating'] = avg_rating
            print(f"Average rating: {avg_rating:.2f}")
            
        # Decade analysis if year column exists
        year_cols = [col for col in data_clean.columns if 'year' in str(col).lower()]
        if year_cols:
            year_col = year_cols[0]
            data_clean['decade'] = (data_clean[year_col] // 10) * 10
            decade_counts = data_clean['decade'].value_counts()
            most_common_decade = decade_counts.index[0]
            decade_count = decade_counts.iloc[0]
            answers['most_movies_decade'] = f"{most_common_decade}s"
            answers['decade_movie_count'] = decade_count
            print(f"Decade with most top movies: {most_common_decade}s ({decade_count} movies)")
    
    def _answer_temporal_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, year_col: str):
        """Answer time-based questions"""
        # Count items before year 2000
        before_2000 = len(data_clean[data_clean[year_col] < 2000])
        answers['items_before_2000'] = before_2000
        print(f"Items before year 2000: {before_2000}")
        
        # Find earliest item above threshold (for financial data)
        if 'billion' in str(analysis_col).lower():
            above_threshold = data_clean[data_clean[analysis_col] > 1500].sort_values(year_col)
            if len(above_threshold) > 0:
                earliest_item = above_threshold.iloc[0][name_col]
                earliest_year = above_threshold.iloc[0][year_col]
                answers['earliest_above_threshold'] = earliest_item
                answers['earliest_year'] = earliest_year
                print(f"Earliest item above 1.5bn: {earliest_item} ({earliest_year})")
    
    def _identify_data_type(self, analysis_col: str, task_description: str) -> str:
        """Identify the type of data being analyzed"""
        if self._is_financial_data(analysis_col, task_description):
            return 'financial'
        elif self._is_health_data(analysis_col, task_description):
            return 'health'
        elif self._is_sports_data(analysis_col, task_description):
            return 'sports'
        elif self._is_economic_data(analysis_col, task_description):
            return 'economic'
        elif self._is_entertainment_data(analysis_col, task_description):
            return 'entertainment'
        else:
            return 'general'
    
    def _identify_domain(self, task_description: str) -> str:
        """Identify the domain/industry of the data"""
        task_lower = task_description.lower()
        
        if 'wikipedia' in task_lower and 'films' in task_lower:
            return 'movies'
        elif 'imdb' in task_lower:
            return 'movies'
        elif 'covid' in task_lower or 'coronavirus' in task_lower:
            return 'health'
        elif 'cricket' in task_lower or 'espn' in task_lower:
            return 'sports'
        elif 'inflation' in task_lower or 'trading' in task_lower:
            return 'economics'
        else:
            return 'general'