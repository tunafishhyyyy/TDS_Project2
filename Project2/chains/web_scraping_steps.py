import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

class ScrapeTableStep:
    """
    Step 1: Web scraping and data extraction
    - Use pandas.read_html (preferred) for scraping tables
    - Print number of tables found
    - Select the table with the most rows (CRITICAL for Wikipedia)
    - Print table info for verification
    - Robust error handling
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
            
            # Select the table with the most rows (and more than 50 rows to ensure it's data table)
            max_rows = 0
            best_table_idx = 0
            data = tables[0]
            for i, table in enumerate(tables):
                if table.shape[0] > max_rows and table.shape[0] > 50:  # Ensure substantial data
                    max_rows = table.shape[0]
                    best_table_idx = i
                    data = table
            
            print(f"Selected table {best_table_idx} with {data.shape[0]} rows and {data.shape[1]} columns")
            print(f"This should be the main GDP data table")
            return {'data': data, 'tables': tables, 'selected_table_idx': best_table_idx}
        except Exception as e:
            print(f"Error scraping data: {e}")
            raise

class InspectTableStep:
    """
    Step 2: Data inspection
    - Print shape, columns, and head
    - Handle MultiIndex columns (flatten if needed)
    - Check if first row contains headers and set as columns if so
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data'].copy()
        print(f"\nSelected data shape: {data.shape}")
        print(f"Original columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head())
        
        # Handle MultiIndex columns (Wikipedia tables often have these)
        if isinstance(data.columns, pd.MultiIndex):
            print("\nDetected MultiIndex columns, flattening...")
            # For GDP table, we want to flatten properly
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
        
        # Check if first row contains headers (common in Wikipedia tables)
        first_row_is_header = False
        if len(data) > 1:
            # Check if first row looks like headers
            first_row_values = data.iloc[0].astype(str).tolist()
            if any('Country' in str(val) or 'Territory' in str(val) for val in first_row_values):
                first_row_is_header = True
        
        if first_row_is_header:
            print("\nFirst row appears to be headers, setting as column names...")
            data.columns = [str(val) for val in data.iloc[0]]
            data = data[1:].reset_index(drop=True)
            print(f"Updated columns: {data.columns.tolist()}")
        
        print("\nAfter column processing:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head(3))
        
        return {'data': data}

class CleanDataStep:
    """
    Step 3: Data cleaning
    - Remove symbols, footnotes, and convert to numeric
    - CRITICAL: Use select_dtypes to find numeric columns
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
                
                # Remove common symbols and formatting
                cleaned = cleaned.str.replace('$', '', regex=False)
                cleaned = cleaned.str.replace(',', '', regex=False)
                cleaned = cleaned.str.replace('€', '', regex=False)
                cleaned = cleaned.str.replace('£', '', regex=False)
                cleaned = cleaned.str.replace('¥', '', regex=False)
                
                # Remove footnote references like [1], [n 1], etc.
                cleaned = cleaned.str.replace(r'\[.*?\]', '', regex=True)
                
                # Remove any other non-numeric characters except decimal points and minus signs
                cleaned = cleaned.str.replace(r'[^\d.\-]', '', regex=True)
                
                # Handle empty strings
                cleaned = cleaned.replace('', np.nan)
                
                # Convert to numeric
                numeric_data = pd.to_numeric(cleaned, errors='coerce')
                
                # Only replace if we got some valid numbers
                valid_count = numeric_data.notna().sum()
                if valid_count > 0:
                    print(f"  Converted {valid_count} values to numeric")
                    data[col] = numeric_data
                else:
                    print(f"  No valid numeric data found, keeping as text")
        
        print("\nAfter cleaning:")
        print(f"Data types:\n{data.dtypes}")
        print(data.head(3))
        
        # Find numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumeric columns found: {numeric_cols}")
        
        # Print some statistics about numeric columns
        for col in numeric_cols:
            valid_count = data[col].notna().sum()
            print(f"  {col}: {valid_count} valid values, range: {data[col].min():.0f} to {data[col].max():.0f}")
        
        return {'data': data, 'numeric_cols': numeric_cols}

class AnalyzeDataStep:
    """
    Step 4: Data analysis
    - Use first numeric column for analysis (unless specified)
    - Remove NaNs, sort, get top N
    - Print top N
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
        
        # Find the best column for analysis (usually the largest/main GDP column)
        best_col = None
        max_valid_values = 0
        
        for col in numeric_cols:
            valid_count = data[col].notna().sum()
            max_value = data[col].max()
            print(f"Column '{col}': {valid_count} valid values, max value: {max_value}")
            
            # Choose column with most valid values and reasonable max value (GDP should be large)
            if valid_count > max_valid_values and max_value > 1000:  # GDP in billions
                max_valid_values = valid_count
                best_col = col
        
        if not best_col:
            # Fallback to first numeric column
            best_col = numeric_cols[0]
        
        print(f"Selected column '{best_col}' for analysis")
        
        # Clean and analyze
        data_clean = data.dropna(subset=[best_col])
        print(f"After removing NaN values: {data_clean.shape[0]} rows")
        
        if len(data_clean) == 0:
            print("ERROR: No valid data after cleaning")
            return {'top_n_df': pd.DataFrame(), 'analysis_col': best_col}
        
        # Sort by the analysis column
        data_sorted = data_clean.sort_values(best_col, ascending=False)
        top_n_df = data_sorted.head(top_n)
        
        print(f"\nTop {len(top_n_df)} by {best_col}:")
        
        # Find the country/name column (usually first text column)
        text_cols = data.select_dtypes(include=['object']).columns.tolist()
        name_col = text_cols[0] if text_cols else data.columns[0]
        
        print(f"Using '{name_col}' as identifier column")
        
        # Display results
        for i, (idx, row) in enumerate(top_n_df.iterrows()):
            country = row[name_col]
            value = row[best_col]
            print(f"{i+1:2d}. {country}: {value:,.0f}")
        
        return {'top_n_df': top_n_df, 'analysis_col': best_col, 'name_col': name_col}

class VisualizeStep:
    """
    Step 5: Visualization
    - Bar plot of top N
    - Use dynamic column names
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data.get('top_n_df')
        analysis_col = input_data.get('analysis_col')
        name_col = input_data.get('name_col')
        
        if top_n_df is None or top_n_df.empty or analysis_col is None:
            print("No data available for visualization.")
            return {'plot_path': None}
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Create bar plot
            countries = top_n_df[name_col].tolist()
            values = top_n_df[analysis_col].tolist()
            
            bars = plt.bar(range(len(countries)), values)
            plt.xticks(range(len(countries)), countries, rotation=45, ha='right')
            plt.title(f'Top {len(top_n_df)} Countries by {analysis_col}')
            plt.xlabel('Country')
            plt.ylabel(f'{analysis_col} (Billions USD)')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:,.0f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plot_path = 'gdp_analysis_plot.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved as {plot_path}")
            return {'plot_path': plot_path}
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return {'plot_path': None}

class AnswerQuestionsStep:
    """
    Step 6: Answer specific questions
    - Use dynamic column names
    - Store answers in variables for capture
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data.get('top_n_df')
        analysis_col = input_data.get('analysis_col')
        name_col = input_data.get('name_col')
        
        print(f"\n=== ANSWERING QUESTIONS ===")
        
        answers = {}
        
        if top_n_df is None or top_n_df.empty or analysis_col is None:
            print("ERROR: No data available for answering questions")
            answers = {
                'rank_5': 'No data available',
                'total_top_n': 0,
                'top_n_list': []
            }
        else:
            print(f"Analyzing top {len(top_n_df)} countries by {analysis_col}")
            
            # Question 1: Which country ranks 5th?
            if len(top_n_df) >= 5:
                rank_5_country = top_n_df.iloc[4][name_col]
                rank_5_value = top_n_df.iloc[4][analysis_col]
                answers['rank_5'] = rank_5_country
                answers['rank_5_value'] = rank_5_value
                print(f"Country ranking 5th: {rank_5_country} ({rank_5_value:,.0f})")
            else:
                answers['rank_5'] = f'Only {len(top_n_df)} countries available'
                answers['rank_5_value'] = 0
                print(f"Not enough data - only {len(top_n_df)} countries in dataset")
            
            # Question 2: Total GDP of top 10
            total_top_n = top_n_df[analysis_col].sum()
            answers['total_top_n'] = total_top_n
            print(f"Total {analysis_col} of top {len(top_n_df)}: {total_top_n:,.0f}")
            
            # Additional info: Full top N list
            top_list = []
            for i, (idx, row) in enumerate(top_n_df.iterrows()):
                top_list.append({
                    'rank': i + 1,
                    'country': row[name_col],
                    'value': row[analysis_col]
                })
            answers['top_n_list'] = top_list
            
            # Summary
            answers['summary'] = {
                'analysis_column': analysis_col,
                'name_column': name_col,
                'total_countries_analyzed': len(top_n_df),
                'data_source': 'Wikipedia GDP data'
            }
        
        print("\nFINAL ANSWERS:")
        for key, value in answers.items():
            if key != 'top_n_list':  # Don't print the full list
                print(f"  {key}: {value}")
        
        return {'answers': answers}