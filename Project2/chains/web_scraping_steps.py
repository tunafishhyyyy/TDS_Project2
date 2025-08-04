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
            # Select the table with the most rows
            max_rows = 0
            best_table_idx = 0
            data = tables[0]
            for i, table in enumerate(tables):
                if table.shape[0] > max_rows:
                    max_rows = table.shape[0]
                    best_table_idx = i
                    data = table
            print(f"Selected table {best_table_idx} with {data.shape[0]} rows and {data.shape[1]} columns")
            return {'data': data, 'tables': tables}
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
        data = input_data['data']
        print(f"\nSelected data shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head())
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[1] if isinstance(col, tuple) and col[1] else str(col) for col in data.columns]
            print("\nFlattened MultiIndex columns:")
            print(f"Columns: {data.columns.tolist()}")
        # Check if first row contains headers
        if data.iloc[0].dtype == 'object':
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)
            print("\nAfter setting headers:")
            print(f"Columns: {data.columns.tolist()}")
        return {'data': data}

class CleanDataStep:
    """
    Step 3: Data cleaning
    - Remove symbols, footnotes, and convert to numeric
    - CRITICAL: Use select_dtypes to find numeric columns
    - Print after cleaning
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data']
        for col in data.columns:
            if data[col].dtype == 'object':
                cleaned = data[col].astype(str)
                cleaned = cleaned.str.replace('$', '').str.replace(',', '').str.replace('€', '').str.replace('£', '')
                cleaned = cleaned.str.replace(r'\[\d+\]', '', regex=True)
                cleaned = cleaned.str.replace(r'[^\d.]', '', regex=True)
                data[col] = pd.to_numeric(cleaned, errors='coerce')
        print("\nAfter cleaning:")
        print(data.head())
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumeric columns available: {numeric_cols}")
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
        if not numeric_cols:
            print("No numeric columns found for analysis")
            return {'top_n_df': pd.DataFrame()}
        analysis_col = numeric_cols[0]
        print(f"Using column '{analysis_col}' for analysis")
        data_clean = data.dropna(subset=[analysis_col])
        data_sorted = data_clean.sort_values(analysis_col, ascending=False)
        top_n_df = data_sorted.head(top_n)
        print(f"\nTop {top_n} by {analysis_col}:")
        print(top_n_df[[data.columns[0], analysis_col]])
        return {'top_n_df': top_n_df, 'analysis_col': analysis_col}

class VisualizeStep:
    """
    Step 5: Visualization
    - Bar plot of top N
    - Use dynamic column names
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data['top_n_df']
        analysis_col = input_data['analysis_col']
        if top_n_df.empty:
            print("No data to plot.")
            return {}
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_n_df)), top_n_df[analysis_col])
        plt.xticks(range(len(top_n_df)), top_n_df[top_n_df.columns[0]], rotation=45)
        plt.title(f'Top {len(top_n_df)} by {analysis_col}')
        plt.xlabel(top_n_df.columns[0])
        plt.ylabel(analysis_col)
        plt.tight_layout()
        plt.savefig('plot.png')
        plt.close()
        print("Plot saved as plot.png")
        return {'plot_path': 'plot.png'}

class AnswerQuestionsStep:
    """
    Step 6: Answer specific questions
    - Use dynamic column names
    - Store answers in variables for capture
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data['top_n_df']
        analysis_col = input_data['analysis_col']
        answers = {}
        if not top_n_df.empty:
            first_col = top_n_df.columns[0]
            if len(top_n_df) >= 5:
                answers['rank_5'] = top_n_df.iloc[4][first_col]
            answers['total_top_n'] = top_n_df[analysis_col].sum()
            answers['top_n_list'] = top_n_df[[first_col, analysis_col]].to_dict('records')
        else:
            answers['rank_5'] = 'Not enough data'
            answers['total_top_n'] = 0
            answers['top_n_list'] = []
        print("\nANSWERS:")
        print(answers)
        return {'answers': answers} 