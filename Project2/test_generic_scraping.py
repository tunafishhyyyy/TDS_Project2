#!/usr/bin/env python3
"""
Test script to demonstrate generic web scraping approach
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def test_generic_scraping():
    """Test the generic web scraping approach"""
    
    print("Testing generic web scraping approach...")
    
    # Example URL (you can change this to any website with tables)
    url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
    print(f"Scraping data from: {url}")
    
    try:
        # Method 1: pandas read_html (works for tables)
        tables = pd.read_html(url)
        print(f"Found {len(tables)} tables on the page")
        
        # Inspect all tables to understand the data structure
        for i, table in enumerate(tables):
            print(f"\nTable {i}:")
            print(f"  Shape: {table.shape}")
            print(f"  Columns: {table.columns.tolist()}")
            print(f"  Sample data:")
            print(table.head(3))
        
        # Select the most relevant table based on content
        # You can modify this logic based on what you're looking for
        data = tables[1]  # GDP data is usually in table[1]
        print(f"\nSelected table shape: {data.shape}")
        
    except Exception as e:
        print(f"pandas read_html failed: {e}")
        return {"status": "error", "error": str(e)}
    
    # Data inspection and cleaning
    print(f"\nSelected data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print("\nFirst few rows:")
    print(data.head())
    
    # Clean the data structure
    # Check if first row contains headers
    if data.iloc[0].dtype == 'object':
        # Use first row as headers
        data.columns = data.iloc[0]
        data = data[1:].reset_index(drop=True)
        print("\nAfter setting headers:")
        print(f"Columns: {data.columns.tolist()}")
    
    # Clean numeric columns (remove symbols, convert to numeric)
    for col in data.columns:
        if data[col].dtype == 'object':
            # Try to convert to numeric, removing common symbols
            cleaned = data[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('€', '').str.replace('£', '')
            data[col] = pd.to_numeric(cleaned, errors='coerce')
    
    print("\nAfter cleaning:")
    print(data.head())
    
    # Data analysis
    # Find numeric columns for analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns available: {numeric_cols}")
    
    if len(numeric_cols) > 0:
        # Use the first numeric column for analysis
        analysis_col = numeric_cols[0]
        print(f"Using column '{analysis_col}' for analysis")
        
        # Sort by the analysis column
        data_sorted = data.sort_values(analysis_col, ascending=False)
        
        # Get top 10 items
        top_10 = data_sorted.head(10)
        print(f"\nTop 10 by {analysis_col}:")
        print(top_10[[data.columns[0], analysis_col]])  # Show first column and analysis column
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_10)), top_10[analysis_col])
        plt.xticks(range(len(top_10)), top_10[data.columns[0]], rotation=45)
        plt.title(f'Top 10 by {analysis_col}')
        plt.xlabel(data.columns[0])
        plt.ylabel(analysis_col)
        plt.tight_layout()
        plt.show()
        
        # Answer specific questions
        if len(top_10) >= 5:
            fifth_item = top_10.iloc[4][data.columns[0]]
            total_top_10 = top_10[analysis_col].sum()
            
            print(f"\nANSWERS:")
            print(f"Item ranking 5th by {analysis_col}: {fifth_item}")
            print(f"Total {analysis_col} of top 10: {total_top_10}")
            
            return {
                "status": "success",
                "fifth_item": fifth_item,
                "total_top_10": total_top_10,
                "analysis_column": analysis_col,
                "top_10_data": top_10
            }
        else:
            print(f"\nNot enough data for ranking analysis")
            return {"status": "error", "error": "Not enough data"}
    else:
        print("\nNo numeric columns found for analysis")
        return {"status": "error", "error": "No numeric columns found"}

if __name__ == "__main__":
    result = test_generic_scraping()
    print(f"\nTest result: {result}") 