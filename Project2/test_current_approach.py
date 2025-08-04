#!/usr/bin/env python3
"""
Test script to verify the current approach works with actual Wikipedia data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def test_current_approach():
    """Test the current approach with actual Wikipedia data"""
    
    print("Testing current approach with Wikipedia GDP data...")
    
    try:
        # Step 1: Web scraping and data extraction
        url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
        print(f"Scraping data from: {url}")
        
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
        data = tables[2]  # GDP data is usually in table[2]
        print(f"\nSelected table shape: {data.shape}")
        
    except Exception as e:
        print(f"Error scraping data: {e}")
        return {"status": "error", "error": str(e)}
    
    # Data inspection and cleaning
    print(f"\nSelected data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print("\nFirst few rows:")
    print(data.head())
    
    # Clean the data structure
    if data.iloc[0].dtype == 'object':
        data.columns = data.iloc[0]
        data = data[1:].reset_index(drop=True)
        print("\nAfter setting headers:")
        print(f"Columns: {data.columns.tolist()}")
    
    # Clean numeric columns
    for col in data.columns:
        if data[col].dtype == 'object':
            cleaned = data[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('€', '').str.replace('£', '')
            data[col] = pd.to_numeric(cleaned, errors='coerce')
    
    print("\nAfter cleaning:")
    print(data.head())
    
    # Data analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns available: {numeric_cols}")
    
    if len(numeric_cols) > 0:
        analysis_col = numeric_cols[1]  # Use second numeric column
        print(f"Using column '{analysis_col}' for analysis")
        
        data_sorted = data.sort_values(analysis_col, ascending=False)
        top_10 = data_sorted.head(10)
        print(f"\nTop 10 by {analysis_col}:")
        print(top_10[[data.columns[0], analysis_col]])
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_10)), top_10[analysis_col])
        plt.xticks(range(len(top_10)), top_10[data.columns[0]], rotation=45)
        plt.title(f'Top 10 by {analysis_col}')
        plt.xlabel(data.columns[0])
        plt.ylabel(analysis_col)
        plt.tight_layout()
        plt.show()
        
        # Answer questions
        if len(top_10) >= 5:
            first_col = data.columns[0]
            fifth_item = top_10.iloc[4][first_col]
            total_top_10 = top_10[analysis_col].sum()
            
            print(f"\nANSWERS:")
            print(f"Item ranking 5th by {analysis_col}: {fifth_item}")
            print(f"Total {analysis_col} of top 10: {total_top_10}")
            
            print(f"\nFull top 10 list:")
            for i, (idx, row) in enumerate(top_10.iterrows()):
                print(f"{i+1}. {row[first_col]}: {row[analysis_col]}")
            
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
    result = test_current_approach()
    print(f"\nTest result: {result}") 