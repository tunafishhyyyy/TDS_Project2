#!/usr/bin/env python3
"""
Test script to verify correct table selection for Wikipedia GDP data
"""

import pandas as pd
import numpy as np

def test_table_selection():
    """Test the table selection logic"""
    
    print("Testing table selection for Wikipedia GDP data...")
    
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
        # Look for the table with the most rows (usually the main data table)
        data = tables[0]  # Start with first table
        max_rows = 0
        best_table_idx = 0
        
        for i, table in enumerate(tables):
            print(f"Table {i}: {table.shape[0]} rows, {table.shape[1]} columns")
            if table.shape[0] > max_rows:
                max_rows = table.shape[0]
                best_table_idx = i
                data = table
        
        print(f"\nSelected table {best_table_idx} with {data.shape[0]} rows and {data.shape[1]} columns")
        print(f"This should be the main data table with the most rows")
        
        # Verify we have the right table
        if data.shape[0] > 200:
            print("✅ Correct table selected (200+ rows)")
        else:
            print("⚠️ Warning: Selected table has fewer than 200 rows")
        
        return {
            "status": "success",
            "selected_table": best_table_idx,
            "table_shape": data.shape,
            "is_correct": data.shape[0] > 200
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    result = test_table_selection()
    print(f"\nTest result: {result}") 