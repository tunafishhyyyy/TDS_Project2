#!/usr/bin/env python3
"""
Simple test script to analyze IMDb page structure
"""

import requests
from bs4 import BeautifulSoup
import json
import re

def analyze_imdb_structure():
    """Analyze IMDb page structure to understand data format"""
    print("Analyzing IMDb structure...")
    
    url = 'https://www.imdb.com/chart/top'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        print(f"Page title: {soup.title.string if soup.title else 'N/A'}")
        print(f"Page size: {len(response.text)} characters")
        
        # Check for tables
        tables = soup.find_all('table')
        print(f"\nHTML tables found: {len(tables)}")
        for i, table in enumerate(tables[:3]):
            rows = len(table.find_all('tr'))
            print(f"  Table {i}: {rows} rows")
        
        # Check for script tags
        scripts = soup.find_all('script')
        print(f"\nScript tags found: {len(scripts)}")
        
        # Analyze script content
        json_ld_scripts = []
        data_scripts = []
        
        for i, script in enumerate(scripts):
            if script.get('type') == 'application/ld+json':
                json_ld_scripts.append(script)
            elif script.string:
                content = script.string.strip()
                if len(content) > 500:  # Substantial scripts
                    # Look for movie/data patterns
                    if any(keyword in content.lower() for keyword in ['movie', 'rating', 'chart', 'data', 'json']):
                        data_scripts.append((i, content[:500]))
        
        print(f"JSON-LD scripts: {len(json_ld_scripts)}")
        print(f"Data-containing scripts: {len(data_scripts)}")
        
        # Show JSON-LD content
        for i, script in enumerate(json_ld_scripts):
            try:
                data = json.loads(script.string)
                print(f"  JSON-LD {i}: {type(data)} with keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            except:
                print(f"  JSON-LD {i}: Could not parse as JSON")
        
        # Show data script previews
        for i, (script_idx, content) in enumerate(data_scripts[:3]):
            print(f"  Data script {script_idx} preview: {content}...")
        
        # Look for chart/movie containers
        chart_containers = soup.find_all(['div', 'section'], class_=lambda c: c and 'chart' in ' '.join(c).lower())
        print(f"\nChart containers: {len(chart_containers)}")
        
        # Look for list items that might contain movie data
        list_items = soup.find_all('li')
        movie_items = [li for li in list_items if li.get_text() and any(word in li.get_text().lower() for word in ['movie', 'title']) and len(li.get_text()) > 20]
        print(f"Potential movie list items: {len(movie_items)}")
        
        # Check if data is in HTML attributes (common in SPAs)
        elements_with_data = soup.find_all(attrs={'data-testid': True})
        print(f"Elements with data-testid: {len(elements_with_data)}")
        
        # Look for specific IMDb patterns
        title_columns = soup.find_all(['div', 'td'], class_=lambda c: c and 'title' in ' '.join(c).lower())
        print(f"Title-related elements: {len(title_columns)}")
        
        # Search for rating patterns
        rating_elements = soup.find_all(text=re.compile(r'\d+\.\d+'))
        print(f"Elements containing rating-like numbers: {len(rating_elements)}")
        
        return {
            'has_tables': len(tables) > 0,
            'has_json_ld': len(json_ld_scripts) > 0,
            'has_data_scripts': len(data_scripts) > 0,
            'has_structured_data': len(elements_with_data) > 0,
            'recommended_approach': 'javascript_data' if len(data_scripts) > 0 else 'structured_divs' if len(elements_with_data) > 0 else 'html_tables'
        }
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None

if __name__ == "__main__":
    result = analyze_imdb_structure()
    if result:
        print(f"\nRecommended extraction approach: {result['recommended_approach']}")
    print("\nAnalysis completed!")
