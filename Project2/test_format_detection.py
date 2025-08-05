#!/usr/bin/env python3
"""
Test script to validate the enhanced data format detection for IMDb
"""

import sys
import os
sys.path.append('.')

from chains.web_scraping_steps import DetectDataFormatStep
import requests
from bs4 import BeautifulSoup

def test_imdb_format_detection():
    """Test the format detection on IMDb to verify it detects JavaScript data"""
    print("Testing IMDb format detection...")
    
    # Test input
    input_data = {
        'url': 'https://www.imdb.com/chart/top',
        'task_description': 'Scrape IMDb top movies with ratings'
    }
    
    try:
        # Create and run the detection step
        detector = DetectDataFormatStep()
        result = detector.run(input_data)
        
        print(f"Format detected: {result['format_analysis']['format']}")
        print(f"Strategy: {result['format_analysis']['strategy']}")
        print(f"Confidence: {result['format_analysis']['confidence']}")
        print(f"Reasoning: {result['format_analysis'].get('reasoning', 'N/A')}")
        
        if 'json_data' in result['format_analysis']:
            print(f"JSON data length: {len(result['format_analysis']['json_data'])}")
            
        return result
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def manual_imdb_analysis():
    """Manually check IMDb page structure"""
    print("\nManual IMDb analysis...")
    
    url = 'https://www.imdb.com/chart/top'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for tables
        tables = soup.find_all('table')
        print(f"HTML tables found: {len(tables)}")
        
        # Check for script tags
        scripts = soup.find_all('script')
        print(f"Script tags found: {len(scripts)}")
        
        # Look for JSON-like content in scripts
        json_scripts = 0
        data_scripts = 0
        for script in scripts:
            if script.string:
                if 'application/ld+json' in script.get('type', ''):
                    json_scripts += 1
                if any(keyword in script.string.lower() for keyword in ['data', 'movie', 'rating', 'chart']):
                    data_scripts += 1
                    if len(script.string) > 500:  # Substantial script
                        print(f"Found substantial data script: {len(script.string)} chars")
                        # Show first 200 chars
                        print(f"Preview: {script.string[:200]}...")
        
        print(f"JSON-LD scripts: {json_scripts}")
        print(f"Data-containing scripts: {data_scripts}")
        
        # Look for chart containers
        chart_containers = soup.find_all(['div', 'section'], class_=lambda c: c and 'chart' in ' '.join(c).lower())
        print(f"Chart containers found: {len(chart_containers)}")
        
        # Look for movie list structures
        movie_elements = soup.find_all(['li', 'div'], class_=lambda c: c and any(word in ' '.join(c).lower() for word in ['movie', 'title', 'item']))
        print(f"Movie-related elements: {len(movie_elements)}")
        
    except Exception as e:
        print(f"Error in manual analysis: {e}")

if __name__ == "__main__":
    # Test format detection
    result = test_imdb_format_detection()
    
    # Manual analysis for comparison
    manual_imdb_analysis()
    
    print("\nTest completed!")
