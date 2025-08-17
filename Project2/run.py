#!/usr/bin/env python3
"""
Script to interface with the data analysis API for promptfoo testing.
Takes a URL and uploads the appropriate files based on test context.
"""
import sys
import os
import requests
import json


def main():
    if len(sys.argv) != 2:
        print("Usage: python run.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Determine test type based on which files exist in current directory
    if os.path.exists("edges.csv") and os.path.exists("test_network_questions.txt"):
        # Network analysis test
        questions_file = "test_network_questions.txt"
        data_files = ["edges.csv"]
    elif os.path.exists("sample-weather.csv") and os.path.exists("test_weather_questions.txt"):
        # Weather analysis test
        questions_file = "test_weather_questions.txt"
        data_files = ["sample-weather.csv"]
    elif os.path.exists("sample-sales.csv") and os.path.exists("test_sales_questions.txt"):
        # Sales analysis test
        questions_file = "test_sales_questions.txt"
        data_files = ["sample-sales.csv"]
    else:
        print("Error: Cannot determine test type. Missing required files.")
        sys.exit(1)
    
    # Make API request
    try:
        # Prepare files for upload
        files_to_upload = {}
        
        # Add questions file
        files_to_upload['questions.txt'] = open(questions_file, 'rb')
        
        # Add data files
        for data_file in data_files:
            if os.path.exists(data_file):
                files_to_upload['files'] = open(data_file, 'rb')
        
        response = requests.post(
            f"{url}/api/",
            files=files_to_upload,
            timeout=180
        )
        
        # Close file handles
        for f in files_to_upload.values():
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            # Extract just the result part for promptfoo
            if 'result' in result and 'result' in result['result']:
                print(json.dumps(result['result']['result']))
            elif 'result' in result:
                print(json.dumps(result['result']))
            else:
                print(json.dumps(result))
        else:
            print(f"Error: {response.status_code} - {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
