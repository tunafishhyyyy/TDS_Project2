#!/usr/bin/env python3
"""Enhanced CAPTCHA test script"""

import json
import time
import requests
from datetime import datetime

def test_api_with_output():
    """Test the API and show full output"""
    url = "http://localhost:8000/api/"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting CAPTCHA test...")
    print(f"Target URL: {url}")
    
    try:
        with open('test_ecourts_questions.txt', 'rb') as f:
            files = {'questions_txt': ('test_ecourts_questions.txt', f, 'text/plain')}
            
            print("Sending request...")
            response = requests.post(url, files=files, timeout=120)
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\n=== RESPONSE CONTENT ===")
                print(json.dumps(result, indent=2))
                
                # Check for CAPTCHA-related messages
                error = result.get('result', {}).get('error', '')
                if 'captcha' in error.lower():
                    print("\n🔍 CAPTCHA DETECTION: Working!")
                    if 'solved captcha:' in error.lower():
                        print("✅ CAPTCHA SOLVING: Success!")
                    else:
                        print("❌ CAPTCHA SOLVING: Failed")
                        print(f"Error details: {error}")
                else:
                    print("✅ No CAPTCHA encountered or successfully bypassed!")
                    
            except json.JSONDecodeError:
                print("Response is not valid JSON:")
                print(response.text)
        else:
            print(f"Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    test_api_with_output()
