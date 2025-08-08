#!/usr/bin/env python3
"""Test script to verify CAPTCHA solving functionality"""

import requests
import json

def test_captcha_scraping():
    """Test the CAPTCHA solving on judgments.ecourts.gov.in"""
    
    # Test the API endpoint with the ecourts URL
    url = "http://localhost:8000/api/"
    
    # Prepare the test data
    test_data = {
        "task_description": "The Indian high court judgement dataset contains judgements from the Indian High Courts, downloaded from [ecourts website](https://judgments.ecourts.gov.in/). It contains judgments of 25 high courts, and here we will analyse the Allahabad High court judgements. It contains 1,500 judgments in different languages. We'll summarise this data by answering some questions about the judgements.",
        "questions": [
            "How many cases were filed?",
            "What is the distribution of case types?",
            "Which are the most common judgment types?"
        ]
    }
    
    print("Testing CAPTCHA solving functionality...")
    print(f"Sending request to: {url}")
    
    try:
        response = requests.post(url, json=test_data, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response received successfully!")
            print(f"Workflow Type: {result.get('workflow_type', 'N/A')}")
            print(f"Status: {result.get('status', 'N/A')}")
            
            if result.get('status') == 'error':
                print(f"Error: {result.get('error', 'Unknown error')}")
                
                # Check if it's still a CAPTCHA-related issue
                if 'captcha' in result.get('error', '').lower():
                    print("❌ CAPTCHA detection/solving failed")
                    return False
                elif '404' in result.get('error', ''):
                    print("⚠️  Still getting 404 errors - CAPTCHA may not be detected properly")
                    return False
            else:
                print("✅ Request processed successfully!")
                return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_captcha_scraping()
    if success:
        print("\n🎉 CAPTCHA solving test completed successfully!")
    else:
        print("\n❌ CAPTCHA solving test failed. Check the logs above.")
