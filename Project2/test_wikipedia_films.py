#!/usr/bin/env python3
"""
Test script for the Wikipedia highest-grossing films example from prompt.txt
This tests the specific case that needs image generation with scatterplot.
"""
import requests
import json
import base64
import sys
import os

# API URL
API_URL = "http://localhost:8000/api/"

def test_wikipedia_films():
    """
    Test the Wikipedia highest-grossing films scraping with the exact questions from prompt.txt
    """
    # The exact task from prompt.txt
    task_description = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""
    
    print("Testing Wikipedia highest-grossing films scraping...")
    print("=" * 70)
    print(f"Task: {task_description}")
    print("=" * 70)
    
    try:
        # Send the request as form data (like the working test examples)
        response = requests.post(
            API_URL,
            data={"questions_txt": task_description},
            timeout=300  # 5 minutes timeout for scraping
        )
        
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print("\nResponse Structure:")
            print(f"- Type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"- Keys: {list(result.keys())}")
                
                # Check for answer field
                if 'answer' in result:
                    answer = result['answer']
                    print(f"\nAnswer Type: {type(answer)}")
                    
                    if isinstance(answer, list):
                        print(f"Number of answers: {len(answer)}")
                        for i, ans in enumerate(answer, 1):
                            print(f"\nAnswer {i}:")
                            if isinstance(ans, str) and ans.startswith("data:image/png;base64,"):
                                # This is the base64 image
                                print(f"  📊 Image data URI (length: {len(ans)} chars)")
                                print(f"  Format: {ans[:50]}...")
                                
                                # Validate base64 size requirement
                                base64_part = ans.split(',', 1)[1] if ',' in ans else ans
                                if len(base64_part) <= 100000:
                                    print(f"  ✅ Image size requirement met ({len(base64_part)} <= 100,000 chars)")
                                else:
                                    print(f"  ❌ Image size requirement failed ({len(base64_part)} > 100,000 chars)")
                                    
                                # Test if it's valid base64
                                try:
                                    decoded = base64.b64decode(base64_part)
                                    print(f"  ✅ Valid base64 encoding ({len(decoded)} bytes)")
                                except Exception as e:
                                    print(f"  ❌ Invalid base64: {e}")
                                    
                            else:
                                print(f"  📝 Text answer: {ans}")
                                
                # Check for plot data
                if 'plot_base64' in result:
                    print(f"\n📊 Direct plot data found (length: {len(result['plot_base64'])} chars)")
                    
                # Check for errors
                if 'error' in result:
                    print(f"\n❌ Error in result: {result['error']}")
                    
            else:
                print(f"Raw result: {result}")
                
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🧪 Testing Wikipedia Films Scraping with Image Generation")
    print("=" * 70)
    
    # Check if API is running
    if not check_api_health():
        print("❌ API server is not running at http://localhost:8000/")
        print("Please start the server with: uvicorn main:app --reload")
        sys.exit(1)
    
    print("✅ API server is running")
    
    # Run the test
    test_wikipedia_films()
    
    print("\n" + "=" * 70)
    print("🏁 Test completed!")
