"""
Test script for the updated File Upload API
Demonstrates how to use the new file upload functionality
"""

import requests
import time
import json

# Base URL - change this to your actual API URL
BASE_URL = "http://localhost:8000"

def test_file_upload():
    """Test file upload functionality"""
    print("🔄 Testing file upload...")
    
    # Test with question.txt file
    with open("question.txt", "rb") as f:
        files = {"file": ("question.txt", f, "text/plain")}
        data = {
            "workflow_type": "data_analysis",
            "business_context": "E-commerce platform analysis"
        }
        
        response = requests.post(f"{BASE_URL}/api/", files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        print("✅ File upload successful!")
        print(f"Task ID: {result['task_id']}")
        print(f"Status: {result['status']}")
        return result['task_id']
    else:
        print(f"❌ File upload failed: {response.status_code}")
        print(response.text)
        return None

def test_form_data():
    """Test form data without file"""
    print("\n🔄 Testing form data submission...")
    
    data = {
        "task_description": "Analyze website traffic patterns and user behavior",
        "workflow_type": "exploratory_data_analysis",
        "business_context": "Monthly traffic analysis for optimization"
    }
    
    response = requests.post(f"{BASE_URL}/api/", data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Form data submission successful!")
        print(f"Task ID: {result['task_id']}")
        return result['task_id']
    else:
        print(f"❌ Form data submission failed: {response.status_code}")
        print(response.text)
        return None

def test_json_legacy():
    """Test legacy JSON endpoint"""
    print("\n🔄 Testing legacy JSON endpoint...")
    
    task_data = {
        "task_description": "Perform sentiment analysis on customer reviews",
        "workflow_type": "data_analysis",
        "dataset_info": {
            "source": "customer_reviews.csv",
            "columns": ["review_text", "rating", "date"],
            "size": "10000 rows"
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/analyze", json=task_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ JSON submission successful!")
        print(f"Task ID: {result['task_id']}")
        return result['task_id']
    else:
        print(f"❌ JSON submission failed: {response.status_code}")
        print(response.text)
        return None

def check_task_status(task_id):
    """Check the status of a task"""
    if not task_id:
        return
        
    print(f"\n🔄 Checking status for task: {task_id}")
    
    # Wait a bit for processing
    time.sleep(3)
    
    response = requests.get(f"{BASE_URL}/api/tasks/{task_id}/status")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Status: {result['status']}")
        
        if result['status'] == 'completed' and 'result' in result:
            print("📊 Analysis Results:")
            print(f"  - Analysis Type: {result['result']['analysis_type']}")
            print(f"  - Summary: {result['result']['summary']}")
            print("  - Insights:")
            for insight in result['result']['insights']:
                print(f"    • {insight}")
            print("  - Recommendations:")
            for rec in result['result']['recommendations']:
                print(f"    • {rec}")
                
            if 'file_analysis' in result['result']:
                print("  - File Analysis:")
                print(f"    • File: {result['result']['file_analysis']['file_name']}")
                print(f"    • Content Length: {result['result']['file_analysis']['content_length']} characters")
                print(f"    • Preview: {result['result']['file_analysis']['content_preview']}")
        
    else:
        print(f"❌ Status check failed: {response.status_code}")
        print(response.text)

def test_curl_command():
    """Show equivalent curl commands for testing"""
    print("\n📝 Equivalent curl commands for testing:")
    print("\n1. File upload:")
    print('curl "http://localhost:8000/api/" -F "file=@question.txt" -F "workflow_type=data_analysis"')
    
    print("\n2. Form data:")
    print('curl "http://localhost:8000/api/" -F "task_description=Analyze sales data" -F "workflow_type=data_analysis"')
    
    print("\n3. JSON (legacy):")
    print('curl -X POST "http://localhost:8000/api/analyze" -H "Content-Type: application/json" -d \'{"task_description": "Analyze data", "workflow_type": "data_analysis"}\'')

if __name__ == "__main__":
    print("🚀 Starting API Tests")
    print("=" * 50)
    
    # Test file upload
    file_task_id = test_file_upload()
    
    # Test form data
    form_task_id = test_form_data()
    
    # Test legacy JSON
    json_task_id = test_json_legacy()
    
    # Check status of all tasks
    for task_id in [file_task_id, form_task_id, json_task_id]:
        check_task_status(task_id)
    
    # Show curl commands
    test_curl_command()
    
    print("\n✅ All tests completed!")
