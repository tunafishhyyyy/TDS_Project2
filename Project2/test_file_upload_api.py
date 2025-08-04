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
    """Test file upload functionality (LangChain workflow)"""
    print("🔄 Testing file upload...")
    with open("dummy.txt", "rb") as f:
        files = {"file": ("dummy.txt", f, "text/plain")}
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
    """Test form data without file (LangChain workflow)"""
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

## Legacy endpoint test removed

def check_task_status(task_id):
    """Check the status of a task (LangChain workflow)"""
    if not task_id:
        return
    print(f"\n🔄 Checking status for task: {task_id}")
    time.sleep(3)
    response = requests.get(f"{BASE_URL}/api/tasks/{task_id}/status")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Status: {result['status']}")
        if result['status'] == 'completed' and 'result' in result:
            print("📊 Analysis Results:")
            if isinstance(result['result'], dict):
                for k, v in result['result'].items():
                    print(f"  - {k}: {v}")
            else:
                print(f"  - Result: {result['result']}")
        elif result['status'] == 'failed':
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"❌ Status check failed: {response.status_code}")
        print(response.text)

## Dead code removed

if __name__ == "__main__":
    print("🚀 Starting API Tests")
    print("=" * 50)
    # Test file upload
    file_task_id = test_file_upload()
    # Test form data
    form_task_id = test_form_data()
    # Check status of all tasks
    for task_id in [file_task_id, form_task_id]:
        check_task_status(task_id)
    print("\n✅ All tests completed!")
