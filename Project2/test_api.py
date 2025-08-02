#!/usr/bin/env python3
"""
Test script for the Data Analysis API

This script tests various endpoints and input formats for the FastAPI application.
Run this script after starting the FastAPI server.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api"

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.task_ids = []
    
    def test_health_check(self):
        """Test the health check endpoint"""
        print("🔍 Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                print(f"   Response: {response.json()}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        print("\n🔍 Testing root endpoint...")
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                print("✅ Root endpoint working")
                print(f"   Response: {response.json()}")
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def test_analyze_with_structured_json(self):
        """Test analyze endpoint with structured JSON"""
        print("\n🔍 Testing analyze endpoint with structured JSON...")
        
        payload = {
            "task_description": "Analyze sales data for Q4 2024",
            "data_source": "sales_database.csv",
            "parameters": {
                "time_period": "Q4 2024",
                "metrics": ["revenue", "profit", "customer_count"],
                "visualizations": True
            },
            "priority": "high"
        }
        
        try:
            response = self.session.post(
                f"{API_URL}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.task_ids.append(result["task_id"])
                print("✅ Structured JSON test passed")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Status: {result['status']}")
                return True
            else:
                print(f"❌ Structured JSON test failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Structured JSON test error: {e}")
            return False
    
    def test_analyze_with_simple_json(self):
        """Test analyze endpoint with simple JSON"""
        print("\n🔍 Testing analyze endpoint with simple JSON...")
        
        payload = {
            "task_description": "Generate monthly report with charts and graphs"
        }
        
        try:
            response = self.session.post(
                f"{API_URL}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.task_ids.append(result["task_id"])
                print("✅ Simple JSON test passed")
                print(f"   Task ID: {result['task_id']}")
                return True
            else:
                print(f"❌ Simple JSON test failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Simple JSON test error: {e}")
            return False
    
    def test_analyze_with_generic_dict(self):
        """Test analyze endpoint with generic dictionary"""
        print("\n🔍 Testing analyze endpoint with generic dictionary...")
        
        payload = {
            "request": "Create data visualization dashboard",
            "dataset": "customer_data.xlsx",
            "output_format": "interactive_html"
        }
        
        try:
            response = self.session.post(
                f"{API_URL}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.task_ids.append(result["task_id"])
                print("✅ Generic dictionary test passed")
                print(f"   Task ID: {result['task_id']}")
                return True
            else:
                print(f"❌ Generic dictionary test failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Generic dictionary test error: {e}")
            return False
    
    def test_task_status(self):
        """Test task status endpoint"""
        if not self.task_ids:
            print("\n⚠️  No task IDs available for status testing")
            return False
        
        print("\n🔍 Testing task status endpoint...")
        task_id = self.task_ids[0]
        
        try:
            response = self.session.get(f"{API_URL}/tasks/{task_id}/status")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Task status test passed")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Status: {result['status']}")
                return True
            else:
                print(f"❌ Task status test failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Task status test error: {e}")
            return False
    
    def test_list_tasks(self):
        """Test list tasks endpoint"""
        print("\n🔍 Testing list tasks endpoint...")
        
        try:
            response = self.session.get(f"{API_URL}/tasks")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ List tasks test passed")
                print(f"   Total tasks: {result['total_tasks']}")
                print(f"   Tasks found: {len(result['tasks'])}")
                return True
            else:
                print(f"❌ List tasks test failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ List tasks test error: {e}")
            return False
    
    def test_invalid_task_status(self):
        """Test task status with invalid ID"""
        print("\n🔍 Testing task status with invalid ID...")
        
        try:
            response = self.session.get(f"{API_URL}/tasks/invalid-task-id/status")
            
            if response.status_code == 404:
                print("✅ Invalid task status test passed (correctly returned 404)")
                return True
            else:
                print(f"❌ Invalid task status test failed: expected 404, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Invalid task status test error: {e}")
            return False
    
    def test_delete_task(self):
        """Test task deletion"""
        if not self.task_ids:
            print("\n⚠️  No task IDs available for deletion testing")
            return False
        
        print("\n🔍 Testing task deletion...")
        task_id = self.task_ids[-1]  # Delete the last created task
        
        try:
            response = self.session.delete(f"{API_URL}/tasks/{task_id}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Task deletion test passed")
                print(f"   Deleted task: {result['deleted_task']['task_id']}")
                self.task_ids.remove(task_id)
                return True
            else:
                print(f"❌ Task deletion test failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Task deletion test error: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling with invalid data"""
        print("\n🔍 Testing error handling...")
        
        # Test with empty payload
        try:
            response = self.session.post(
                f"{API_URL}/analyze",
                json={},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                print("✅ Empty payload error handling test passed")
                return True
            else:
                print(f"❌ Empty payload test failed: expected 400, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error handling test error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all test cases"""
        print("🚀 Starting API Test Suite")
        print("=" * 50)
        
        results = []
        
        # Run tests
        results.append(("Health Check", self.test_health_check()))
        results.append(("Root Endpoint", self.test_root_endpoint()))
        results.append(("Structured JSON", self.test_analyze_with_structured_json()))
        results.append(("Simple JSON", self.test_analyze_with_simple_json()))
        results.append(("Generic Dictionary", self.test_analyze_with_generic_dict()))
        results.append(("Task Status", self.test_task_status()))
        results.append(("List Tasks", self.test_list_tasks()))
        results.append(("Invalid Task Status", self.test_invalid_task_status()))
        results.append(("Task Deletion", self.test_delete_task()))
        results.append(("Error Handling", self.test_error_handling()))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<20} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == total

def main():
    """Main function to run the test suite"""
    print("Data Analysis API Test Suite")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("You can start it with: uvicorn src.main:app --reload")
    input("Press Enter to continue with testing...")
    
    tester = APITester()
    success = tester.run_all_tests()
    
    if success:
        exit(0)
    else:
        exit(1)

if __name__ == "__main__":
    main()
