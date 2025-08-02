#!/usr/bin/env python3
"""
Enhanced test script for the Data Analysis API with LangChain integration

This script tests LangChain workflows and advanced API features.
Make sure to set up your .env file with API keys before running.
"""

import requests
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api"

class LangChainAPITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.task_ids = []
    
    def test_capabilities(self):
        """Test the capabilities endpoint"""
        print("🔍 Testing LangChain capabilities endpoint...")
        try:
            response = self.session.get(f"{API_URL}/capabilities")
            if response.status_code == 200:
                capabilities = response.json()
                print("✅ Capabilities endpoint working")
                print(f"   Available workflows: {capabilities.get('available_workflows', [])}")
                print(f"   LangChain integration: {capabilities.get('langchain_integration', False)}")
                return True, capabilities
            else:
                print(f"❌ Capabilities test failed: {response.status_code}")
                return False, {}
        except Exception as e:
            print(f"❌ Capabilities test error: {e}")
            return False, {}
    
    def test_basic_data_analysis_workflow(self):
        """Test basic data analysis workflow"""
        print("\n🔍 Testing basic data analysis workflow...")
        
        payload = {
            "task_description": "Analyze customer churn data to identify key factors",
            "workflow_type": "data_analysis",
            "dataset_info": {
                "description": "Customer churn dataset with demographics and usage data",
                "columns": ["customer_id", "age", "tenure", "monthly_charges", "total_charges", "churn"],
                "data_types": {
                    "customer_id": "string",
                    "age": "integer",
                    "tenure": "integer", 
                    "monthly_charges": "float",
                    "total_charges": "float",
                    "churn": "boolean"
                },
                "sample_size": 7043
            },
            "parameters": {
                "analysis_type": "comprehensive",
                "include_visualizations": True
            }
        }
        
        return self._execute_and_wait(payload, "data_analysis_workflow")
    
    def test_exploratory_data_analysis_workflow(self):
        """Test EDA workflow"""
        print("\n🔍 Testing Exploratory Data Analysis workflow...")
        
        workflow_request = {
            "workflow_type": "exploratory_data_analysis",
            "input_data": {
                "dataset_info": {
                    "description": "Sales performance dataset",
                    "columns": ["date", "product_category", "sales_amount", "units_sold", "region"],
                    "data_types": {
                        "date": "datetime",
                        "product_category": "categorical",
                        "sales_amount": "float",
                        "units_sold": "integer",
                        "region": "categorical"
                    },
                    "sample_size": 10000
                },
                "business_context": "Quarterly sales analysis for strategic planning",
                "parameters": {
                    "time_period": "Q4 2024",
                    "focus_areas": ["seasonality", "regional_performance", "product_trends"]
                }
            }
        }
        
        return self._execute_workflow_request(workflow_request, "eda_workflow")
    
    def test_code_generation_workflow(self):
        """Test code generation workflow"""
        print("\n🔍 Testing code generation workflow...")
        
        workflow_request = {
            "workflow_type": "code_generation",
            "input_data": {
                "task_description": "Create data visualization dashboard for sales data",
                "data_context": "CSV file with sales data including date, amount, region, product",
                "libraries": "pandas, matplotlib, seaborn, plotly",
                "output_format": "interactive dashboard with multiple charts"
            }
        }
        
        return self._execute_workflow_request(workflow_request, "code_generation_workflow")
    
    def test_predictive_modeling_workflow(self):
        """Test predictive modeling workflow"""
        print("\n🔍 Testing predictive modeling workflow...")
        
        workflow_request = {
            "workflow_type": "predictive_modeling",
            "input_data": {
                "problem_statement": "Predict customer lifetime value based on historical data",
                "target_variable": "customer_lifetime_value",
                "dataset_characteristics": {
                    "size": 50000,
                    "features": ["demographics", "transaction_history", "engagement_metrics"],
                    "target_type": "continuous",
                    "missing_data": "minimal"
                },
                "business_requirements": "Model should be interpretable and achieve 85% accuracy",
                "performance_requirements": "Fast inference for real-time scoring"
            }
        }
        
        return self._execute_workflow_request(workflow_request, "predictive_modeling_workflow")
    
    def test_multi_step_pipeline(self):
        """Test multi-step workflow pipeline"""
        print("\n🔍 Testing multi-step workflow pipeline...")
        
        pipeline_request = {
            "pipeline_type": "custom_analysis",
            "steps": [
                {
                    "workflow_type": "exploratory_data_analysis",
                    "input_data": {
                        "dataset_info": {
                            "description": "E-commerce transaction data",
                            "columns": ["transaction_id", "customer_id", "product_id", "amount", "timestamp"],
                            "sample_size": 25000
                        },
                        "business_context": "Monthly performance review"
                    }
                },
                {
                    "workflow_type": "data_visualization",
                    "input_data": {
                        "data_description": "E-commerce transactions with temporal and customer patterns",
                        "variables": ["amount", "timestamp", "customer_id", "product_id"],
                        "analysis_goals": "Identify trends and patterns in customer behavior",
                        "target_audience": "executive team"
                    }
                },
                {
                    "workflow_type": "report_generation",
                    "input_data": {
                        "analysis_results": "Results from previous steps",
                        "data_summary": "E-commerce transaction analysis",
                        "key_findings": "Customer behavior patterns and revenue trends",
                        "audience": "executive team"
                    }
                }
            ]
        }
        
        return self._execute_pipeline_request(pipeline_request, "multi_step_pipeline")
    
    def test_complete_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        print("\n🔍 Testing complete analysis pipeline...")
        
        payload = {
            "task_description": "Comprehensive analysis of customer behavior data",
            "dataset_info": {
                "description": "Customer behavior and transaction dataset",
                "columns": ["customer_id", "age", "income", "purchase_frequency", "avg_order_value"],
                "data_types": {
                    "customer_id": "string",
                    "age": "integer",
                    "income": "float",
                    "purchase_frequency": "float",
                    "avg_order_value": "float"
                },
                "sample_size": 15000
            },
            "include_modeling": True,
            "target_audience": "business stakeholders",
            "business_context": "Customer segmentation and retention strategy",
            "problem_statement": "Identify customer segments and predict churn risk",
            "target_variable": "churn_risk"
        }
        
        try:
            response = self.session.post(
                f"{API_URL}/analyze/complete",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]
                self.task_ids.append(task_id)
                
                print("✅ Complete analysis pipeline initiated")
                print(f"   Task ID: {task_id}")
                print(f"   Estimated duration: {result.get('estimated_duration', 'unknown')}")
                
                # Wait for completion (longer timeout for complete analysis)
                return self._wait_for_completion(task_id, timeout=300)
            else:
                print(f"❌ Complete analysis pipeline failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Complete analysis pipeline error: {e}")
            return False
    
    def _execute_and_wait(self, payload: Dict[str, Any], test_name: str, timeout: int = 120) -> bool:
        """Execute a task and wait for completion"""
        try:
            response = self.session.post(
                f"{API_URL}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]
                self.task_ids.append(task_id)
                
                print(f"✅ {test_name} initiated")
                print(f"   Task ID: {task_id}")
                
                return self._wait_for_completion(task_id, timeout)
            else:
                print(f"❌ {test_name} failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            return False
    
    def _execute_workflow_request(self, request: Dict[str, Any], test_name: str, timeout: int = 120) -> bool:
        """Execute a workflow request and wait for completion"""
        try:
            response = self.session.post(
                f"{API_URL}/workflow",
                json=request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]
                self.task_ids.append(task_id)
                
                print(f"✅ {test_name} initiated")
                print(f"   Task ID: {task_id}")
                print(f"   Workflow type: {result['workflow_type']}")
                
                return self._wait_for_completion(task_id, timeout)
            else:
                print(f"❌ {test_name} failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            return False
    
    def _execute_pipeline_request(self, request: Dict[str, Any], test_name: str, timeout: int = 180) -> bool:
        """Execute a pipeline request and wait for completion"""
        try:
            response = self.session.post(
                f"{API_URL}/pipeline",
                json=request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]
                self.task_ids.append(task_id)
                
                print(f"✅ {test_name} initiated")
                print(f"   Task ID: {task_id}")
                print(f"   Pipeline type: {result['pipeline_type']}")
                print(f"   Total steps: {result['total_steps']}")
                
                return self._wait_for_completion(task_id, timeout)
            else:
                print(f"❌ {test_name} failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            return False
    
    def _wait_for_completion(self, task_id: str, timeout: int = 120) -> bool:
        """Wait for task completion and show results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{API_URL}/tasks/{task_id}/status")
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data["status"]
                    
                    if status == "completed":
                        print(f"   ✅ Task completed successfully")
                        if "workflow_result" in status_data:
                            result = status_data["workflow_result"]
                            print(f"   📊 Result type: {type(result).__name__}")
                            if isinstance(result, dict) and "status" in result:
                                print(f"   📊 Workflow status: {result['status']}")
                        return True
                    elif status == "failed":
                        print(f"   ❌ Task failed")
                        if "error" in status_data:
                            print(f"   Error: {status_data['error']}")
                        return False
                    elif status in ["queued", "processing"]:
                        print(f"   ⏳ Status: {status}")
                    
                    time.sleep(5)
                else:
                    print(f"   ❌ Error checking status: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Error waiting for completion: {e}")
                return False
        
        print(f"   ⏰ Timeout reached ({timeout}s)")
        return False
    
    def run_langchain_tests(self):
        """Run comprehensive LangChain workflow tests"""
        print("🚀 Starting LangChain API Test Suite")
        print("=" * 60)
        
        # Check capabilities first
        success, capabilities = self.test_capabilities()
        if not success:
            print("❌ Cannot proceed without capabilities - check API configuration")
            return False
        
        # Check if we have LangChain integration
        if not capabilities.get("langchain_integration", False):
            print("⚠️  LangChain integration not detected - check environment setup")
        
        results = []
        
        # Basic tests
        results.append(("Basic Data Analysis", self.test_basic_data_analysis_workflow()))
        results.append(("EDA Workflow", self.test_exploratory_data_analysis_workflow()))
        results.append(("Code Generation", self.test_code_generation_workflow()))
        results.append(("Predictive Modeling", self.test_predictive_modeling_workflow()))
        
        # Advanced tests
        results.append(("Multi-Step Pipeline", self.test_multi_step_pipeline()))
        results.append(("Complete Analysis", self.test_complete_analysis_pipeline()))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 LangChain Test Results Summary")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<25} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All LangChain tests passed!")
        else:
            print("⚠️  Some tests failed. Check API logs and environment setup.")
        
        return passed == total

def main():
    """Main function to run the LangChain test suite"""
    print("Data Analysis API with LangChain - Test Suite")
    print("=" * 60)
    print("Prerequisites:")
    print("1. FastAPI server running on http://localhost:8000")
    print("2. .env file with OpenAI API key configured")
    print("3. All requirements installed")
    print("\nStart server with: uvicorn src.main:app --reload")
    
    choice = input("\nSelect test mode:\n1. Basic tests only\n2. Full LangChain tests\nEnter choice (1 or 2): ")
    
    tester = LangChainAPITester()
    
    if choice == "1":
        # Import and run basic tests
        from test_api import APITester
        basic_tester = APITester()
        success = basic_tester.run_all_tests()
    elif choice == "2":
        success = tester.run_langchain_tests()
    else:
        print("Invalid choice. Running basic tests.")
        from test_api import APITester
        basic_tester = APITester()
        success = basic_tester.run_all_tests()
    
    if success:
        exit(0)
    else:
        exit(1)

if __name__ == "__main__":
    main()
