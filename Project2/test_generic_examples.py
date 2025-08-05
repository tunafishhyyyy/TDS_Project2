#!/usr/bin/env python3
"""
Test script to verify generic web scraping approach works for all examples from prompt.txt
"""

import requests
import json
import time
import sys

# API base URL (update if different)
API_BASE = "http://localhost:8000"

def test_example(name: str, task_description: str) -> dict:
    """Test a specific example"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        # Submit task
        response = requests.post(f"{API_BASE}/api/analyze", json={
            "task_description": task_description,
            "workflow_type": "multi_step_web_scraping"
        })
        
        if response.status_code != 200:
            return {"error": f"Failed to submit task: {response.status_code}", "response": response.text}
        
        task_data = response.json()
        task_id = task_data.get("task_id")
        
        if not task_id:
            return {"error": "No task ID returned", "response": task_data}
        
        print(f"Task submitted: {task_id}")
        
        # Poll for completion
        max_wait = 180  # 3 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            status_response = requests.get(f"{API_BASE}/api/tasks/{task_id}/status")
            
            if status_response.status_code != 200:
                return {"error": f"Failed to get status: {status_response.status_code}"}
            
            status_data = status_response.json()
            status = status_data.get("status", "unknown")
            
            print(f"Status: {status} (waited {wait_time}s)")
            
            if status == "completed":
                result = status_data.get("result", {})
                return {
                    "success": True,
                    "task_id": task_id,
                    "result": result,
                    "execution_time": wait_time
                }
            elif status == "error":
                return {
                    "error": status_data.get("error", "Unknown error"),
                    "task_id": task_id,
                    "status_data": status_data
                }
            
            time.sleep(10)
            wait_time += 10
        
        return {"error": "Timeout waiting for completion", "task_id": task_id}
        
    except Exception as e:
        return {"error": str(e)}

def main():
    """Test all examples from prompt.txt"""
    
    # Test examples from the prompt
    examples = [
        {
            "name": "Example 1: Wikipedia Highest Grossing Films",
            "task": """Scrape the list of highest grossing films from Wikipedia:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions:
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it."""
        },
        
        {
            "name": "Example 2: IMDb Top Movies",
            "task": """Scrape the top 50 movies from:
https://www.imdb.com/chart/top

Extract movie name, year, rating.
Create a histogram of IMDb ratings.
Answer:
What is the average rating?
Which decade has the most top-rated movies?"""
        },
        
        {
            "name": "Example 3: India Inflation Data",
            "task": """Scrape inflation rate data for India from:
https://tradingeconomics.com/india/inflation-cpi

Plot a time series of inflation over the last 12 months.
Answer:
What is the current inflation rate?
What was the highest rate in the last year?"""
        },
        
        {
            "name": "Example 4: COVID-19 Cases Data",
            "task": """Scrape the table from:
https://www.worldometers.info/coronavirus/

Extract top 20 countries by total cases.
Plot total cases vs. deaths (scatter plot).
Answer:
Which country has the highest death-to-case ratio?
What's the global average recovery rate?"""
        },
        
        {
            "name": "Example 5: Cricket Stats from ESPN",
            "task": """Scrape top 10 batsmen from:
https://stats.espncricinfo.com/ci/content/records/

Extract player name, country, total runs, average.
Plot total runs vs batting average.
Answer:
Who has the highest average among players with over 8000 runs?
Which country has the most players in top 10?"""
        }
    ]
    
    results = []
    
    print("Starting comprehensive test of generic web scraping capabilities...")
    print(f"Testing against API at: {API_BASE}")
    
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {example['name']}")
        
        result = test_example(example['name'], example['task'])
        result['example_name'] = example['name']
        result['example_number'] = i
        
        results.append(result)
        
        # Print summary
        if result.get('success'):
            print(f"✅ SUCCESS - Completed in {result.get('execution_time', 0)}s")
            
            # Print key results if available
            if 'result' in result and 'results' in result['result']:
                answers = result['result'].get('results', {})
                print(f"   Answers found: {len(answers)} items")
                
                # Print some key answers
                for key, value in answers.items():
                    if key in ['item_ranking_5th', 'average_rating', 'current_inflation_rate', 
                             'highest_death_rate_country', 'highest_average_player_8000_runs']:
                        print(f"   {key}: {value}")
        else:
            print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"Total examples tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
    
    if successful:
        avg_time = sum(r.get('execution_time', 0) for r in successful) / len(successful)
        print(f"Average execution time: {avg_time:.1f}s")
    
    # Print failed examples
    if failed:
        print(f"\nFailed examples:")
        for r in failed:
            print(f"  - {r['example_name']}: {r.get('error', 'Unknown error')}")
    
    # Save detailed results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: test_results.json")
    
    return len(successful) == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
