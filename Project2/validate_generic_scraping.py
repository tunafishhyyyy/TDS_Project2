#!/usr/bin/env python3
"""
Local validation script to test web scraping steps without running the full API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chains.web_scraping_steps import (
    ScrapeTableStep, InspectTableStep, CleanDataStep, 
    AnalyzeDataStep, VisualizeStep, AnswerQuestionsStep
)

def test_wikipedia_films():
    """Test the Wikipedia films example locally"""
    
    print("Testing Wikipedia highest-grossing films...")
    
    task_description = """Scrape the list of highest grossing films from Wikipedia:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions:
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it."""
    
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    
    try:
        # Step 1: Scrape
        step1 = ScrapeTableStep()
        data = step1.run({'url': url, 'task_description': task_description})
        print("✅ Step 1: Scraping completed")
        
        # Step 2: Inspect
        step2 = InspectTableStep()
        data.update(step2.run(data))
        print("✅ Step 2: Inspection completed")
        
        # Step 3: Clean
        step3 = CleanDataStep()
        data.update(step3.run(data))
        print("✅ Step 3: Cleaning completed")
        
        # Step 4: Analyze
        step4 = AnalyzeDataStep()
        data['top_n'] = 20
        data['task_description'] = task_description
        data.update(step4.run(data))
        print("✅ Step 4: Analysis completed")
        
        # Step 5: Visualize
        step5 = VisualizeStep()
        data['return_base64'] = True
        data.update(step5.run(data))
        print("✅ Step 5: Visualization completed")
        
        # Step 6: Answer questions
        step6 = AnswerQuestionsStep()
        result = step6.run(data)
        print("✅ Step 6: Questions answered")
        
        # Print summary
        answers = result.get('answers', {})
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"Analysis column: {answers.get('summary', {}).get('analysis_column', 'N/A')}")
        print(f"Total items: {answers.get('summary', {}).get('total_items_in_dataset', 0)}")
        print(f"Data type: {answers.get('summary', {}).get('data_type', 'N/A')}")
        
        # Key answers
        key_answers = {
            'items_above_2_billion': 'Movies above $2bn',
            'items_before_2000': 'Movies before 2000',
            'earliest_above_threshold': 'Earliest $1.5bn+ movie',
            'item_ranking_5th': '5th ranked movie'
        }
        
        for key, description in key_answers.items():
            if key in answers:
                print(f"{description}: {answers[key]}")
        
        # Check if visualization was created
        if 'plot_base64' in data and data['plot_base64']:
            print(f"📈 Visualization created: {data['chart_type']} chart")
            print(f"   Base64 size: {len(data['plot_base64'])} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imdb_ratings():
    """Test IMDB ratings example"""
    
    print("\nTesting IMDB top movies...")
    
    task_description = """Scrape the top 50 movies from:
https://www.imdb.com/chart/top

Extract movie name, year, rating.
Create a histogram of IMDb ratings."""
    
    url = "https://www.imdb.com/chart/top"
    
    try:
        # Quick test of first few steps
        step1 = ScrapeTableStep()
        data = step1.run({'url': url, 'task_description': task_description})
        
        if 'data' in data and not data['data'].empty:
            print("✅ IMDB scraping successful")
            print(f"   Found {data['data'].shape[0]} rows, {data['data'].shape[1]} columns")
            return True
        else:
            print("❌ IMDB scraping failed - no data found")
            return False
            
    except Exception as e:
        print(f"❌ IMDB test error: {e}")
        return False

def validate_generic_capabilities():
    """Validate that the generic capabilities work"""
    
    print("🔍 VALIDATING GENERIC WEB SCRAPING CAPABILITIES")
    print("="*60)
    
    results = []
    
    # Test 1: Wikipedia Films (most reliable)
    results.append(test_wikipedia_films())
    
    # Test 2: IMDB (quick check)
    results.append(test_imdb_ratings())
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(results)
    total = len(results)
    
    print(f"Tests passed: {successful}/{total}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    if successful == total:
        print("🎉 All validation tests passed!")
        print("✅ Generic web scraping capabilities are working correctly")
    else:
        print("⚠️  Some validation tests failed")
        print("❌ Generic capabilities may need further improvement")
    
    return successful == total

if __name__ == "__main__":
    success = validate_generic_capabilities()
    sys.exit(0 if success else 1)
