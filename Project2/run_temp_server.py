#!/usr/bin/env python3
"""
Run the enhanced main_app.py system on port 8001 for the TDS Data Analysis API
"""
import os
import sys
import uvicorn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import the app from main_app.py
    from chains.main_app import app
    
    print("🚀 Starting TDS Data Analysis API on port 8001...")
    print("📊 Enhanced multi-modal analysis system")
    print("🔧 Features: LLM orchestration, web scraping, image processing, PDF analysis")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
