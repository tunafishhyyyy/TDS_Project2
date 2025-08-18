#!/usr/bin/env python3
"""
Run the temp.py system on port 8001 for comparison with main system
"""
import os
import sys
import uvicorn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import the app from temp.py
    from chains.temp import app
    
    print("🚀 Starting enhanced data analysis system on port 8001...")
    print("📊 Main system still running on port 8000")
    print("🔧 Enhanced features: Multi-LLM fallback, better diagnostics, image optimization")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
