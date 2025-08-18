#!/usr/bin/env python3
"""
TDS Data Analysis API - Main Server Launcher

This script launches the enhanced TDS Data Analysis API on port 8001.
The system provides comprehensive multi-modal data analysis capabilities
including web scraping, document processing, image analysis, and statistical computation.

Features:
- Multi-LLM orchestration with intelligent fallback
- Real-time data analysis and visualization
- Secure sandboxed code execution
- Comprehensive system diagnostics
- Enterprise-grade error handling and logging

Usage:
    python run_main_server.py

The server will be available at http://localhost:8001
API documentation at http://localhost:8001/docs
System diagnostics at http://localhost:8001/summary
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
