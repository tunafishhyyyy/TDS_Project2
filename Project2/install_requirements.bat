@echo off
echo Installing required packages for Data Analysis API...
echo ====================================================

echo Installing core FastAPI packages...
pip install fastapi uvicorn[standard] python-multipart

echo Installing data processing packages...
pip install pandas matplotlib requests python-dotenv

echo Installing LangChain packages...
pip install langchain langchain-openai langchain-community langchain-core

echo Installing AI/ML packages...
pip install openai tiktoken langsmith

echo Installing database and utility packages...
pip install duckdb faiss-cpu chromadb jinja2 pydantic

echo ====================================================
echo Installation complete!
echo.
echo Running verification script...
python check_installation.py

echo.
echo To start the server, run:
echo uvicorn main:app --reload
pause
