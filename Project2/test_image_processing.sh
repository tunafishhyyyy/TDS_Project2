#!/bin/bash

# Comprehensive Image Processing Test Suite
# Tests various image types and OCR capabilities

echo "=== Image Processing Test Suite ==="
echo ""

# Test 1: Simple Text Image
echo "Test 1: Simple Text Image with OCR"
echo "File: test_simple_text.png"
echo "Expected: OCR should extract text and analyze sales data"
echo ""

response1=$(curl -s -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_simple_text_questions.txt" \
  -F "files=@test_simple_text.png" \
  --max-time 60)

echo "Response: $response1"
echo ""
echo "----------------------------------------"
echo ""

# Test 2: Employee Table Image  
echo "Test 2: Employee Table Image with OCR"
echo "File: test_employee_table.png"
echo "Expected: OCR should extract table data and analyze employee info"
echo ""

response2=$(curl -s -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_employee_table_questions.txt" \
  -F "files=@test_employee_table.png" \
  --max-time 60)

echo "Response: $response2"
echo ""
echo "----------------------------------------"
echo ""

# Test 3: Sales Chart Image
echo "Test 3: Sales Chart Image"
echo "File: test_sales_chart.png"
echo "Expected: Should process chart image (may rely on LLM vision capabilities)"
echo ""

response3=$(curl -s -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_sales_chart_questions.txt" \
  -F "files=@test_sales_chart.png" \
  --max-time 60)

echo "Response: $response3"
echo ""
echo "----------------------------------------"
echo ""

# Test 4: Correlation Plot Image
echo "Test 4: Correlation Plot Image"
echo "File: test_correlation_plot.png"
echo "Expected: Should analyze scatter plot and correlation data"
echo ""

response4=$(curl -s -X POST "http://localhost:8001/api/" \
  -F "questions.txt=@test_correlation_questions.txt" \
  -F "files=@test_correlation_plot.png" \
  --max-time 60)

echo "Response: $response4"
echo ""
echo "========================================="
echo "Image Processing Test Suite Complete"
echo "========================================="
