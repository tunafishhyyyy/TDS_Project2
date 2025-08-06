#!/bin/bash

# Script to fix common Flake8 issues in web_scraping_steps.py

FILE="chains/workflows.py"


echo "Fixing Flake8 issues in $FILE..."

# Fix bare except clauses (E722) by replacing with Exception
sed -i 's/except:/except Exception:/g' "$FILE"

# Remove unused variables by commenting them out (F841)
# This is safer than deleting them entirely

echo "Running Black formatter again to ensure consistent formatting..."
black "$FILE" --line-length=120

echo "Checking remaining issues..."
flake8 "$FILE" | head -20
echo "Total remaining issues:"
flake8 "$FILE" 2>&1 | wc -l
