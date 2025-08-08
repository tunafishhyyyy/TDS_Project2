#!/bin/bash

# Script to fix common Flake8 issues across the entire project

echo "Fixing lint issues across the project..."

# Function to fix issues in a specific file
fix_file() {
    local file=$1
    echo "Processing $file..."
    
    # Fix bare except clauses (E722)
    sed -i 's/except:/except Exception:/g' "$file"
    
    # Fix whitespace issues (remove trailing whitespace and fix blank lines)
    # Remove trailing whitespace (W291, W292)
    sed -i 's/[[:space:]]*$//' "$file"
    
    # Fix blank lines with whitespace (W293)
    sed -i '/^[[:space:]]*$/s/.*//' "$file"
    
    # Fix arithmetic operators spacing (E226)
    sed -i 's/\([0-9a-zA-Z_]\)\*\([0-9a-zA-Z_]\)/\1 * \2/g' "$file"
    sed -i 's/\([0-9a-zA-Z_]\)+\([0-9a-zA-Z_]\)/\1 + \2/g' "$file"
    sed -i 's/\([0-9a-zA-Z_]\)-\([0-9a-zA-Z_]\)/\1 - \2/g' "$file"
    sed -i 's/\([0-9a-zA-Z_]\)\/\([0-9a-zA-Z_]\)/\1 \/ \2/g' "$file"
    
    echo "Fixed basic issues in $file"
}

# Process all Python files
for file in main.py config.py install_dependencies.py test_file_upload_api.py; do
    if [[ -f "$file" ]]; then
        fix_file "$file"
    fi
done

for file in chains/*.py utils/*.py; do
    if [[ -f "$file" ]]; then
        fix_file "$file"
    fi
done

echo "Running basic fixes complete. Manual fixes still needed for:"
echo "- F401 (unused imports) - need manual review"
echo "- F821 (undefined names) - need manual review"
echo "- E501 (line too long) - need manual line breaking"
echo "- E128/E131 (continuation line indentation) - need manual fixing"

echo "Checking remaining critical issues..."
source venv/bin/activate
flake8 --config=/work/tanmay/TDS/TDS_Project2/Project2/.flake8 --select=F401,F821,E722 --statistics . 2>/dev/null || echo "Flake8 not available"
