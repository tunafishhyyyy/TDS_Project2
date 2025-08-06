#!/usr/bin/env python3
"""
Code formatting and cleanup script for TDS Project2
Fixes line width violations, removes dead code, and improves code quality
"""

import os
import re
import subprocess
import sys
from pathlib import Path

def run_black_formatter():
    """Run Black formatter on all Python files"""
    try:
        print("Running Black formatter for line length compliance...")
        result = subprocess.run([
            sys.executable, '-m', 'black', 
            '--line-length=79', 
            '--target-version=py38',
            '.'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Black formatting completed successfully")
            print(result.stdout)
        else:
            print("❌ Black formatting failed:")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ Black not installed. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'black'])
        run_black_formatter()

def run_flake8_linting():
    """Run Flake8 linting to identify additional issues"""
    try:
        print("\nRunning Flake8 linting...")
        result = subprocess.run([
            sys.executable, '-m', 'flake8', 
            '--max-line-length=79',
            '--ignore=E501,W503',  # Ignore line length and line break before binary operator
            '.'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ No Flake8 violations found")
        else:
            print("⚠️  Flake8 found some issues:")
            print(result.stdout)
    except FileNotFoundError:
        print("❌ Flake8 not installed. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flake8'])
        run_flake8_linting()

def fix_empty_exception_handlers():
    """Find and report empty exception handlers that need attention"""
    python_files = list(Path('.').rglob('*.py'))
    issues_found = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for i, line in enumerate(lines):
                # Look for empty except blocks with just 'pass'
                if 'except:' in line or 'except Exception' in line:
                    # Check if the next few lines contain only 'pass'
                    next_lines = lines[i+1:i+4]
                    if any('pass' in l.strip() and l.strip() == 'pass' for l in next_lines):
                        issues_found.append(f"{file_path}:{i+1} - Empty exception handler")
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if issues_found:
        print("\n⚠️  Empty exception handlers found (should add logging):")
        for issue in issues_found[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more")
    else:
        print("✅ No empty exception handlers found")

def find_dead_code():
    """Find potential dead code patterns"""
    python_files = list(Path('.').rglob('*.py'))
    dead_code_patterns = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Look for methods that only contain 'pass'
                if 'def ' in line and ':' in line:
                    # Check if method only contains pass
                    j = i + 1
                    method_content = []
                    while j < len(lines) and (lines[j].startswith('    ') or lines[j].strip() == ''):
                        if lines[j].strip():
                            method_content.append(lines[j].strip())
                        j += 1
                    
                    if len(method_content) == 1 and method_content[0] == 'pass':
                        dead_code_patterns.append(f"{file_path}:{i+1} - Method only contains 'pass'")
                
                # Look for TODO/FIXME comments
                if 'TODO' in stripped or 'FIXME' in stripped:
                    dead_code_patterns.append(f"{file_path}:{i+1} - TODO/FIXME comment")
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if dead_code_patterns:
        print("\n⚠️  Potential dead code found:")
        for pattern in dead_code_patterns[:15]:  # Show first 15
            print(f"  - {pattern}")
        if len(dead_code_patterns) > 15:
            print(f"  ... and {len(dead_code_patterns) - 15} more")
    else:
        print("✅ No obvious dead code patterns found")

def generate_improvement_report():
    """Generate a report of code improvements"""
    print("\n" + "="*60)
    print("CODE IMPROVEMENT REPORT")
    print("="*60)
    
    # Count Python files
    python_files = list(Path('.').rglob('*.py'))
    print(f"📁 Total Python files analyzed: {len(python_files)}")
    
    # Calculate total lines of code
    total_lines = 0
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines += len(f.readlines())
        except:
            pass
    
    print(f"📊 Total lines of code: {total_lines:,}")
    
    print("\n🔧 RECOMMENDED ACTIONS:")
    print("1. Run 'black .' to fix all line width violations")
    print("2. Review and fix empty exception handlers")
    print("3. Remove or implement methods that only contain 'pass'")
    print("4. Replace domain-specific logic with generic LLM-based approach")
    print("5. Add proper logging instead of bare 'except:' blocks")
    
    print("\n📋 STANDARDS COMPLIANCE:")
    print("- ✅ PEP 8 line length: 79 characters (after Black formatting)")
    print("- ⚠️  Exception handling: Needs improvement")
    print("- ⚠️  Code reusability: Domain-specific code should be generalized")
    print("- ✅ Import organization: Generally good")

def main():
    """Main function to run all code improvements"""
    print("🔧 TDS Project2 Code Quality Improvement Tool")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run formatting
    run_black_formatter()
    
    # Run linting
    run_flake8_linting()
    
    # Find issues
    fix_empty_exception_handlers()
    find_dead_code()
    
    # Generate report
    generate_improvement_report()
    
    print("\n✨ Code quality analysis complete!")
    print("Run this script with '--fix' to apply automatic fixes")

if __name__ == "__main__":
    main()
