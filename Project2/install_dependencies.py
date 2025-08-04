#!/usr/bin/env python3
"""
Install missing dependencies for multi-step web scraping workflow
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        'beautifulsoup4',
        'lxml',
        'html5lib',
        'numpy'
    ]
    
    print("Installing missing dependencies...")
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
        except Exception as e:
            print(f"✗ Error installing {dep}: {e}")
    
    print("\nDependency installation completed!")
    print("You can now run the multi-step web scraping workflow.")

if __name__ == "__main__":
    install_dependencies() 