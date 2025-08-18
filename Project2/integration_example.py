"""
Integration example showing how to use enhanced features in current system
"""

# Example 1: Using Enhanced LLM in existing workflows


def integrate_enhanced_llm():
    """Replace existing LLM with enhanced version"""
    from utils.enhanced_llm import create_enhanced_llm
    from config import USE_ENHANCED_LLM, TEMPERATURE
    
    if USE_ENHANCED_LLM:
        # Use enhanced LLM with fallback
        llm = create_enhanced_llm(temperature=TEMPERATURE)
        print("✅ Using enhanced LLM with multi-key fallback")
    else:
        # Use standard LLM
        from langchain_google_genai import ChatGoogleGenerativeAI
        from config import GEMINI_API_KEY, DEFAULT_GEMINI_MODEL
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_GEMINI_MODEL,
            temperature=TEMPERATURE,
            google_api_key=GEMINI_API_KEY
        )
        print("ℹ️  Using standard LLM")
    
    return llm


# Example 2: Using Enhanced Image Optimization in workflows
def integrate_image_optimization():
    """Add enhanced image optimization to existing visualization workflows"""
    from utils.image_optimization import plot_to_base64_enhanced
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Example plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("Example Plot with Enhanced Optimization")
    
    # Use enhanced optimization instead of standard base64 encoding
    optimized_image = plot_to_base64_enhanced(max_bytes=100000)
    plt.close()
    
    print(f"✅ Generated optimized image: {len(optimized_image)} characters")
    return optimized_image


# Example 3: Adding diagnostics endpoint to existing system
def add_diagnostics_to_main():
    """How to add diagnostics to main.py"""
    diagnostic_code = '''
# Add this to your main.py after existing imports:

@app.get("/health/detailed")
async def detailed_health():
    """Enhanced health check with system diagnostics"""
    import psutil
    import platform
    from datetime import datetime
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "disk_usage_percent": psutil.disk_usage('/').percent
        },
        "environment": {
            "gemini_keys_configured": len([k for k in [os.getenv(f"gemini_api_{i}") for i in range(1, 11)] if k]),
            "enhanced_llm_enabled": os.getenv("USE_ENHANCED_LLM", "true").lower() == "true"
        }
    }
'''
    print("📋 Add this code to your main.py for enhanced diagnostics:")
    print(diagnostic_code)


if __name__ == "__main__":
    print("🔧 Integration Examples for Enhanced Features\n")
    
    # Test enhanced LLM
    print("1. Testing Enhanced LLM Integration:")
    try:
        llm = integrate_enhanced_llm()
        print(f"   LLM Type: {type(llm).__name__}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n2. Testing Enhanced Image Optimization:")
    try:
        image_b64 = integrate_image_optimization()
        print(f"   ✅ Image generated successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n3. Diagnostics Integration:")
    add_diagnostics_to_main()
    
    print("\n🚀 Integration complete! Your system now has enhanced capabilities.")
