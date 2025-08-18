# Enhanced Data Analysis System - Usage Guide

## 🚀 Quick Start Options

### Option 1: Run Enhanced System Alongside Current System

```bash
# Keep your current system running on port 8000
# Run enhanced system on port 8001
./run_temp_docker.sh
```

**Benefits:**
- ✅ Compare both systems side-by-side
- ✅ Zero risk to existing setup
- ✅ Enhanced features: Multi-LLM fallback, better diagnostics, image optimization

**Test Commands:**
```bash
# Test current system (port 8000)
curl "http://localhost:8000/api/" -F "questions.txt=@test_wikipedia.txt"

# Test enhanced system (port 8001) 
curl "http://localhost:8001/api/" -F "questions.txt=@test_wikipedia.txt"

# Enhanced diagnostics
curl "http://localhost:8001/summary"
```

### Option 2: Integrate Enhanced Features into Current System

Add these environment variables to your `.env` file:
```bash
# Multiple Gemini API keys for fallback
gemini_api_1=your_primary_key_here
gemini_api_2=your_backup_key_here
gemini_api_3=your_third_key_here

# Enable enhanced features
USE_ENHANCED_LLM=true
LLM_MAX_RETRIES=3
LLM_TIMEOUT_SECONDS=240
```

## 🆚 **System Comparison**

| Feature | Current System | Enhanced System (temp.py) |
|---------|---------------|---------------------------|
| **Architecture** | Workflow-based orchestration | Agent-based with tools |
| **LLM Reliability** | Single API key | Multi-key fallback |
| **Error Handling** | Basic try-catch | Advanced retry mechanisms |
| **Image Optimization** | Basic base64 | Size-aware optimization |
| **Diagnostics** | Health endpoint | Comprehensive system checks |
| **Code Execution** | Direct execution | Sandboxed with injection |
| **Data Sources** | File upload + web scraping | Enhanced scraping + formats |

## 🔧 **Key Advantages of Enhanced System**

### 1. **Multi-LLM Fallback**
```python
# Automatically tries multiple API keys and models:
# gemini-2.5-pro → gemini-2.5-flash → gemini-2.0-flash
# If quota exceeded, switches to next key
```

### 2. **Smart Image Optimization**
```python
# Automatically keeps images under 100KB:
# 1. Reduces DPI progressively
# 2. Converts to WEBP if possible  
# 3. Adjusts quality settings
# 4. Resizes if necessary
```

### 3. **Comprehensive Diagnostics**
```bash
curl "http://localhost:8001/summary"
# Returns: System resources, network connectivity, 
# LLM key validation, package status, etc.
```

### 4. **Enhanced Error Recovery**
- Automatic retries with exponential backoff
- Graceful degradation when services fail
- Detailed error logging and reporting

## 📊 **Performance Comparison**

Run this test to compare both systems:

```bash
#!/bin/bash
echo "🧪 Testing both systems with same input..."

echo "📊 Current System (port 8000):"
time curl -s "http://localhost:8000/api/" \
  -F "questions.txt=@test_wikipedia.txt" | \
  jq '.workflow_type, .status'

echo "📊 Enhanced System (port 8001):"  
time curl -s "http://localhost:8001/api/" \
  -F "questions.txt=@test_wikipedia.txt" | \
  jq '.status'
```

## 🔄 **Migration Strategy**

### Phase 1: Parallel Testing (Recommended)
1. Run both systems simultaneously
2. Compare outputs and performance
3. Identify which features you want to adopt

### Phase 2: Selective Integration
1. Add enhanced LLM fallback to current system
2. Integrate image optimization utilities
3. Add enhanced diagnostics endpoints

### Phase 3: Full Migration (Optional)
1. Switch to agent-based architecture
2. Adopt sandboxed code execution
3. Use enhanced error handling throughout

## 🛠️ **Quick Integration Examples**

### Add Enhanced LLM to Current System
```python
# In your existing workflow files, replace:
# llm = ChatGoogleGenerativeAI(...)

# With:
from utils.enhanced_llm import create_enhanced_llm
llm = create_enhanced_llm(temperature=0.3)
```

### Add Image Optimization
```python
# Replace existing base64 generation with:
from utils.image_optimization import plot_to_base64_enhanced
image_data = plot_to_base64_enhanced(max_bytes=100000)
```

### Add Enhanced Diagnostics
```python
# Add to main.py:
@app.get("/health/detailed")
async def detailed_health():
    # [See integration_example.py for full code]
    return system_diagnostics
```

## 🚨 **Troubleshooting**

### Enhanced System Won't Start
```bash
# Check if port 8001 is available
netstat -tlnp | grep 8001

# Check Docker logs
docker logs temp-analysis-api
```

### API Key Issues
```bash
# Test API key validity
curl "http://localhost:8001/summary?full=true"
# Look for "llm_keys_models" section
```

### Performance Issues
```bash
# Monitor system resources
curl "http://localhost:8001/summary" | jq '.checks.system'
```

## 📈 **Next Steps**

1. **Start with Option 1** - Run both systems in parallel
2. **Test with your datasets** - Compare outputs and performance
3. **Identify preferred features** - Choose what to integrate
4. **Gradual migration** - Adopt features one by one
5. **Full optimization** - Fine-tune for your specific use cases

## 🎯 **Recommendation**

For your current working setup, I recommend **Option 1** - running both systems in parallel. This gives you:

- ✅ **Zero risk** to your current working system
- ✅ **Side-by-side comparison** of capabilities
- ✅ **Learning opportunity** to see different architectural approaches
- ✅ **Flexibility** to choose the best features from each

Start with: `./run_temp_docker.sh` and test with your existing question files!
