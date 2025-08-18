"""
Enhanced image optimization utilities extracted from temp.py
"""

import base64
import matplotlib.pyplot as plt
from io import BytesIO

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def plot_to_base64_enhanced(max_bytes=100000, quality_steps=[100, 80, 60, 40, 30]):
    """
    Enhanced plot_to_base64 that ensures images stay under size limit
    
    Args:
        max_bytes: Maximum file size in bytes (default 100KB)
        quality_steps: DPI values to try in order
        
    Returns:
        str: Base64 encoded image data
    """
    # Try standard PNG with decreasing DPI
    for dpi in quality_steps:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        img_bytes = buf.getvalue()
        
        if len(img_bytes) <= max_bytes:
            return base64.b64encode(img_bytes).decode('ascii')
        buf.close()
    
    # If PIL available, try WEBP compression
    if PIL_AVAILABLE:
        try:
            for quality in [80, 60, 40]:
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
                buf.seek(0)
                im = Image.open(buf)
                
                out_buf = BytesIO()
                im.save(out_buf, format='WEBP', quality=quality, method=6)
                out_buf.seek(0)
                webp_bytes = out_buf.getvalue()
                
                if len(webp_bytes) <= max_bytes:
                    return base64.b64encode(webp_bytes).decode('ascii')
                
                out_buf.close()
                buf.close()
        except Exception:
            pass
    
    # Last resort: return smallest PNG even if over limit
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def optimize_existing_base64_image(base64_data, max_bytes=100000):
    """
    Optimize an existing base64 image to fit size constraints
    
    Args:
        base64_data: Base64 encoded image data
        max_bytes: Maximum file size in bytes
        
    Returns:
        str: Optimized base64 encoded image data
    """
    if not PIL_AVAILABLE:
        return base64_data
    
    try:
        # Decode the image
        img_bytes = base64.b64decode(base64_data)
        
        # If already under limit, return as-is
        if len(img_bytes) <= max_bytes:
            return base64_data
        
        # Try to optimize
        img = Image.open(BytesIO(img_bytes))
        
        # Try WEBP compression with different quality levels
        for quality in [80, 60, 40, 20]:
            buf = BytesIO()
            img.save(buf, format='WEBP', quality=quality, method=6)
            buf.seek(0)
            optimized_bytes = buf.getvalue()
            
            if len(optimized_bytes) <= max_bytes:
                return base64.b64encode(optimized_bytes).decode('ascii')
            buf.close()
        
        # Try resizing if still too large
        for scale in [0.8, 0.6, 0.4]:
            new_size = (int(img.width * scale), int(img.height * scale))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buf = BytesIO()
            resized.save(buf, format='WEBP', quality=60, method=6)
            buf.seek(0)
            resized_bytes = buf.getvalue()
            
            if len(resized_bytes) <= max_bytes:
                return base64.b64encode(resized_bytes).decode('ascii')
            buf.close()
        
        # If all else fails, return heavily compressed version
        buf = BytesIO()
        small_img = img.resize((200, 150), Image.Resampling.LANCZOS)
        small_img.save(buf, format='WEBP', quality=20, method=6)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('ascii')
        
    except Exception:
        # Return original if optimization fails
        return base64_data
