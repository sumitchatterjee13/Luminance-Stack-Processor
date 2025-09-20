"""
Luminance Stack Processor - ComfyUI Custom Nodes
Professional HDR processing nodes using the Debevec Algorithm

This module provides two custom nodes for ComfyUI:
1. Luminance Stack Processor (3 Stops) - For processing EV+2, EV+0, EV-2 exposures
2. Luminance Stack Processor (5 Stops) - For processing EV+4, EV+2, EV+0, EV-2, EV-4 exposures

Requirements:
- OpenCV (cv2) - NumPy and PyTorch are provided by ComfyUI

Installation:
1. Place this folder in your ComfyUI/custom_nodes/ directory
2. Install OpenCV: python_embeded\python.exe -m pip install opencv-python (for portable)
3. Restart ComfyUI

Author: Sumit Chatterjee
Version: 1.0.0
License: MIT
"""

import os
import sys

# Import version information
try:
    from .version import __version__, get_version_string, get_full_version_info
except ImportError:
    __version__ = "1.0.0"
    get_version_string = lambda: f"v{__version__}"
    get_full_version_info = lambda: {"version": __version__}

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Import the node classes
    from .luminance_stack_processor import (
        LuminanceStackProcessor3Stops,
        LuminanceStackProcessor5Stops,
        NODE_CLASS_MAPPINGS as NODES,
        NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_NAMES
    )
    
    # Export the mappings that ComfyUI expects
    NODE_CLASS_MAPPINGS = NODES
    NODE_DISPLAY_NAME_MAPPINGS = DISPLAY_NAMES
    
    print("✅ Luminance Stack Processor nodes loaded successfully!")
    print(f"   - Version: {get_version_string()}")
    print(f"   - Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
    
except ImportError as e:
    print(f"❌ Failed to import Luminance Stack Processor nodes: {e}")
    print("   Please ensure OpenCV is installed:")
    print("   For ComfyUI Portable: python_embeded\\python.exe -m pip install opencv-python")
    print("   For Standard ComfyUI: pip install opencv-python")
    
    # Provide empty mappings to prevent ComfyUI errors
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

except Exception as e:
    print(f"❌ Unexpected error loading Luminance Stack Processor nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Metadata for ComfyUI
__all__ = [
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS',
    '__version__'
]

# Version info for external access
__version__ = __version__
