"""
Version information for Luminance Stack Processor
"""

__version__ = "1.0.3"
__version_info__ = (1, 0, 3)

# Semantic versioning components
MAJOR = 1
MINOR = 0
PATCH = 3

# Build metadata
BUILD_DATE = "2025-01-20"
AUTHOR = "Sumit Chatterjee"
DESCRIPTION = "Professional HDR processing nodes using the Debevec Algorithm"

# ComfyUI compatibility
COMFYUI_MIN_VERSION = "0.1.0"
PYTHON_MIN_VERSION = "3.8"

def get_version_string():
    """Get formatted version string"""
    return f"v{__version__}"

def get_full_version_info():
    """Get complete version information"""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build_date": BUILD_DATE,
        "author": AUTHOR,
        "description": DESCRIPTION,
        "comfyui_min_version": COMFYUI_MIN_VERSION,
        "python_min_version": PYTHON_MIN_VERSION
    }
