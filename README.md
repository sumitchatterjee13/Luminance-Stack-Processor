# Luminance Stack Processor - ComfyUI Custom Nodes

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor/blob/main/LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor)

Professional HDR (High Dynamic Range) processing nodes for ComfyUI using the **Debevec Algorithm** for merging multiple exposure images into a single high dynamic range image.

**Version: 1.0.0** | **Release Date: 2025-01-20**

## 🎯 Features

- **Professional HDR Processing**: Implements the industry-standard Debevec Algorithm for HDR image creation
- **Two Processing Modes**:
  - **3-Stop Processor**: Merges EV+2, EV+0, EV-2 exposures
  - **5-Stop Processor**: Merges EV+4, EV+2, EV+0, EV-2, EV-4 exposures
- **16-bit Output**: Preserves maximum dynamic range in the final image
- **Automatic Camera Response Function**: Estimates and applies camera response curves
- **Fallback Safety**: Gracefully handles errors with fallback to middle exposure
- **ComfyUI Integration**: Seamless integration with ComfyUI workflows

## 📋 Requirements

- ComfyUI (includes NumPy and PyTorch)
- Python 3.8+
- OpenCV (cv2) >= 4.8.0

## 🚀 Installation

1. **Clone or Download** this repository to your ComfyUI custom nodes directory:
   ```bash
   # Method 1: Clone directly into ComfyUI custom nodes directory
   cd ComfyUI/custom_nodes/
   git clone https://github.com/sumitchatterjee13/Luminance-Stack-Processor.git luminance-stack-processor
   
   # Method 2: Download ZIP and extract to:
   # ComfyUI/custom_nodes/luminance-stack-processor/
   ```

2. **Install Dependencies**:
   
   **For ComfyUI Portable (Recommended)**:
   ```bash
   # Navigate to your ComfyUI portable directory and use embedded Python
   cd path/to/ComfyUI_windows_portable
   python_embeded\python.exe -m pip install opencv-python>=4.8.0
   ```
   
   **For Standard ComfyUI Installation**:
   ```bash
   # Use your system Python or ComfyUI's virtual environment
   pip install opencv-python>=4.8.0
   ```
   
   **Or install from requirements.txt**:
   ```bash
   # For portable version
   python_embeded\python.exe -m pip install -r requirements.txt
   
   # For standard installation
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI** to load the new nodes

## 🎨 Usage

### Luminance Stack Processor (3 Stops)

Perfect for standard HDR bracketing with 3 exposures:

**Inputs:**
- `ev_plus_2`: Overexposed image (+2 EV)
- `ev_0`: Normal exposure image (0 EV)
- `ev_minus_2`: Underexposed image (-2 EV)
- `exposure_step`: (Optional) EV step size (default: 2.0)

**Output:**
- `hdr_image`: Merged 16-bit HDR image

### Luminance Stack Processor (5 Stops)

For extended dynamic range with 5 exposures:

**Inputs:**
- `ev_plus_4`: Most overexposed image (+4 EV)
- `ev_plus_2`: Overexposed image (+2 EV)
- `ev_0`: Normal exposure image (0 EV)
- `ev_minus_2`: Underexposed image (-2 EV)
- `ev_minus_4`: Most underexposed image (-4 EV)
- `exposure_step`: (Optional) EV step size (default: 2.0)

**Output:**
- `hdr_image`: Merged 16-bit HDR image

## 🔬 How It Works

The nodes implement the **Debevec Algorithm** (Paul E. Debevec, Jitendra Malik, 1997) which:

1. **Analyzes Multiple Exposures**: Takes differently exposed images of the same scene
2. **Estimates Camera Response Function**: Determines how the camera sensor responds to light
3. **Recovers Scene Radiance**: Calculates the actual light values in the scene
4. **Merges to HDR**: Combines all exposures into a single high dynamic range image
5. **Tone Maps for Output**: Converts to 16-bit format for maximum compatibility

## 📸 Best Practices

### For Capturing Source Images:
- Use a **tripod** for perfect alignment
- Keep the **same white balance** across all exposures
- Use **manual focus** to prevent focus shifts
- Capture in **RAW format** when possible
- Use **exposure compensation** or **manual mode**

### EV (Exposure Value) Guidelines:
- **3-Stop**: +2, 0, -2 EV (4x range)
- **5-Stop**: +4, +2, 0, -2, -4 EV (16x range)
- Adjust `exposure_step` parameter if using different increments

## ⚙️ Technical Details

- **Algorithm**: Debevec & Malik HDR reconstruction
- **Input Format**: ComfyUI IMAGE tensors (0-1 float range)
- **Output Format**: 16-bit images for extended dynamic range
- **Processing**: OpenCV's `createCalibrateDebevec()` and `createMergeDebevec()`
- **Memory**: Efficient processing with automatic cleanup
- **Error Handling**: Graceful fallbacks with detailed logging

## 🔧 Troubleshooting

### Common Issues:

1. **"Module not found" error**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Restart ComfyUI completely

2. **Poor HDR results**:
   - Ensure input images are properly exposed (not all over/under)
   - Check that images are aligned (use tripod)
   - Verify EV differences match your capture method

3. **Memory issues**:
   - Process smaller images first
   - Ensure adequate RAM available
   - Close unnecessary applications

### Debug Information:
The nodes provide detailed logging. Check your ComfyUI console for processing information and error details.

## 🏗️ Development

### Project Structure:
```
luminance-stack-processor/
├── __init__.py                 # ComfyUI registration
├── luminance_stack_processor.py # Main node implementations
├── requirements.txt            # Dependencies
└── README.md                  # Documentation
```

### Contributing:
1. Fork the repository
2. Create a feature branch
3. Test with various HDR image sets
4. Submit a pull request

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and detailed release notes.

## 🔧 Development

### Version Management

This project uses [Semantic Versioning](https://semver.org/). To bump version:

```bash
# Install bumpversion
pip install bump2version

# Bump patch version (1.0.0 -> 1.0.1)
bump2version patch

# Bump minor version (1.0.0 -> 1.1.0) 
bump2version minor

# Bump major version (1.0.0 -> 2.0.0)
bump2version major
```

### Project Structure
```
luminance-stack-processor/
├── __init__.py                 # ComfyUI node registration & version info
├── luminance_stack_processor.py # Main HDR processing nodes
├── version.py                  # Centralized version management
├── pyproject.toml             # Modern Python packaging configuration
├── requirements.txt           # Runtime dependencies
├── README.md                  # Documentation
└── CHANGELOG.md              # Version history
```

## 📚 References

- **Debevec, P. E., & Malik, J.** (1997). Recovering high dynamic range radiance maps from photographs. *ACM SIGGRAPH Computer Graphics*, 31(Annual Conference Series), 367-378.
- **OpenCV HDR Documentation**: https://docs.opencv.org/4.x/d3/db7/tutorial_hdr_imaging.html
- **ComfyUI Custom Node Guidelines**: https://docs.comfy.org/custom-nodes/
- **Semantic Versioning**: https://semver.org/

## 📜 License

MIT License - Feel free to use and modify for your projects.

## 🤝 Support

For issues, questions, or contributions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/sumitchatterjee13/Luminance-Stack-Processor/issues)
- **GitHub Discussions**: [Ask questions or share workflows](https://github.com/sumitchatterjee13/Luminance-Stack-Processor/discussions)
- Check the troubleshooting section above
- Review ComfyUI console logs for detailed error information
- Check [CHANGELOG.md](CHANGELOG.md) for version-specific issues
- Ensure all dependencies are properly installed

### Contributing
We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Happy HDR Processing!** 📸✨ | **Version 1.0.0**
