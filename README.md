# Luminance Stack Processor - ComfyUI Custom Nodes

[![Version](https://img.shields.io/badge/version-1.0.1-blue.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor/blob/main/LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor)

Professional HDR (High Dynamic Range) processing nodes for ComfyUI using the **Debevec Algorithm** for merging multiple exposure images into a single high dynamic range image.

**Version: 1.0.1** | **Release Date: 2025-01-20**

## 🎯 Features

- **🆕 HDR Export Node**: Dedicated EXR export that preserves full HDR dynamic range
- **🚨 TRUE HDR Values Above 1.0**: Proper HDR data preservation in EXR files
- **Four Distinct HDR Algorithms**: Each produces different visual results
  - **Natural Blend**: EV0 appearance with enhanced dynamic range
  - **Mertens**: Adobe Lightroom style exposure fusion
  - **Debevec**: Classic HDR with maximum dynamic range
  - **Robertson**: Alternative robust HDR processing
- **Three Custom Nodes**:
  - **3-Stop Processor**: Merges EV+2, EV+0, EV-2 exposures
  - **5-Stop Processor**: Merges EV+4, EV+2, EV+0, EV-2, EV-4 exposures
  - **HDR Export**: Saves EXR files with preserved HDR data
- **ComfyUI-Style Interface**: Filename and path inputs like built-in save nodes
- **HDR Verification**: Automatic checking that HDR data is preserved in exports

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

### 🚨 **CRITICAL: Proper HDR Workflow**

1. **HDR Processing**: Use "Luminance Stack Processor" nodes
2. **🔥 HDR Export**: **ALWAYS** connect to "HDR Export to EXR" node  
3. **❌ Never use ComfyUI's built-in save nodes** for HDR - they normalize to 0-1!

### Luminance Stack Processor (3 Stops)

Perfect for standard HDR bracketing with 3 exposures:

**Inputs:**
- `ev_plus_2`: Overexposed image (+2 EV)
- `ev_0`: Normal exposure image (0 EV)  
- `ev_minus_2`: Underexposed image (-2 EV)
- `exposure_step`: (Optional) EV step size (default: 2.0)
- `hdr_algorithm`: Choose "natural_blend" (default), "mertens", "debevec", "robertson"

**Output:**
- `hdr_image`: HDR tensor with values potentially above 1.0

### HDR Export to EXR

**REQUIRED for true HDR preservation:**

**Inputs:**
- `hdr_image`: HDR tensor from processing nodes
- `filename_prefix`: Base filename (e.g., "My_HDR_Image")
- `output_path`: Directory path (empty = ComfyUI output folder)

**Output:**
- `filepath`: Path to saved EXR file with preserved HDR values

### Luminance Stack Processor (5 Stops)

For extended dynamic range with 5 exposures:

**Inputs:**
- `ev_plus_4`: Overexposed image (+4 EV)
- `ev_plus_2`: Overexposed image (+2 EV)
- `ev_0`: Normal exposure image (0 EV)
- `ev_minus_2`: Underexposed image (-2 EV)
- `ev_minus_4`: Underexposed image (-4 EV)
- `exposure_step`: (Optional) EV step size (default: 2.0)
- `hdr_algorithm`: Choose "natural_blend" (default), "mertens", "debevec", "robertson"

**Output:**
- `hdr_image`: HDR tensor with values potentially above 1.0

### 📋 **Complete HDR Workflow Example:**

1. **Load Images**: Load your bracketed exposures (3 or 5 images)
2. **Add Processing Node**: "Luminance Stack Processor (3/5 Stops)"
3. **Connect Exposures**: Connect each EV image to corresponding input
4. **Choose Algorithm**: Select HDR algorithm (Natural Blend recommended)
5. **Add Export Node**: "HDR Export to EXR" 
6. **Connect HDR Output**: From processor to export node
7. **Set Filename**: Enter desired filename prefix
8. **Set Path**: Output directory (or leave empty for default)
9. **Execute**: Get true HDR EXR file with values above 1.0!

## 🔬 How It Works

The nodes implement **multiple HDR algorithms** with **Natural Blend** as the default (preserves natural appearance):

1. **Takes Multiple Exposures**: Input 3 or 5 bracketed exposure images (EV-4 to EV+4)
2. **Selects HDR Algorithm**: Choose between Natural Blend, Mertens, Debevec, or Robertson
3. **Processes HDR Data**: Merges exposures preserving dynamic range above 1.0
4. **Outputs HDR Tensor**: 16-bit linear data ready for EXR export
5. **🚨 CRITICAL: Use HDR Export Node**: ComfyUI's built-in save nodes normalize to 0-1, use our HDR Export for true EXR

### 🎯 **HDR Algorithm Options:**

#### **Natural Blend (Default - Recommended)**  
- **HDR Range**: 1-8 (moderate HDR values for natural look)
- **Perfect for natural look**: Uses EV0 as base, adds dynamic range from other exposures
- **Gentle blending**: 30% blend strength with smooth luminance masks 
- **Preserved EV0 appearance**: Looks like original EV0 with enhanced dynamic range data

#### **Mertens Exposure Fusion**
- **HDR Range**: 1-12 (medium HDR values for balanced results)
- **Adobe Lightroom style**: Natural-looking results similar to professional HDR software
- **Enhanced contrast**: Produces more dynamic results than Natural Blend
- **Fastest processing**: No camera response function estimation required

#### **Debevec Algorithm (Classic HDR)**
- **HDR Range**: 1-100+ (maximum HDR values for extreme dynamic range)
- **Industry standard**: Original HDR reconstruction method from 1997
- **True scene radiance**: Recovers actual physical light values
- **Maximum dynamic range**: Best for scenes with extreme lighting (sun, reflections, etc.)

#### **Robertson Algorithm**
- **HDR Range**: 1-80+ (high HDR values with alternative processing)
- **Alternative to Debevec**: Different camera response function estimation
- **Robust processing**: Often produces cleaner results than Debevec

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

### Algorithm Selection Guide:
- **Use Natural Blend (Default)**: When you want to keep the exact look of your EV0 image with enhanced dynamic range
- **Use Mertens**: For natural HDR look similar to Adobe Lightroom (may add some contrast)
- **Use Debevec**: For scientific/research work requiring precise scene radiance values  
- **Use Robertson**: Alternative to Debevec with different mathematical approach

## ⚙️ Technical Details

- **Algorithms**: Natural Blend (default), Mertens Exposure Fusion, Debevec, Robertson
- **Input Format**: 8-bit ComfyUI IMAGE tensors (0-1 float range from 8-bit sources)
- **Output Format**: 16-bit linear colorspace images for extended dynamic range
- **Processing**: OpenCV's `createCalibrateDebevec()` and `createMergeDebevec()`
- **Memory**: Efficient processing with automatic cleanup
- **Error Handling**: Graceful fallbacks with detailed logging

## 🔧 Troubleshooting

### Common Issues:

1. **"Module not found" error**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Restart ComfyUI completely

2. **Color inversion or wrong colors**:
   - **Natural Blend algorithm** (default) - uses original EV0 colors, no issues expected
   - **Mertens algorithm** - produces natural colors, no issues expected
   - **Debevec algorithm** - now includes automatic Reinhard tone mapping for proper colors
   - **Robertson algorithm** - also includes tone mapping for natural appearance

3. **Image too bright or too dark**:
   - **Natural Blend** (default) - keeps exact brightness of your EV0 image
   - **All other algorithms** produce proper brightness levels automatically  
   - **Mertens** - natural brightness similar to Adobe Lightroom
   - **Debevec/Robertson** - include automatic Reinhard tone mapping for proper brightness

4. **Poor HDR results**:
   - Ensure input images are properly exposed (not all over/under)
   - Check that images are aligned (use tripod)  
   - Verify EV differences match your capture method
   - **For Natural Blend**: Make sure the EV0 image is your desired base appearance
   - Try different algorithms: Natural Blend preserves natural look, Mertens for more artistic results

5. **Memory issues**:
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

**Happy HDR Processing!** 📸✨ | **Version 1.0.1**
