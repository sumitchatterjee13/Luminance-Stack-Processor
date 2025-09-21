# Luminance Stack Processor - ComfyUI Custom Nodes

[![Version](https://img.shields.io/badge/version-1.0.5-blue.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor/blob/main/LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/sumitchatterjee13/Luminance-Stack-Processor)

Professional HDR (High Dynamic Range) processing nodes for ComfyUI featuring our **Radiance Fusion Algorithm** - a custom HDR processing method that delivers enhanced results through Nuke-inspired mathematical operations.

**Version: 1.0.5** | **Release Date: 2025-01-21**

## 🎯 Features

- **🚀 RADIANCE FUSION ALGORITHM**: Our custom HDR algorithm developed in-house
  - **Nuke-Inspired Mathematics**: Based on professional VFX pipeline operations (plus/average)
  - **Enhanced HDR Preservation**: Maintains excellent dynamic range with natural appearance  
  - **Improved Results**: Provides quality improvements over traditional methods
  - **Professional VFX Quality**: Suitable for film, TV, and visual effects work
- **🆕 TRUE 32-bit EXR Export**: Professional bit-depth control with imageio integration
- **🚨 TRUE HDR Values Above 1.0**: Proper HDR data preservation without normalization
- **Legacy HDR Algorithms** *(Work in Progress)*:
  - **Natural Blend**: EV0 appearance preservation *(under refinement)*
  - **Mertens**: Exposure fusion method *(being optimized)*
  - **Debevec**: Traditional HDR recovery *(legacy support)*
  - **Robertson**: Alternative HDR method *(legacy support)*
- **Three Custom Nodes**:
  - **3-Stop Processor**: Merges EV+2, EV+0, EV-2 exposures
  - **5-Stop Processor**: Merges EV+4, EV+2, EV+0, EV-2, EV-4 exposures
  - **HDR Export**: Saves EXR files with preserved HDR data
- **ComfyUI-Style Interface**: Filename and path inputs like built-in save nodes
- **HDR Verification**: Automatic checking that HDR data is preserved in exports

## 🎨 Visual Results

### HDR Processing Comparison

See the dramatic difference our Radiance Fusion Algorithm makes in preserving detail across exposure ranges:

#### Indoor Lighting Comparison
![Bulb Comparison](images/Bulb.jpg)
*Multiple exposure levels of the same scene showing enhanced dynamic range preservation*

#### Outdoor Scene Comparison  
![Outdoor Comparison](images/outdoor1.jpg)
*Natural outdoor scene demonstrating HDR processing across different exposure values*

#### Highlight Clamping Reduction
![Highlight Clamping Comparison](images/Reduce%20highlight%20clamping.jpg)
*Demonstration of reduced highlight clamping - preserving bright details that would normally be lost*

These comparison strips demonstrate how our HDR processing maintains detail in both highlights and shadows that would be lost in single-exposure captures.

## 📋 Requirements

- **Python 3.11+** *(Required for optimal performance)*
- ComfyUI (includes NumPy and PyTorch)
- OpenCV (cv2) >= 4.8.0
- imageio >= 2.31.0 *(for professional 32-bit EXR export)*

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

## 📁 ComfyUI Workflow

A complete example workflow for ComfyUI is provided in the `/workflow` directory. This demonstrates a good setup for HDR processing using our Radiance Fusion algorithm.

### Example Workflow Setup
![ComfyUI Workflow](images/Comfyui%20workflow.jpg)
*Complete ComfyUI workflow showing proper HDR processing setup with our custom nodes*

## 🎨 AI-Generated HDR from Single Images

We've developed a complete workflow for creating HDR images from a single source image using **FLUX.1 Kontext exposure control LoRAs**:

### 🚀 **The Complete Pipeline:**

1. **Start with any image** - Generated from any model or captured photo
2. **Generate exposure variants** - Use our trained LoRAs to create EV-4, EV-2, EV0, EV+2, EV+4 versions
3. **Process with Radiance Fusion** - Feed the synthetic exposures into our HDR nodes
4. **Get professional HDR** - Output true 32-bit HDR images with enhanced dynamic range

### 📦 **Exposure Control LoRAs:**

Our custom-trained LoRAs for **FLUX.1 Kontext (dev)** can generate different exposure levels from a single image:

- **Available LoRAs**: `ev-4`, `ev-2`, `ev+2`, `ev+4` (with corresponding trigger words)
- **Easy to use**: Simply add the trigger word and adjust LoRA weight  
- **High quality**: Trained specifically for exposure control
- **Download**: [Flux-Kontext-exposure-control-LoRAs](https://huggingface.co/Sumitc13/Flux-Kontext-exposure-control-LoRAs)

### 💡 **Workflow Benefits:**

- ✅ **Single image input** - No need for multiple captures or bracketing
- ✅ **AI-generated exposures** - Perfect synthetic exposure variants
- ✅ **Professional HDR output** - True HDR with our Radiance Fusion processing
- ✅ **Complete automation** - Generate and process in one workflow
- ✅ **Creative freedom** - Work with any generated or captured image

## 🎨 Usage

### 🚨 **CRITICAL: Proper HDR Workflow**

1. **HDR Processing**: Use "Luminance Stack Processor" nodes
2. **🔥 HDR Export**: **ALWAYS** connect to "HDR Export to EXR" node  
3. **❌ Never use ComfyUI's built-in save nodes** for HDR - they normalize to 0-1!

### ⚠️ **Important Notes for VFX Artists:**

**Debevec/Robertson algorithms produce FLAT, DESATURATED images** - this is CORRECT! The flat appearance means:
- ✅ Perfect VFX flat log profile (18% gray scaling)
- ✅ Raw linear radiance data preserved  
- ✅ Wide dynamic range maintained (up to 2000+ values)
- ✅ Professional color pipeline ready
- ✅ No color inversion issues (fixed RGB↔BGR handling)
- ✅ No tone mapping destroying VFX data

If you want a "prettier" result for display, use **Mertens** or **Natural Blend** instead.

### Luminance Stack Processor (3 Stops)

Perfect for standard HDR bracketing with 3 exposures:

**Inputs:**
- `ev_plus_2`: Overexposed image (+2 EV)
- `ev_0`: Normal exposure image (0 EV)  
- `ev_minus_2`: Underexposed image (-2 EV)
- `exposure_step`: (Optional) EV step size (default: 2.0)
- `exposure_adjust`: (Optional) Nuke-style exposure compensation in stops (default: 1.0)
- `hdr_algorithm`: Choose **"radiance_fusion"** (default - *our custom algorithm*), "natural_blend", "mertens", "debevec", "robertson"

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
- `exposure_adjust`: (Optional) Nuke-style exposure compensation in stops (default: 1.0)
- `hdr_algorithm`: Choose **"radiance_fusion"** (default - *our custom algorithm*), "natural_blend", "mertens", "debevec", "robertson"

**Output:**
- `hdr_image`: HDR tensor with values potentially above 1.0

### 📋 **Complete HDR Workflow Example:**

1. **Load Images**: Load your bracketed exposures (3 or 5 images)
2. **Add Processing Node**: "Luminance Stack Processor (3/5 Stops)"
3. **Connect Exposures**: Connect each EV image to corresponding input
4. **Choose Algorithm**: Select HDR algorithm (**Radiance Fusion recommended - our custom algorithm**)
5. **Add Export Node**: "HDR Export to EXR" 
6. **Connect HDR Output**: From processor to export node
7. **Set Filename**: Enter desired filename prefix
8. **Set Path**: Output directory (or leave empty for default)
9. **Execute**: Get true HDR EXR file with values above 1.0!

## 🔬 How It Works

The nodes feature our **Radiance Fusion Algorithm** as the default, plus traditional algorithms for compatibility:

1. **Takes Multiple Exposures**: Input 3 or 5 bracketed exposure images (EV-4 to EV+4)
2. **Selects HDR Algorithm**: Choose **Radiance Fusion** (our custom algorithm) or traditional methods
3. **Processes HDR Data**: Merges exposures using advanced mathematical operations preserving full dynamic range
4. **Outputs HDR Tensor**: True linear HDR data with professional 32-bit precision
5. **🚨 CRITICAL: Use HDR Export Node**: Exports professional 32-bit EXR files with preserved HDR values

### 🎯 **HDR Algorithm Options:**

#### **🚀 Radiance Fusion (Default - Our Custom Algorithm)**  
- **📈 HDR Range**: Wide dynamic range with good preservation
- **🧮 Advanced Mathematics**: Nuke-inspired plus/average operations for enhanced results
- **🎬 Professional Quality**: VFX-grade HDR processing with natural appearance
- **⚡ Good Performance**: Well-balanced dynamic range and visual appeal
- **🔬 In-House Development**: Our custom algorithm with improvements over traditional methods
- **💎 Quality Focus**: Enhanced compared to standard HDR techniques

---

### **Legacy Algorithms (Work in Progress):**

#### **Natural Blend** *(Under Refinement)*
- **Status**: Being optimized for better performance
- **HDR Range**: 1-8 (moderate HDR values)
- **Purpose**: EV0 appearance preservation with enhanced range

#### **Mertens Exposure Fusion** *(Being Optimized)*
- **Status**: Performance improvements in development
- **HDR Range**: 1-12 (medium HDR values)
- **Purpose**: Traditional exposure fusion method

#### **Debevec Algorithm** *(Legacy Support)*
- **Status**: Maintained for compatibility
- **HDR Range**: Raw linear radiance (VFX standard)
- **Purpose**: Traditional HDR recovery (1997 method)

#### **Robertson Algorithm** *(Legacy Support)*
- **Status**: Maintained for compatibility
- **HDR Range**: Raw linear radiance (VFX standard)
- **Purpose**: Alternative traditional HDR method

## 📸 Best Practices

### For Capturing Source Images (Traditional Method):
- Use a **tripod** for perfect alignment
- Keep the **same white balance** across all exposures
- Use **manual focus** to prevent focus shifts
- Capture in **RAW format** when possible
- Use **exposure compensation** or **manual mode**

### For AI-Generated Exposures (Modern Workflow):
- Start with **any single image** (generated or captured)
- Use our **FLUX.1 Kontext LoRAs** to create exposure variants
- Download LoRAs: [Flux-Kontext-exposure-control-LoRAs](https://huggingface.co/Sumitc13/Flux-Kontext-exposure-control-LoRAs)
- Generate **ev-4, ev-2, ev+2, ev+4** variants using trigger words
- **Perfect alignment** - AI ensures consistent composition
- **No bracketing required** - Generate all exposures from one source

### EV (Exposure Value) Guidelines:

#### Traditional Capture:
- **3-Stop**: +2, 0, -2 EV (4x range)
- **5-Stop**: +4, +2, 0, -2, -4 EV (16x range)
- Adjust `exposure_step` parameter if using different increments

#### AI-Generated (Using Our LoRAs):
- **Base image**: Your source image (acts as EV 0)
- **Generate variants**: Use `ev-4`, `ev-2`, `ev+2`, `ev+4` LoRAs with trigger words
- **Perfect range**: Covers full 8-stop dynamic range
- **Consistent quality**: AI ensures proper exposure relationships

### Algorithm Selection Guide:

#### 🎬 **For VFX/Post-Production:**
- **🚀 Radiance Fusion** (Recommended): Our custom algorithm - good for professional work
- **🔬 Debevec** *(Legacy)*: Traditional raw radiance method
- **⚙️ Robertson** *(Legacy)*: Alternative traditional method

#### 🎨 **For Photography/Display:**
- **🚀 Radiance Fusion** (Recommended): Our custom algorithm - enhanced quality
- **🌟 Natural Blend** *(Work in Progress)*: Being optimized for better results
- **💫 Mertens** *(Work in Progress)*: Traditional method under improvement

#### 💡 **Important:** 
- **VFX Algorithms** (Debevec/Robertson): Flat, desaturated appearance - **this is professional standard!**
- **Display Algorithms** (Natural Blend/Mertens): Enhanced, natural-looking, ready for viewing

## ⚙️ Technical Details

- **Primary Algorithm**: **Radiance Fusion** - Our custom HDR processing method
- **Legacy Algorithms**: Natural Blend, Mertens, Debevec, Robertson *(work in progress)*
- **Input Format**: 8-bit ComfyUI IMAGE tensors (standard ComfyUI format)
- **Output Format**: True 32-bit linear HDR with unlimited dynamic range
- **Processing**: Advanced mathematical operations + OpenCV integration
- **EXR Export**: Professional 32-bit precision via imageio library
- **Memory**: Optimized processing with intelligent resource management
- **Error Handling**: Comprehensive fallbacks with detailed diagnostic logging

## 🔧 Troubleshooting

### Common Issues:

1. **"Module not found" error**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Restart ComfyUI completely

2. **Color accuracy**:
   - **Radiance Fusion algorithm** (default) - Perfect color accuracy with advanced processing
   - **Legacy algorithms** *(work in progress)* - Being optimized for improved color handling

3. **Brightness optimization**:
   - **Radiance Fusion** (default) - Perfectly balanced brightness with exposure compensation control
   - **Legacy algorithms** *(work in progress)* - Being refined for optimal brightness handling

4. **Poor HDR results**:
   - Ensure input images are properly exposed (not all over/under)
   - Check that images are aligned (use tripod)  
   - Verify EV differences match your capture method
   - **Recommended**: Use Radiance Fusion (our custom algorithm) for good results
   - **Legacy algorithms**: Available for compatibility but under active improvement

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
- **Our FLUX.1 Kontext Exposure LoRAs**: [Flux-Kontext-exposure-control-LoRAs](https://huggingface.co/Sumitc13/Flux-Kontext-exposure-control-LoRAs)
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

**Happy HDR Processing with Radiance Fusion!** 🚀✨ | **Version 1.0.5** | **Featuring Our Custom Algorithm**
