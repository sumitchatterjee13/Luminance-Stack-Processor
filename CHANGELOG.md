# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.8] - 2025-10-02

### Changed
- **⚙️ Default exposure_adjust changed to 0.0** - No additional exposure adjustment by default
  - Previous default: 1.0 (+1 stop brighter)
  - New default: 0.0 (use automatic brightness compensation only)
  - Applies to both 3-stop and 5-stop processors
  - Users can still adjust manually if needed (+/- 5 stops range)

## [1.1.7] - 2025-10-02

### Changed
- **🎨 Improved highlight/shadow mask generation** - Smoother, more gradual transitions
  - **Wider threshold ranges**: Highlight masks start earlier (0.6 vs 0.7), cover more area
  - **21×21 Gaussian smoothing**: Applied to all masks with sigma=7.0 for very gradual feathering
  - **No image blurring**: Smooths the blending masks, not the image itself
  - **Eliminates sharp cutoffs**: Gradual transitions prevent visible blend boundaries
  - **Consistent across highlights and shadows**: Both use improved mask generation

### Removed
- **Removed HDR region post-processing blur** - Caused unwanted image blurring

### Technical Details
- Masks now use both wider Hermite interpolation ranges AND Gaussian smoothing
- 21×21 kernel with sigma=7.0 creates very soft feathering (14 pixel transition zone)
- Affects blending behavior, not final image sharpness

## [1.1.6] - 2025-10-02

### Changed
- **🎨 HDR highlight post-processing** - Direct smoothing of HDR regions for grain reduction
  - **Selective Gaussian blur**: Only affects pixels with values > 1.0 (HDR highlights)
  - **9×9 kernel with sigma=2.0**: Effective grain reduction without over-blurring
  - **Smooth transition mask**: 15×15 Gaussian feathering prevents hard edges between smoothed/unsmoothed regions
  - **Preserves displayable range**: 0-1 values remain untouched (sharp EV0 base preserved)
  - **Fixes**: Directly smooths the final HDR values that appear grainy when exposed down in Nuke

### Technical Details
- Post-processing applied after detail injection (more effective than pre-filtering sources)
- Mask-based selective blending prevents affecting non-HDR regions
- Addresses grain created during injection process, not just source noise

## [1.1.5] - 2025-10-02

### Added
- **🎨 Bilateral filtering for grain reduction** - Smoother highlight details in Nuke/Resolve
  - **Highlights (EV-2)**: 5x5 bilateral filter (sigmaColor=0.1, sigmaSpace=5)
  - **Extreme highlights (EV-4)**: 7x7 bilateral filter (sigmaColor=0.15, sigmaSpace=7) - stronger smoothing
  - **Shadows (EV+2/EV+4)**: 3x3 bilateral filter (sigmaColor=0.05, sigmaSpace=3) - light smoothing
  - **Edge-preserving smoothing**: Reduces AI-generated grain without losing important detail
  - **Fixes**: Highlights no longer appear grainy/sharp when exposed down in Nuke

### Technical Details
- Bilateral filtering applied to detail sources before injection
- Preserves edges and important detail while removing noise
- Stronger filtering for more underexposed images (more grain)
- Maintains HDR range and color accuracy

## [1.1.4] - 2025-10-02

### Fixed
- **🔆 Improved brightness compensation algorithm** - Better histogram distribution
  - **Now uses median-based scaling** for very dark images (median < 0.05)
  - **Uses mean-based scaling** for normal brightness images
  - **More permissive adjustment range**: 0.3x to 8.0x (was 0.5x to 4.0x)
  - **Better handling of shadow-heavy images** - prevents histogram crush to left
  - **Fixes**: Histogram no longer compressed to shadows, better tonal distribution

## [1.1.3] - 2025-10-02

### Changed
- **🎨 Detail Injection is now the DEFAULT algorithm** - Best results for AI-generated images
- **Simplified output: Linear HDR only** - Removed sRGB display option
  - Always outputs true linear HDR (perfect for EXR export)
  - Linear data is correct format for professional color grading
  - Preview may look contrasty in ComfyUI (this is expected and correct)
- **Algorithm order updated**: detail_injection listed first in dropdown menu

### Removed
- **Removed `output_colorspace` parameter** - Streamlined interface, always linear HDR output

## [1.1.2] - 2025-10-02

### Added
- **🎨 Output Colorspace Selection** - New parameter for all HDR processors
  - **`srgb_display` (NEW DEFAULT)**: Tone-mapped output with Reinhard operator for natural preview in ComfyUI
  - **`linear_hdr`**: True linear HDR output for EXR export (may look contrasty in preview)
  - **Applies to all algorithms** - Consistent colorspace handling across radiance_fusion, detail_injection, etc.
  - **Smart tone mapping**: Reinhard operator compresses HDR values >1.0 into displayable range
  - **Preserves workflow flexibility**: Use sRGB for preview, linear for final EXR export

### Changed
- **Default output is now `srgb_display`** - Better out-of-box experience for ComfyUI users
- **Linear output properly documented** - Clear warnings that linear HDR looks contrasty in preview

## [1.1.1] - 2025-10-02

### Fixed
- **🚨 CRITICAL: Fixed inverted exposure adjustment formula** - Exposure adjustment was backwards!
  - **Bug**: `+1.0` stop was darkening (0.5x) instead of brightening (2.0x)
  - **Fix**: Removed negative sign from formula: `2^exposure_adjust` (was `2^(-exposure_adjust)`)
  - **Impact**: All exposure adjustments now work correctly (positive = brighter, negative = darker)
- **🔆 Fixed Detail Injection producing dark images** - Added automatic brightness compensation
  - **Issue**: Output had mean 0.13 instead of target 0.18 (18% middle gray standard)
  - **Solution**: Automatic brightness scaling targets 0.18 mean for proper exposure
  - **Uses percentile-based mean** (middle 80%) for robustness against outliers
  - **Clamped to 0.5x-4.0x range** to prevent over-correction
  - **Preserves HDR values proportionally** while normalizing overall brightness

## [1.1.0] - 2025-10-02

### Added
- **🎨 Detail Injection Algorithm** - Revolutionary AI-aware HDR processing for AI-generated exposure stacks
  - **Automatic gamma analysis** - Detects sRGB gamma encoding (2.2) in AI-generated images
  - **Proper sRGB to linear conversion** - Accurate color space transformation for HDR processing
  - **Intelligent highlight detail injection**:
    - Maps EV-2 (underexposed) detail into 1.0-2.0 HDR range for moderate highlights
    - Maps EV-4 (very underexposed) detail into 2.0-4.0 HDR range for extreme highlights
  - **Intelligent shadow detail injection**:
    - Recovers shadow detail from EV+2 (overexposed) for moderate shadows
    - Recovers deep shadow detail from EV+4 (very overexposed) for extreme shadows
  - **Hermite interpolation** - Smooth S-curve blending prevents harsh transitions
  - **Color preservation** - Maintains hue and color ratios using luminance-based scaling
  - **EV0 as base** - Preserves natural appearance while extending dynamic range
  - **Perfect for AI workflows** - Handles non-photometric AI-generated exposure stacks
  - **Comprehensive logging** - Detailed step-by-step processing information
  - **Supports both 3-stop and 5-stop processing** - Works with any exposure stack size

### Added  
- **🆕 Latent Stack Processor (5 Stops)** - New node for averaging latent representations from 5 different exposures
- **👑 Ultimate Quality-Aware Blending (NEW DEFAULT!)** - Merges Laplacian pyramid + enhanced quality metrics!
  - **Combines the best of both technologies** - multi-scale decomposition + sophisticated quality analysis
  - **4-level Laplacian pyramid decomposition** - separates frequency bands for optimal blending
  - **Adaptive quality power per frequency level**:
    - Level 0 (finest details): Power = 3.8x (very selective for edges like tree leaves)
    - Level 1 (fine details): Power = 2.8x (highly selective)
    - Level 2 (medium details): Power = 2.1x (selective)
    - Level 3+ (coarse/smooth): Power = 1.7x (balanced for smooth areas)
  - **Enhanced quality metrics with edge emphasis**:
    - Contrast: 60% weight (edges/details prioritized)
    - Saturation: 25% weight (color richness)
    - Exposedness: 15% weight (well-exposed regions)
  - **Completely eliminates ALL ghosting artifacts** - tree leaves perfectly sharp!
  - **Seamless smooth area blending** - sky remains perfectly smooth
  - **Professional production-ready results** matching top HDR software!
- **⭐ Variance-Adaptive Blending** - Alternative approach with spatial smoothing
  - Analyzes local variance across latents to detect problem areas
  - 7x7 spatial smoothing prevents checkerboard patterns
  - Adaptive per-pixel weighting based on variance analysis
- **🎯 Six Professional Blend Modes** - Complete arsenal of blending strategies:
  - `quality_aware`: Multi-scale pyramid + enhanced quality (NEW DEFAULT!) 👑
  - `variance_adaptive`: Spatial smoothing with variance analysis
  - `weighted_center`: Simple center-biased weighting
  - `strong_center`: Maximum noise reduction approach
  - `median_blend`: Outlier rejection for robustness
  - `simple_average`: Equal weights for maximum range
- **Detail Preservation Control** - New parameter (0.0-1.0) controls quality distinction sharpness (quality_aware) or artifact reduction strength (variance_adaptive)
- **Center Bias Control** - Adjustable weighting (0.0-0.8) to balance noise vs dynamic range
- **Fast latent processing** - Works with latent space instead of image space
- **Direct latent manipulation** - Processes before VAE decode for efficiency
- **NEW: Natural Blend Algorithm** - Now default! Preserves exact EV0 appearance with enhanced dynamic range
- **Smart luminance masking** - Seamless highlight and shadow detail recovery

### Changed
- **Optimized for 8-bit input workflow** - Removed unnecessary 8-bit conversion since inputs are already 8-bit
- **Enhanced 16-bit linear output** - Improved scaling algorithm for better 16-bit range utilization
- **Changed default algorithm** - Natural Blend is now default instead of Mertens for natural results

### Added
- **🆕 HDR Export Node** - Dedicated EXR export node that preserves full HDR dynamic range
- **True HDR file output** - EXR files maintain values above 1.0 without normalization
- **ComfyUI-style interface** - Filename prefix and output path inputs like built-in save nodes
- **HDR verification** - Automatic verification that HDR data is preserved in exported files

### Fixed
- **🚨 CRITICAL: Fixed Debevec color inversion** - Proper RGB↔BGR conversion fixes inverted colors completely
- **🚨 CRITICAL: Perfect VFX flat log profile** - Research-based implementation with 18% gray scaling
- **Fixed sRGB to Linear conversion** - Proper gamma correction applied at correct stage for camera response
- **VFX Pipeline compliance** - Debevec now outputs professional flat, desaturated appearance (CORRECT)  
- **Removed tone mapping completely** - Raw linear radiance preserved for professional color pipeline
- **Created proper HDR workflow** - Use HDR Export Node instead of built-in save nodes for EXR
- **🆕 CLEAN FILENAME INTERFACE** - No more automatic timestamps! Professional filename control like standard ComfyUI nodes

## [1.0.1] - 2025-01-20

### Fixed
- **CRITICAL: Fixed HDR processing for proper linear colorspace output**
- **Fixed 8-bit input requirement for OpenCV HDR functions** (createMergeDebevec expects 8-bit input)
- **Added proper gamma correction preprocessing** (sRGB to linear conversion before HDR processing)
- **Improved HDR data preservation** - no longer clips values to 0-1 range, preserves HDR information
- **Enhanced error handling and logging** for better debugging of HDR processing issues
- **Fixed color artifacts** in HDR output by proper colorspace handling
- **Added linear colorspace validation** to ensure proper scene radiance recovery

### Technical Improvements
- Added proper sRGB to linear gamma correction before HDR processing
- Implemented 99.9th percentile normalization to handle extreme HDR values
- Enhanced logging with detailed HDR processing information
- Added validation for image format and HDR output quality
- Improved fallback handling with proper linear space conversion

## [1.0.0] - 2025-01-20

### Added
- **Luminance Stack Processor (3 Stops)** - Professional HDR processing node for merging EV+2, EV+0, EV-2 exposures
- **Luminance Stack Processor (5 Stops)** - Professional HDR processing node for merging EV+4, EV+2, EV+0, EV-2, EV-4 exposures
- **Debevec Algorithm Implementation** - Industry-standard HDR reconstruction using Paul Debevec's method
- **16-bit Output Support** - Maximum dynamic range preservation in final images
- **Automatic Camera Response Function Estimation** - Intelligent sensor response curve calculation
- **Configurable Exposure Steps** - Adjustable EV step sizes for different capture methods
- **Error Handling & Fallbacks** - Graceful degradation with middle exposure fallback
- **Professional Documentation** - Complete README, installation guide, and usage examples
- **Modern Python Packaging** - Full pyproject.toml configuration with semantic versioning
- **Comprehensive Testing** - Unit tests for core HDR processing functionality
- **ComfyUI Integration** - Seamless tensor format handling and node registration

### Technical Features
- OpenCV-based HDR processing pipeline
- Automatic image alignment preparation 
- Memory-efficient processing with cleanup
- Detailed logging for debugging
- Cross-platform compatibility (Windows, Linux, macOS)

### Dependencies
- OpenCV >= 4.8.0 (only external dependency)
- NumPy and PyTorch provided by ComfyUI
- Python 3.8+ support

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backward compatible manner  
- **PATCH** version when you make backward compatible bug fixes

### Version History Guidelines

- **1.x.x** - Stable production releases with Debevec HDR algorithm
- **0.x.x** - Pre-release/beta versions (not used in this project)
- **x.x.0** - Major feature releases
- **x.x.1+** - Bug fixes and minor improvements
