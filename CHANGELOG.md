# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added  
- **NEW: EV0-Based Blending Algorithm** - Now default! Preserves exact EV0 appearance with enhanced dynamic range
- **Smart luminance masking** - Seamless highlight and shadow detail recovery

### Changed
- **Optimized for 8-bit input workflow** - Removed unnecessary 8-bit conversion since inputs are already 8-bit
- **Enhanced 16-bit linear output** - Improved scaling algorithm for better 16-bit range utilization
- **Changed default algorithm** - EV0-Based is now default instead of Mertens for natural results

### Fixed
- **CRITICAL: Fixed contrast issues** - EV0-Based algorithm maintains original image contrast
- **CRITICAL: Fixed color inversion issues** - Added proper BGR↔RGB conversion for correct color handling
- **Fixed brightness problems** - Improved scaling and algorithm selection  
- **Added Mertens Exposure Fusion algorithm** - Alternative for Lightroom-style results
- **Added Robertson algorithm** - Alternative HDR method for different use cases
- **Enhanced algorithm selection** - Users can choose between EV0-Based (default), Mertens, Debevec, and Robertson
- **CRITICAL: Fixed Debevec algorithm issues** - Added Reinhard tone mapping to fix brightness/color inversion
- **Improved Debevec output** - Now produces natural results similar to Mertens
- **Enhanced per-algorithm scaling** - Different scaling strategies for different algorithms

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
