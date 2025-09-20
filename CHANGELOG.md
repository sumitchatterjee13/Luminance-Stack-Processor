# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

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
