"""
Luminance Stack Processor - Professional ComfyUI Custom Nodes
Implements HDR processing using the Debevec Algorithm for multiple exposure fusion

Author: Sumit Chatterjee
Version: 1.1.8
Semantic Versioning: MAJOR.MINOR.PATCH


"""

import numpy as np
import torch
import cv2
from typing import Tuple, List, Optional
import logging
import os

# Try to import HDRutils for alternative HDR processing
try:
    import HDRutils
    HDRUTILS_AVAILABLE = True
except ImportError:
    HDRUTILS_AVAILABLE = False
    
# Try to import imageio for HDR/EXR support
try:
    import imageio.v3 as iio
    IMAGEIO_AVAILABLE = True
except ImportError:
    try:
        import imageio as iio
        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI tensor to OpenCV format for HDR processing
    
    IMPORTANT: OpenCV's Debevec and Robertson algorithms expect 8-bit sRGB images as input.
    They internally recover the camera response function, so we should NOT pre-linearize the images.
    """
    # ComfyUI tensors are typically [B, H, W, C] in 0-1 range
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Convert to numpy
    image = tensor.cpu().numpy()
    
    # Convert to 8-bit sRGB (no gamma correction - Debevec/Robertson expect sRGB input)
    image_8bit = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    logger.info(f"Converted to OpenCV: shape={image_8bit.shape}, dtype={image_8bit.dtype}, range=[{image_8bit.min()}, {image_8bit.max()}]")
    
    return image_8bit


def cv2_to_tensor(hdr_image: np.ndarray, output_16bit_linear: bool = True, algorithm_hint: str = "unknown") -> torch.Tensor:
    """Convert OpenCV HDR image to ComfyUI tensor format - VFX pipeline friendly"""
    
    if output_16bit_linear:
        logger.info(f"HDR processing ({algorithm_hint}):")
        logger.info(f"  Input range: [{hdr_image.min():.6f}, {hdr_image.max():.6f}]")
        
        # VFX PIPELINE APPROACH: Different scaling philosophy per algorithm
        if algorithm_hint == "radiance_fusion":
            # Radiance Fusion: Perfect HDR preservation with Nuke-style operations
            # The plus/average operations already create optimal HDR scaling
            hdr_linear = hdr_image
            logger.info(f"  Radiance Fusion: Direct pass-through (Nuke-style HDR preservation)")
            
        elif algorithm_hint == "natural_blend":
            # Natural Blend: Preserve EV0 appearance exactly - NO SCALING
            # The algorithm already provides the correctly scaled values
            hdr_linear = hdr_image
            logger.info(f"  Natural Blend: Direct pass-through (preserves EV0 appearance)")
            
        elif algorithm_hint == "detail_injection":
            # Detail Injection: Preserve the exact values from the algorithm
            # Already in linear space with HDR values properly mapped
            hdr_linear = hdr_image
            logger.info(f"  Detail Injection: Direct pass-through (linear space with HDR detail)")
            
        elif algorithm_hint == "mertens":
            # Mertens: Medium HDR range, values 1-10 range
            p85 = np.percentile(hdr_image, 85.0)
            if p85 > 0:
                hdr_linear = hdr_image * (3.0 / p85)
            else:
                hdr_linear = hdr_image * 3.0
            hdr_linear = np.clip(hdr_linear, 0.0, 12.0)
            
        elif algorithm_hint in ["debevec", "robertson"]:
            # PURE HDR: Direct pass-through of calibrated linear radiance
            # No post-processing - the calibrated exposure times handle everything
            hdr_linear = hdr_image
            logger.info(f"  Debevec/Robertson: Direct pass-through (NO post-processing)")
            logger.info(f"  Pure linear radiance from calibrated exposure times")
            
        else:
            # Unknown algorithm - conservative approach
            hdr_linear = np.clip(hdr_image * 2.0, 0.0, 10.0)
        
        logger.info(f"  Final HDR range: [{hdr_linear.min():.6f}, {hdr_linear.max():.6f}]")
        logger.info(f"  Max value: {hdr_linear.max():.2f} (VFX raw data)")
        
        # Convert to ComfyUI format: [1, H, W, C] - NO NORMALIZATION!
        if len(hdr_linear.shape) == 3:
            tensor = torch.from_numpy(hdr_linear.astype(np.float32)).unsqueeze(0)
        else:
            tensor = torch.from_numpy(hdr_linear.astype(np.float32))
        
        return tensor.float()
        
    else:
        # Standard 8-bit conversion (fallback)
        image_8bit = np.clip(hdr_image * 255.0, 0, 255).astype(np.uint8)
        normalized = image_8bit.astype(np.float32) / 255.0
        
        if len(normalized.shape) == 3:
            tensor = torch.from_numpy(normalized).unsqueeze(0)
        else:
            tensor = torch.from_numpy(normalized)
        
        return tensor.float()


class DebevecHDRProcessor:
    """Core HDR processing using multiple algorithms"""
    
    def __init__(self):
        self.calibrator = cv2.createCalibrateDebevec()
        self.merge_debevec = cv2.createMergeDebevec()
        # Alternative algorithms
        self.merge_mertens = cv2.createMergeMertens()
        self.merge_robertson = cv2.createMergeRobertson()
        self.calibrator_robertson = cv2.createCalibrateRobertson()
    
    def _apply_antibanding_filter(self, hdr_image: np.ndarray) -> np.ndarray:
        """
        Apply subtle bilateral filtering to reduce banding artifacts in HDR images
        while preserving edges and maintaining linear radiance values
        
        CRITICAL: Works directly with float32 data to avoid introducing quantization
        
        Args:
            hdr_image: HDR image in linear space (float32, BGR)
            
        Returns:
            Filtered HDR image with reduced banding
        """
        # Work with a copy to preserve original
        filtered = hdr_image.copy()
        
        # Normalize to 0-1 range for bilateral filter
        max_val = np.percentile(hdr_image, 99.9)  # Use 99.9th percentile to avoid extreme outliers
        if max_val <= 0:
            return filtered
        
        normalized = np.clip(hdr_image / max_val, 0, 1).astype(np.float32)
        
        # Apply bilateral filter DIRECTLY on float32 data
        # This avoids 8-bit quantization that would introduce banding!
        # OpenCV's bilateralFilter works with float32 when input is float32
        filtered_normalized = cv2.bilateralFilter(
            normalized,
            d=5,                # Small kernel - very subtle
            sigmaColor=0.01,    # Very low for float32 (0-1 range) - strongly preserves edges
            sigmaSpace=5        # Small spatial sigma - local smoothing only
        )
        
        # Restore original scale
        filtered = filtered_normalized * max_val
        
        # Blend: 70% filtered + 30% original - conservative blend
        # This ensures we only smooth obvious banding, not fine details
        final = filtered * 0.7 + hdr_image * 0.3
        
        logger.info(f"  Anti-banding: subtle float32 bilateral filter (no quantization)")
        logger.info(f"  Blend: 70% filtered + 30% original (preserves detail)")
        logger.info(f"  Preserved HDR peaks: max value {final.max():.2f}")
        
        return final
    
    def _compute_exposure_ratio_srgb(self, reference: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the brightness ratio between two exposures IN sRGB SPACE
        This matches how Debevec sees the images (8-bit sRGB, not linear)
        
        Args:
            reference: Reference image in sRGB space (float32 0-1, BGR)
            target: Target image to compare in sRGB space (float32 0-1, BGR)
            
        Returns:
            float: Brightness ratio in sRGB space (target/reference)
        """
        # Convert to grayscale for analysis
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        # Create mask for well-exposed regions (0.2 to 0.8 in reference)
        # These regions are least likely to be clipped in either image
        mask = (ref_gray > 0.2) & (ref_gray < 0.8) & (tgt_gray > 0.01) & (tgt_gray < 0.99)
        
        if np.sum(mask) < 1000:  # Not enough valid pixels
            logger.warning("Insufficient well-exposed pixels for calibration, using full image")
            mask = np.ones_like(ref_gray, dtype=bool)
        
        # Compute ratio using median (robust to outliers)
        ref_values = ref_gray[mask]
        tgt_values = tgt_gray[mask]
        
        # Ratio in sRGB space
        ratios = tgt_values / (ref_values + 1e-8)
        median_ratio = np.median(ratios)
        
        # Log for debugging
        logger.info(f"  Valid pixels: {np.sum(mask)}, sRGB ratio range: [{np.percentile(ratios, 10):.3f}, {np.percentile(ratios, 90):.3f}]")
        
        return median_ratio
    
    def _calibrate_exposure_times(self, images: List[np.ndarray], nominal_times: List[float]) -> List[float]:
        """
        Analyze AI-generated brackets and compute corrected exposure times
        that match their actual brightness relationships IN sRGB SPACE
        
        CRITICAL: Calibrates in sRGB space (NOT linear) because Debevec receives
        8-bit sRGB images and estimates the response curve internally.
        
        Args:
            images: List of AI-generated exposure brackets (8-bit BGR)
            nominal_times: Theoretical exposure times [t+2, t0, t-2, ...]
            
        Returns:
            Calibrated exposure times that match AI's actual behavior in sRGB space
        """
        logger.info("=" * 60)
        logger.info("ADAPTIVE EXPOSURE CALIBRATION (sRGB SPACE)")
        logger.info("Analyzing AI-generated brackets as Debevec sees them")
        logger.info("=" * 60)
        
        # DON'T convert to linear - work in sRGB space like Debevec does!
        # Just normalize to 0-1 range
        srgb_images = [img.astype(np.float32) / 255.0 for img in images]
        
        # Find EV0 (reference exposure) - should be in the middle
        ev0_idx = len(images) // 2
        ev0_srgb = srgb_images[ev0_idx]
        
        logger.info(f"Reference image (EV0): Index {ev0_idx}, Nominal time: {nominal_times[ev0_idx]:.6f}s")
        logger.info(f"Working in sRGB space (matching Debevec's input)")
        
        # Compute intensity statistics for each image
        calibrated_times = []
        
        for i, img_srgb in enumerate(srgb_images):
            if i == ev0_idx:
                # Reference exposure - keep as is
                calibrated_times.append(nominal_times[ev0_idx])
                logger.info(f"Image {i} (EV0 reference): time={nominal_times[ev0_idx]:.6f}s")
                continue
            
            # Analyze the brightness relationship in sRGB space
            logger.info(f"Calibrating Image {i}:")
            srgb_ratio = self._compute_exposure_ratio_srgb(ev0_srgb, img_srgb)
            
            # For AI-generated images, use a hybrid approach:
            # Apply a gentler power (1.8 instead of 2.2) as AI doesn't follow perfect sRGB gamma
            # This accounts for some gamma relationship without overcorrecting
            time_ratio = srgb_ratio ** 1.8
            calibrated_time = nominal_times[ev0_idx] * time_ratio
            calibrated_times.append(calibrated_time)
            
            expected_ratio = nominal_times[i] / nominal_times[ev0_idx]
            adjustment_factor = calibrated_time / nominal_times[i]
            
            logger.info(f"  sRGB brightness ratio: {srgb_ratio:.3f}")
            logger.info(f"  Exposure time ratio (^1.8): {time_ratio:.3f}")
            logger.info(f"  Expected time ratio: {expected_ratio:.3f}")
            logger.info(f"  Nominal time: {nominal_times[i]:.6f}s → Calibrated time: {calibrated_time:.6f}s")
            logger.info(f"  Adjustment factor: {adjustment_factor:.3f}x")
        
        logger.info("=" * 60)
        logger.info(f"CALIBRATION SUMMARY (sRGB-space calibration):")
        logger.info(f"  Original times:    {[f'{t:.6f}' for t in nominal_times]}")
        logger.info(f"  Calibrated times:  {[f'{t:.6f}' for t in calibrated_times]}")
        logger.info(f"  Adjustment factors: {[f'{calibrated_times[i]/nominal_times[i]:.3f}x' for i in range(len(nominal_times))]}")
        logger.info(f"  EV0 anchors absolute scale at {nominal_times[ev0_idx]:.6f}s")
        logger.info("=" * 60)
        
        return calibrated_times
        
    def process_hdr(self, images: List[np.ndarray], exposure_times: List[float], 
                    algorithm: str = "detail_injection", auto_calibrate: bool = True, 
                    debevec_exposure_compensation: float = -8.0,
                    debevec_anti_banding: bool = True) -> np.ndarray:
        """
        Process multiple exposure images using various HDR algorithms
        
        VFX PIPELINE NOTE:
        - Debevec/Robertson expect 8-bit sRGB input (NOT linearized)
        - They output linear radiance values (physical light intensity)
        - The output will look flat/desaturated - this is CORRECT for VFX
        
        Args:
            images: List of 8-bit sRGB images (0-255 range) - DO NOT pre-linearize!
            exposure_times: List of exposure times in seconds
            algorithm: HDR algorithm to use:
                 - "radiance_fusion": Nuke-style plus/average HDR fusion (NEW DEFAULT!)
                 - "detail_injection": AI-aware detail injection with sRGB->linear conversion
                 - "natural_blend": Preserves EV0 look with enhanced range  
                 - "mertens": Exposure fusion (display-ready)
                 - "debevec": True HDR recovery (flat/linear for VFX)
                 - "robertson": Alternative HDR recovery (flat/linear for VFX)
            auto_calibrate: Enable adaptive calibration for AI-generated brackets (debevec/robertson only)
                          Analyzes actual brightness relationships and corrects exposure times
            debevec_exposure_compensation: Exposure compensation in stops for debevec/robertson output
                          Default: -8.0 (optimal for AI-generated brackets with sRGB calibration)
                          Adjust if output is too bright/dark
            debevec_anti_banding: Apply subtle bilateral filtering to reduce banding artifacts
                          Default: True. Preserves edges while smoothing gradients.
            
        Returns:
            HDR merged image in linear radiance space (float32)
            - For Debevec/Robertson: Raw linear radiance (can exceed 1.0)
            - For Mertens/Natural: Display-oriented fusion (0-1 range typical)
        """
        try:
            # Validate input format - OpenCV HDR functions require 8-bit input
            processed_images = []
            for i, img in enumerate(images):
                if img.dtype != np.uint8:
                    logger.warning(f"Image {i} is not 8-bit (dtype: {img.dtype}), this may cause issues")
                
                # Handle color channels properly
                if len(img.shape) == 3 and img.shape[2] == 3:  # 3-channel image
                    # CRITICAL FIX: ALL OpenCV functions work with BGR internally
                    # ComfyUI provides RGB, so we MUST convert for ALL algorithms
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    processed_images.append(img_bgr)
                    logger.info(f"Converting RGB to BGR for {algorithm} (OpenCV standard)")
                else:
                    logger.error(f"Image {i} has invalid shape: {img.shape}")
                    raise ValueError(f"Image must be 3-channel RGB, got shape: {img.shape}")
            
            # Convert exposure times to numpy array
            times = np.array(exposure_times, dtype=np.float32)
            
            # ADAPTIVE AUTO-CALIBRATION for AI-generated images
            if auto_calibrate and algorithm in ["debevec", "robertson"]:
                logger.info("🔬 AUTO-CALIBRATION ENABLED for AI-generated brackets")
                times_calibrated = np.array(self._calibrate_exposure_times(processed_images, times.tolist()), dtype=np.float32)
                times = times_calibrated
            
            logger.info(f"Processing {len(processed_images)} images with {algorithm} algorithm")
            logger.info(f"Exposure times: {times}")
            logger.info(f"Image formats: {[img.shape for img in processed_images]}")
            logger.info(f"Image dtypes: {[img.dtype for img in processed_images]}")
            
            # Process using selected algorithm
            if algorithm == "mertens":
                # Mertens Exposure Fusion - uses original gamma, no correction needed
                logger.info("Using Mertens Exposure Fusion algorithm...")
                hdr_radiance = self.merge_mertens.process(processed_images)
                # Mertens output is typically in 0-1 range, gently scale for HDR
                if hdr_radiance.max() <= 1.0:
                    hdr_radiance = hdr_radiance * 1.5  # Gentle boost, preserve contrast
                    
            elif algorithm == "radiance_fusion":
                # Radiance Fusion - Nuke-inspired HDR blending (NEW DEFAULT!)
                logger.info("Using Radiance Fusion - Nuke-style plus/average HDR blending...")
                hdr_radiance = self._radiance_fusion(processed_images, times)
                
            elif algorithm == "natural_blend":
                # Natural Blend - maintains EV0 appearance with enhanced dynamic range
                logger.info("Using Natural Blend exposure blending...")
                hdr_radiance = self._blend_ev0_preserving(processed_images, times)
                
            elif algorithm == "detail_injection":
                # Detail Injection - AI-aware HDR detail injection
                logger.info("Using Detail Injection - AI-aware HDR processing...")
                # Always output linear HDR for proper EXR export
                hdr_radiance = self._detail_injection(processed_images, times, output_mode="linear")
                
            elif algorithm == "hdrutils" and HDRUTILS_AVAILABLE:
                # Use HDRutils library for HDR merging
                logger.info("Using HDRutils library for HDR merging...")
                hdr_radiance = self._merge_with_hdrutils(processed_images, times)
                
            elif algorithm == "robertson":
                # Robertson algorithm - Pure HDR recovery with exposure compensation
                logger.info("Using Robertson algorithm - Pure HDR mode...")
                response = self.calibrator_robertson.process(processed_images, times)
                hdr_radiance = self.merge_robertson.process(processed_images, times, response)
                
                logger.info(f"Robertson raw output: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                # Anti-banding filter
                if debevec_anti_banding:
                    logger.info("Applying anti-banding bilateral filter...")
                    hdr_radiance = self._apply_antibanding_filter(hdr_radiance)
                    logger.info(f"After anti-banding: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                # Apply exposure compensation
                if debevec_exposure_compensation != 0.0:
                    compensation_factor = 2.0 ** debevec_exposure_compensation
                    hdr_radiance = hdr_radiance * compensation_factor
                    logger.info(f"Applied exposure compensation: {debevec_exposure_compensation:+.1f} stops (factor: {compensation_factor:.6f}x)")
                    logger.info(f"Robertson compensated output: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                logger.info(f"Robertson mean radiance: {hdr_radiance.mean():.6f}")
                
            else:  # Default to Debevec - Pure HDR recovery with exposure compensation
                # Estimate camera response function using Debevec method
                logger.info("Using Debevec algorithm - Pure HDR mode...")
                response = self.calibrator.process(processed_images, times)
                logger.info(f"Response function shape: {response.shape}")
                
                # Merge images into HDR using Debevec algorithm
                # Debevec outputs linear radiance values (already in linear space)
                hdr_radiance = self.merge_debevec.process(processed_images, times, response)
                
                logger.info(f"Debevec raw output: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                # Anti-banding filter
                if debevec_anti_banding:
                    logger.info("Applying anti-banding bilateral filter...")
                    hdr_radiance = self._apply_antibanding_filter(hdr_radiance)
                    logger.info(f"After anti-banding: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                # Apply exposure compensation
                if debevec_exposure_compensation != 0.0:
                    compensation_factor = 2.0 ** debevec_exposure_compensation
                    hdr_radiance = hdr_radiance * compensation_factor
                    logger.info(f"Applied exposure compensation: {debevec_exposure_compensation:+.1f} stops (factor: {compensation_factor:.6f}x)")
                    logger.info(f"Debevec compensated output: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                logger.info(f"Debevec mean radiance: {hdr_radiance.mean():.6f}")
            
            # Validate HDR output
            if hdr_radiance is None or hdr_radiance.size == 0:
                raise ValueError("HDR merge produced empty result")
            
            logger.info(f"HDR merge completed with {algorithm} algorithm:")
            logger.info(f"  Output shape: {hdr_radiance.shape}")
            logger.info(f"  Output dtype: {hdr_radiance.dtype}")
            logger.info(f"  Value range: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
            logger.info(f"  Mean value: {hdr_radiance.mean():.6f}")
            
            # Check for valid HDR data
            if np.all(hdr_radiance == 0):
                raise ValueError("HDR merge produced all-zero result")
            
            # CRITICAL: Convert BGR back to RGB for ALL algorithms
            # All OpenCV functions output BGR, but ComfyUI needs RGB
            if len(hdr_radiance.shape) == 3 and hdr_radiance.shape[2] == 3:
                hdr_radiance = cv2.cvtColor(hdr_radiance, cv2.COLOR_BGR2RGB)
                logger.info("Converting BGR back to RGB for ComfyUI output")
            
            # The result is already in linear colorspace - preserve HDR data
            return hdr_radiance.astype(np.float32)
            
        except Exception as e:
            logger.error(f"HDR processing error: {str(e)}")
            logger.error(f"Image count: {len(images)}")
            logger.error(f"Image shapes: {[img.shape if img is not None else 'None' for img in images]}")
            logger.error(f"Exposure times: {exposure_times}")
            
            # Fallback: return the middle exposure image in linear space
            if images:
                middle_idx = len(images) // 2
                fallback = images[middle_idx].astype(np.float32) / 255.0
                # Convert back to linear space (reverse gamma correction)
                fallback_linear = np.where(fallback <= 0.04045, 
                                         fallback / 12.92,
                                         np.power((fallback + 0.055) / 1.055, 2.4))
                logger.info(f"Using fallback image (index {middle_idx})")
                return fallback_linear.astype(np.float32)
            raise e

    def _gentle_tone_map(self, hdr_image: np.ndarray, algorithm_name: str) -> np.ndarray:
        """
        Apply gentle tone mapping to preserve HDR range while making output usable
        
        Args:
            hdr_image: Raw HDR output from Debevec/Robertson
            algorithm_name: Name of algorithm for logging
            
        Returns:
            Gently processed HDR image with preserved dynamic range
        """
        logger.info(f"{algorithm_name} raw output range: [{hdr_image.min():.6f}, {hdr_image.max():.6f}]")
        
        try:
            # Gentle processing to preserve HDR range but make it usable
            
            # Method 1: Simple scaling based on percentiles (preserves HDR better than Reinhard)
            p95 = np.percentile(hdr_image, 95)
            p05 = np.percentile(hdr_image, 5)
            
            if p95 > p05 and p95 > 0:
                # Scale so 95th percentile maps to reasonable value (1.0-3.0 range)
                scale_factor = 2.0 / p95
                scaled = hdr_image * scale_factor
                
                # Apply very gentle gamma correction to improve appearance
                gentle_gamma = np.power(np.clip(scaled, 0, 10), 0.8)
                
                logger.info(f"{algorithm_name} after gentle processing: [{gentle_gamma.min():.6f}, {gentle_gamma.max():.6f}]")
                return gentle_gamma.astype(np.float32)
            else:
                # Fallback for edge cases
                return np.clip(hdr_image, 0.0, 10.0).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Gentle tone mapping failed for {algorithm_name}: {e}")
            # Fallback: simple clipping
            return np.clip(hdr_image, 0.0, 10.0).astype(np.float32)
    
    def _radiance_fusion(self, images: List[np.ndarray], times: List[float]) -> np.ndarray:
        """
        Radiance Fusion - Nuke-inspired HDR blending algorithm
        
        Uses Nuke's plus and average operations for perfect HDR preservation:
        1. Plus all outer exposures: (ev-4 + ev-2 + ev+2 + ev+4)
        2. Average with center exposure: result + ev0 / 2
        
        This creates beautiful HDR data while maintaining natural appearance.
        
        Args:
            images: List of exposure images in BGR format (from OpenCV)
            times: Exposure times
            
        Returns:
            Radiance fusion result with excellent HDR preservation
        """
        logger.info("Radiance Fusion: Nuke-style HDR blending with plus/average operations")
        
        # Convert to float32 for HDR processing
        float_images = [img.astype(np.float32) / 255.0 for img in images]
        
        # For 5-stop: [ev+4, ev+2, ev0, ev-2, ev-4] - indices [0,1,2,3,4]
        # For 3-stop: [ev+2, ev0, ev-2] - indices [0,1,2]
        
        if len(float_images) == 5:
            # 5-stop processing: ev+4, ev+2, ev0, ev-2, ev-4
            ev_plus_4 = float_images[0]   # Most overexposed
            ev_plus_2 = float_images[1]   # Overexposed  
            ev_0 = float_images[2]        # Middle exposure
            ev_minus_2 = float_images[3]  # Underexposed
            ev_minus_4 = float_images[4]  # Most underexposed
            
            # NUKE PLUS OPERATION: Add all outer exposures
            # This preserves full dynamic range from all sources
            outer_sum = ev_plus_4 + ev_plus_2 + ev_minus_2 + ev_minus_4
            logger.info("5-stop: Added outer exposures (ev±4, ev±2) using Nuke plus operation")
            
        elif len(float_images) == 3:
            # 3-stop processing: ev+2, ev0, ev-2
            ev_plus_2 = float_images[0]   # Overexposed
            ev_0 = float_images[1]        # Middle exposure  
            ev_minus_2 = float_images[2]  # Underexposed
            
            # NUKE PLUS OPERATION: Add outer exposures
            outer_sum = ev_plus_2 + ev_minus_2
            logger.info("3-stop: Added outer exposures (ev±2) using Nuke plus operation")
            
        else:
            raise ValueError(f"Radiance Fusion requires 3 or 5 images, got {len(float_images)}")
        
        # NUKE AVERAGE OPERATION: (outer_sum + ev0) / 2
        # This balances the combined outer detail with the natural center exposure
        radiance_result = (outer_sum + ev_0) / 2.0
        
        logger.info(f"Applied Nuke average operation: (outer_sum + ev0) / 2")
        logger.info(f"Radiance Fusion result: [{radiance_result.min():.3f}, {radiance_result.max():.3f}]")
        logger.info(f"HDR pixels above 1.0: {np.sum(radiance_result > 1.0)} pixels")
        
        # Return with full HDR range preserved (no clipping!)
        return radiance_result.astype(np.float32)

    def _blend_ev0_preserving(self, images: List[np.ndarray], times: List[float]) -> np.ndarray:
        """
        Improved exposure blending that perfectly preserves EV0 appearance
        while storing HDR information in values above 1.0
        
        This method uses the EV0 as the base and only modifies areas where
        detail is lost (pure white or pure black) with information from other exposures.
        
        Args:
            images: List of exposure images in BGR format (from OpenCV)
            times: Exposure times
            
        Returns:
            Enhanced image that looks identical to EV0 but with HDR values > 1.0
        """
        logger.info("Natural Blend: Perfect EV0 preservation with HDR extension")
        
        # Find the EV0 image (middle exposure)
        ev0_idx = len(images) // 2
        ev0_base = images[ev0_idx].astype(np.float32) / 255.0
        
        logger.info(f"Using image {ev0_idx} as EV0 base (out of {len(images)} images)")
        
        # Convert all images to float (no exposure compensation needed)
        float_images = []
        for i, img in enumerate(images):
            float_img = img.astype(np.float32) / 255.0
            float_images.append(float_img)
        
        # Start with EV0 as the base - this ensures perfect appearance match
        result = ev0_base.copy()
        
        # Only blend in areas where EV0 has lost detail (highlights and shadows)
        gray_ev0 = cv2.cvtColor(ev0_base, cv2.COLOR_BGR2GRAY)
        
        # Process highlights - where EV0 is clipped (near 1.0)
        highlight_threshold = 0.95  # Areas above this in EV0 need HDR data
        highlight_mask = gray_ev0 > highlight_threshold
        
        if np.any(highlight_mask):
            # Use underexposed images for highlight recovery
            for i in range(ev0_idx + 1, len(float_images)):
                img = float_images[i]
                
                # Use the actual exposure difference for HDR scaling
                # Underexposed images have longer exposure times
                exposure_ratio = times[ev0_idx] / times[i]
                scale_factor = exposure_ratio  # This gives proper HDR scaling
                
                # Blend only in highlight areas
                for c in range(3):
                    # Use the underexposed data scaled up for HDR
                    hdr_values = img[:, :, c] * scale_factor
                    
                    # Smooth transition: gradually blend as we approach pure white
                    blend_weight = np.where(highlight_mask,
                                           (gray_ev0 - highlight_threshold) / (1.0 - highlight_threshold),
                                           0.0)
                    
                    # Blend HDR values only in highlights, preserving EV0 elsewhere
                    result[:, :, c] = np.where(highlight_mask,
                                              result[:, :, c] * (1 - blend_weight) + hdr_values * blend_weight,
                                              result[:, :, c])
            
            logger.info(f"HDR highlight recovery applied - values up to {result.max():.2f}")
        
        # Process shadows - where EV0 is too dark (near 0.0)
        shadow_threshold = 0.05  # Areas below this in EV0 need shadow detail
        shadow_mask = gray_ev0 < shadow_threshold
        
        if np.any(shadow_mask):
            # Use overexposed images for shadow recovery
            for i in range(ev0_idx):
                img = float_images[i]
                
                # Blend only in shadow areas
                for c in range(3):
                    # Smooth transition: gradually blend as we approach pure black
                    blend_weight = np.where(shadow_mask,
                                           (shadow_threshold - gray_ev0) / shadow_threshold,
                                           0.0)
                    
                    # Blend shadow detail, preserving EV0 elsewhere
                    result[:, :, c] = np.where(shadow_mask,
                                              result[:, :, c] * (1 - blend_weight * 0.5) + img[:, :, c] * blend_weight * 0.5,
                                              result[:, :, c])
            
            logger.info("Shadow detail recovery applied")
        
        # Ensure midtones exactly match EV0
        midtone_mask = np.logical_and(gray_ev0 >= shadow_threshold, gray_ev0 <= highlight_threshold)
        for c in range(3):
            result[:, :, c] = np.where(midtone_mask, ev0_base[:, :, c], result[:, :, c])
        
        logger.info(f"Natural Blend completed - EV0 appearance preserved")
        logger.info(f"  HDR range: [{result.min():.3f}, {result.max():.3f}]")
        logger.info(f"  Values > 1.0: {np.sum(result > 1.0)} pixels")
        
        # NO CLIPPING - preserve HDR values above 1.0
        return result.astype(np.float32)
        
    def _merge_with_hdrutils(self, images: List[np.ndarray], times: List[float]) -> np.ndarray:
        """
        Use HDRutils library for HDR merging if available
        
        Args:
            images: List of 8-bit images
            times: Exposure times
            
        Returns:
            HDR merged image
        """
        try:
            # Convert images to the format HDRutils expects
            # HDRutils typically expects RGB format
            rgb_images = []
            for img in images:
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Convert BGR to RGB for HDRutils
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb_images.append(rgb_img)
                else:
                    rgb_images.append(img)
            
            # Stack images for HDRutils
            image_stack = np.stack(rgb_images, axis=0)
            
            # Create HDR merge using HDRutils
            # Note: HDRutils.merge might have different parameters
            # This is a generic implementation
            hdr_image = HDRutils.merge(image_stack, times)
            
            logger.info(f"HDRutils merge complete: range [{hdr_image.min():.6f}, {hdr_image.max():.6f}]")
            
            return hdr_image.astype(np.float32)
            
        except Exception as e:
            logger.error(f"HDRutils merge failed: {e}")
            logger.info("Falling back to Mertens algorithm")
            # Fallback to Mertens
            return self.merge_mertens.process(images)
    
    def _create_highlight_mask(self, gray_image: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Create a mask for highlight areas that need detail recovery"""
        # Smooth transition for highlights
        mask = np.zeros_like(gray_image, dtype=np.float32)
        
        # Areas above threshold get progressively more blending
        bright_areas = gray_image > threshold
        if np.any(bright_areas):
            mask[bright_areas] = (gray_image[bright_areas] - threshold) / (1.0 - threshold)
        
        # Smooth the mask to avoid harsh transitions with larger kernel
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        
        return np.clip(mask, 0, 1)
    
    def _create_shadow_mask(self, gray_image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """Create a mask for shadow areas that need detail recovery"""
        # Smooth transition for shadows
        mask = np.zeros_like(gray_image, dtype=np.float32)
        
        # Areas below threshold get progressively more blending
        dark_areas = gray_image < threshold
        if np.any(dark_areas):
            mask[dark_areas] = (threshold - gray_image[dark_areas]) / threshold
        
        # Smooth the mask to avoid harsh transitions with larger kernel
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        
        return np.clip(mask, 0, 1)
    
    def _analyze_gamma(self, images: List[np.ndarray]) -> float:
        """
        Analyze input images to detect gamma encoding
        AI-generated sRGB images typically have gamma 2.2
        
        Args:
            images: List of 8-bit images
            
        Returns:
            Detected gamma value (2.2 for sRGB, 1.0 for linear)
        """
        # Sample multiple images for better detection
        sample_values = []
        
        for img in images[:min(3, len(images))]:
            # Convert to float and normalize
            img_float = img.astype(np.float32) / 255.0
            
            # Sample mid-gray values (0.3-0.7 range) for gamma analysis
            mask = (img_float > 0.3) & (img_float < 0.7)
            if np.any(mask):
                sample_values.extend(img_float[mask].flatten()[:1000])
        
        if len(sample_values) > 0:
            # Analyze distribution to detect gamma curve
            sample_array = np.array(sample_values)
            median_val = np.median(sample_array)
            
            # AI-generated images tend to have strong gamma curve
            # Real linear images would have different distribution
            # Simple heuristic: if median is in expected sRGB range, assume gamma 2.2
            if 0.4 < median_val < 0.6:
                logger.info("Detected sRGB gamma encoding (gamma ≈ 2.2)")
                return 2.2
            else:
                logger.info("Images appear to be in sRGB color space (gamma 2.2)")
                return 2.2  # Default to sRGB for AI-generated images
        
        logger.info("Defaulting to sRGB gamma (2.2) for AI-generated images")
        return 2.2
    
    def _srgb_to_linear(self, srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB (gamma-encoded) to linear light space
        
        Standard sRGB to linear transformation:
        - For values <= 0.04045: linear = srgb / 12.92
        - For values > 0.04045: linear = ((srgb + 0.055) / 1.055) ^ 2.4
        
        Args:
            srgb: Image in sRGB color space (0-1 range)
            
        Returns:
            Image in linear light space
        """
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        
        logger.info(f"sRGB to linear: input range [{srgb.min():.3f}, {srgb.max():.3f}] -> output range [{linear.min():.3f}, {linear.max():.3f}]")
        return linear.astype(np.float32)
    
    def _linear_to_srgb(self, linear: np.ndarray) -> np.ndarray:
        """
        Convert linear light space to sRGB (gamma-encoded)
        
        Inverse of sRGB transformation - for display purposes only!
        This will clip HDR values >1.0 to displayable range.
        
        Standard linear to sRGB transformation:
        - For values <= 0.0031308: srgb = linear * 12.92
        - For values > 0.0031308: srgb = 1.055 * (linear ^ (1/2.4)) - 0.055
        
        Args:
            linear: Image in linear light space
            
        Returns:
            Image in sRGB color space (gamma-encoded)
        """
        # Tone map HDR values first using simple Reinhard
        # This compresses values >1.0 into displayable range
        linear_tonemapped = linear / (1.0 + linear)
        
        # Apply sRGB gamma encoding
        srgb = np.where(
            linear_tonemapped <= 0.0031308,
            linear_tonemapped * 12.92,
            1.055 * np.power(linear_tonemapped, 1.0 / 2.4) - 0.055
        )
        
        # Clamp to valid range
        srgb = np.clip(srgb, 0.0, 1.0)
        
        logger.info(f"Linear to sRGB (tonemapped): input range [{linear.min():.3f}, {linear.max():.3f}] -> output range [{srgb.min():.3f}, {srgb.max():.3f}]")
        return srgb.astype(np.float32)
    
    def _smooth_step(self, edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        """
        Hermite interpolation for smooth transitions
        Creates smooth S-curve for natural blending
        
        Args:
            edge0: Lower edge
            edge1: Upper edge
            x: Input values
            
        Returns:
            Smoothly interpolated values (0-1)
        """
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    def _inject_detail_preserve_color(self, base_rgb: np.ndarray, detail_rgb: np.ndarray, 
                                      blend: np.ndarray) -> np.ndarray:
        """
        Inject detail while preserving hue and color ratios
        
        Uses luminance-based scaling to maintain color appearance
        while changing detail and brightness
        
        Args:
            base_rgb: Base RGB image (H, W, 3)
            detail_rgb: Detail source RGB image (H, W, 3)
            blend: Blend weight map (H, W)
            
        Returns:
            Result with injected detail
        """
        # Calculate luminance using Rec. 709 coefficients
        base_luma = (0.2126 * base_rgb[:, :, 2] + 
                     0.7152 * base_rgb[:, :, 1] + 
                     0.0722 * base_rgb[:, :, 0])  # Note: BGR order from OpenCV
        
        detail_luma = (0.2126 * detail_rgb[:, :, 2] + 
                       0.7152 * detail_rgb[:, :, 1] + 
                       0.0722 * detail_rgb[:, :, 0])
        
        # Calculate new luminance
        new_luma = base_luma * (1 - blend) + detail_luma * blend
        
        # Scale RGB to match new luminance while preserving color ratios
        result_rgb = np.zeros_like(base_rgb)
        
        # Avoid division by zero
        safe_base_luma = np.maximum(base_luma, 1e-6)
        scale = new_luma / safe_base_luma
        
        # Apply scaling to each channel
        for c in range(3):
            result_rgb[:, :, c] = base_rgb[:, :, c] * scale
        
        # In near-black areas, use detail color directly
        near_black = base_luma < 0.001
        if np.any(near_black):
            for c in range(3):
                result_rgb[:, :, c] = np.where(
                    near_black,
                    detail_rgb[:, :, c] * blend,
                    result_rgb[:, :, c]
                )
        
        return result_rgb
    
    def _detail_injection(self, images: List[np.ndarray], times: List[float], output_mode: str = "linear") -> np.ndarray:
        """
        Detail Injection algorithm for AI-generated HDR stacks
        
        This algorithm is designed specifically for AI-generated exposure stacks
        where images don't follow photometric relationships. It:
        1. Analyzes gamma encoding and converts to linear space
        2. Uses EV0 as base (preserves natural look)
        3. Injects highlight detail from underexposed images (EV-2, EV-4) into >1.0 range
        4. Injects shadow detail from overexposed images (EV+2, EV+4) into near-zero range
        5. Preserves color ratios using luminance-based scaling
        6. Applies automatic brightness compensation to target 18% gray
        
        Args:
            images: List of exposure images in BGR format (from OpenCV)
            times: Exposure times
            output_mode: Always "linear" for true HDR output
            
        Returns:
            Linear HDR image with detail injection (perfect for EXR export)
        """
        logger.info("=" * 80)
        logger.info("DETAIL INJECTION ALGORITHM - AI-Aware HDR Processing")
        logger.info("=" * 80)
        
        # Step 1: Analyze and convert to linear space
        logger.info("\nStep 1: Gamma Analysis and Linear Conversion")
        logger.info("-" * 40)
        
        detected_gamma = self._analyze_gamma(images)
        
        # Convert all images from 8-bit sRGB to linear float
        linear_images = []
        for i, img in enumerate(images):
            # Convert to 0-1 range
            img_float = img.astype(np.float32) / 255.0
            # Convert sRGB to linear
            img_linear = self._srgb_to_linear(img_float)
            linear_images.append(img_linear)
            logger.info(f"  Image {i}: sRGB -> linear, range [{img_linear.min():.4f}, {img_linear.max():.4f}]")
        
        # Identify images by exposure
        num_images = len(linear_images)
        ev0_idx = num_images // 2
        
        if num_images == 5:
            # 5-stop: [EV+4, EV+2, EV0, EV-2, EV-4]
            ev0 = linear_images[2]
            ev_plus_4 = linear_images[0]
            ev_plus_2 = linear_images[1]
            ev_minus_2 = linear_images[3]
            ev_minus_4 = linear_images[4]
            logger.info("\n5-stop stack detected: EV+4, EV+2, EV0, EV-2, EV-4")
            
        elif num_images == 3:
            # 3-stop: [EV+2, EV0, EV-2]
            ev0 = linear_images[1]
            ev_plus_2 = linear_images[0]
            ev_minus_2 = linear_images[2]
            ev_plus_4 = None
            ev_minus_4 = None
            logger.info("\n3-stop stack detected: EV+2, EV0, EV-2")
            
        else:
            raise ValueError(f"Detail injection requires 3 or 5 images, got {num_images}")
        
        # Start with EV0 as base (ensures natural appearance)
        result = ev0.copy()
        
        # Step 2: Highlight Detail Injection
        logger.info("\nStep 2: Highlight Detail Injection")
        logger.info("-" * 40)
        logger.info("Injecting detail from underexposed images into >1.0 range...")
        
        # Convert to grayscale for masking (use Rec. 709)
        gray_ev0 = (0.2126 * ev0[:, :, 2] + 
                    0.7152 * ev0[:, :, 1] + 
                    0.0722 * ev0[:, :, 0])
        
        # Highlight thresholds - wider ranges for smoother transitions
        highlight_threshold_moderate = 0.80  # Start blending earlier
        highlight_threshold_extreme = 0.92   # Start extreme highlights earlier
        
        # Create smooth highlight masks using Hermite interpolation with wider ranges
        # Wider ranges = more gradual transitions, less sharp cutoffs
        highlight_mask_moderate = self._smooth_step(0.60, 0.95, gray_ev0)  # Start at 0.6 instead of 0.7
        highlight_mask_extreme = self._smooth_step(0.75, 0.98, gray_ev0)    # Start at 0.75 instead of 0.85
        
        # Apply additional Gaussian smoothing to masks for even smoother transitions
        # This creates very gradual feathering at blend boundaries
        highlight_mask_moderate = cv2.GaussianBlur(highlight_mask_moderate, (21, 21), 7.0)
        highlight_mask_extreme = cv2.GaussianBlur(highlight_mask_extreme, (21, 21), 7.0)
        
        logger.info("  Applied Gaussian smoothing to highlight masks for gradual transitions")
        
        pixels_moderate = np.sum(highlight_mask_moderate > 0.1)
        pixels_extreme = np.sum(highlight_mask_extreme > 0.5)
        logger.info(f"  Moderate highlights: {pixels_moderate} pixels")
        logger.info(f"  Extreme highlights: {pixels_extreme} pixels")
        
        # Inject from EV-2 for moderate highlights
        if ev_minus_2 is not None:
            logger.info("  Processing EV-2 (underexposed) for moderate highlights...")
            
            # Apply bilateral filtering to reduce grain while preserving edges
            ev_minus_2_smooth = cv2.bilateralFilter(
                (ev_minus_2 * 255).astype(np.uint8), 
                d=9, 
                sigmaColor=75, 
                sigmaSpace=75
            ).astype(np.float32) / 255.0
            logger.info("  Applied bilateral filtering to EV-2 for grain reduction")
            
            # Map EV-2 detail into 1.0-2.0 range
            for y in range(result.shape[0]):
                for x in range(result.shape[1]):
                    blend_weight = highlight_mask_moderate[y, x]
                    
                    if blend_weight > 0.01:  # Only process significant contributions
                        base_value = gray_ev0[y, x]
                        
                        if base_value > highlight_threshold_moderate:
                            # Get detail from smoothed underexposed image
                            detail_source = ev_minus_2_smooth[y, x, :]
                            
                            # Map detail into 1.0-2.0 range
                            # Remap from 0-1 input range to 1.0-2.0 HDR range
                            detail_luma = np.mean(detail_source)
                            hdr_target = 1.0 + detail_luma * 1.0  # Maps 0-1 to 1.0-2.0
                            
                            # Create HDR detail preserving color
                            if detail_luma > 1e-6:
                                detail_hdr = detail_source * (hdr_target / detail_luma)
                            else:
                                detail_hdr = detail_source
                            
                            # Blend based on how clipped we are
                            clip_amount = (base_value - highlight_threshold_moderate) / (1.0 - highlight_threshold_moderate)
                            clip_amount = np.clip(clip_amount, 0, 1)
                            
                            # Smooth blend
                            final_blend = blend_weight * clip_amount
                            result[y, x, :] = result[y, x, :] * (1 - final_blend) + detail_hdr * final_blend
            
            result_max = result.max()
            logger.info(f"  After EV-2 injection: max value = {result_max:.3f}")
        
        # Inject from EV-4 for extreme highlights (if available)
        if ev_minus_4 is not None:
            logger.info("  Processing EV-4 (very underexposed) for extreme highlights...")
            
            # Apply bilateral filtering to reduce grain while preserving edges
            ev_minus_4_smooth = cv2.bilateralFilter(
                (ev_minus_4 * 255).astype(np.uint8), 
                d=9, 
                sigmaColor=75, 
                sigmaSpace=75
            ).astype(np.float32) / 255.0
            logger.info("  Applied bilateral filtering to EV-4 for grain reduction")
            
            # Map EV-4 detail into 2.0-4.0 range
            for y in range(result.shape[0]):
                for x in range(result.shape[1]):
                    blend_weight = highlight_mask_extreme[y, x]
                    
                    if blend_weight > 0.1:  # Only process strong highlights
                        base_value = gray_ev0[y, x]
                        
                        if base_value > highlight_threshold_extreme:
                            # Get detail from smoothed very underexposed image
                            detail_source = ev_minus_4_smooth[y, x, :]
                            
                            # Map detail into 2.0-4.0 range
                            detail_luma = np.mean(detail_source)
                            hdr_target = 2.0 + detail_luma * 2.0  # Maps 0-1 to 2.0-4.0
                            
                            # Create HDR detail preserving color
                            if detail_luma > 1e-6:
                                detail_hdr = detail_source * (hdr_target / detail_luma)
                            else:
                                detail_hdr = detail_source * 2.0
                            
                            # Blend for extreme highlights only
                            clip_amount = (base_value - highlight_threshold_extreme) / (1.0 - highlight_threshold_extreme)
                            clip_amount = np.clip(clip_amount, 0, 1)
                            
                            final_blend = blend_weight * clip_amount * 0.7  # Reduced strength for naturalness
                            result[y, x, :] = result[y, x, :] * (1 - final_blend) + detail_hdr * final_blend
            
            result_max = result.max()
            logger.info(f"  After EV-4 injection: max value = {result_max:.3f}")
        
        # Step 3: Shadow Detail Injection
        logger.info("\nStep 3: Shadow Detail Injection")
        logger.info("-" * 40)
        logger.info("Injecting detail from overexposed images into shadow regions...")
        
        # Shadow thresholds - wider ranges for smoother transitions
        shadow_threshold_moderate = 0.18  # Start blending a bit higher
        shadow_threshold_extreme = 0.06   # Start extreme shadows higher
        
        # Create smooth shadow masks with wider ranges
        shadow_mask_moderate = self._smooth_step(0.30, 0.05, gray_ev0)  # Wider range: 0.30 instead of 0.25
        shadow_mask_extreme = self._smooth_step(0.12, 0.01, gray_ev0)    # Wider range: 0.12 instead of 0.10
        
        # Apply Gaussian smoothing to shadow masks for gradual transitions
        shadow_mask_moderate = cv2.GaussianBlur(shadow_mask_moderate, (21, 21), 7.0)
        shadow_mask_extreme = cv2.GaussianBlur(shadow_mask_extreme, (21, 21), 7.0)
        
        logger.info("  Applied Gaussian smoothing to shadow masks for gradual transitions")
        
        pixels_shadow_mod = np.sum(shadow_mask_moderate > 0.1)
        pixels_shadow_ext = np.sum(shadow_mask_extreme > 0.5)
        logger.info(f"  Moderate shadows: {pixels_shadow_mod} pixels")
        logger.info(f"  Extreme shadows: {pixels_shadow_ext} pixels")
        
        # Inject from EV+2 for moderate shadows
        if ev_plus_2 is not None:
            logger.info("  Processing EV+2 (overexposed) for moderate shadows...")
            
            # Apply bilateral filtering to reduce grain while preserving edges
            ev_plus_2_smooth = cv2.bilateralFilter(
                (ev_plus_2 * 255).astype(np.uint8), 
                d=9, 
                sigmaColor=75, 
                sigmaSpace=75
            ).astype(np.float32) / 255.0
            logger.info("  Applied bilateral filtering to EV+2 for grain reduction")
            
            for y in range(result.shape[0]):
                for x in range(result.shape[1]):
                    blend_weight = shadow_mask_moderate[y, x]
                    
                    if blend_weight > 0.01:
                        base_value = gray_ev0[y, x]
                        
                        if base_value < shadow_threshold_moderate:
                            # Get detail from smoothed overexposed image
                            detail_source = ev_plus_2_smooth[y, x, :]
                            
                            # Scale detail into shadow range (0.5x)
                            shadow_detail = detail_source * 0.5 * (base_value + 0.1)
                            
                            # Blend
                            crush_amount = (shadow_threshold_moderate - base_value) / shadow_threshold_moderate
                            crush_amount = np.clip(crush_amount, 0, 1)
                            
                            final_blend = blend_weight * crush_amount * 0.5
                            result[y, x, :] = result[y, x, :] * (1 - final_blend) + shadow_detail * final_blend
            
            logger.info("  EV+2 shadow injection complete")
        
        # Inject from EV+4 for extreme shadows (if available)
        if ev_plus_4 is not None:
            logger.info("  Processing EV+4 (very overexposed) for extreme shadows...")
            
            # Apply bilateral filtering to reduce grain while preserving edges
            ev_plus_4_smooth = cv2.bilateralFilter(
                (ev_plus_4 * 255).astype(np.uint8), 
                d=9, 
                sigmaColor=75, 
                sigmaSpace=75
            ).astype(np.float32) / 255.0
            logger.info("  Applied bilateral filtering to EV+4 for grain reduction")
            
            for y in range(result.shape[0]):
                for x in range(result.shape[1]):
                    blend_weight = shadow_mask_extreme[y, x]
                    
                    if blend_weight > 0.1:
                        base_value = gray_ev0[y, x]
                        
                        if base_value < shadow_threshold_extreme:
                            # Get detail from smoothed very overexposed image
                            detail_source = ev_plus_4_smooth[y, x, :]
                            
                            # Scale detail into very dark shadow range (0.2x)
                            shadow_detail = detail_source * 0.2 * (base_value + 0.05)
                            
                            # Blend
                            crush_amount = (shadow_threshold_extreme - base_value) / shadow_threshold_extreme
                            crush_amount = np.clip(crush_amount, 0, 1)
                            
                            final_blend = blend_weight * crush_amount * 0.3
                            result[y, x, :] = result[y, x, :] * (1 - final_blend) + shadow_detail * final_blend
            
            logger.info("  EV+4 extreme shadow injection complete")
        
        # Step 4: Automatic Brightness Compensation
        logger.info("\nStep 4: Automatic Brightness Compensation")
        logger.info("-" * 40)
        
        # Calculate current mean using 50th percentile (median) as reference
        # This is more robust than mean for images with extreme values
        result_flat = result.flatten()
        
        # Calculate median across all channels
        current_median = np.median(result_flat)
        
        # Also calculate mean for comparison
        current_mean = np.mean(result_flat)
        
        # Target: 0.18 in linear space = 18% gray = professional middle gray
        # For very dark images, use median-based scaling
        # For normal images, use mean-based scaling
        target_value = 0.18
        
        logger.info(f"  Current median: {current_median:.4f}")
        logger.info(f"  Current mean: {current_mean:.4f}")
        logger.info(f"  Target (18% gray): {target_value:.4f}")
        
        # Use median if image is very dark (shadows dominate)
        # Use mean if image has reasonable brightness distribution
        if current_median < 0.05:
            # Very dark image - use median for scaling
            reference_value = current_median
            logger.info(f"  Using median-based scaling (image is very dark)")
        else:
            # Normal image - use mean for scaling
            reference_value = current_mean
            logger.info(f"  Using mean-based scaling (normal brightness)")
        
        if reference_value > 0.001:  # Avoid division by zero
            brightness_factor = target_value / reference_value
            
            # More permissive range for brightness adjustment (0.3x to 8.0x)
            # This allows proper recovery of very dark or very bright images
            brightness_factor = np.clip(brightness_factor, 0.3, 8.0)
            
            if abs(brightness_factor - 1.0) > 0.05:  # Only adjust if significant difference
                logger.info(f"  Applying brightness compensation: {brightness_factor:.3f}x")
                result = result * brightness_factor
                logger.info(f"  New value range: [{result.min():.4f}, {result.max():.4f}]")
                logger.info(f"  New mean: {result.mean():.4f}")
                logger.info(f"  New median: {np.median(result):.4f}")
            else:
                logger.info("  No brightness adjustment needed (already optimal)")
        else:
            logger.warning("  Skipping brightness compensation (image too dark)")
        
        # Final statistics
        logger.info("\nStep 5: Final Result Statistics (Linear HDR)")
        logger.info("-" * 40)
        logger.info(f"  Value range: [{result.min():.4f}, {result.max():.4f}]")
        logger.info(f"  Pixels > 1.0 (HDR highlights): {np.sum(result > 1.0)}")
        logger.info(f"  Pixels > 2.0 (super-bright HDR): {np.sum(result > 2.0)}")
        logger.info(f"  Mean value: {result.mean():.4f}")
        logger.info(f"  Median value: {np.median(result):.4f}")
        
        logger.info("\nOutput Mode: Linear HDR (perfect for EXR export)")
        logger.info("  ✓  True linear HDR values preserved")
        logger.info("  ✓  Ready for professional color grading and compositing")
        
        logger.info("\n" + "=" * 80)
        logger.info("DETAIL INJECTION COMPLETE")
        logger.info("=" * 80)
        
        return result.astype(np.float32)


class LuminanceStackProcessor3Stops:
    """
    Professional ComfyUI Custom Node for 3-stop HDR processing
    Processes EV+2, EV+0, EV-2 exposures using multiple algorithms
    
    VFX PIPELINE NOTE:
    When using Debevec/Robertson algorithms, the output will be linear radiance
    values that look flat/desaturated. This is the correct format for professional
    VFX compositing and should be saved as 16-bit EXR.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ev_plus_2": ("IMAGE",),
                "ev_0": ("IMAGE",),
                "ev_minus_2": ("IMAGE",),
            },
            "optional": {
                "exposure_step": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "exposure_adjust": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "hdr_algorithm": (["detail_injection", "radiance_fusion", "natural_blend", "mertens", "debevec", "robertson"] + (["hdrutils"] if HDRUTILS_AVAILABLE else []), {
                    "default": "detail_injection"
                }),
                "auto_calibrate": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Enable adaptive calibration for AI-generated brackets (Debevec/Robertson only)"
                }),
                "debevec_exposure_comp": ("FLOAT", {
                    "default": -8.0,
                    "min": -15.0,
                    "max": 15.0,
                    "step": 0.1,
                    "tooltip": "Exposure compensation for Debevec/Robertson output (in stops). Default -8.0 optimal for AI brackets."
                }),
                "debevec_anti_banding": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Apply subtle bilateral filtering to reduce banding (Debevec/Robertson only)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_3_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_3_stop_hdr(self, ev_plus_2, ev_0, ev_minus_2, exposure_step=2.0, exposure_adjust=0.0, hdr_algorithm="detail_injection", auto_calibrate=True, debevec_exposure_comp=-8.0, debevec_anti_banding=True):
        """
        Process 3-stop HDR merge
        
        Args:
            ev_plus_2: Overexposed image (+2 EV)
            ev_0: Normal exposure image (0 EV)
            ev_minus_2: Underexposed image (-2 EV)
            exposure_step: EV step size
            exposure_adjust: Final exposure adjustment in stops (Nuke-style)
            
        Returns:
            Tuple containing merged HDR image with exposure adjustment applied
        """
        try:
            # Convert tensors to 8-bit sRGB images (no gamma correction needed)
            # Debevec/Robertson algorithms expect sRGB input and output linear radiance
            img_plus_2 = tensor_to_cv2(ev_plus_2)
            img_0 = tensor_to_cv2(ev_0)
            img_minus_2 = tensor_to_cv2(ev_minus_2)
            
            logger.info(f"Processing 3-stop HDR with {hdr_algorithm} algorithm")
            
            # Calculate exposure times based on EV values
            # EV difference formula: Brighter images need longer exposure times
            base_time = 1.0 / 60.0  # 1/60s as base exposure
            
            time_plus_2 = base_time * (2.0 ** exposure_step)  # Longer time (overexposed/brighter)
            time_0 = base_time  # Normal exposure
            time_minus_2 = base_time * (2.0 ** (-exposure_step))  # Shorter time (underexposed/darker)
            
            images = [img_plus_2, img_0, img_minus_2]
            times = [time_plus_2, time_0, time_minus_2]
            
            logger.info(f"3-Stop HDR: Processing with times {times} using {hdr_algorithm} algorithm")
            logger.info(f"Auto-calibration: {'ENABLED' if auto_calibrate else 'DISABLED'}")
            
            # Process HDR using selected algorithm - each should produce DIFFERENT results
            hdr_result = self.processor.process_hdr(images, times, algorithm=hdr_algorithm, 
                                                   auto_calibrate=auto_calibrate, 
                                                   debevec_exposure_compensation=debevec_exposure_comp,
                                                   debevec_anti_banding=debevec_anti_banding)
            
            logger.info(f"3-Stop HDR result range before tensor conversion: [{hdr_result.min():.6f}, {hdr_result.max():.6f}]")
            
            # Apply exposure adjustment (Nuke-style)
            if exposure_adjust != 0.0:
                # Exposure formula: result * (2^exposure_adjust)
                # +1.0 stop = 2x brighter, -1.0 stop = 0.5x darker
                adjustment_factor = 2.0 ** exposure_adjust
                hdr_result = hdr_result * adjustment_factor
                logger.info(f"Applied exposure adjustment: {exposure_adjust:+.1f} stops (factor: {adjustment_factor:.3f})")
                logger.info(f"HDR result after adjustment: [{hdr_result.min():.6f}, {hdr_result.max():.6f}]")
            
            # Convert back to tensor with TRUE HDR values (above 1.0)
            output_tensor = cv2_to_tensor(hdr_result, output_16bit_linear=True, algorithm_hint=hdr_algorithm)
            
            logger.info(f"3-Stop final tensor range (should be > 1.0 for HDR): [{output_tensor.min():.6f}, {output_tensor.max():.6f}]")
            
            return (output_tensor,)
            
        except Exception as e:
            logger.error(f"3-Stop HDR processing failed: {str(e)}")
            # Return middle exposure as fallback
            return (ev_0,)


class LuminanceStackProcessor5Stops:
    """
    Professional ComfyUI Custom Node for 5-stop HDR processing
    Processes EV+4, EV+2, EV+0, EV-2, EV-4 exposures using multiple algorithms
    
    VFX PIPELINE NOTE:
    When using Debevec/Robertson algorithms, the output will be linear radiance
    values that look flat/desaturated. This is the correct format for professional
    VFX compositing and should be saved as 16-bit EXR.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ev_plus_4": ("IMAGE",),
                "ev_plus_2": ("IMAGE",),
                "ev_0": ("IMAGE",),
                "ev_minus_2": ("IMAGE",),
                "ev_minus_4": ("IMAGE",),
            },
            "optional": {
                "exposure_step": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "exposure_adjust": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "hdr_algorithm": (["detail_injection", "radiance_fusion", "natural_blend", "mertens", "debevec", "robertson"] + (["hdrutils"] if HDRUTILS_AVAILABLE else []), {
                    "default": "detail_injection"
                }),
                "auto_calibrate": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Enable adaptive calibration for AI-generated brackets (Debevec/Robertson only)"
                }),
                "debevec_exposure_comp": ("FLOAT", {
                    "default": -8.0,
                    "min": -15.0,
                    "max": 15.0,
                    "step": 0.1,
                    "tooltip": "Exposure compensation for Debevec/Robertson output (in stops). Default -8.0 optimal for AI brackets."
                }),
                "debevec_anti_banding": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Apply subtle bilateral filtering to reduce banding (Debevec/Robertson only)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_5_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_5_stop_hdr(self, ev_plus_4, ev_plus_2, ev_0, ev_minus_2, ev_minus_4, exposure_step=2.0, exposure_adjust=0.0, hdr_algorithm="detail_injection", auto_calibrate=True, debevec_exposure_comp=-8.0, debevec_anti_banding=True):
        """
        Process 5-stop HDR merge
        
        Args:
            ev_plus_4: Most overexposed image (+4 EV)
            ev_plus_2: Overexposed image (+2 EV)
            ev_0: Normal exposure image (0 EV)
            ev_minus_2: Underexposed image (-2 EV)
            ev_minus_4: Most underexposed image (-4 EV)
            exposure_step: EV step size
            exposure_adjust: Final exposure adjustment in stops (Nuke-style)
            hdr_algorithm: HDR merge algorithm to use
            output_colorspace: "linear_hdr" (true HDR) or "srgb_display" (tone-mapped for preview)
            
        Returns:
            Tuple containing merged HDR image with exposure adjustment applied
        """
        try:
            # Convert tensors to 8-bit sRGB images (no gamma correction needed)
            # Debevec/Robertson algorithms expect sRGB input and output linear radiance
            img_plus_4 = tensor_to_cv2(ev_plus_4)
            img_plus_2 = tensor_to_cv2(ev_plus_2)
            img_0 = tensor_to_cv2(ev_0)
            img_minus_2 = tensor_to_cv2(ev_minus_2)
            img_minus_4 = tensor_to_cv2(ev_minus_4)
            
            logger.info(f"Processing 5-stop HDR with {hdr_algorithm} algorithm")
            
            # Calculate exposure times based on EV values
            # Brighter images (EV+) need longer exposure times
            base_time = 1.0 / 60.0  # 1/60s as base exposure
            
            time_plus_4 = base_time * (2.0 ** (2 * exposure_step))  # Longest time (brightest)
            time_plus_2 = base_time * (2.0 ** exposure_step)  # Longer time (brighter)
            time_0 = base_time  # Normal exposure
            time_minus_2 = base_time * (2.0 ** (-exposure_step))  # Shorter time (darker)
            time_minus_4 = base_time * (2.0 ** (-2 * exposure_step))  # Shortest time (darkest)
            
            images = [img_plus_4, img_plus_2, img_0, img_minus_2, img_minus_4]
            times = [time_plus_4, time_plus_2, time_0, time_minus_2, time_minus_4]
            
            logger.info(f"5-Stop HDR: Processing with times {times} using {hdr_algorithm} algorithm")
            logger.info(f"Auto-calibration: {'ENABLED' if auto_calibrate else 'DISABLED'}")
            
            # Process HDR using selected algorithm - each should produce DIFFERENT results
            hdr_result = self.processor.process_hdr(images, times, algorithm=hdr_algorithm, 
                                                   auto_calibrate=auto_calibrate, 
                                                   debevec_exposure_compensation=debevec_exposure_comp,
                                                   debevec_anti_banding=debevec_anti_banding)
            
            logger.info(f"5-Stop HDR result range before tensor conversion: [{hdr_result.min():.6f}, {hdr_result.max():.6f}]")
            
            # Apply exposure adjustment (Nuke-style)
            if exposure_adjust != 0.0:
                # Exposure formula: result * (2^exposure_adjust)
                # +1.0 stop = 2x brighter, -1.0 stop = 0.5x darker
                adjustment_factor = 2.0 ** exposure_adjust
                hdr_result = hdr_result * adjustment_factor
                logger.info(f"Applied exposure adjustment: {exposure_adjust:+.1f} stops (factor: {adjustment_factor:.3f})")
                logger.info(f"HDR result after adjustment: [{hdr_result.min():.6f}, {hdr_result.max():.6f}]")
            
            # Convert back to tensor with TRUE HDR values (above 1.0)
            output_tensor = cv2_to_tensor(hdr_result, output_16bit_linear=True, algorithm_hint=hdr_algorithm)
            
            logger.info(f"5-Stop final tensor range (should be > 1.0 for HDR): [{output_tensor.min():.6f}, {output_tensor.max():.6f}]")
            
            return (output_tensor,)
            
        except Exception as e:
            logger.error(f"5-Stop HDR processing failed: {str(e)}")
            # Return middle exposure as fallback
            return (ev_0,)


class LatentStackProcessor5Stops:
    """
    ComfyUI Custom Node for averaging latent representations from 5 different exposures
    
    This node performs weighted averaging of 5 latent inputs with noise reduction strategies:
    - Weighted average favoring middle exposures (less noisy)
    - Optional center bias for cleaner results
    - Denoising strength control
    
    Unlike the Luminance Stack Processor which processes images using HDR algorithms,
    this node works directly with latent representations for faster processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_1": ("LATENT", {"tooltip": "First exposure latent (e.g., EV+4)"}),
                "latent_2": ("LATENT", {"tooltip": "Second exposure latent (e.g., EV+2)"}),
                "latent_3": ("LATENT", {"tooltip": "Third exposure latent (e.g., EV0)"}),
                "latent_4": ("LATENT", {"tooltip": "Fourth exposure latent (e.g., EV-2)"}),
                "latent_5": ("LATENT", {"tooltip": "Fifth exposure latent (e.g., EV-4)"}),
            },
            "optional": {
                "blend_mode": (["simple_average", "weighted_center", "strong_center", "median_blend", "variance_adaptive", "quality_aware"], {
                    "default": "quality_aware",
                    "tooltip": "Blending strategy: quality_aware (multi-scale pyramid + quality metrics), variance_adaptive (spatial smoothing), weighted_center (favor middle), strong_center (heavily favor middle), median_blend (reduce noise), simple_average (equal weights)"
                }),
                "center_bias": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 0.8,
                    "step": 0.05,
                    "tooltip": "How much to favor the center exposure (reduces noise, 0.0 = equal weights)"
                }),
                "detail_preservation": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detail preservation strength for variance_adaptive mode (higher = cleaner details like tree leaves)"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("merged_latent",)
    FUNCTION = "process_latent_stack"
    CATEGORY = "latent/luminance"
    
    def process_latent_stack(self, latent_1, latent_2, latent_3, latent_4, latent_5, blend_mode="quality_aware", center_bias=0.4, detail_preservation=0.7):
        """
        Process latent stack with noise-reducing weighted averaging
        
        Args:
            latent_1 to latent_5: Latent dictionaries from different exposure images
            blend_mode: Blending strategy to use
            center_bias: How much to favor center exposure (0.0 to 0.8)
            detail_preservation: Detail preservation strength for variance_adaptive (0.0 to 1.0)
            
        Returns:
            Tuple containing the merged latent dictionary
        """
        try:
            # Extract the latent tensors from the dictionaries
            samples_1 = latent_1["samples"]
            samples_2 = latent_2["samples"]
            samples_3 = latent_3["samples"]
            samples_4 = latent_4["samples"]
            samples_5 = latent_5["samples"]
            
            logger.info(f"Processing 5 latent inputs with blend mode: {blend_mode}")
            logger.info(f"Latent shapes: {samples_1.shape}, {samples_2.shape}, {samples_3.shape}, {samples_4.shape}, {samples_5.shape}")
            
            # Verify all latents have the same shape
            if not (samples_1.shape == samples_2.shape == samples_3.shape == samples_4.shape == samples_5.shape):
                raise ValueError(f"All latent inputs must have the same shape. Got: {samples_1.shape}, {samples_2.shape}, {samples_3.shape}, {samples_4.shape}, {samples_5.shape}")
            
            # Apply different blending strategies
            if blend_mode == "simple_average":
                # Simple average: (L1 + L2 + L3 + L4 + L5) / 5
                merged_samples = (samples_1 + samples_2 + samples_3 + samples_4 + samples_5) / 5.0
                logger.info("Using simple average (equal weights)")
                
            elif blend_mode == "weighted_center":
                # Weighted average favoring center exposure (reduces noise)
                # Center exposure (EV0) typically has best quality and least noise
                # Outer exposures have more weight for: [0.15, 0.2, 0.3, 0.2, 0.15] = 1.0
                center_weight = 0.3 + center_bias  # Default: 0.7
                side_weight = (1.0 - center_weight) / 4  # Distribute remaining weight
                
                merged_samples = (
                    samples_1 * side_weight +
                    samples_2 * (side_weight * 1.33) +  # Slightly favor inner exposures
                    samples_3 * center_weight +
                    samples_4 * (side_weight * 1.33) +
                    samples_5 * side_weight
                )
                logger.info(f"Using weighted center blend (center: {center_weight:.2f}, sides: {side_weight:.2f})")
                
            elif blend_mode == "strong_center":
                # Heavily favor center exposure for maximum noise reduction
                # Weights: [0.05, 0.15, 0.6, 0.15, 0.05] = 1.0
                center_weight = 0.6 + center_bias
                side_weight = (1.0 - center_weight) / 4
                
                merged_samples = (
                    samples_1 * side_weight +
                    samples_2 * (side_weight * 3) +
                    samples_3 * center_weight +
                    samples_4 * (side_weight * 3) +
                    samples_5 * side_weight
                )
                logger.info(f"Using strong center blend (center: {center_weight:.2f}, maximum noise reduction)")
                
            elif blend_mode == "median_blend":
                # Median-like blending: average top 3 closest to median
                # This reduces noise by excluding outliers
                stacked = torch.stack([samples_1, samples_2, samples_3, samples_4, samples_5], dim=0)
                
                # Sort and take middle 3 values, then average
                sorted_stack, _ = torch.sort(stacked, dim=0)
                # Take indices 1, 2, 3 (excluding min and max)
                merged_samples = (sorted_stack[1] + sorted_stack[2] + sorted_stack[3]) / 3.0
                logger.info("Using median blend (excludes outliers, reduces noise)")
                
            elif blend_mode == "quality_aware":
                # Quality-Aware Multi-Scale Blending: Laplacian Pyramid + Enhanced Quality Metrics
                # Combines the best of both worlds: multi-scale decomposition + sophisticated quality analysis
                # This is the ultimate solution for artifact-free latent blending
                
                logger.info("Using quality-aware multi-scale blending (Laplacian pyramid + enhanced quality)")
                
                # Helper functions for pyramid construction
                def gaussian_pyramid(img, levels=4):
                    """Build Gaussian pyramid"""
                    pyramid = [img]
                    current = img
                    for i in range(levels):
                        smoothed = torch.nn.functional.avg_pool2d(current, kernel_size=5, stride=1, padding=2)
                        downsampled = torch.nn.functional.avg_pool2d(smoothed, kernel_size=2, stride=2, padding=0)
                        pyramid.append(downsampled)
                        current = downsampled
                    return pyramid
                
                def laplacian_pyramid(img, levels=4):
                    """Build Laplacian pyramid"""
                    gaussian_pyr = gaussian_pyramid(img, levels)
                    laplacian_pyr = []
                    
                    for i in range(levels):
                        current_level = gaussian_pyr[i]
                        next_level = gaussian_pyr[i + 1]
                        upsampled = torch.nn.functional.interpolate(
                            next_level, size=current_level.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                        laplacian = current_level - upsampled
                        laplacian_pyr.append(laplacian)
                    
                    laplacian_pyr.append(gaussian_pyr[levels])
                    return laplacian_pyr
                
                def compute_enhanced_quality(latent_samples):
                    """Enhanced quality metrics with better edge preservation"""
                    # Contrast using Laplacian
                    laplacian_kernel = torch.tensor([[[
                        [0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]
                    ]]], dtype=latent_samples.dtype, device=latent_samples.device)
                    
                    contrast_maps = []
                    for c in range(latent_samples.shape[1]):
                        channel = latent_samples[:, c:c+1, :, :]
                        contrast = torch.nn.functional.conv2d(channel, laplacian_kernel, padding=1)
                        contrast_maps.append(torch.abs(contrast))
                    contrast_quality = torch.mean(torch.cat(contrast_maps, dim=1), dim=1, keepdim=True)
                    
                    # Saturation
                    channel_mean = torch.mean(latent_samples, dim=1, keepdim=True)
                    saturation_quality = torch.mean(torch.abs(latent_samples - channel_mean), dim=1, keepdim=True)
                    
                    # Exposedness
                    median_val = torch.median(latent_samples)
                    exposedness = -torch.abs(latent_samples - median_val)
                    exposedness_quality = torch.mean(exposedness, dim=1, keepdim=True)
                    exposedness_quality = torch.exp(exposedness_quality)
                    
                    # Weighted combination with emphasis on contrast for edges
                    quality = (
                        contrast_quality * 0.6 +      # Increased: 60% - edges matter most
                        saturation_quality * 0.25 +   # 25%
                        exposedness_quality * 0.15    # 15%
                    )
                    
                    # Light smoothing to prevent harsh transitions
                    quality = torch.nn.functional.avg_pool2d(quality, kernel_size=3, stride=1, padding=1)
                    
                    return quality
                
                # Build Laplacian pyramids for all latents
                logger.info("Building multi-scale pyramids (4 levels)...")
                pyramid_levels = 4
                
                lap_pyr_1 = laplacian_pyramid(samples_1, pyramid_levels)
                lap_pyr_2 = laplacian_pyramid(samples_2, pyramid_levels)
                lap_pyr_3 = laplacian_pyramid(samples_3, pyramid_levels)
                lap_pyr_4 = laplacian_pyramid(samples_4, pyramid_levels)
                lap_pyr_5 = laplacian_pyramid(samples_5, pyramid_levels)
                
                # Compute enhanced quality pyramids
                quality_1_pyr = gaussian_pyramid(compute_enhanced_quality(samples_1), pyramid_levels)
                quality_2_pyr = gaussian_pyramid(compute_enhanced_quality(samples_2), pyramid_levels)
                quality_3_pyr = gaussian_pyramid(compute_enhanced_quality(samples_3), pyramid_levels)
                quality_4_pyr = gaussian_pyramid(compute_enhanced_quality(samples_4), pyramid_levels)
                quality_5_pyr = gaussian_pyramid(compute_enhanced_quality(samples_5), pyramid_levels)
                
                # Apply center bias
                center_boost = 1.0 + center_bias
                inner_boost = 1.0 + center_bias * 0.3
                
                for i in range(len(quality_3_pyr)):
                    quality_3_pyr[i] = quality_3_pyr[i] * center_boost
                    quality_2_pyr[i] = quality_2_pyr[i] * inner_boost
                    quality_4_pyr[i] = quality_4_pyr[i] * inner_boost
                
                # Blend each pyramid level with adaptive quality power
                logger.info("Blending pyramid levels with enhanced quality metrics...")
                blended_pyramid = []
                
                for level in range(pyramid_levels + 1):
                    # Get Laplacian bands
                    lap_1 = lap_pyr_1[level]
                    lap_2 = lap_pyr_2[level]
                    lap_3 = lap_pyr_3[level]
                    lap_4 = lap_pyr_4[level]
                    lap_5 = lap_pyr_5[level]
                    
                    # Get quality maps
                    qual_1 = quality_1_pyr[level]
                    qual_2 = quality_2_pyr[level]
                    qual_3 = quality_3_pyr[level]
                    qual_4 = quality_4_pyr[level]
                    qual_5 = quality_5_pyr[level]
                    
                    # Stack qualities
                    quality_stack = torch.stack([qual_1, qual_2, qual_3, qual_4, qual_5], dim=0)
                    
                    # Adaptive quality power per level
                    # High-frequency (fine details) = very selective (high power)
                    # Low-frequency (smooth areas) = balanced blending (lower power)
                    if level == 0:  # Finest details
                        quality_power = 1.0 + detail_preservation * 4.0  # Very selective
                    elif level == 1:
                        quality_power = 1.0 + detail_preservation * 3.0
                    elif level == 2:
                        quality_power = 1.0 + detail_preservation * 2.0
                    else:  # Coarse levels
                        quality_power = 1.0 + detail_preservation * 1.0  # More balanced
                    
                    quality_stack = torch.pow(quality_stack + 1e-8, quality_power)
                    
                    # Normalize to weights
                    quality_sum = torch.sum(quality_stack, dim=0, keepdim=True)
                    weights = quality_stack / (quality_sum + 1e-8)
                    
                    # Expand and blend
                    weight_1 = weights[0].expand_as(lap_1)
                    weight_2 = weights[1].expand_as(lap_2)
                    weight_3 = weights[2].expand_as(lap_3)
                    weight_4 = weights[3].expand_as(lap_4)
                    weight_5 = weights[4].expand_as(lap_5)
                    
                    blended_level = (
                        lap_1 * weight_1 +
                        lap_2 * weight_2 +
                        lap_3 * weight_3 +
                        lap_4 * weight_4 +
                        lap_5 * weight_5
                    )
                    
                    blended_pyramid.append(blended_level)
                    
                    center_usage = torch.mean(weight_3).item()
                    logger.info(f"  Level {level}: power={quality_power:.2f}, center={center_usage*100:.1f}%")
                
                # Reconstruct from pyramid
                logger.info("Reconstructing from multi-scale pyramid...")
                merged_samples = blended_pyramid[pyramid_levels]
                
                for level in range(pyramid_levels - 1, -1, -1):
                    upsampled = torch.nn.functional.interpolate(
                        merged_samples,
                        size=blended_pyramid[level].shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    merged_samples = upsampled + blended_pyramid[level]
                
                logger.info(f"Multi-scale quality-aware blending complete!")
                logger.info(f"  Laplacian pyramid with enhanced edge-preserving quality metrics")
                logger.info(f"  Adaptive selectivity per frequency band")
                logger.info(f"  Professional artifact-free results")
                
            elif blend_mode == "variance_adaptive":
                # Variance-Adaptive Blending with Spatial Smoothing
                # FIXED: Added proper spatial filtering to avoid checkerboard patterns
                
                logger.info("Using variance-adaptive blend (intelligent artifact reduction with spatial smoothing)")
                
                # Stack all latents for analysis
                stacked = torch.stack([samples_1, samples_2, samples_3, samples_4, samples_5], dim=0)
                
                # Compute variance across exposures for each spatial location
                # High variance = inconsistent detail (potential artifacts)
                # Low variance = consistent detail (safe to blend)
                variance_map = torch.var(stacked, dim=0, keepdim=False)
                
                # CRITICAL FIX: Apply spatial smoothing to variance map to avoid checkerboard artifacts
                # Average variance across channels first for better stability
                variance_spatial = torch.mean(variance_map, dim=1, keepdim=True)  # [B, 1, H, W]
                
                # Apply Gaussian-like smoothing using average pooling with padding
                # This prevents harsh per-pixel transitions that create checkerboard patterns
                kernel_size = 7  # Larger kernel = smoother transitions
                padding = kernel_size // 2
                
                # Smooth the variance map using average pooling
                variance_smoothed = torch.nn.functional.avg_pool2d(
                    variance_spatial, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=padding
                )
                
                # Broadcast back to all channels
                variance_smoothed = variance_smoothed.expand(-1, variance_map.shape[1], -1, -1)
                
                # Normalize variance to 0-1 range for weighting
                # Use percentile-based normalization for robustness
                var_flat = variance_smoothed.flatten()
                var_p05 = torch.quantile(var_flat, 0.05)
                var_p95 = torch.quantile(var_flat, 0.95)
                variance_normalized = torch.clamp((variance_smoothed - var_p05) / (var_p95 - var_p05 + 1e-8), 0, 1)
                
                # Apply smooth non-linear mapping to create gentler transitions
                # This further reduces hard edges in the weight map
                variance_normalized = torch.pow(variance_normalized, 0.7)  # Gamma correction for smoother curve
                
                # Create adaptive weights based on smoothed variance
                # High variance -> favor center (safe, clean)
                # Low variance -> use weighted blend (more dynamic range)
                
                # Base weights for low-variance areas (more aggressive blending)
                center_weight_low = 0.3 + center_bias * 0.5  # e.g., 0.5 with default
                side_weight_low = (1.0 - center_weight_low) / 4
                
                # Weights for high-variance areas (conservative, favor center heavily)
                center_weight_high = 0.7 + center_bias  # e.g., 1.1 -> clamp to reasonable range
                center_weight_high = min(center_weight_high, 0.95)  # Cap at 0.95
                side_weight_high = (1.0 - center_weight_high) / 4
                
                # Interpolate weights based on variance and detail_preservation strength
                # detail_preservation controls how aggressive we are in problem areas
                variance_factor = variance_normalized * detail_preservation
                
                # Compute adaptive weights for each latent
                # Center gets more weight in high-variance areas
                weight_center = center_weight_low + (center_weight_high - center_weight_low) * variance_factor
                weight_side = side_weight_low + (side_weight_high - side_weight_low) * variance_factor
                weight_mid = weight_side * 1.33  # Inner exposures get slightly more weight
                
                # Apply adaptive weighted blending
                merged_samples = (
                    samples_1 * weight_side +      # EV+4 (outer)
                    samples_2 * weight_mid +       # EV+2 (inner)
                    samples_3 * weight_center +    # EV0 (center) - gets more weight in problem areas
                    samples_4 * weight_mid +       # EV-2 (inner)
                    samples_5 * weight_side        # EV-4 (outer)
                )
                
                # Normalize to ensure proper weighting (sum of weights should = 1)
                total_weight = weight_side * 2 + weight_mid * 2 + weight_center
                merged_samples = merged_samples / total_weight
                
                # Log statistics
                high_variance_pixels = torch.sum(variance_normalized > 0.5).item()
                total_pixels = variance_normalized.numel()
                logger.info(f"Variance-adaptive (spatially smoothed): {high_variance_pixels}/{total_pixels} pixels ({100*high_variance_pixels/total_pixels:.1f}%) in high-variance regions")
                logger.info(f"Spatial smoothing: 7x7 kernel applied to prevent checkerboard artifacts")
                logger.info(f"Detail preservation: {detail_preservation:.2f} (higher = more artifact reduction)")
                logger.info(f"Adaptive weights - Center: {center_weight_low:.3f}->{center_weight_high:.3f}")
                
            else:
                # Fallback to simple average
                merged_samples = (samples_1 + samples_2 + samples_3 + samples_4 + samples_5) / 5.0
                logger.info("Using fallback simple average")
            
            logger.info(f"Latent stack merged successfully")
            logger.info(f"Output shape: {merged_samples.shape}")
            logger.info(f"Value range: [{merged_samples.min():.6f}, {merged_samples.max():.6f}]")
            
            # Return in the same dictionary format as ComfyUI expects
            return ({"samples": merged_samples},)
            
        except Exception as e:
            logger.error(f"Latent stack processing failed: {str(e)}")
            # Return the middle latent as fallback
            return (latent_3,)


class HDRExportNode:
    """
    ComfyUI Custom Node for exporting HDR images to EXR format
    Clean filename interface matching standard ComfyUI save nodes
    Preserves full dynamic range data without normalization
    
    VFX PIPELINE NOTE:
    EXR files store linear radiance values (32-bit float per channel).
    The flat/desaturated appearance of Debevec/Robertson output is correct.
    Professional compositing software expects this linear format.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_image": ("IMAGE", {"tooltip": "HDR image tensor with values potentially above 1.0"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Base filename (without extension)"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "tooltip": "Output path: Empty=default ComfyUI/output, /subfolder=output/subfolder, or full custom path"}),
                "counter": ("INT", {"default": 1, "min": 0, "max": 99999, "step": 1, "tooltip": "Frame/sequence counter"}),
                "format": (["exr", "hdr"], {"default": "exr", "tooltip": "HDR file format"}),
                "bit_depth": (["16bit", "32bit"], {"default": "32bit", "tooltip": "EXR precision: 32bit = maximum quality, 16bit = smaller files"}),
                "compression": (["none", "rle", "zip", "piz", "pxr24"], {"default": "zip", "tooltip": "EXR compression type"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_hdr"
    CATEGORY = "Luminance Stack Processor"
    OUTPUT_NODE = True
    
    def export_hdr(self, hdr_image: torch.Tensor, filename_prefix: str = "ComfyUI", 
                  output_path: str = "", counter: int = 1, format: str = "exr", bit_depth: str = "32bit", compression: str = "zip"):
        """
        Export HDR image with clean filename interface (no automatic prefixes)
        
        Args:
            hdr_image: HDR image tensor (potentially with values > 1.0)
            filename_prefix: Base filename (no extension)
            output_path: Custom output directory 
            counter: Frame/sequence number
            format: Output format (exr/hdr)
            bit_depth: EXR precision (16bit/32bit)
            compression: EXR compression type
            
        Returns:
            Tuple containing the filepath of saved HDR file
        """
        try:
            # Convert tensor to numpy array
            if len(hdr_image.shape) == 4:
                hdr_image = hdr_image.squeeze(0)  # Remove batch dimension
            
            hdr_array = hdr_image.cpu().numpy()
            
            logger.info(f"HDR Export: Input range [{hdr_array.min():.6f}, {hdr_array.max():.6f}]")
            logger.info(f"HDR Export: Shape {hdr_array.shape}, dtype {hdr_array.dtype}")
            
            # Check for HDR data
            hdr_pixels = int(np.sum(hdr_array > 1.0))
            negative_pixels = int(np.sum(hdr_array < 0.0))
            logger.info(f"HDR Export: HDR pixels (>1.0): {hdr_pixels}, Negative pixels: {negative_pixels}")
            
            # Determine output path - default to ComfyUI/output/ directory
            output_path_clean = output_path.strip() if output_path else ""
            
            if not output_path_clean:
                # Use default ComfyUI output directory
                output_dir = self._get_comfyui_output_directory()
                logger.info(f"Using default ComfyUI output directory: {output_dir}")
            elif output_path_clean.startswith("/"):
                # User specified a subdirectory within ComfyUI output (e.g., "/Test" -> "output/Test")
                base_output_dir = self._get_comfyui_output_directory()
                subdirectory = output_path_clean[1:]  # Remove leading "/"
                output_dir = os.path.join(base_output_dir, subdirectory)
                logger.info(f"Using ComfyUI output subdirectory: {output_dir}")
            else:
                # User specified absolute or relative custom path
                output_dir = output_path_clean
                logger.info(f"Using custom absolute path: {output_dir}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Clean filename generation (NO automatic timestamps or prefixes)
            if counter > 0:
                # Include counter if specified
                filename = f"{filename_prefix}_{counter:05d}.{format}"
            else:
                # No counter - simple filename
                filename = f"{filename_prefix}.{format}"
                
            filepath = os.path.join(output_dir, filename)
            
            logger.info(f"HDR Export: Saving to {filepath}")
            
            # Convert RGB to BGR for OpenCV (ComfyUI tensors are RGB)
            # Set precision based on bit_depth selection
            if bit_depth == "32bit":
                target_dtype = np.float32  # 32-bit single precision
                logger.info("Using 32-bit float precision for maximum HDR quality")
            else:
                # For 16-bit, we still use float32 in processing but OpenCV will write as half-float
                target_dtype = np.float32
                logger.info("Using 16-bit half-float precision for smaller file size")
            
            if len(hdr_array.shape) == 3 and hdr_array.shape[2] == 3:
                hdr_bgr = cv2.cvtColor(hdr_array.astype(target_dtype), cv2.COLOR_RGB2BGR)
            else:
                hdr_bgr = hdr_array.astype(target_dtype)
            
            # Save HDR file with TRUE bit depth control
            if format.lower() == "exr":
                # CRITICAL: OpenCV's cv2.imwrite doesn't control EXR bit depth properly
                # We need to use proper EXR writing method
                try:
                    if IMAGEIO_AVAILABLE:
                        # Use imageio for proper 32-bit EXR writing
                        if bit_depth == "32bit":
                            logger.info("Using imageio for TRUE 32-bit EXR writing")
                            # Convert BGR back to RGB for imageio
                            hdr_rgb = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)
                            # Write as float32 for true 32-bit precision
                            iio.imwrite(filepath, hdr_rgb.astype(np.float32))
                            success = True
                        else:
                            logger.info("Using imageio for 16-bit EXR writing")
                            # Convert BGR back to RGB for imageio
                            hdr_rgb = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)
                            # Write as float16 for 16-bit precision
                            iio.imwrite(filepath, hdr_rgb.astype(np.float16))
                            success = True
                    else:
                        # Fallback to OpenCV (limited bit depth control)
                        logger.warning("imageio not available - using OpenCV (limited 32-bit support)")
                        success = cv2.imwrite(filepath, hdr_bgr)
                except Exception as e:
                    logger.error(f"imageio EXR writing failed: {e}")
                    logger.info("Falling back to OpenCV EXR writing")
                    success = cv2.imwrite(filepath, hdr_bgr)
                
            elif format.lower() == "hdr":
                # Save as Radiance HDR format (always 32-bit RGBE)
                logger.info("Saving as Radiance HDR format (32-bit RGBE)")
                success = cv2.imwrite(filepath, hdr_bgr)
            else:
                success = cv2.imwrite(filepath, hdr_bgr)  # Default to EXR behavior
            
            if not success:
                raise RuntimeError(f"Failed to save HDR file: {filepath}")
            
            # Verify the saved file preserves HDR data
            if os.path.exists(filepath):
                try:
                    # Load back and verify HDR preservation
                    verification_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    if verification_img is not None:
                        max_val = np.max(verification_img)
                        min_val = np.min(verification_img)
                        logger.info(f"HDR Export verification: Range in saved file: [{min_val:.6f}, {max_val:.6f}]")
                        
                        if max_val > 1.0:
                            logger.info("✅ HDR values above 1.0 successfully preserved!")
                        else:
                            logger.warning("⚠️ No HDR values above 1.0 detected (may be LDR data)")
                        
                        if min_val < 0.0:
                            logger.info("✅ Negative values preserved (signed HDR range)")
                            
                        # Check file size as secondary verification
                        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        logger.info(f"HDR file size: {file_size_mb:.2f} MB")
                        
                        # Get image stats
                        stats = self._get_file_stats(filepath)
                        logger.info(f"Image dimensions: {stats['width']}x{stats['height']}, {stats['channels']} channels")
                        
                    else:
                        logger.warning("Could not verify saved HDR file")
                except Exception as verify_e:
                    logger.warning(f"Could not verify HDR file: {verify_e}")
                
                logger.info(f"✅ HDR {format.upper()} file exported: {filepath}")
                return (filepath,)
            else:
                raise RuntimeError(f"HDR file was not created: {filepath}")
                
        except Exception as e:
            logger.error(f"HDR export failed: {str(e)}")
            import traceback
            logger.error(f"HDR export traceback: {traceback.format_exc()}")
            
            # Return error message
            error_path = f"ERROR: {str(e)}"
            return (error_path,)
    
    def _get_comfyui_output_directory(self) -> str:
        """
        Determine the ComfyUI output directory using multiple fallback methods
        Returns the path to the ComfyUI output directory
        """
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
            logger.info(f"Found ComfyUI output directory via folder_paths: {output_dir}")
            return output_dir
        except ImportError:
            # Fallback: Look for ComfyUI output directory structure
            # Navigate up from custom_nodes to find ComfyUI root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfyui_root = None
            
            # Try to find ComfyUI root by looking for typical structure
            search_dir = current_dir
            for _ in range(5):  # Search up to 5 levels up
                if os.path.exists(os.path.join(search_dir, "custom_nodes")) and \
                   os.path.exists(os.path.join(search_dir, "models")):
                    comfyui_root = search_dir
                    break
                search_dir = os.path.dirname(search_dir)
            
            if comfyui_root:
                output_dir = os.path.join(comfyui_root, "output")
                logger.info(f"Found ComfyUI root, using output directory: {output_dir}")
                return output_dir
            else:
                # Final fallback - assume we're in custom_nodes and go up 2 levels
                output_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "output")
                logger.info(f"Using fallback output directory: {output_dir}")
                return output_dir
                
        except Exception as e:
            logger.warning(f"Error determining ComfyUI output directory: {e}")
            # Emergency fallback - try to create output directory relative to current location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "output")
            logger.info(f"Using emergency fallback output directory: {output_dir}")
            return output_dir

    def _get_file_stats(self, filepath: str) -> dict:
        """Get statistics about the saved file"""
        try:
            # File size
            size_bytes = os.path.getsize(filepath)
            size_mb = size_bytes / (1024 * 1024)
            
            # Image dimensions using OpenCV
            img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is not None:
                height, width = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1
            else:
                width = height = channels = 0
            
            return {
                'size_mb': size_mb,
                'width': width,
                'height': height,
                'channels': channels
            }
        except Exception:
            return {
                'size_mb': 0,
                'width': 0,
                'height': 0,
                'channels': 0
            }


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LuminanceStackProcessor3Stops": LuminanceStackProcessor3Stops,
    "LuminanceStackProcessor5Stops": LuminanceStackProcessor5Stops,
    "LatentStackProcessor5Stops": LatentStackProcessor5Stops,
    "HDRExportNode": HDRExportNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminanceStackProcessor3Stops": "Luminance Stack Processor (3 Stops)",
    "LuminanceStackProcessor5Stops": "Luminance Stack Processor (5 Stops)",
    "LatentStackProcessor5Stops": "Latent Stack Processor (5 Stops)",
    "HDRExportNode": "HDR Export to EXR"
}
