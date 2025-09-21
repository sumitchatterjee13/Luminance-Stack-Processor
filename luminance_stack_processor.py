"""
Luminance Stack Processor - Professional ComfyUI Custom Nodes
Implements HDR processing using the Debevec Algorithm for multiple exposure fusion

Author: Sumit Chatterjee
  Version: 1.0.3
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
            
        elif algorithm_hint == "mertens":
            # Mertens: Medium HDR range, values 1-10 range
            p85 = np.percentile(hdr_image, 85.0)
            if p85 > 0:
                hdr_linear = hdr_image * (3.0 / p85)
            else:
                hdr_linear = hdr_image * 3.0
            hdr_linear = np.clip(hdr_linear, 0.0, 12.0)
            
        elif algorithm_hint in ["debevec", "robertson"]:
            # VFX PIPELINE: Raw linear radiance values
            # IMPORTANT: The flat/desaturated appearance is CORRECT for professional VFX
            # This is linear radiance data meant for compositing, not direct viewing
            
            logger.info(f"  Processing VFX linear radiance - flat appearance is CORRECT for compositing")
            
            # VFX STANDARD: Minimal scaling for professional flat log appearance
            if hdr_image.max() > 0:
                # Research-based scaling: Use median (50th percentile) as reference
                # This creates the proper flat log appearance VFX artists expect
                p50 = np.percentile(hdr_image, 50.0)  # Middle gray reference
                
                if p50 > 0:
                    # Scale to VFX standard: 18% gray = 0.18 (professional standard)
                    # This creates the flat, log-like appearance
                    scale_factor = 0.18 / p50
                    hdr_linear = hdr_image * scale_factor
                    
                    # VFX RANGE: Allow wide dynamic range (no aggressive clipping)
                    # Professional VFX needs values up to 100+ for bright sources
                    hdr_linear = np.clip(hdr_linear, 0.0, 2000.0)  # Wide range for VFX work
                    
                    logger.info(f"  VFX scaling applied: middle gray -> 0.18 (scale: {scale_factor:.3f})")
                else:
                    hdr_linear = hdr_image
                    logger.info(f"  No scaling needed - preserving original values")
            else:
                hdr_linear = hdr_image
                logger.info(f"  Zero image detected - no processing applied")
                
            logger.info(f"  VFX FLAT LOG: Appearance will be flat/desaturated - this is PROFESSIONAL STANDARD")
            
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
        
    def process_hdr(self, images: List[np.ndarray], exposure_times: List[float], algorithm: str = "natural_blend") -> np.ndarray:
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
                 - "natural_blend": Preserves EV0 look with enhanced range  
                 - "mertens": Exposure fusion (display-ready)
                 - "debevec": True HDR recovery (flat/linear for VFX)
                 - "robertson": Alternative HDR recovery (flat/linear for VFX)
            
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
                
            elif algorithm == "hdrutils" and HDRUTILS_AVAILABLE:
                # Use HDRutils library for HDR merging
                logger.info("Using HDRutils library for HDR merging...")
                hdr_radiance = self._merge_with_hdrutils(processed_images, times)
                
            elif algorithm == "robertson":
                # Robertson algorithm - VFX pipeline style (raw linear radiance)
                logger.info("Using Robertson algorithm - VFX pipeline mode...")
                response = self.calibrator_robertson.process(processed_images, times)
                hdr_radiance = self.merge_robertson.process(processed_images, times, response)
                
                logger.info(f"Robertson raw radiance range: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                # VFX PIPELINE: NO TONE MAPPING! Keep raw linear radiance
                if hdr_radiance.max() > 1000.0:  # Only if extremely bright values
                    scale_factor = 100.0 / np.percentile(hdr_radiance, 99.5)
                    hdr_radiance = hdr_radiance * scale_factor
                    logger.info(f"Applied minimal scaling for numerical stability: {scale_factor:.3f}")
                
                logger.info(f"Robertson VFX output range: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
            else:  # Default to Debevec - VFX pipeline style (raw linear radiance)
                # Estimate camera response function using Debevec method
                logger.info("Using Debevec algorithm - VFX pipeline mode (raw linear radiance)...")
                response = self.calibrator.process(processed_images, times)
                logger.info(f"Response function shape: {response.shape}")
                
                # Merge images into HDR using Debevec algorithm
                # Debevec outputs linear radiance values (already in linear space)
                hdr_radiance = self.merge_debevec.process(processed_images, times, response)
                
                logger.info(f"Debevec raw radiance range: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                # NOTE: OpenCV's Debevec sometimes has issues with color
                # Check if values are reasonable
                if hdr_radiance.mean() < 0.001:
                    logger.warning("Debevec produced very dark image, applying recovery")
                    # Try to recover by scaling
                    hdr_radiance = hdr_radiance * 100.0
                elif hdr_radiance.mean() > 100.0:
                    logger.warning("Debevec produced extremely bright image, scaling down")
                    hdr_radiance = hdr_radiance / 100.0
                
                # Apply minimal scaling for VFX pipeline
                if hdr_radiance.max() > 1000.0:
                    scale_factor = 100.0 / np.percentile(hdr_radiance, 99.5)
                    hdr_radiance = hdr_radiance * scale_factor
                    logger.info(f"Applied minimal scaling for numerical stability: {scale_factor:.3f}")
                
                logger.info(f"Debevec VFX output (linear radiance): [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
            
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
                    "default": -1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "hdr_algorithm": (["radiance_fusion", "natural_blend", "mertens", "debevec", "robertson"] + (["hdrutils"] if HDRUTILS_AVAILABLE else []), {
                    "default": "radiance_fusion"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_3_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_3_stop_hdr(self, ev_plus_2, ev_0, ev_minus_2, exposure_step=2.0, exposure_adjust=-1.0, hdr_algorithm="radiance_fusion"):
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
            # EV difference formula: time = base_time * 2^(-EV_difference)
            base_time = 1.0 / 60.0  # 1/60s as base exposure
            
            time_plus_2 = base_time * (2.0 ** (-exposure_step))  # Shorter time (overexposed)
            time_0 = base_time  # Normal exposure
            time_minus_2 = base_time * (2.0 ** exposure_step)  # Longer time (underexposed)
            
            images = [img_plus_2, img_0, img_minus_2]
            times = [time_plus_2, time_0, time_minus_2]
            
            logger.info(f"3-Stop HDR: Processing with times {times} using {hdr_algorithm} algorithm")
            
            # Process HDR using selected algorithm - each should produce DIFFERENT results
            hdr_result = self.processor.process_hdr(images, times, algorithm=hdr_algorithm)
            
            logger.info(f"3-Stop HDR result range before tensor conversion: [{hdr_result.min():.6f}, {hdr_result.max():.6f}]")
            
            # Apply exposure adjustment (Nuke-style)
            if exposure_adjust != 0.0:
                # Standard exposure formula: result * (2^(-exposure_adjust))
                adjustment_factor = 2.0 ** (-exposure_adjust)
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
                    "default": -1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "hdr_algorithm": (["radiance_fusion", "natural_blend", "mertens", "debevec", "robertson"] + (["hdrutils"] if HDRUTILS_AVAILABLE else []), {
                    "default": "radiance_fusion"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_5_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_5_stop_hdr(self, ev_plus_4, ev_plus_2, ev_0, ev_minus_2, ev_minus_4, exposure_step=2.0, exposure_adjust=-1.0, hdr_algorithm="radiance_fusion"):
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
            base_time = 1.0 / 60.0  # 1/60s as base exposure
            
            time_plus_4 = base_time * (2.0 ** (-2 * exposure_step))
            time_plus_2 = base_time * (2.0 ** (-exposure_step))
            time_0 = base_time
            time_minus_2 = base_time * (2.0 ** exposure_step)
            time_minus_4 = base_time * (2.0 ** (2 * exposure_step))
            
            images = [img_plus_4, img_plus_2, img_0, img_minus_2, img_minus_4]
            times = [time_plus_4, time_plus_2, time_0, time_minus_2, time_minus_4]
            
            logger.info(f"5-Stop HDR: Processing with times {times} using {hdr_algorithm} algorithm")
            
            # Process HDR using selected algorithm - each should produce DIFFERENT results
            hdr_result = self.processor.process_hdr(images, times, algorithm=hdr_algorithm)
            
            logger.info(f"5-Stop HDR result range before tensor conversion: [{hdr_result.min():.6f}, {hdr_result.max():.6f}]")
            
            # Apply exposure adjustment (Nuke-style)
            if exposure_adjust != 0.0:
                # Standard exposure formula: result * (2^(-exposure_adjust))
                adjustment_factor = 2.0 ** (-exposure_adjust)
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
                "output_path": ("STRING", {"default": "", "tooltip": "Custom output directory (leave empty for ComfyUI output folder)"}),
                "counter": ("INT", {"default": 1, "min": 0, "max": 99999, "step": 1, "tooltip": "Frame/sequence counter"}),
                "format": (["exr", "hdr"], {"default": "exr", "tooltip": "HDR file format"}),
                "compression": (["none", "rle", "zip", "piz", "pxr24"], {"default": "zip", "tooltip": "EXR compression type"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_hdr"
    CATEGORY = "Luminance Stack Processor"
    OUTPUT_NODE = True
    
    def export_hdr(self, hdr_image: torch.Tensor, filename_prefix: str = "ComfyUI", 
                  output_path: str = "", counter: int = 1, format: str = "exr", compression: str = "zip"):
        """
        Export HDR image with clean filename interface (no automatic prefixes)
        
        Args:
            hdr_image: HDR image tensor (potentially with values > 1.0)
            filename_prefix: Base filename (no extension)
            output_path: Custom output directory 
            counter: Frame/sequence number
            format: Output format (exr/hdr)
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
            
            # Determine output path
            if not output_path or output_path.strip() == "":
                # Use ComfyUI's output directory
                try:
                    import folder_paths
                    output_dir = folder_paths.get_output_directory()
                except ImportError:
                    # Fallback if folder_paths module doesn't exist
                    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
                except Exception:
                    # Secondary fallback
                    output_dir = os.path.join(os.getcwd(), "output")
            else:
                output_dir = output_path.strip()
            
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
            if len(hdr_array.shape) == 3 and hdr_array.shape[2] == 3:
                hdr_bgr = cv2.cvtColor(hdr_array.astype(np.float32), cv2.COLOR_RGB2BGR)
            else:
                hdr_bgr = hdr_array.astype(np.float32)
            
            # Save HDR file based on format
            if format.lower() == "exr":
                # Save as EXR with specified compression
                # Note: OpenCV's EXR compression options may be limited
                success = cv2.imwrite(filepath, hdr_bgr)
            elif format.lower() == "hdr":
                # Save as Radiance HDR format
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
                        logger.info(f"HDR Export verification: Max value in saved file: {max_val:.6f}")
                        
                        if max_val > 1.0:
                            logger.info("✅ HDR values above 1.0 successfully preserved!")
                        else:
                            logger.warning("⚠️ No HDR values above 1.0 detected (may be LDR data)")
                            
                        # Check file size as secondary verification
                        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        logger.info(f"HDR file size: {file_size_mb:.2f} MB")
                        
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
    "HDRExportNode": HDRExportNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminanceStackProcessor3Stops": "Luminance Stack Processor (3 Stops)",
    "LuminanceStackProcessor5Stops": "Luminance Stack Processor (5 Stops)",
    "HDRExportNode": "HDR Export to EXR"
}
