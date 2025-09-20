"""
Luminance Stack Processor - Professional ComfyUI Custom Nodes
Implements HDR processing using the Debevec Algorithm for multiple exposure fusion

Author: Sumit Chatterjee
Version: 1.0.1
Semantic Versioning: MAJOR.MINOR.PATCH
"""

import numpy as np
import torch
import cv2
from typing import Tuple, List
import logging
import os


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tensor_to_cv2(tensor: torch.Tensor, apply_gamma_correction: bool = False) -> np.ndarray:
    """Convert ComfyUI tensor to OpenCV format for HDR processing"""
    # ComfyUI tensors are typically [B, H, W, C] in 0-1 range
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Convert to numpy and scale to 8-bit (input images are already 8-bit)
    image = tensor.cpu().numpy()
    
    if apply_gamma_correction:
        # Apply gamma correction (sRGB to linear) for algorithms that need it (Debevec/Robertson)
        # This is crucial for proper camera response function recovery
        image_linear = np.where(image <= 0.04045, 
                               image / 12.92,
                               np.power((image + 0.055) / 1.055, 2.4))
        # Convert back to 8-bit for OpenCV HDR functions
        image_8bit = np.clip(image_linear * 255.0, 0, 255).astype(np.uint8)
        logger.info(f"Applied gamma correction for HDR processing")
    else:
        # For Mertens and Natural Blend - use original gamma (no correction)
        image_8bit = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        logger.info(f"No gamma correction applied - using original image gamma")
    
    logger.info(f"Converted tensor to CV2: shape={image_8bit.shape}, dtype={image_8bit.dtype}, range=[{image_8bit.min()}, {image_8bit.max()}]")
    
    return image_8bit


def cv2_to_tensor(hdr_image: np.ndarray, output_16bit_linear: bool = True, algorithm_hint: str = "unknown") -> torch.Tensor:
    """Convert OpenCV HDR image to ComfyUI tensor format - VFX pipeline friendly"""
    
    if output_16bit_linear:
        logger.info(f"HDR processing ({algorithm_hint}):")
        logger.info(f"  Input range: [{hdr_image.min():.6f}, {hdr_image.max():.6f}]")
        
        # VFX PIPELINE APPROACH: Different scaling philosophy per algorithm
        if algorithm_hint == "natural_blend":
            # Natural Blend: Conservative scaling, values 1-5 range
            p90 = np.percentile(hdr_image, 90.0)
            if p90 > 0:
                hdr_linear = hdr_image * (2.0 / p90)
            else:
                hdr_linear = hdr_image * 2.0
            hdr_linear = np.clip(hdr_linear, 0.0, 8.0)
            
        elif algorithm_hint == "mertens":
            # Mertens: Medium HDR range, values 1-10 range
            p85 = np.percentile(hdr_image, 85.0)
            if p85 > 0:
                hdr_linear = hdr_image * (3.0 / p85)
            else:
                hdr_linear = hdr_image * 3.0
            hdr_linear = np.clip(hdr_linear, 0.0, 12.0)
            
        elif algorithm_hint in ["debevec", "robertson"]:
            # VFX PIPELINE: RAW LINEAR RADIANCE - minimal processing!
            # This should look FLAT and LOG-LIKE - that's CORRECT for VFX!
            
            # Only apply gentle scaling to bring into reasonable range for EXR
            # Don't make it "pretty" - VFX artists want raw data
            if hdr_image.max() > 0:
                # Conservative scaling - preserve as much raw data as possible
                p99 = np.percentile(hdr_image, 99.0)  # Use 99th percentile, not lower
                if p99 > 0:
                    # Very gentle scaling - target range around 0.18 (18% gray equivalent)
                    # This matches VFX pipeline expectations
                    hdr_linear = hdr_image * (0.18 / np.percentile(hdr_image, 50.0))  # Scale middle gray
                    
                    # Allow VERY wide range for VFX work - no aggressive clipping
                    hdr_linear = np.clip(hdr_linear, 0.0, 1000.0)  # Wide range for sun, sky, etc.
                else:
                    hdr_linear = hdr_image
            else:
                hdr_linear = hdr_image
                
            logger.info(f"  VFX Pipeline: Raw linear radiance preserved (flat/log appearance is CORRECT)")
            
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
        
        Args:
            images: List of 8-bit images (OpenCV format) - CRITICAL: Must be 8-bit for OpenCV HDR
            exposure_times: List of exposure times in seconds  
            algorithm: HDR algorithm to use ("mertens", "debevec", "robertson")
            
        Returns:
            HDR merged image in linear colorspace (float32)
        """
        try:
            # Validate input format - OpenCV HDR functions require 8-bit input
            processed_images = []
            for i, img in enumerate(images):
                if img.dtype != np.uint8:
                    logger.warning(f"Image {i} is not 8-bit (dtype: {img.dtype}), this may cause issues")
                
                # Handle color channels properly for each algorithm
                if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
                    if algorithm in ["debevec", "robertson"]:
                        # VFX PIPELINE FIX: Test both RGB and BGR to find which works correctly
                        # Many Debevec color issues come from wrong channel order assumption
                        processed_images.append(img)  # Try original first (ComfyUI tensors are typically RGB)
                        logger.info(f"Using original color order for {algorithm} (assuming RGB)")
                    else:
                        # For Mertens and Natural Blend, keep original format
                        processed_images.append(img)
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
                    
            elif algorithm == "natural_blend":
                # Natural Blend - maintains EV0 appearance with enhanced dynamic range
                logger.info("Using Natural Blend exposure blending...")
                hdr_radiance = self._blend_ev0_based(processed_images, times)
                
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
                hdr_radiance = self.merge_debevec.process(processed_images, times, response)
                
                logger.info(f"Debevec raw radiance range: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
                
                # VFX PIPELINE: NO TONE MAPPING! Keep raw linear radiance for professional workflow
                # This should look flat/log-like - that's correct for VFX
                # Only apply minimal scaling if needed for numerical stability
                if hdr_radiance.max() > 1000.0:  # Only if extremely bright values
                    scale_factor = 100.0 / np.percentile(hdr_radiance, 99.5)
                    hdr_radiance = hdr_radiance * scale_factor
                    logger.info(f"Applied minimal scaling for numerical stability: {scale_factor:.3f}")
                
                logger.info(f"Debevec VFX output range: [{hdr_radiance.min():.6f}, {hdr_radiance.max():.6f}]")
            
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
            
            # VFX PIPELINE: Keep output in same format as input (no additional conversions)
            # ComfyUI expects consistent color order throughout the pipeline
            
            # The result is in linear colorspace - preserve HDR data
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
    
    def _blend_ev0_based(self, images: List[np.ndarray], times: List[float]) -> np.ndarray:
        """
        Blend exposures using EV0 as base appearance with enhanced dynamic range
        
        This method preserves the exact look of the EV0 image while adding
        highlight and shadow detail from other exposures.
        
        Args:
            images: List of exposure images (should be ordered from overexposed to underexposed)
            times: Exposure times
            
        Returns:
            Enhanced image that looks like EV0 but with extended dynamic range
        """
        logger.info("Natural Blend: Preserving EV0 appearance with enhanced dynamic range")
        
        # Find the EV0 image (middle exposure - should be the normal exposure)
        ev0_idx = len(images) // 2  # Middle image is typically EV0
        ev0_base = images[ev0_idx].astype(np.float32) / 255.0
        
        logger.info(f"Using image {ev0_idx} as EV0 base (out of {len(images)} images)")
        
        # Convert all images to float for processing
        float_images = [img.astype(np.float32) / 255.0 for img in images]
        
        # Create luminance masks for blending - use BGR format (as images come in BGR)
        if ev0_base.shape[2] == 3:
            # Convert BGR to grayscale for luminance calculation
            ev0_gray = cv2.cvtColor(ev0_base, cv2.COLOR_BGR2GRAY)
        else:
            ev0_gray = ev0_base[:, :, 0]  # Fallback
        
        # Start with EV0 as the base result
        result = ev0_base.copy()
        
        # Blend highlights from underexposed images (better highlight detail)
        for i in range(ev0_idx + 1, len(float_images)):
            if i >= len(float_images):
                continue
                
            img = float_images[i]
            
            # Create highlight mask - where EV0 is bright but this image has detail
            highlight_mask = self._create_highlight_mask(ev0_gray, threshold=0.8)  # Higher threshold
            
            # Gentle blending with reduced strength
            blend_strength = 0.3  # Reduce blend strength to 30%
            
            # Blend highlight areas with reduced strength
            for c in range(min(3, result.shape[2])):  # RGB/BGR channels
                result[:, :, c] = (1 - highlight_mask * blend_strength) * result[:, :, c] + (highlight_mask * blend_strength) * img[:, :, c]
                
            logger.info(f"Gently blended highlights from underexposed image {i} (strength: {blend_strength})")
        
        # Blend shadows from overexposed images (better shadow detail)  
        for i in range(ev0_idx):
            if i < 0:
                continue
                
            img = float_images[i]
            
            # Create shadow mask - where EV0 is dark but this image has detail
            shadow_mask = self._create_shadow_mask(ev0_gray, threshold=0.2)  # Lower threshold
            
            # Gentle blending with reduced strength
            blend_strength = 0.3  # Reduce blend strength to 30%
            
            # Blend shadow areas with reduced strength
            for c in range(min(3, result.shape[2])):  # RGB/BGR channels
                result[:, :, c] = (1 - shadow_mask * blend_strength) * result[:, :, c] + (shadow_mask * blend_strength) * img[:, :, c]
                
            logger.info(f"Gently blended shadows from overexposed image {i} (strength: {blend_strength})")
        
        logger.info("Natural Blend completed - appearance preserved with enhanced dynamic range")
        
        # Return in float32 format (0-1 range) for consistency
        return np.clip(result, 0.0, 1.0).astype(np.float32)
    
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
    Processes EV+2, EV+0, EV-2 exposures using Debevec Algorithm
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
                "hdr_algorithm": (["natural_blend", "mertens", "debevec", "robertson"], {
                    "default": "natural_blend"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_3_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_3_stop_hdr(self, ev_plus_2, ev_0, ev_minus_2, exposure_step=2.0, hdr_algorithm="natural_blend"):
        """
        Process 3-stop HDR merge
        
        Args:
            ev_plus_2: Overexposed image (+2 EV)
            ev_0: Normal exposure image (0 EV)  
            ev_minus_2: Underexposed image (-2 EV)
            exposure_step: EV step size
            
        Returns:
            Tuple containing merged HDR image
        """
        try:
            # Convert tensors to OpenCV format - apply gamma correction only for certain algorithms
            apply_gamma = hdr_algorithm in ["debevec", "robertson"]
            img_plus_2 = tensor_to_cv2(ev_plus_2, apply_gamma_correction=apply_gamma)
            img_0 = tensor_to_cv2(ev_0, apply_gamma_correction=apply_gamma)
            img_minus_2 = tensor_to_cv2(ev_minus_2, apply_gamma_correction=apply_gamma)
            
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
    Processes EV+4, EV+2, EV+0, EV-2, EV-4 exposures using Debevec Algorithm
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
                "hdr_algorithm": (["natural_blend", "mertens", "debevec", "robertson"], {
                    "default": "natural_blend"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_5_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_5_stop_hdr(self, ev_plus_4, ev_plus_2, ev_0, ev_minus_2, ev_minus_4, exposure_step=2.0, hdr_algorithm="natural_blend"):
        """
        Process 5-stop HDR merge
        
        Args:
            ev_plus_4: Most overexposed image (+4 EV)
            ev_plus_2: Overexposed image (+2 EV)
            ev_0: Normal exposure image (0 EV)
            ev_minus_2: Underexposed image (-2 EV)
            ev_minus_4: Most underexposed image (-4 EV)
            exposure_step: EV step size
            
        Returns:
            Tuple containing merged HDR image
        """
        try:
            # Convert tensors to OpenCV format - apply gamma correction only for certain algorithms
            apply_gamma = hdr_algorithm in ["debevec", "robertson"]
            img_plus_4 = tensor_to_cv2(ev_plus_4, apply_gamma_correction=apply_gamma)
            img_plus_2 = tensor_to_cv2(ev_plus_2, apply_gamma_correction=apply_gamma)
            img_0 = tensor_to_cv2(ev_0, apply_gamma_correction=apply_gamma)
            img_minus_2 = tensor_to_cv2(ev_minus_2, apply_gamma_correction=apply_gamma)
            img_minus_4 = tensor_to_cv2(ev_minus_4, apply_gamma_correction=apply_gamma)
            
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
    Preserves full dynamic range data without normalization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_image": ("IMAGE", {"tooltip": "HDR image tensor with values potentially above 1.0"}),
                "filename_prefix": ("STRING", {"default": "HDR_Export", "tooltip": "Filename prefix for the EXR file"}),
                "output_path": ("STRING", {"default": "", "tooltip": "Output directory (leave empty for ComfyUI output folder)"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_hdr"
    CATEGORY = "Luminance Stack Processor"
    OUTPUT_NODE = True
    
    def export_hdr(self, hdr_image: torch.Tensor, filename_prefix: str = "HDR_Export", output_path: str = ""):
        """
        Export HDR image to EXR format preserving full dynamic range
        
        Args:
            hdr_image: HDR image tensor (potentially with values > 1.0)
            filename_prefix: Base filename for the output
            output_path: Output directory path
            
        Returns:
            Tuple containing the filepath of saved EXR
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
                    import os as os_module
                    output_dir = os_module.path.join(os_module.getcwd(), "output")
            else:
                output_dir = output_path.strip()
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.exr"
            filepath = os.path.join(output_dir, filename)
            
            logger.info(f"HDR Export: Saving to {filepath}")
            
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if len(hdr_array.shape) == 3 and hdr_array.shape[2] == 3:
                # Assume ComfyUI tensors are RGB, but OpenCV expects BGR for writing
                hdr_bgr = cv2.cvtColor(hdr_array.astype(np.float32), cv2.COLOR_RGB2BGR)
            else:
                hdr_bgr = hdr_array.astype(np.float32)
            
            # Save as EXR using OpenCV - this preserves HDR data!
            success = cv2.imwrite(filepath, hdr_bgr)
            
            if success:
                # Verify the saved file
                saved_stats = self._get_file_stats(filepath)
                logger.info(f"HDR Export: Successfully saved EXR")
                logger.info(f"  File: {filepath}")
                logger.info(f"  Size: {saved_stats['size_mb']:.2f} MB")
                logger.info(f"  Dimensions: {saved_stats['width']}x{saved_stats['height']}")
                logger.info(f"  Channels: {saved_stats['channels']}")
                
                # Verify HDR values are preserved
                test_read = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if test_read is not None:
                    logger.info(f"  Verification: Saved range [{test_read.min():.6f}, {test_read.max():.6f}]")
                    if test_read.max() > 1.0:
                        logger.info(f"  ✅ HDR data preserved (max value: {test_read.max():.2f})")
                    else:
                        logger.warning(f"  ⚠️ HDR data may be clipped (max value: {test_read.max():.2f})")
                
                return (filepath,)
            else:
                error_msg = f"Failed to save EXR file to {filepath}"
                logger.error(f"HDR Export: {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"HDR Export failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
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
