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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI tensor to OpenCV format for HDR processing"""
    # ComfyUI tensors are typically [B, H, W, C] in 0-1 range
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Convert to numpy and scale to 8-bit (input images are already 8-bit)
    image = tensor.cpu().numpy()
    
    # Apply gamma correction (sRGB to linear) before HDR processing
    # This is crucial for proper camera response function recovery
    image_linear = np.where(image <= 0.04045, 
                           image / 12.92,
                           np.power((image + 0.055) / 1.055, 2.4))
    
    # Convert back to 8-bit for OpenCV HDR functions (since inputs are 8-bit)
    image_8bit = np.clip(image_linear * 255.0, 0, 255).astype(np.uint8)
    
    logger.info(f"Converted tensor to CV2: shape={image_8bit.shape}, dtype={image_8bit.dtype}, range=[{image_8bit.min()}, {image_8bit.max()}]")
    
    return image_8bit


def cv2_to_tensor(hdr_image: np.ndarray, output_16bit_linear: bool = True, algorithm_hint: str = "unknown") -> torch.Tensor:
    """Convert OpenCV HDR image to ComfyUI tensor format"""
    
    if output_16bit_linear:
        # Convert HDR linear data to 16-bit linear format
        # Scale to 16-bit range while preserving linear characteristics
        
        # Different scaling strategies based on algorithm
        if algorithm_hint == "natural_blend":
            # Natural Blend should have similar range to original images, use 85th percentile
            scale_reference = np.percentile(hdr_image, 85.0)
            target_scale = 0.85  # 85% of 16-bit range
        elif algorithm_hint == "mertens":
            # Mertens typically produces 0-2 range, use 90th percentile
            scale_reference = np.percentile(hdr_image, 90.0)
            target_scale = 0.8  # 80% of 16-bit range
        else:
            # Debevec/Robertson can have very wide ranges, use 95th percentile
            scale_reference = np.percentile(hdr_image, 95.0)  
            target_scale = 0.7  # 70% of 16-bit range for more headroom
        
        if scale_reference > 0:
            # Scale so that the percentile maps to target % of 16-bit range
            scale_factor = (target_scale * 65535.0) / scale_reference
            scaled_linear = hdr_image * scale_factor
        else:
            scaled_linear = hdr_image * 65535.0
        
        # Clamp to 16-bit range but preserve linear values above 1.0
        hdr_16bit_linear = np.clip(scaled_linear, 0.0, 65535.0)
        
        # Convert to ComfyUI tensor format (0-1 range) while preserving 16-bit precision
        normalized_16bit = hdr_16bit_linear / 65535.0
        
        logger.info(f"HDR 16-bit linear output ({algorithm_hint}):")
        logger.info(f"  Original range: [{hdr_image.min():.6f}, {hdr_image.max():.6f}]")
        
        # Choose percentile based on algorithm
        if algorithm_hint == "natural_blend":
            percentile_used = 85
        elif algorithm_hint == "mertens":
            percentile_used = 90
        else:
            percentile_used = 95
            
        logger.info(f"  Scale reference ({percentile_used}th percentile): {scale_reference:.6f}")
        logger.info(f"  16-bit range: [{hdr_16bit_linear.min():.1f}, {hdr_16bit_linear.max():.1f}]")
        logger.info(f"  Final range: [{normalized_16bit.min():.6f}, {normalized_16bit.max():.6f}]")
        logger.info(f"  Scale factor: {scale_factor:.2f}")
        
    else:
        # Legacy floating point output (not recommended for 16-bit workflow)
        max_val = np.percentile(hdr_image, 99.9)
        if max_val > 0:
            normalized_16bit = hdr_image / max_val
        else:
            normalized_16bit = hdr_image
        
        normalized_16bit = np.clip(normalized_16bit, 0.0, 10.0)
    
    # Add batch dimension and convert to tensor
    tensor = torch.from_numpy(normalized_16bit.astype(np.float32)).unsqueeze(0)
    return tensor


class DebevecHDRProcessor:
    """Core HDR processing using multiple algorithms"""
    
    def __init__(self):
        self.calibrator = cv2.createCalibrateDebevec()
        self.merge_debevec = cv2.createMergeDebevec()
        # Alternative algorithms
        self.merge_mertens = cv2.createMergeMertens()
        self.merge_robertson = cv2.createMergeRobertson()
        self.calibrator_robertson = cv2.createCalibrateRobertson()
        
    def process_hdr(self, images: List[np.ndarray], exposure_times: List[float], algorithm: str = "mertens") -> np.ndarray:
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
                
                # CRITICAL FIX: Convert BGR to RGB for proper color handling
                # OpenCV uses BGR by default, but HDR processing expects RGB
                if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    processed_images.append(img_rgb)
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
                # Mertens Exposure Fusion - often produces better results
                logger.info("Using Mertens Exposure Fusion algorithm...")
                hdr_radiance = self.merge_mertens.process(processed_images)
                # Mertens output is typically in 0-1 range, scale appropriately
                if hdr_radiance.max() <= 1.0:
                    hdr_radiance = hdr_radiance * 2.0  # Boost for better dynamic range
                    
            elif algorithm == "natural_blend":
                # Natural Blend - maintains EV0 appearance with enhanced dynamic range
                logger.info("Using Natural Blend exposure blending...")
                hdr_radiance = self._blend_ev0_based(processed_images, times)
                
            elif algorithm == "robertson":
                # Robertson algorithm - alternative to Debevec
                logger.info("Using Robertson algorithm...")
                response = self.calibrator_robertson.process(processed_images, times)
                hdr_radiance = self.merge_robertson.process(processed_images, times, response)
                # Apply same tone mapping as Debevec
                hdr_radiance = self._tone_map_debevec_output(hdr_radiance, "Robertson")
                
            else:  # Default to Debevec
                # Estimate camera response function using Debevec method
                logger.info("Using Debevec algorithm...")
                response = self.calibrator.process(processed_images, times)
                logger.info(f"Response function shape: {response.shape}")
                
                # Merge images into HDR using Debevec algorithm
                hdr_radiance = self.merge_debevec.process(processed_images, times, response)
                # Apply tone mapping to fix brightness and make output similar to Mertens
                hdr_radiance = self._tone_map_debevec_output(hdr_radiance, "Debevec")
            
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
            
            # CRITICAL FIX: Convert back from RGB to BGR for consistency
            # This ensures the output color channels are in the expected order
            if len(hdr_radiance.shape) == 3 and hdr_radiance.shape[2] == 3:
                hdr_radiance = cv2.cvtColor(hdr_radiance, cv2.COLOR_RGB2BGR)
            
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

    def _tone_map_debevec_output(self, hdr_image: np.ndarray, algorithm_name: str) -> np.ndarray:
        """
        Apply tone mapping to Debevec/Robertson output to fix brightness and color issues
        
        Args:
            hdr_image: Raw HDR output from Debevec/Robertson
            algorithm_name: Name of algorithm for logging
            
        Returns:
            Tone-mapped HDR image with proper brightness and color balance
        """
        logger.info(f"{algorithm_name} raw output range: [{hdr_image.min():.6f}, {hdr_image.max():.6f}]")
        
        # Debevec often produces extremely bright values - apply gamma correction and scaling
        try:
            # Method 1: Reinhard tone mapping for natural results
            # This is similar to what Lightroom/Photoshop do for HDR
            reinhard = cv2.createTonemapReinhard(gamma=1.8, intensity=-1.0, light_adapt=0.8, color_adapt=0.0)
            tone_mapped = reinhard.process(hdr_image.astype(np.float32))
            
            logger.info(f"{algorithm_name} after Reinhard tone mapping: [{tone_mapped.min():.6f}, {tone_mapped.max():.6f}]")
            
            # Ensure reasonable dynamic range
            if tone_mapped.max() > 0:
                # Scale to have similar range as Mertens (0-2 range typical)
                max_val = np.percentile(tone_mapped, 98)  # Use 98th percentile
                if max_val > 0:
                    tone_mapped = tone_mapped * (2.0 / max_val)
                    
            # Clamp extreme values while preserving HDR range
            tone_mapped = np.clip(tone_mapped, 0.0, 10.0)
            
            logger.info(f"{algorithm_name} final tone mapped range: [{tone_mapped.min():.6f}, {tone_mapped.max():.6f}]")
            
            return tone_mapped.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Tone mapping failed for {algorithm_name}: {e}")
            # Fallback: simple normalization
            if hdr_image.max() > 0:
                normalized = hdr_image / np.percentile(hdr_image, 95)
                return np.clip(normalized, 0.0, 5.0).astype(np.float32)
            return hdr_image.astype(np.float32)
    
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
        
        # Create luminance masks for blending
        ev0_gray = cv2.cvtColor(ev0_base, cv2.COLOR_RGB2GRAY)
        
        # Start with EV0 as the base result
        result = ev0_base.copy()
        
        # Blend highlights from underexposed images (better highlight detail)
        for i in range(ev0_idx + 1, len(float_images)):
            img = float_images[i]
            
            # Create highlight mask - where EV0 is bright but this image has detail
            highlight_mask = self._create_highlight_mask(ev0_gray)
            
            # Blend highlight areas
            for c in range(3):  # RGB channels
                result[:, :, c] = (1 - highlight_mask) * result[:, :, c] + highlight_mask * img[:, :, c]
                
            logger.info(f"Blended highlights from underexposed image {i}")
        
        # Blend shadows from overexposed images (better shadow detail)  
        for i in range(ev0_idx):
            img = float_images[i]
            
            # Create shadow mask - where EV0 is dark but this image has detail
            shadow_mask = self._create_shadow_mask(ev0_gray)
            
            # Blend shadow areas
            for c in range(3):  # RGB channels
                result[:, :, c] = (1 - shadow_mask) * result[:, :, c] + shadow_mask * img[:, :, c]
                
            logger.info(f"Blended shadows from overexposed image {i}")
        
        # Convert back to 0-255 range for consistency with other algorithms
        result_8bit = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        logger.info("Natural Blend completed - appearance preserved with enhanced dynamic range")
        
        return result_8bit.astype(np.float32) / 255.0  # Return as float32 in 0-1 range
    
    def _create_highlight_mask(self, gray_image: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """Create a mask for highlight areas that need detail recovery"""
        # Smooth transition for highlights
        mask = np.zeros_like(gray_image, dtype=np.float32)
        
        # Areas above threshold get progressively more blending
        bright_areas = gray_image > threshold
        mask[bright_areas] = (gray_image[bright_areas] - threshold) / (1.0 - threshold)
        
        # Smooth the mask to avoid harsh transitions
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return np.clip(mask, 0, 1)
    
    def _create_shadow_mask(self, gray_image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Create a mask for shadow areas that need detail recovery"""
        # Smooth transition for shadows
        mask = np.zeros_like(gray_image, dtype=np.float32)
        
        # Areas below threshold get progressively more blending
        dark_areas = gray_image < threshold
        mask[dark_areas] = (threshold - gray_image[dark_areas]) / threshold
        
        # Smooth the mask to avoid harsh transitions
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
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
            # Convert tensors to OpenCV format
            img_plus_2 = tensor_to_cv2(ev_plus_2)
            img_0 = tensor_to_cv2(ev_0)
            img_minus_2 = tensor_to_cv2(ev_minus_2)
            
            # Calculate exposure times based on EV values
            # EV difference formula: time = base_time * 2^(-EV_difference)
            base_time = 1.0 / 60.0  # 1/60s as base exposure
            
            time_plus_2 = base_time * (2.0 ** (-exposure_step))  # Shorter time (overexposed)
            time_0 = base_time  # Normal exposure
            time_minus_2 = base_time * (2.0 ** exposure_step)  # Longer time (underexposed)
            
            images = [img_plus_2, img_0, img_minus_2]
            times = [time_plus_2, time_0, time_minus_2]
            
            logger.info(f"3-Stop HDR: Processing with times {times} using {hdr_algorithm} algorithm")
            
            # Process HDR using selected algorithm (default: Mertens for better results)
            hdr_result = self.processor.process_hdr(images, times, algorithm=hdr_algorithm)
            
            # Convert back to tensor - output 16-bit linear data
            output_tensor = cv2_to_tensor(hdr_result, output_16bit_linear=True, algorithm_hint=hdr_algorithm)
            
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
            # Convert tensors to OpenCV format
            img_plus_4 = tensor_to_cv2(ev_plus_4)
            img_plus_2 = tensor_to_cv2(ev_plus_2)
            img_0 = tensor_to_cv2(ev_0)
            img_minus_2 = tensor_to_cv2(ev_minus_2)
            img_minus_4 = tensor_to_cv2(ev_minus_4)
            
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
            
            # Process HDR using selected algorithm (default: Mertens for better results)
            hdr_result = self.processor.process_hdr(images, times, algorithm=hdr_algorithm)
            
            # Convert back to tensor - output 16-bit linear data  
            output_tensor = cv2_to_tensor(hdr_result, output_16bit_linear=True, algorithm_hint=hdr_algorithm)
            
            return (output_tensor,)
            
        except Exception as e:
            logger.error(f"5-Stop HDR processing failed: {str(e)}")
            # Return middle exposure as fallback
            return (ev_0,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LuminanceStackProcessor3Stops": LuminanceStackProcessor3Stops,
    "LuminanceStackProcessor5Stops": LuminanceStackProcessor5Stops,
}

# Display names for nodes in ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminanceStackProcessor3Stops": "Luminance Stack Processor (3 Stops)",
    "LuminanceStackProcessor5Stops": "Luminance Stack Processor (5 Stops)",
}
