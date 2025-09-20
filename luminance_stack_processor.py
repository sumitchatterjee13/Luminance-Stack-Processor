"""
Luminance Stack Processor - Professional ComfyUI Custom Nodes
Implements HDR processing using the Debevec Algorithm for multiple exposure fusion

Author: Sumit Chatterjee
Version: 1.0.0
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
    """Convert ComfyUI tensor to OpenCV format"""
    # ComfyUI tensors are typically [B, H, W, C] in 0-1 range
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Convert to numpy and scale to 0-255
    image = tensor.cpu().numpy()
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    return image


def cv2_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert OpenCV image to ComfyUI tensor format"""
    # Ensure image is float32 in 0-1 range
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.float32:
        image = np.clip(image, 0.0, 1.0)
    
    # Add batch dimension and convert to tensor
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor


class DebevecHDRProcessor:
    """Core HDR processing using Debevec Algorithm"""
    
    def __init__(self):
        self.calibrator = cv2.createCalibrateDebevec()
        self.merge_debevec = cv2.createMergeDebevec()
        
    def process_hdr(self, images: List[np.ndarray], exposure_times: List[float]) -> np.ndarray:
        """
        Process multiple exposure images using Debevec algorithm
        
        Args:
            images: List of images (OpenCV format)
            exposure_times: List of exposure times in seconds
            
        Returns:
            HDR merged image as 16-bit
        """
        try:
            # Ensure images are in uint8 format for calibration
            processed_images = []
            for img in images:
                if img.dtype != np.uint8:
                    if img.dtype == np.float32:
                        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                processed_images.append(img)
            
            # Convert exposure times to numpy array
            times = np.array(exposure_times, dtype=np.float32)
            
            logger.info(f"Processing {len(processed_images)} images with exposure times: {times}")
            
            # Estimate camera response function
            response = self.calibrator.process(processed_images, times)
            
            # Merge images into HDR
            hdr = self.merge_debevec.process(processed_images, times, response)
            
            # Convert HDR to 16-bit for better dynamic range preservation
            # Apply tone mapping to fit into 16-bit range
            hdr_normalized = cv2.normalize(hdr, None, 0, 65535, cv2.NORM_MINMAX)
            hdr_16bit = hdr_normalized.astype(np.uint16)
            
            logger.info(f"HDR processing completed. Output shape: {hdr_16bit.shape}")
            
            return hdr_16bit
            
        except Exception as e:
            logger.error(f"HDR processing error: {str(e)}")
            # Fallback: return the middle exposure image
            if images:
                middle_idx = len(images) // 2
                fallback = images[middle_idx]
                if fallback.dtype != np.uint16:
                    fallback = (fallback.astype(np.float32) * 257).astype(np.uint16)
                return fallback
            raise e


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
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_3_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_3_stop_hdr(self, ev_plus_2, ev_0, ev_minus_2, exposure_step=2.0):
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
            
            logger.info(f"3-Stop HDR: Processing with times {times}")
            
            # Process HDR
            hdr_result = self.processor.process_hdr(images, times)
            
            # Convert back to tensor
            output_tensor = cv2_to_tensor(hdr_result)
            
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
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "process_5_stop_hdr"
    CATEGORY = "image/luminance"
    
    def __init__(self):
        self.processor = DebevecHDRProcessor()
    
    def process_5_stop_hdr(self, ev_plus_4, ev_plus_2, ev_0, ev_minus_2, ev_minus_4, exposure_step=2.0):
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
            
            logger.info(f"5-Stop HDR: Processing with times {times}")
            
            # Process HDR
            hdr_result = self.processor.process_hdr(images, times)
            
            # Convert back to tensor
            output_tensor = cv2_to_tensor(hdr_result)
            
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
