import cv2
import numpy as np


def edge_adaptive_tv_denoise(image, lambda_val=0.125, tau=0.125, iterations=50):
    """
    Performs edge-adaptive TV denoising on a color image (per-channel).
    
    Parameters:
      image: Input image in uint8.
      lambda_val: Regularization parameter (controls noise removal).
      tau: Step size for dual variable update.
      iterations: Number of iterations for convergence.
      
    Returns:
      Denoised image (uint8) with reduced salt-and-pepper noise.
    """
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    channels = cv2.split(image)
    denoised_channels = []
    
    for channel in channels:
        u = channel.copy()
        px = np.zeros_like(u)
        py = np.zeros_like(u)
        
        for i in range(iterations):
            # Forward differences (gradients)
            gradx = np.roll(u, -1, axis=1) - u
            grady = np.roll(u, -1, axis=0) - u
            
            # Adaptive weight: less smoothing near strong gradients
            weight = 1.0 / (1.0 + np.sqrt(gradx**2 + grady**2))
            
            # Update dual variables
            px_new = px + (tau / lambda_val) * gradx
            py_new = py + (tau / lambda_val) * grady
            norm = np.maximum(1, np.sqrt(px_new**2 + py_new**2) * weight)
            px = px_new / norm
            py = py_new / norm
            
            # Compute divergence of the dual field
            div_p = (np.roll(px, 1, axis=1) - px) + (np.roll(py, 1, axis=0) - py)
            u = channel + lambda_val * div_p
        
        denoised_channels.append(u)
    
    # Merge channels and convert back to uint8
    denoised_image = cv2.merge(denoised_channels)
    denoised_image = np.uint8(np.clip(denoised_image * 255, 0, 255))
    return denoised_image
