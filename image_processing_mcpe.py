import cv2
import numpy as np


def denoise_image(image, threshold_value=185, morph_kernel_size=(3, 3), adaptive=True):
    """
    Denoises an image using segmentation, region-based filtering, and morphological operations.
    
    Parameters:
        image (np.array): Input noisy image.
        threshold_value (int, optional): Threshold for segmentation. If None, auto-detect using Otsu.
        morph_kernel_size (tuple): Kernel size for morphological operations.
        adaptive (bool): If True, dynamically calculates threshold instead of using a fixed one.
    
    Returns:
        np.array: Denoised image.
    """
    # Normalize to [0, 1] range
    image = image.astype(np.float32) / 255.0

    # Split channels
    channels = cv2.split(image)
    denoised_channels = []

    for channel in channels:
        # Auto-detect threshold using Otsu if adaptive is True
        if adaptive or threshold_value is None:
            hist, bins = np.histogram(channel.ravel(), 256, [0, 1])
            threshold_value = bins[np.argmax(hist.cumsum() > hist.sum() * 0.5)]  # Approximate Otsu threshold

        # Step 1: Segmentation using thresholding
        _, segmented_image = cv2.threshold(channel, float(threshold_value), 1.0, cv2.THRESH_BINARY)
        
        # Step 2: Compute region mean intensities
        unique_labels = np.unique(segmented_image)
        region_means = {}
        
        for label in unique_labels:
            region_mask = (segmented_image == label)
            region_mean = np.mean(channel[region_mask]) if np.any(region_mask) else 0
            region_means[label] = region_mean
        
        # Step 3: Soft blending of noisy regions
        denoised_channel = np.copy(channel)
        for label, mean_value in region_means.items():
            region_mask = (segmented_image == label)
            denoised_channel[region_mask] = (denoised_channel[region_mask] * 0.6) + (mean_value * 0.4)

        # Step 4: Morphological post-processing
        kernel = np.ones(morph_kernel_size, np.uint8)
        denoised_channel = cv2.morphologyEx(denoised_channel, cv2.MORPH_CLOSE, kernel)
        denoised_channel = cv2.morphologyEx(denoised_channel, cv2.MORPH_OPEN, kernel)

        # Append channel
        denoised_channels.append(denoised_channel)

    # Merge back channels
    denoised_image = cv2.merge(denoised_channels)

    # Convert back to uint8
    return np.uint8(np.clip(denoised_image * 255, 0, 255))



# Function to process the image
def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        return
    
    image = denoise_image(image)
    cv2.imwrite(output_path, image)
