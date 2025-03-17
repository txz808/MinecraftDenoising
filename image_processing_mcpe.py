import cv2
import numpy as np
import random


def add_colored_salt_and_pepper_noise(image, color, amount=0.05):
    """
    Add colored salt-and-pepper noise to an image.
    
    :param image: Input image (H, W, C)
    :param color: Color of the noise (e.g., 'red', 'green', 'blue', 'purple', etc.)
    :param amount: Proportion of pixels to be affected by noise
    :return: Noisy image
    """
    noisy_image = image.copy()
    h, w, c = image.shape
    num_pixels = int(amount * h * w)

    # Define color mapping
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'purple': (255, 0, 255),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }

    if color not in color_map:
        raise ValueError(f"Unsupported color '{color}'. Supported colors: {list(color_map.keys())}")

    noise_color = color_map[color]

    # Add salt noise
    for _ in range(num_pixels // 2):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        noisy_image[y, x] = noise_color

    # Add pepper noise (black pixels)
    for _ in range(num_pixels // 2):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        noisy_image[y, x] = (0, 0, 0)

    return noisy_image


def denoise_image(image, threshold_value=185, morph_kernel_size=(3, 3), adaptive=True):
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
def process_image(input_path, output_path, noise_color=None, noise_amount=0.01):
    image = cv2.imread(input_path)
    if image is None:
        return

    if noise_color:
        image = add_colored_salt_and_pepper_noise(image, noise_color, noise_amount)

    image = denoise_image(image)
    cv2.imwrite(output_path, image)
