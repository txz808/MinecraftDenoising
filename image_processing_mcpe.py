import cv2
import numpy as np


# This function fixes image warping using perspective transformation
def fix_image_warping(image):
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Define the source points
    source_points = np.float32([[15, 15], [width - 20, 25], [30, height - 20], [width - 20, height - 20]]) 

    # Define the destination points
    destination_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Construct a transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    
    # Apply the transformation to the image and return the result
    return cv2.warpPerspective(image, transformation_matrix, (width, height)) 

   
# This function balances the colour channels of the image (reference: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc)
def balance_colour_channels(image, percent = 1):
    
    # Ensure the image has 3 channels and the percentage is between 0 and 100
    assert image.shape[2] == 3
    assert 0 <= percent <= 100

    # Calculate the percentile values for each channel
    half_percent = percent / 200.0

   # Split the image into its channels and sort the pixels
    channels = cv2.split(image)
    out_channels = []

    # Iterate over each channel
    for channel in channels:
        flat = np.sort(channel.flatten()) # Flatten the channel and sort the pixels
        num_pixels = flat.shape[0] # Get the number of pixels

        # Calculate the low and high percentile values
        low_val = flat[int(num_pixels * half_percent)]
        high_val = flat[int(num_pixels * (1.0 - half_percent))]

        # Clip the channel between the low and high percentile values
        channel = np.clip(channel, low_val, high_val)

        # Normalise the channel
        normalised_channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Append the normalised channel to the output channels
        out_channels.append(normalised_channel)

    # Merge the output channels
    return cv2.merge(out_channels)

def adjust_image_brightness_and_contrast_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit = 2.5, tileGridSize = (8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def inpaint_missing_region(image):
    # Convert to grayscale for better contrast analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to identify the missing region dynamically
    _, mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY_INV)

    # Refine the mask using morphology to remove small noise and connect gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)


    # Inpaint the detected missing region
    inpainted_image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

    return inpainted_image


def denoise_image(image, threshold_value=None, morph_kernel_size=(3, 3), adaptive=True):
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

    image = fix_image_warping(image)
    image = balance_colour_channels(image)
    image = adjust_image_brightness_and_contrast_clahe(image)
    image = denoise_image(image)
    image = inpaint_missing_region(image)
    cv2.imwrite(output_path, image)
