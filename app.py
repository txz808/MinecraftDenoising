import cv2
import numpy as np
import random
import os
from flask import Flask, render_template, request, jsonify, url_for


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Resize images before processing
def resize_image(image, max_width=300):
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

# Add colored salt-and-pepper noise
def add_colored_salt_and_pepper_noise(image, color, amount=0.03, edge_weight=0.7):
    noisy_image = image.copy()
    h, w, c = image.shape
    num_pixels = int(amount * h * w)

    color_map = {
        'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0),
        'purple': (255, 0, 255), 'yellow': (0, 255, 255), 'cyan': (255, 255, 0),
        'white': (255, 255, 255), 'black': (0, 0, 0)
    }

    noise_color = color_map.get(color, (255, 255, 255))

    # Detect edges using Canny edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)

    # Add noise near edges
    edge_indices = np.column_stack(np.where(edges > 0))
    edge_pixels = int(num_pixels * edge_weight)

    for _ in range(edge_pixels):
        if len(edge_indices) > 0:
            y, x = random.choice(edge_indices)
            noisy_image[y, x] = noise_color

    # Add noise randomly across the image
    remaining_pixels = num_pixels - edge_pixels
    for _ in range(remaining_pixels // 2):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        noisy_image[y, x] = noise_color

    for _ in range(remaining_pixels // 2):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        noisy_image[y, x] = (0, 0, 0)

    return noisy_image

# Minecraft denoising function
def denoise_image_minecraft(image, threshold_value=185, morph_kernel_size=(3, 3), adaptive=True):
    image = image.astype(np.float32) / 255.0
    channels = cv2.split(image)
    denoised_channels = []

    for channel in channels:
        if adaptive or threshold_value is None:
            hist, bins = np.histogram(channel.ravel(), 256, [0, 1])
            threshold_value = bins[np.argmax(hist.cumsum() > hist.sum() * 0.5)]

        _, segmented_image = cv2.threshold(channel, float(threshold_value), 1.0, cv2.THRESH_BINARY)
        
        unique_labels = np.unique(segmented_image)
        region_means = {label: np.mean(channel[segmented_image == label]) for label in unique_labels}

        denoised_channel = np.copy(channel)
        for label, mean_value in region_means.items():
            region_mask = (segmented_image == label)
            denoised_channel[region_mask] = (denoised_channel[region_mask] * 0.6) + (mean_value * 0.4)

        kernel = np.ones(morph_kernel_size, np.uint8)
        denoised_channel = cv2.morphologyEx(denoised_channel, cv2.MORPH_CLOSE, kernel)
        denoised_channel = cv2.morphologyEx(denoised_channel, cv2.MORPH_OPEN, kernel)

        denoised_channels.append(denoised_channel)

    denoised_image = cv2.merge(denoised_channels)
    return np.uint8(np.clip(denoised_image * 255, 0, 255))

# Denoise image using comic denoising

def denoise_image_comic(image, lambda_val=0.1, tau=0.2, iterations=20):
    # Normalize image to [0, 1]
    image = image.astype(np.float32) / 255.0
    channels = cv2.split(image)
    denoised_channels = []
    epsilon = 1e-8

    for channel in channels:
        u = channel.copy()
        px = np.zeros_like(u)
        py = np.zeros_like(u)

        for i in range(iterations):
            # Compute forward differences with clamping to preserve object details
            gradx = np.roll(u, -1, axis=1) - u
            grady = np.roll(u, -1, axis=0) - u
            gradx = np.clip(gradx, -0.5, 0.5)
            grady = np.clip(grady, -0.5, 0.5)

            # Compute adaptive weight to enhance edges
            weight = 1.0 / (1.0 + np.sqrt(gradx**2 + grady**2) + epsilon)

            # Update dual variables
            px_new = px + (tau / lambda_val) * gradx
            py_new = py + (tau / lambda_val) * grady
            norm = np.maximum(1, np.sqrt(px_new**2 + py_new**2) * weight)
            px = px_new / norm
            py = py_new / norm

            # Compute divergence of the dual field
            div_p = (np.roll(px, 1, axis=1) - px) + (np.roll(py, 1, axis=0) - py)

            # Update primal variable
            u = channel + lambda_val * div_p

            # Apply clipping to suppress extreme values
            u = np.clip(u, 0.0, 1.0)

        denoised_channels.append(u)

    # Merge channels back and convert to uint8
    denoised_image = cv2.merge(denoised_channels)
    denoised_image = np.uint8(np.clip(denoised_image * 255, 0, 255))
    return denoised_image

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    noise_color = request.form.get("color", "red")
    mode = request.form.get("mode", "minecraft")  # Get the selected denoising mode

    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        image = cv2.imread(filename)

        resized_image = resize_image(image)
        original_path = os.path.join(PROCESSED_FOLDER, "original.jpg")
        cv2.imwrite(original_path, resized_image)

        noisy_image = add_colored_salt_and_pepper_noise(resized_image, noise_color)

        # Apply the selected denoising mode
        if mode == "minecraft":
            denoised_image = denoise_image_minecraft(noisy_image)
        elif mode == "comic":
            denoised_image = denoise_image_comic(noisy_image)
        else:
            return jsonify({"error": "Invalid mode selected"}), 400

        noisy_path = os.path.join(PROCESSED_FOLDER, "noisy.jpg")
        denoised_path = os.path.join(PROCESSED_FOLDER, "denoised.jpg")

        cv2.imwrite(noisy_path, noisy_image)
        cv2.imwrite(denoised_path, denoised_image)

        # Use `url_for` to generate URLs for the images
        return jsonify({
            "original": url_for('static', filename=f'processed/original.jpg'),
            "noisy": url_for('static', filename=f'processed/noisy.jpg'),
            "denoised": url_for('static', filename=f'processed/denoised.jpg')
        })

if __name__ == "__main__":
    app.run(debug=True)
