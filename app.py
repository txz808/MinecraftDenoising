import cv2
import numpy as np
import random
import os
from flask import Flask, render_template, request, jsonify

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
def add_colored_salt_and_pepper_noise(image, color, amount=0.02):
    noisy_image = image.copy()
    h, w, c = image.shape
    num_pixels = int(amount * h * w)

    color_map = {
        'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0),
        'purple': (255, 0, 255), 'yellow': (0, 255, 255), 'cyan': (255, 255, 0),
        'white': (255, 255, 255), 'black': (0, 0, 0)
    }

    noise_color = color_map.get(color, (255, 255, 255))

    for _ in range(num_pixels // 2):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        noisy_image[y, x] = noise_color

    for _ in range(num_pixels // 2):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        noisy_image[y, x] = (0, 0, 0)

    return noisy_image

# Denoising function
def denoise_image(image, threshold_value=185, morph_kernel_size=(3, 3), adaptive=True):
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    noise_color = request.form.get("color", "red")

    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        image = cv2.imread(filename)

        resized_image = resize_image(image)
        original_path = os.path.join(PROCESSED_FOLDER, "original.jpg")
        cv2.imwrite(original_path, resized_image)

        noisy_image = add_colored_salt_and_pepper_noise(resized_image, noise_color)
        denoised_image = denoise_image(noisy_image)

        noisy_path = os.path.join(PROCESSED_FOLDER, "noisy.jpg")
        denoised_path = os.path.join(PROCESSED_FOLDER, "denoised.jpg")

        cv2.imwrite(noisy_path, noisy_image)
        cv2.imwrite(denoised_path, denoised_image)

        return jsonify({
            "original": original_path,
            "noisy": noisy_path,
            "denoised": denoised_path
        })

if __name__ == "__main__":
    app.run(debug=True)
