let uploaded = false;
let originalImageData = null; // Store original image data

document.getElementById("uploadButton").addEventListener("click", uploadImage);
document.getElementById("changeColorButton").addEventListener("click", changeColor);

function uploadImage() {
    const fileInput = document.getElementById("imageUpload").files[0];
    if (!fileInput) {
        alert("Please upload an image first.");
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput);

    fetch("/upload", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            document.getElementById("originalImage").src = data.original;
            document.getElementById("originalImage").style.maxWidth = "250px"; // Resize for better visibility
            document.getElementById("originalImage").style.maxHeight = "250px"; // Resize for better visibility
            uploaded = true;
            originalImageData = data.original; // Store original image data
            changeColor(); // Automatically apply noise after upload
            document.getElementById("imageUpload").value = ""; // Clear the file input
        })
        .catch(error => console.error("Error uploading image:", error));
}

function changeColor() {
    if (!uploaded) {
        alert("Please upload an image first.");
        return;
    }

    const color = document.getElementById("colorSelect").value;

    fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ color: color, original: originalImageData }) // Send original image data
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("noisyImage").src = data.noisy;
        document.getElementById("noisyImage").style.maxWidth = "250px"; // Resize for better visibility
        document.getElementById("noisyImage").style.maxHeight = "250px"; // Resize for better visibility
        document.getElementById("denoisedImage").src = data.denoised;
        document.getElementById("denoisedImage").style.maxWidth = "250px"; // Resize for better visibility
        document.getElementById("denoisedImage").style.maxHeight = "250px"; // Resize for better visibility
    })
    .catch(error => console.error("Error processing image:", error));
}
