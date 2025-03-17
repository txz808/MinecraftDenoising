document.getElementById("uploadForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let formData = new FormData();
    formData.append("image", document.getElementById("image").files[0]);
    formData.append("noise_color", document.getElementById("noise_color").value);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        let url = URL.createObjectURL(blob);
        document.getElementById("processedImage").src = url;
        let downloadLink = document.getElementById("downloadLink");
        downloadLink.href = url;
        downloadLink.download = "processed_image.png";
        downloadLink.style.display = "block";
        downloadLink.textContent = "Download Processed Image";
    })
    .catch(error => console.error("Error:", error));
});
