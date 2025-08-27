// Show Preview of Selected Image
document.getElementById("fileInput").addEventListener("change", function (event) {
  const preview = document.getElementById("preview");
  preview.innerHTML = "";
  const file = event.target.files[0];

  if (file) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file); // was: img.scr
    img.alt = "Preview";
    preview.appendChild(img);
  }
});
