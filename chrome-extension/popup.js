let uploadedCollectionName = "";

const uploadStatus = document.getElementById("uploadStatus");
const askButton = document.getElementById("ask");
askButton.disabled = true;

document.getElementById("pdfFile").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  uploadStatus.innerText = "Uploading PDF...";
  askButton.disabled = true;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("collection_name", file.name);

  try {
    const res = await fetch("http://localhost:8000/upload_pdf", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (data.collection_name) {
      uploadedCollectionName = data.collection_name;
      uploadStatus.innerHTML = "<b>PDF uploaded!</b>";
      askButton.disabled = false;
    } else {
      uploadStatus.innerText = "Upload response missing collection name.";
      console.log("Upload response:", data);
    }
  } catch (err) {
    console.error("Upload failed:", err);
    uploadStatus.innerText = "Failed to upload PDF.";
  }
});
