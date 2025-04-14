(async () => {
    try {
      const pdfUrl = window.location.href;
      const filename = pdfUrl.split("/").pop() || "unnamed.pdf";
  
      const res = await fetch(pdfUrl);
      const blob = await res.blob();
  
      const formData = new FormData();
      formData.append("file", blob, filename);
      formData.append("collection_name", filename); // 🚨 added
  
      await fetch("http://localhost:8000/upload_pdf", {
        method: "POST",
        body: formData
      });
  
      console.log("PDF uploaded to backend.");
    } catch (err) {
      console.error("Error uploading PDF:", err);
    }
  })();
  