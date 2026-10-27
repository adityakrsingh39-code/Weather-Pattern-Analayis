"use strict";

// Dashboard navigation, selected-file feedback and upload progress.
document.addEventListener("DOMContentLoaded", () => {
  const menuToggle = document.getElementById("menu-toggle");
  const sidebar = document.getElementById("sidebar");
  const fileInput = document.getElementById("file-input");
  const selectedFile = document.getElementById("selected-file");
  const uploadForm = document.getElementById("upload-form");
  const uploadButton = document.getElementById("upload-button");
  const uploadProgress = document.getElementById("upload-progress");

  menuToggle?.addEventListener("click", () => {
    const isOpen = sidebar.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
  });

  sidebar?.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      sidebar.classList.remove("open");
      menuToggle?.setAttribute("aria-expanded", "false");
    });
  });

  fileInput?.addEventListener("change", () => {
    if (fileInput.files.length && selectedFile) {
      selectedFile.textContent = fileInput.files[0].name;
    }
  });

  uploadForm?.addEventListener("submit", () => {
    if (!uploadForm.reportValidity()) return;
    uploadForm.setAttribute("aria-busy", "true");
    uploadProgress?.removeAttribute("hidden");
    if (uploadButton) {
      uploadButton.disabled = true;
      uploadButton.textContent = "Processing…";
    }
  });
});
