const form = document.getElementById("predict-form");
const input = document.getElementById("xray-input");
const fileNameLabel = document.getElementById("file-name");
const uploadArea = document.getElementById("upload-area");
const imagePreview = document.getElementById("image-preview");
const heatmapPreview = document.getElementById("heatmap-preview");
const previewContainer = document.getElementById("preview-container");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const historyList = document.getElementById("history-list");
const emptyHistory = document.getElementById("empty-history");

// Diagnostic workflow elements
const diagAuthNotice = document.getElementById("diag-auth-notice");
const diagWorkflow = document.getElementById("diag-workflow");
const diagLoginLink = document.getElementById("diag-login-link");
const docsInput = document.getElementById("docs-input");
const docsFileName = document.getElementById("docs-file-name");
const docsUploadLabel = document.getElementById("docs-upload-label");
const processDocsBtn = document.getElementById("process-docs-btn");
const docsStatus = document.getElementById("docs-status");
const ragJsonDisplay = document.getElementById("rag-json-display");
const diagBoxXray = document.getElementById("diag-box-xray");
const diagBoxHeatmap = document.getElementById("diag-box-heatmap");
const diagBoxImgPlaceholder = document.getElementById("diag-box-img-placeholder");
const finalDiagBtn = document.getElementById("final-diag-btn");
const finalDiagStatus = document.getElementById("final-diag-status");
const finalResult = document.getElementById("final-result");

let mergedRagJson = null;
let selectedDocs = [];
let selectedXray = null;

async function loadMetrics() {
  try {
    const res = await fetch("/metrics");
    const data = await res.json();
    if (data && data.accuracy) {
      document.getElementById("metric-accuracy").textContent = (data.accuracy * 100).toFixed(1) + "%";
    }
    if (data && data.auc) {
      document.getElementById("metric-auc").textContent = (data.auc * 100).toFixed(1) + "%";
    }
  } catch (e) {
    console.error("Could not load metrics", e);
  }
}
loadMetrics();

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const parent = tab.closest('.glass-panel, .modal-content');
    if (!parent) return;
    parent.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.target).classList.add('active');
  });
});

function updateFinalDiagButton() {
  finalDiagBtn.disabled = !(selectedXray && token);
}

function syncDiagnosticBoxImage() {
  if (selectedXray && imagePreview.src) {
    diagBoxImgPlaceholder.classList.add("hidden");
    diagBoxXray.src = imagePreview.src;
    diagBoxXray.classList.remove("hidden");
  } else {
    diagBoxImgPlaceholder.classList.remove("hidden");
    diagBoxXray.classList.add("hidden");
    diagBoxHeatmap.classList.add("hidden");
  }
}

function handleFileSelect(file) {
  heatmapPreview.classList.remove("show");
  heatmapPreview.classList.add("hidden");
  heatmapPreview.src = "";
  diagBoxHeatmap.classList.add("hidden");

  if (file && file.type.startsWith("image/")) {
    selectedXray = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewContainer.classList.remove("hidden");
      uploadPlaceholder.classList.add("hidden");
      syncDiagnosticBoxImage();
    };
    reader.readAsDataURL(file);
    fileNameLabel.textContent = file.name;
    updateFinalDiagButton();
  } else {
    selectedXray = null;
    previewContainer.classList.add("hidden");
    uploadPlaceholder.classList.remove("hidden");
    syncDiagnosticBoxImage();
    updateFinalDiagButton();
  }
}

input.addEventListener("change", () => handleFileSelect(input.files?.[0]));

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const file = e.dataTransfer.files?.[0];
  if (file && file.type.startsWith("image/")) {
    input.files = e.dataTransfer.files;
    handleFileSelect(file);
  }
});

form.addEventListener("submit", (e) => e.preventDefault());

// Document upload
docsInput.addEventListener("change", () => {
  selectedDocs = Array.from(docsInput.files || []);
  mergedRagJson = null;
  if (selectedDocs.length) {
    docsFileName.textContent = selectedDocs.map(f => f.name).join(", ");
    docsUploadLabel.classList.add("has-file");
    processDocsBtn.disabled = false;
    ragJsonDisplay.textContent = "Documents selected. They will be processed when you generate the final diagnostic.";
  } else {
    docsFileName.textContent = "Click to upload PDF medical reports (multiple allowed)";
    docsUploadLabel.classList.remove("has-file");
    processDocsBtn.disabled = true;
    ragJsonDisplay.textContent = "Upload and process medical documents to see JSON data here.";
  }
});

processDocsBtn.addEventListener("click", async () => {
  if (!selectedDocs.length || !token) return;

  processDocsBtn.disabled = true;
  docsStatus.classList.remove("hidden");
  docsStatus.innerHTML = `<div class="spinner"></div><span>Processing documents with RAG…</span>`;

  const formData = new FormData();
  selectedDocs.forEach(f => formData.append("files", f));

  try {
    const res = await fetch("/process-documents", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Document processing failed.");

    mergedRagJson = payload.merged;
    ragJsonDisplay.textContent = JSON.stringify(payload.merged, null, 2);
    docsStatus.innerHTML = `<span style="color:#10b981;">✓ ${selectedDocs.length} document(s) processed successfully.</span>`;
  } catch (err) {
    docsStatus.innerHTML = `<span style="color:#ef4444;">⚠ ${err.message}</span>`;
  } finally {
    processDocsBtn.disabled = false;
  }
});

finalDiagBtn.addEventListener("click", async () => {
  if (!selectedXray || !token) return;

  finalDiagBtn.disabled = true;
  finalDiagStatus.classList.remove("hidden");
  finalResult.classList.add("hidden");

  const statusMsg = selectedDocs.length
    ? "Processing medical report with RAG, then generating diagnostic…"
    : "Generating final diagnostic…";
  finalDiagStatus.innerHTML = `<div class="spinner"></div><span>${statusMsg}</span>`;

  const formData = new FormData();
  formData.append("xray", selectedXray);
  formData.append("rag_json", JSON.stringify(mergedRagJson || {}));
  selectedDocs.forEach(f => formData.append("documents", f));

  try {
    const res = await fetch("/final-diagnostic", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Final diagnostic failed.");

    const blocked = data.identity_check?.mismatch_detected || data.final_diagnostic?.blocked;

    if (data.gradcam_base64 && !blocked) {
      heatmapPreview.src = "data:image/jpeg;base64," + data.gradcam_base64;
      heatmapPreview.classList.remove("hidden");
      setTimeout(() => heatmapPreview.classList.add("show"), 50);

      diagBoxHeatmap.src = heatmapPreview.src;
      diagBoxHeatmap.classList.remove("hidden");
    }

    displayFinalDiagnostic(data.final_diagnostic, data.rag_data, data.identity_check);
    lastDiagnosticData = data;
    addHistoryItem(selectedXray.name, data.final_diagnostic, blocked);
    finalDiagStatus.classList.add("hidden");
  } catch (err) {
    finalDiagStatus.innerHTML = `<span style="color:#ef4444;">⚠ ${err.message}</span>`;
  } finally {
    finalDiagBtn.disabled = false;
    updateFinalDiagButton();
  }
});

function displayFinalDiagnostic(diag, ragData, identityCheck) {
  console.log("displayFinalDiagnostic:", { diag, ragData, identityCheck });
  finalResult.classList.remove("hidden");

  // Safety fallbacks
  diag = diag || {};
  ragData = ragData || {};
  identityCheck = identityCheck || {};

  const blocked = identityCheck?.mismatch_detected || diag?.blocked;
  const diagnosticBody = document.getElementById("final-diagnostic-body");
  const resultTitle = document.getElementById("final-result-title");

  if (resultTitle) {
    resultTitle.innerHTML = blocked
      ? '<span class="step-num">4</span> Identity Verification Failed'
      : '<span class="step-num">4</span> Final Diagnostic';
  }

  if (diagnosticBody) {
    diagnosticBody.classList.toggle("hidden", !!blocked);
  }

  const reportCount = ragData?.documents_count || ragData?.documents?.length || 0;
  const mismatch = identityCheck?.mismatch_detected;
  const identityVerified = identityCheck?.match === true;

  const identityBanner = document.getElementById("identity-banner");
  if (identityBanner) {
    identityBanner.classList.remove("hidden", "identity-match", "identity-mismatch", "identity-unknown");
    if (mismatch) {
      identityBanner.classList.add("identity-mismatch");
      const ap = identityCheck.account_patient || {};
      const rp = identityCheck.report_patient || {};
      identityBanner.innerHTML = `
        <strong>Identity mismatch detected</strong><br>
        Connected account: <b>${ap.name || "?"}</b>, age ${ap.age ?? "?"}<br>
        Medical report: <b>${rp.name || "unknown"}</b>, age ${rp.age ?? "?"}, ${rp.gender || "unknown"}<br>
        <span style="font-size:0.85rem;">${identityCheck.message || ""}</span>
        <p style="margin:12px 0 0;font-size:0.9rem;">${diag.mismatch_explanation || ""}</p>
        <p style="margin:8px 0 0;font-size:0.85rem;color:#fca5a5;">${diag.follow_up || "Upload the correct medical report and try again."}</p>
      `;
    } else if (identityVerified) {
      identityBanner.classList.add("identity-match");
      identityBanner.innerHTML = `<strong>Patient identity verified</strong> — account and report describe the same patient.`;
    } else if (reportCount > 0) {
      identityBanner.classList.add("identity-unknown");
      identityBanner.innerHTML = `<strong>Identity not verified</strong> — ${identityCheck?.message || "Could not compare account with report."}`;
    } else {
      identityBanner.classList.add("hidden");
    }
  }

  if (blocked) {
    const mismatchEl = document.getElementById("final-report-note");
    if (mismatchEl) mismatchEl.classList.add("hidden");
    return;
  }

  const badge = document.getElementById("final-diagnosis");
  if (badge) {
    badge.textContent = diag.diagnosis || "Unknown";
    badge.className = "badge " + (diag.severity === "High" || diag.severity === "Critical" ? "pneumonia" : "normal");
  }

  const severityEl = document.getElementById("final-severity");
  if (severityEl) {
    severityEl.textContent = `Severity: ${diag.severity || "Unknown"} · Confidence: ${diag.confidence || "Unknown"} · Reports: ${reportCount}`;
  }

  const summaryEl = document.getElementById("final-summary");
  if (summaryEl) {
    summaryEl.textContent = diag.clinical_summary || diag.summary || "";
  }

  const contributions = document.getElementById("final-contributions");
  if (contributions) {
    contributions.innerHTML = `
      <div class="contribution-card"><strong>Your profile</strong><p>${diag.account_contribution || "—"}</p></div>
      <div class="contribution-card"><strong>Medical report</strong><p>${diag.report_contribution || "—"}</p></div>
      <div class="contribution-card"><strong>X-ray</strong><p>${diag.xray_contribution || "—"}</p></div>
    `;
  }

  const mismatchEl = document.getElementById("final-report-note");
  if (mismatchEl) {
    mismatchEl.classList.add("hidden");
  }

  if (ragData && reportCount > 0) {
    const ragDisplay = document.getElementById("rag-json-display");
    if (ragDisplay) ragDisplay.textContent = JSON.stringify(ragData, null, 2);
  }

  const followupEl = document.getElementById("final-followup");
  if (followupEl) {
    followupEl.textContent = diag.follow_up || "";
  }

  const disclaimerEl = document.getElementById("final-disclaimer");
  if (disclaimerEl) {
    disclaimerEl.textContent = diag.disclaimer || "";
  }

  const factorsList = document.getElementById("final-factors");
  if (factorsList) {
    factorsList.innerHTML = "";
    (diag.key_factors || []).forEach(f => {
      const li = document.createElement("li");
      li.textContent = f;
      factorsList.appendChild(li);
    });
  }

  const recList = document.getElementById("final-recommendations");
  if (recList) {
    recList.innerHTML = "";
    (diag.recommendations || []).forEach(r => {
      const li = document.createElement("li");
      li.textContent = r;
      recList.appendChild(li);
    });
  }
}

let lastDiagnosticData = null;

function addHistoryItem(filename, diag, blocked = false) {
  if (!historyList) return;
  if (emptyHistory) emptyHistory.remove();
  const historyItem = document.createElement("div");
  historyItem.className = "history-item";
  const severity = diag.severity || "Unknown";
  const badgeClass = blocked ? "pneumonia" : (severity === "High" || severity === "Critical" ? "pneumonia" : "normal");
  const label = blocked ? "Identity mismatch" : (diag.diagnosis || "Diagnostic");
  const ts = new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  historyItem.innerHTML = `
    <div>
      <span class="history-item-name" title="${filename}">${filename}</span>
      <div class="history-item-date">${ts}</div>
    </div>
    <span class="badge ${badgeClass}">${label}</span>
  `;
  // Store data for click-to-view
  const itemData = {
    xray_filename: filename,
    final_diagnostic: diag,
    rag_data: lastDiagnosticData?.rag_data || null,
    xray_analysis: lastDiagnosticData?.xray_analysis || null,
    patient_context: lastDiagnosticData?.patient_context || null,
    timestamp: new Date().toISOString(),
  };
  historyItem.addEventListener("click", () => openHistoryDetail(itemData));
  if (historyList) historyList.prepend(historyItem);
}

// Modal Logic
const modal = document.getElementById("image-modal");
const modalImage = document.getElementById("modal-image");
const modalHeatmap = document.getElementById("modal-heatmap");
const modalClose = document.getElementById("modal-close");
const expandBtn = document.getElementById("expand-btn");

expandBtn.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (previewContainer.classList.contains("hidden")) return;
  modalImage.src = imagePreview.src;
  if (!heatmapPreview.classList.contains("hidden") && heatmapPreview.src) {
    modalHeatmap.src = heatmapPreview.src;
    modalHeatmap.classList.remove("hidden");
    setTimeout(() => modalHeatmap.classList.add("show"), 50);
  } else {
    modalHeatmap.classList.add("hidden");
    modalHeatmap.classList.remove("show");
  }
  modal.classList.remove("hidden");
});

modalClose.addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.add("hidden"); });

// Auth & Profile
const profileBtn = document.getElementById("profile-btn");
const registerBtn = document.getElementById("register-btn");
const profileName = document.getElementById("profile-name");
const authModal = document.getElementById("auth-modal");
const authModalClose = document.getElementById("auth-modal-close");
const authError = document.getElementById("auth-error");
const profileModal = document.getElementById("profile-modal");
const profileModalClose = document.getElementById("profile-modal-close");
const profileSuccess = document.getElementById("profile-success");
const profileIncompleteNotice = document.getElementById("profile-incomplete-notice");
const loginForm = document.getElementById("login-form");
const profileForm = document.getElementById("profile-form");
const logoutBtn = document.getElementById("logout-btn");
const onboardingOverlay = document.getElementById("onboarding-overlay");
const onboardingForm = document.getElementById("onboarding-form");
const onboardingError = document.getElementById("onboarding-error");

let token = localStorage.getItem("jwt_token");

function openAuthModal() {
  authModal.classList.remove("hidden");
  authError.classList.add("hidden");
}

function openProfileModal() {
  profileModal.classList.remove("hidden");
  profileSuccess.classList.add("hidden");
  loadProfile();
}

function updateAuthButtons() {
  if (token) {
    if (registerBtn) registerBtn.classList.add("hidden");
  } else {
    if (registerBtn) registerBtn.classList.remove("hidden");
    profileName.textContent = "Connect";
  }
}

profileBtn.addEventListener("click", () => {
  if (token) openProfileModal();
  else openAuthModal();
});

authModalClose.addEventListener("click", () => authModal.classList.add("hidden"));
profileModalClose.addEventListener("click", () => profileModal.classList.add("hidden"));
authModal.addEventListener("click", (e) => { if (e.target === authModal) authModal.classList.add("hidden"); });
profileModal.addEventListener("click", (e) => { if (e.target === profileModal) profileModal.classList.add("hidden"); });
diagLoginLink.addEventListener("click", () => openAuthModal());

function getRadioValue(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value === "yes" : false;
}

function setRadioValue(name, value) {
  const val = value ? "yes" : "no";
  const radio = document.querySelector(`input[name="${name}"][value="${val}"]`);
  if (radio) radio.checked = true;
}

function getOnboardingRadio(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value === "yes" : false;
}

function isProfileIncomplete(user) {
  return !user.blood_type || !user.avg_heartbeat || !user.age || !user.height || !user.weight
    || !user.medical_history || !user.current_illness;
}

function showOnboarding(user) {
  document.getElementById("ob-age").value = user.age || "";
  document.getElementById("ob-blood-type").value = user.blood_type || "";
  document.getElementById("ob-height").value = user.height || "";
  document.getElementById("ob-weight").value = user.weight || "";
  document.getElementById("ob-heartbeat").value = user.avg_heartbeat || "";
  document.getElementById("ob-family-details").value = user.family_illness_history || "";
  document.getElementById("ob-medical-history").value = user.medical_history || "";
  document.getElementById("ob-current-illness").value = user.current_illness || "";
  setRadioValue("ob-family-illness", user.family_has_illness);
  setRadioValue("ob-smokes", user.smokes);
  onboardingOverlay.classList.remove("hidden");
}

function hideOnboarding() {
  onboardingOverlay.classList.add("hidden");
  onboardingError.classList.add("hidden");
}

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("login-email").value;
  const password = document.getElementById("login-password").value;
  const formData = new URLSearchParams();
  formData.append("username", email);
  formData.append("password", password);

  try {
    const res = await fetch("/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Login failed");
    }
    const data = await res.json();
    token = data.access_token;
    localStorage.setItem("jwt_token", token);
    authModal.classList.add("hidden");
    updateAuthButtons();
    loadProfile();
    updateDiagPanel();
  } catch (err) {
    authError.textContent = err.message;
    authError.classList.remove("hidden");
  }
});

onboardingForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  onboardingError.classList.add("hidden");

  const payload = {
    age: parseInt(document.getElementById("ob-age").value, 10),
    height: parseFloat(document.getElementById("ob-height").value),
    weight: parseFloat(document.getElementById("ob-weight").value),
    blood_type: document.getElementById("ob-blood-type").value,
    avg_heartbeat: parseInt(document.getElementById("ob-heartbeat").value, 10),
    family_has_illness: getOnboardingRadio("ob-family-illness"),
    family_illness_history: document.getElementById("ob-family-details").value.trim() || null,
    smokes: getOnboardingRadio("ob-smokes"),
    medical_history: document.getElementById("ob-medical-history").value.trim(),
    current_illness: document.getElementById("ob-current-illness").value.trim(),
  };

  try {
    const res = await fetch("/users/me", {
      method: "PUT",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to save medical profile");
    hideOnboarding();
    loadProfile();
  } catch (err) {
    onboardingError.textContent = err.message;
    onboardingError.classList.remove("hidden");
  }
});

async function loadProfile() {
  if (!token) {
    profileName.textContent = "Connect";
    historyList.innerHTML = '<p class="text-muted" id="empty-history">No analyses performed yet.</p>';
    updateAuthButtons();
    return;
  }

  try {
    const res = await fetch("/users/me", { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) throw new Error("Invalid token");
    const user = await res.json();

    profileName.textContent = user.first_name || user.email.split("@")[0];
    document.getElementById("first-name").value = user.first_name || "";
    document.getElementById("last-name").value = user.last_name || "";
    document.getElementById("age").value = user.age || "";
    document.getElementById("medical-id").value = user.medical_id || "";
    document.getElementById("height").value = user.height || "";
    document.getElementById("weight").value = user.weight || "";
    document.getElementById("blood-type").value = user.blood_type || "";
    document.getElementById("avg-heartbeat").value = user.avg_heartbeat || "";
    document.getElementById("family-illness-history").value = user.family_illness_history || "";
    document.getElementById("medical-history").value = user.medical_history || "";
    document.getElementById("current-illness").value = user.current_illness || "";
    setRadioValue("family-illness", user.family_has_illness);
    setRadioValue("smokes", user.smokes);

    const incomplete = isProfileIncomplete(user);
    profileIncompleteNotice.classList.toggle("hidden", !incomplete);

    if (incomplete) {
      showOnboarding(user);
    } else {
      hideOnboarding();
    }

    updateAuthButtons();
    fetchHistory();
  } catch (err) {
    logout();
  }
}

async function fetchHistory() {
  if (!token || !historyList) return;
  try {
    const res = await fetch("/history/diagnostics", {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      const fallback = await fetch("/history", { headers: { Authorization: `Bearer ${token}` } });
      if (!fallback.ok) return;
      const history = await fallback.json();
      renderAnalysisHistory(history);
      return;
    }
    const history = await res.json();
    historyList.innerHTML = "";
    if (history.length === 0) {
      historyList.innerHTML = '<p class="text-muted" id="empty-history">No analyses performed yet.</p>';
      return;
    }
    history.forEach(item => {
      const historyItem = document.createElement("div");
      historyItem.className = "history-item";
      const diag = item.final_diagnostic;
      const severity = diag.severity || "Unknown";
      const badgeClass = severity === "High" || severity === "Critical" ? "pneumonia" : "normal";
      const ts = new Date(item.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
      historyItem.innerHTML = `
        <div>
          <span class="history-item-name" title="${item.xray_filename}">${item.xray_filename}</span>
          <div class="history-item-date">${ts}</div>
        </div>
        <span class="badge ${badgeClass}">${diag.diagnosis || "Diagnostic"}</span>
      `;
      historyItem.addEventListener("click", () => openHistoryDetail(item));
      historyList.appendChild(historyItem);
    });
  } catch (e) {
    console.error("Failed to fetch history", e);
  }
}

function renderAnalysisHistory(history) {
  if (!historyList) return;
  historyList.innerHTML = "";
  if (history.length === 0) {
    historyList.innerHTML = '<p class="text-muted" id="empty-history">No analyses performed yet.</p>';
    return;
  }
  history.forEach(item => {
    const historyItem = document.createElement("div");
    historyItem.className = "history-item";
    const badgeClass = item.prediction.toLowerCase() === "normal" ? "normal" : "pneumonia";
    const confidencePct = (item.confidence * 100).toFixed(1);
    const ts = new Date(item.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    historyItem.innerHTML = `
      <div>
        <span class="history-item-name" title="${item.filename}">${item.filename}</span>
        <div class="history-item-date">${ts}</div>
      </div>
      <span class="badge ${badgeClass}">${item.prediction} (${confidencePct}%)</span>
    `;
    historyItem.addEventListener("click", () => openSimpleHistoryDetail(item));
    historyList.appendChild(historyItem);
  });
}

profileForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    first_name: document.getElementById("first-name").value,
    last_name: document.getElementById("last-name").value,
    age: document.getElementById("age").value ? parseInt(document.getElementById("age").value, 10) : null,
    height: document.getElementById("height").value ? parseFloat(document.getElementById("height").value) : null,
    weight: document.getElementById("weight").value ? parseFloat(document.getElementById("weight").value) : null,
    blood_type: document.getElementById("blood-type").value || null,
    avg_heartbeat: document.getElementById("avg-heartbeat").value ? parseInt(document.getElementById("avg-heartbeat").value, 10) : null,
    family_has_illness: getRadioValue("family-illness"),
    family_illness_history: document.getElementById("family-illness-history").value || null,
    smokes: getRadioValue("smokes"),
    medical_history: document.getElementById("medical-history").value || null,
    current_illness: document.getElementById("current-illness").value || null,
  };

  try {
    const res = await fetch("/users/me", {
      method: "PUT",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to update profile");
    profileSuccess.classList.remove("hidden");
    setTimeout(() => profileSuccess.classList.add("hidden"), 3000);
    loadProfile();
  } catch (err) {
    console.error(err);
  }
});

function logout() {
  token = null;
  localStorage.removeItem("jwt_token");
  profileName.textContent = "Log In";
  profileModal.classList.add("hidden");
  hideOnboarding();
  mergedRagJson = null;
  selectedDocs = [];
  selectedXray = null;
  ragJsonDisplay.textContent = "Upload and process medical documents to see JSON data here.";
  finalResult.classList.add("hidden");
  updateAuthButtons();
}

logoutBtn.addEventListener("click", () => {
  logout();
  updateDiagPanel();
});

function updateDiagPanel() {
  if (token) {
    diagAuthNotice.classList.add("hidden");
    diagWorkflow.classList.remove("hidden");
  } else {
    diagAuthNotice.classList.remove("hidden");
    diagWorkflow.classList.add("hidden");
    finalResult.classList.add("hidden");
  }
  updateFinalDiagButton();
}

loadProfile().then(() => updateDiagPanel());

// ── History Detail Modal ──────────────────────────────────
const historyDetailModal = document.getElementById("history-detail-modal");
const historyDetailClose = document.getElementById("history-detail-close");

if (historyDetailClose && historyDetailModal) {
  historyDetailClose.addEventListener("click", () => historyDetailModal.classList.add("hidden"));
  historyDetailModal.addEventListener("click", (e) => {
    if (e.target === historyDetailModal) historyDetailModal.classList.add("hidden");
  });
}

function openHistoryDetail(item) {
  const diag = item.final_diagnostic || {};
  const ragData = item.rag_data || {};
  const xray = item.xray_analysis || {};

  // Title and date
  const titleEl = document.getElementById("history-detail-title");
  titleEl.innerHTML = `Diagnostic Result <span class="history-filename">— ${item.xray_filename || 'Unknown'}</span>`;
  const dateEl = document.getElementById("history-detail-date");
  if (item.timestamp) {
    dateEl.textContent = new Date(item.timestamp).toLocaleString("en-US", {
      dateStyle: "medium", timeStyle: "short"
    });
  } else {
    dateEl.textContent = "";
  }

  // Identity banner
  const banner = document.getElementById("history-identity-banner");
  banner.classList.add("hidden");
  banner.classList.remove("identity-match", "identity-mismatch", "identity-unknown");

  // Diagnosis badge
  const badge = document.getElementById("history-detail-diagnosis");
  badge.textContent = diag.diagnosis || "Unknown";
  badge.className = "badge " + ((diag.severity === "High" || diag.severity === "Critical") ? "pneumonia" : "normal");

  // Severity line
  const reportCount = ragData?.documents_count || ragData?.documents?.length || 0;
  document.getElementById("history-detail-severity").textContent =
    `Severity: ${diag.severity || "Unknown"} · Confidence: ${diag.confidence || xray.confidence || "Unknown"} · Reports: ${reportCount}`;

  // Summary
  document.getElementById("history-detail-summary").textContent = diag.clinical_summary || diag.summary || "";

  // Contributions
  const contributions = document.getElementById("history-detail-contributions");
  contributions.innerHTML = `
    <div class="contribution-card"><strong>Your profile</strong><p>${diag.account_contribution || "—"}</p></div>
    <div class="contribution-card"><strong>Medical report</strong><p>${diag.report_contribution || "—"}</p></div>
    <div class="contribution-card"><strong>X-ray</strong><p>${diag.xray_contribution || "—"}</p></div>
  `;

  // Key factors
  const factorsList = document.getElementById("history-detail-factors");
  factorsList.innerHTML = "";
  (diag.key_factors || []).forEach(f => {
    const li = document.createElement("li");
    li.textContent = f;
    factorsList.appendChild(li);
  });

  // Recommendations
  const recList = document.getElementById("history-detail-recommendations");
  recList.innerHTML = "";
  (diag.recommendations || []).forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    recList.appendChild(li);
  });

  // Follow up
  const followup = diag.follow_up || "";
  document.getElementById("history-detail-followup").textContent = followup;
  document.getElementById("history-detail-followup-box").classList.toggle("hidden", !followup);

  // Disclaimer
  document.getElementById("history-detail-disclaimer").textContent = diag.disclaimer || "";

  // Show modal
  historyDetailModal.classList.remove("hidden");
}

function openSimpleHistoryDetail(item) {
  // Title and date
  const titleEl = document.getElementById("history-detail-title");
  titleEl.innerHTML = `X-Ray Analysis <span class="history-filename">— ${item.filename || 'Unknown'}</span>`;
  const dateEl = document.getElementById("history-detail-date");
  if (item.timestamp) {
    dateEl.textContent = new Date(item.timestamp).toLocaleString("en-US", {
      dateStyle: "medium", timeStyle: "short"
    });
  } else {
    dateEl.textContent = "";
  }

  // Identity banner
  const banner = document.getElementById("history-identity-banner");
  banner.classList.add("hidden");

  // Diagnosis badge
  const badge = document.getElementById("history-detail-diagnosis");
  badge.textContent = item.prediction || "Unknown";
  badge.className = "badge " + (item.prediction.toLowerCase() === "normal" ? "normal" : "pneumonia");

  // Severity line
  const confidencePct = (item.confidence * 100).toFixed(1);
  document.getElementById("history-detail-severity").textContent =
    `Confidence: ${confidencePct}%`;

  // Summary
  document.getElementById("history-detail-summary").textContent = "Simple X-ray analysis result. No combined clinical diagnostic history is available for this record.";

  // Contributions
  const contributions = document.getElementById("history-detail-contributions");
  contributions.innerHTML = `
    <div class="contribution-card" style="grid-column: span 3;">
      <strong>X-ray Model Prediction</strong>
      <p>The neural network predicted <b>${item.prediction}</b> with a confidence of <b>${confidencePct}%</b>.</p>
    </div>
  `;

  // Key factors / recommendations / followup
  document.getElementById("history-detail-factors").innerHTML = "";
  document.getElementById("history-detail-recommendations").innerHTML = "";
  document.getElementById("history-detail-followup-box").classList.add("hidden");
  document.getElementById("history-detail-disclaimer").textContent = "Disclaimer: This is an automated preliminary screening tool. Please consult with a healthcare professional for a formal diagnosis.";

  // Show modal
  historyDetailModal.classList.remove("hidden");
}
