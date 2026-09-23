const registerForm = document.getElementById("register-form");
const registerError = document.getElementById("register-error");

function splitFullName(fullName) {
  const parts = fullName.trim().split(/\s+/);
  return {
    first_name: parts[0] || "",
    last_name: parts.slice(1).join(" ") || parts[0] || "",
  };
}

function getRadioValue(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value === "yes" : false;
}

function showError(message) {
  registerError.textContent = message;
  registerError.classList.remove("hidden");
}

registerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  registerError.classList.add("hidden");

  const fullName = document.getElementById("reg-fullname").value.trim();
  const bloodType = document.getElementById("reg-blood-type").value;
  const medicalHistory = document.getElementById("reg-medical-history").value.trim();
  const currentIllness = document.getElementById("reg-current-illness").value.trim();

  if (!fullName || !bloodType || !medicalHistory || !currentIllness) {
    showError("Please fill in all required medical fields.");
    return;
  }

  const { first_name, last_name } = splitFullName(fullName);
  const payload = {
    email: document.getElementById("reg-email").value.trim(),
    password: document.getElementById("reg-password").value,
    first_name,
    last_name,
    age: parseInt(document.getElementById("reg-age").value, 10),
    height: parseFloat(document.getElementById("reg-height").value),
    weight: parseFloat(document.getElementById("reg-weight").value),
    blood_type: bloodType,
    family_has_illness: getRadioValue("reg-family-illness"),
    family_illness_history: document.getElementById("reg-family-details").value.trim() || null,
    smokes: getRadioValue("reg-smokes"),
    medical_history: medicalHistory,
    current_illness: currentIllness,
    avg_heartbeat: parseInt(document.getElementById("reg-heartbeat").value, 10),
  };

  try {
    const res = await fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      const detail = err.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg || JSON.stringify(d)).join(", ")
        : detail || "Registration failed";
      throw new Error(msg);
    }

    const formData = new URLSearchParams();
    formData.append("username", payload.email);
    formData.append("password", payload.password);
    const tokenRes = await fetch("/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
    });

    if (!tokenRes.ok) {
      throw new Error("Account created but login failed. Please log in from the dashboard.");
    }

    const data = await tokenRes.json();
    localStorage.setItem("jwt_token", data.access_token);
    window.location.href = "/";
  } catch (err) {
    showError(err.message || "Registration failed");
  }
});
