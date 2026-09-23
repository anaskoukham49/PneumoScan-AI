"""
RAG pipeline for medical report analysis.
Flow: PDF bytes → text extraction → chunking → embedding
      → ChromaDB storage → semantic retrieval → Gemini LLM → structured result
"""
from __future__ import annotations

import os
import re
import uuid
from typing import List
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 1. PDF PARSING
# ──────────────────────────────────────────────────────────────────────────────
def parse_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        raise ValueError(f"Could not parse PDF: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TEXT CHUNKING (sliding window)
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# 3. VECTOR STORE (ChromaDB + sentence-transformers)
# ──────────────────────────────────────────────────────────────────────────────
_chroma_client = None
_embedding_fn = None

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.Client()  # in-memory
    return _chroma_client


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embedding_fn


def build_vectorstore(chunks: List[str], collection_name: str):
    """Embed chunks and store in a temporary ChromaDB collection."""
    client = _get_chroma_client()
    ef = _get_embedding_fn()

    # Delete if exists (fresh per upload)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
    )
    ids = [str(i) for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection


def retrieve_context(question: str, collection, n_results: int = 5) -> str:
    """Retrieve the top-K most relevant chunks for a question."""
    results = collection.query(query_texts=[question], n_results=min(n_results, collection.count()))
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)


# ──────────────────────────────────────────────────────────────────────────────
# 4. LLM GENERATION (Gemini)
# ──────────────────────────────────────────────────────────────────────────────
ANALYSIS_QUESTION = (
    "Based on this medical report, provide a structured clinical analysis. "
    "Focus on: respiratory conditions, pneumonia indicators, lung health findings, "
    "key abnormalities, and any relevant lab values or imaging results."
)

SYSTEM_PROMPT = """You are a clinical AI assistant specialized in respiratory medicine.
Analyze the provided medical report excerpts and return a JSON object with EXACTLY this structure:
{
  "patient_info": {
    "name": "Patient full name from the report, or null if not found",
    "age": "Patient age as integer, or null if not found",
    "gender": "Male" | "Female" | "Unknown"
  },
  "summary": "A 2-3 sentence plain-language summary of the report's findings",
  "risk_level": "Low" | "Medium" | "High",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "recommendation": "A brief clinical recommendation or next step"
}
Extract patient_info from the report header or body (name, age, sex/gender).
Base your analysis ONLY on the provided text. Return ONLY valid JSON, no markdown."""


def call_gemini(context: str) -> dict:
    """Call Google Gemini API with model name retries and detailed logging."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _fallback_analysis(context)

    import httpx
    import json

    # Try different model names/versions if one fails
    # Prioritizing the 2.x models since they are available for this key
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-pro"
    ]
    
    prompt = f"{SYSTEM_PROMPT}\n\n=== REPORT EXCERPTS ===\n{context}\n\n=== QUESTION ===\n{ANALYSIS_QUESTION}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }

    last_error = ""
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            print(f"DEBUG: Attempting RAG analysis with model: {model_name}...")
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    try:
                        raw = data['candidates'][0]['content']['parts'][0]['text'].strip()
                        if "```" in raw:
                            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
                        return json.loads(raw)
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        last_error = f"Response parsing error: {e}"
                        print(f"DEBUG: {last_error}")
                else:
                    last_error = f"Model {model_name} failed: {response.status_code}"
                    print(f"DEBUG: {last_error} - {response.text}")
        except Exception as e:
            last_error = str(e)
            print(f"DEBUG: Error with {model_name}: {e}")

    # ULTIMATE FALLBACK: List models to see what this key can actually do
    try:
        print("DEBUG: All standard models failed. Fetching list of available models for this key...")
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        with httpx.Client(timeout=10.0) as client:
            res = client.get(list_url)
            if res.status_code == 200:
                available = [m["name"].split("/")[-1] for m in res.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
                print(f"DEBUG: Models your key actually supports: {available}")
                if available:
                    return {
                        "summary": "Model Mismatch. Your API key doesn't seem to support the standard 'gemini-1.5-flash' name.",
                        "risk_level": "Unknown",
                        "key_findings": [f"Available models for your key: {', '.join(available[:5])}"],
                        "recommendation": "Check the terminal output for the list of models your key supports."
                    }
    except:
        pass

    # If all models fail
    return {
        "summary": f"AI Analysis failed. Your internet connection or API key is likely blocking the request. Last error: {last_error}",
        "risk_level": "Unknown",
        "key_findings": ["Check if you have a VPN or Firewall blocking Google APIs.", "Verify the key at aistudio.google.com"],
        "recommendation": "If your internet is unstable, try again when it is stronger."
    }


def _extract_patient_info_fallback(text: str) -> dict:
    """Extract patient demographics from raw report text."""
    if not text:
        return {"name": None, "age": None, "gender": "Unknown"}

    text_lower = text.lower()

    age = None
    for pattern in [
        r"(\d{1,3})\s*[-]?\s*year[s]?\s*[-]?\s*old",
        r"(?:age|âge|aged?)\s*[:\s]*(\d{1,3})",
        r"(\d{1,3})\s*(?:y/?o|ans?)\b",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            age = int(m.group(1))
            break

    gender = "Unknown"
    if re.search(r"\b(?:female|woman|femme|féminin)\b", text_lower):
        gender = "Female"
    elif re.search(r"\b(?:male|man|homme|masculin)\b", text_lower):
        gender = "Male"

    name = None
    patterns = [
        r"patient\s*:\s*([^,\n]+?)(?:,\s*\d|\n|$)",
        r"patient\s*name\s*:\s*([^,\n]+)",
        r"nom(?:\s*du\s*patient)?\s*:\s*([^,\n]+)",
        r"patient\s*:\s*([A-Za-z]+(?:\s+[A-Za-z]+)+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            candidate = re.sub(r"\s*\d+.*$", "", candidate).strip()
            if len(candidate) > 2 and not candidate.lower().startswith("id"):
                name = candidate.title() if candidate.isupper() else candidate
                break

    return {"name": name, "age": age, "gender": gender}


def _merge_patient_info(primary: dict | None, fallback: dict) -> dict:
    """Merge Gemini extraction with regex fallback from raw PDF text."""
    primary = primary or {}
    return {
        "name": primary.get("name") or fallback.get("name"),
        "age": primary.get("age") if primary.get("age") is not None else fallback.get("age"),
        "gender": (
            primary.get("gender")
            if primary.get("gender") and primary.get("gender") != "Unknown"
            else fallback.get("gender", "Unknown")
        ),
    }


def _fallback_analysis(context: str) -> dict:
    """Simple keyword-based fallback when no Gemini API key is set."""
    text_lower = context.lower()
    high_kw = ["pneumonia", "consolidation", "opacity", "infiltrate", "effusion", "critical"]
    medium_kw = ["atelectasis", "mild", "moderate", "abnormal", "elevated", "infection"]

    findings = []
    for kw in high_kw + medium_kw:
        if kw in text_lower:
            findings.append(f"Report mentions '{kw}'")

    risk = "Low"
    if any(kw in text_lower for kw in high_kw):
        risk = "High"
    elif any(kw in text_lower for kw in medium_kw):
        risk = "Medium"

    return {
        "patient_info": _extract_patient_info_fallback(context),
        "summary": "Basic keyword analysis performed (no Gemini API key configured). Set GEMINI_API_KEY for AI-powered analysis.",
        "risk_level": risk,
        "key_findings": findings[:5] if findings else ["No specific respiratory keywords detected."],
        "recommendation": "Please consult a physician for a proper diagnosis."
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def analyze_report(pdf_bytes: bytes) -> dict:
    text = parse_pdf(pdf_bytes)
    if not text or len(text.strip()) < 50:
        raise ValueError("Could not extract meaningful text from the uploaded PDF.")

    # SHORT-CIRCUIT: If document is small enough, skip RAG and send full text for perfect accuracy
    if len(text) < 10000:
        print(f"DEBUG: Document is small ({len(text)} chars). Sending full text to AI.")
        context = text
    else:
        print(f"DEBUG: Document is large ({len(text)} chars). Using RAG retrieval.")
        chunks = chunk_text(text)
        collection_name = "report_" + uuid.uuid4().hex[:8]
        collection = build_vectorstore(chunks, collection_name)
        context = retrieve_context(ANALYSIS_QUESTION, collection)
        
        # Cleanup the temporary collection
        try:
            _get_chroma_client().delete_collection(collection_name)
        except Exception:
            pass

    # Final check of context quality
    print(f"DEBUG: Context snippet being sent to AI: {context[:200]}...")

    result = call_gemini(context)
    extracted = _extract_patient_info_fallback(text)
    result["patient_info"] = _merge_patient_info(result.get("patient_info"), extracted)
    print(f"DEBUG: Extracted patient_info: {result['patient_info']}")
    return result


def build_patient_context(user) -> dict:
    """Build JSON patient profile from the user ORM object."""
    full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return {
        "full_name": full_name,
        "age": user.age,
        "height_cm": user.height,
        "weight_kg": user.weight,
        "blood_type": user.blood_type,
        "family_has_illness": bool(user.family_has_illness),
        "family_illness_history": user.family_illness_history,
        "smokes": bool(user.smokes),
        "medical_history": user.medical_history,
        "current_illness": user.current_illness,
        "avg_heartbeat_bpm": user.avg_heartbeat,
        "medical_id": user.medical_id,
    }


def merge_rag_results(results: List[dict]) -> dict:
    """Merge multiple RAG document analyses into one JSON summary."""
    if not results:
        return {
            "summary": "No medical documents were uploaded.",
            "risk_level": "Unknown",
            "key_findings": [],
            "recommendation": "Upload medical reports for a richer analysis.",
            "documents_count": 0,
        }

    risk_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
    highest_risk = "Low"
    all_findings: List[str] = []
    summaries: List[str] = []
    recommendations: List[str] = []

    for item in results:
        risk = item.get("risk_level", "Unknown")
        if risk_order.get(risk, 0) > risk_order.get(highest_risk, 0):
            highest_risk = risk
        summaries.append(item.get("summary", ""))
        all_findings.extend(item.get("key_findings", []))
        if item.get("recommendation"):
            recommendations.append(item.get("recommendation"))

    unique_findings = list(dict.fromkeys(f for f in all_findings if f))

    return {
        "summary": " ".join(s for s in summaries if s).strip() or "Documents processed successfully.",
        "risk_level": highest_risk,
        "key_findings": unique_findings[:10],
        "recommendation": recommendations[0] if recommendations else "Consult a physician for interpretation.",
        "documents_count": len(results),
    }


def build_rag_payload(raw_results: List[dict], filenames: List[str], raw_texts: List[str] | None = None) -> dict:
    """Build full RAG payload with per-document results and merged summary."""
    documents = []
    raw_texts = raw_texts or []
    for i, (filename, result) in enumerate(zip(filenames, raw_results)):
        raw_text = raw_texts[i] if i < len(raw_texts) else ""
        from_raw = _extract_patient_info_fallback(raw_text)
        from_summary = _extract_patient_info_fallback(
            result.get("summary", "") + " " + " ".join(result.get("key_findings", []))
        )
        patient_info = _merge_patient_info(
            result.get("patient_info"),
            _merge_patient_info(from_raw, from_summary),
        )
        documents.append({
            "filename": filename,
            "patient_info": patient_info,
            "summary": result.get("summary", ""),
            "risk_level": result.get("risk_level", "Unknown"),
            "key_findings": result.get("key_findings", []),
            "recommendation": result.get("recommendation", ""),
        })
    merged = merge_rag_results(raw_results)
    report_patient = documents[0]["patient_info"] if documents else {}
    return {
        "merged": merged,
        "documents": documents,
        "documents_count": len(documents),
        "report_patient": report_patient,
    }


def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


def _names_match(account_name: str, report_name: str) -> bool:
    a = _normalize_name(account_name)
    r = _normalize_name(report_name)
    if not r:
        return True
    if not a:
        return False
    a_parts = set(a.split())
    r_parts = set(r.split())
    if not a_parts or not r_parts:
        return a == r
    overlap = a_parts & r_parts
    if overlap:
        return True
    return a in r or r in a


def check_identity_match(account: dict, rag_data: dict) -> dict:
    """Compare logged-in account profile with patient identity extracted from reports."""
    documents = rag_data.get("documents", [])
    report_patient = rag_data.get("report_patient") or (
        documents[0].get("patient_info", {}) if documents else {}
    )

    account_summary = {
        "name": account.get("full_name"),
        "age": account.get("age"),
        "blood_type": account.get("blood_type"),
        "current_illness": account.get("current_illness"),
        "medical_history": account.get("medical_history"),
        "smokes": account.get("smokes"),
    }

    if not documents:
        return {
            "match": None,
            "mismatch_detected": False,
            "account_patient": account_summary,
            "report_patient": report_patient,
            "differences": [],
            "message": "No medical report uploaded — diagnostic will use account profile and X-ray only.",
        }

    differences = []
    account_age = account.get("age")
    report_age = report_patient.get("age")
    report_name = report_patient.get("name")
    account_name = account.get("full_name")

    if report_name and account_name and not _names_match(account_name, report_name):
        differences.append(f"Name: account is '{account_name}', report says '{report_name}'")

    if account_age is not None and report_age is not None:
        try:
            if abs(int(account_age) - int(report_age)) >= 5:
                differences.append(f"Age: account is {account_age}, report says {report_age}")
        except (TypeError, ValueError):
            pass

    if not report_name and not report_age:
        return {
            "match": None,
            "mismatch_detected": False,
            "account_patient": account_summary,
            "report_patient": report_patient,
            "differences": ["Could not extract patient name/age from report"],
            "message": "Could not verify patient identity from the report. Please check the uploaded file.",
        }

    mismatch = len(differences) > 0
    if mismatch:
        message = (
            "Identity mismatch detected: the uploaded medical report describes "
            "a different patient than the logged-in account. "
            "No diagnostic will be generated. Please upload the correct medical report."
        )
    else:
        message = "Account profile and medical report appear to describe the same patient."

    return {
        "match": not mismatch,
        "mismatch_detected": mismatch,
        "account_patient": account_summary,
        "report_patient": report_patient,
        "differences": differences,
        "message": message,
    }


FINAL_DIAGNOSTIC_PROMPT_MATCH = """You are a clinical AI assistant specialized in respiratory medicine.

The logged-in ACCOUNT PROFILE and the uploaded MEDICAL REPORT describe the SAME patient.
Produce a unified diagnostic combining:
1. Account registration medical form (demographics, history, current illness)
2. Medical report clinical findings (RAG analysis)
3. Chest X-ray AI result

Return JSON:
{
  "diagnosis": "Primary diagnostic conclusion",
  "severity": "Low" | "Moderate" | "High" | "Critical",
  "confidence": "Low" | "Medium" | "High",
  "clinical_summary": "2-4 sentences synthesizing form, report, and X-ray",
  "account_contribution": "What the account medical form contributes",
  "report_contribution": "What the medical report contributes",
  "xray_contribution": "What the X-ray analysis contributes",
  "identity_status": "match",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "follow_up": "Next clinical steps",
  "disclaimer": "This is an AI-assisted analysis and does not replace professional medical diagnosis."
}
Return ONLY valid JSON."""


def build_mismatch_blocked_response(identity_check: dict) -> dict:
    """Return a blocked response when account and report describe different patients."""
    ap = identity_check.get("account_patient", {})
    rp = identity_check.get("report_patient", {})
    differences = identity_check.get("differences", [])

    explanation = (
        f"The logged-in account belongs to {ap.get('name', 'unknown')} "
        f"(age {ap.get('age', '?')}), but the uploaded medical report describes "
        f"{rp.get('name', 'unknown')} (age {rp.get('age', '?')}, {rp.get('gender', 'unknown')}). "
        "These are different patients, so no diagnostic can be produced."
    )
    if differences:
        explanation += " " + "; ".join(differences) + "."

    return {
        "blocked": True,
        "identity_status": "mismatch",
        "diagnosis": None,
        "severity": None,
        "confidence": None,
        "clinical_summary": None,
        "account_contribution": None,
        "report_contribution": None,
        "xray_contribution": None,
        "mismatch_explanation": explanation,
        "key_factors": [],
        "recommendations": [
            "Upload a medical report that belongs to your account.",
            "Verify the patient name and age on the report match your profile before retrying.",
        ],
        "follow_up": "Correct the uploaded report and run the diagnostic again.",
        "disclaimer": (
            "No diagnostic was generated because the medical report does not match "
            "the logged-in patient."
        ),
    }


FINAL_DIAGNOSTIC_PROMPT_NO_REPORT = """You are a clinical AI assistant specialized in respiratory medicine.
No medical report was uploaded. Base the diagnostic on the account profile and X-ray only.

Return JSON:
{
  "diagnosis": "Primary diagnostic conclusion",
  "severity": "Low" | "Moderate" | "High" | "Critical",
  "confidence": "Low" | "Medium" | "High",
  "clinical_summary": "2-4 sentences from account profile and X-ray",
  "account_contribution": "What the account form contributes",
  "report_contribution": "No report uploaded",
  "xray_contribution": "What the X-ray contributes",
  "identity_status": "no_report",
  "key_factors": ["factor 1", "factor 2"],
  "recommendations": ["recommendation 1"],
  "follow_up": "Next steps",
  "disclaimer": "This is an AI-assisted analysis and does not replace professional medical diagnosis."
}
Return ONLY valid JSON."""


def _fallback_final_diagnostic(
    patient_context: dict, rag_data: dict, xray_analysis: dict, identity_check: dict
) -> dict:
    xray_pred = xray_analysis.get("prediction", "UNKNOWN")
    merged = rag_data.get("merged", rag_data)
    rag_risk = merged.get("risk_level", "Unknown")
    rag_summary = merged.get("summary", "")
    rag_findings = merged.get("key_findings", [])
    has_report = rag_data.get("documents_count", 0) > 0 or merged.get("documents_count", 0) > 0
    mismatch = identity_check.get("mismatch_detected", False)
    account_name = patient_context.get("full_name", "unknown")
    account_age = patient_context.get("age", "N/A")

    severity = "Low"
    if xray_pred == "PNEUMONIA" or rag_risk == "High":
        severity = "High"
    elif rag_risk == "Medium":
        severity = "Moderate"

    factors = [
        f"Account: {account_name}, age {account_age}, illness: {patient_context.get('current_illness', 'N/A')}",
    ]
    if has_report and rag_findings:
        factors.extend([f"Report: {f}" for f in rag_findings[:2]])
    factors.append(f"X-ray: {xray_pred}")

    if mismatch:
        return build_mismatch_blocked_response(identity_check)

    if has_report:
        clinical_summary = (
            f"Same patient confirmed. {account_name} ({account_age}y): "
            f"report findings — {rag_summary[:150] if rag_summary else 'processed'}. X-ray: {xray_pred}."
        )
        identity_status = "match"
    else:
        clinical_summary = f"No report. Assessment for {account_name} ({account_age}y) based on profile and X-ray ({xray_pred})."
        identity_status = "no_report"

    result = {
        "diagnosis": f"Combined diagnostic for {account_name}: {xray_pred}",
        "severity": severity,
        "confidence": "Medium" if has_report else "Low",
        "clinical_summary": clinical_summary,
        "account_contribution": f"Profile: age {account_age}, {patient_context.get('medical_history', 'no history')}",
        "report_contribution": rag_summary if has_report else "No report uploaded",
        "xray_contribution": f"{xray_pred} ({xray_analysis.get('confidence', 0):.0%} confidence)",
        "identity_status": identity_status,
        "key_factors": factors[:5],
        "recommendations": [
            merged.get("recommendation", "Consult a pulmonologist"),
            "Follow up with clinician",
        ],
        "follow_up": "Schedule clinical review.",
        "disclaimer": "This is an AI-assisted analysis and does not replace professional medical diagnosis.",
    }
    return result


def process_uploaded_pdfs(files: list[tuple[str, bytes]]) -> dict:
    """Run RAG on uploaded PDFs and return structured payload."""
    raw_results = []
    filenames = []
    raw_texts = []
    for filename, pdf_bytes in files:
        filenames.append(filename)
        raw_text = parse_pdf(pdf_bytes)
        raw_texts.append(raw_text)
        raw_results.append(analyze_report(pdf_bytes))
    return build_rag_payload(raw_results, filenames, raw_texts)


def generate_final_diagnostic(
    patient_context: dict, rag_data: dict, xray_analysis: dict, identity_check: dict
) -> dict:
    """Combine account profile, report, and X-ray into final diagnostic."""
    if identity_check.get("mismatch_detected"):
        return build_mismatch_blocked_response(identity_check)

    import json

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _fallback_final_diagnostic(patient_context, rag_data, xray_analysis, identity_check)

    import httpx

    if rag_data.get("documents_count", 0) > 0:
        system_prompt = FINAL_DIAGNOSTIC_PROMPT_MATCH
    else:
        system_prompt = FINAL_DIAGNOSTIC_PROMPT_NO_REPORT

    prompt = (
        f"{system_prompt}\n\n"
        f"=== IDENTITY CHECK ===\n{json.dumps(identity_check, indent=2)}\n\n"
        f"=== ACCOUNT PROFILE (connected user) ===\n{json.dumps(patient_context, indent=2)}\n\n"
        f"=== MEDICAL REPORTS (RAG) ===\n{json.dumps(rag_data, indent=2)}\n\n"
        f"=== X-RAY AI (account holder's upload) ===\n{json.dumps(xray_analysis, indent=2)}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }

    for model_name in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if "```" in raw:
                        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
                    result = json.loads(raw)
                    result["identity_status"] = identity_check.get("mismatch_detected") and "mismatch" or (
                        "match" if rag_data.get("documents_count", 0) > 0 else "no_report"
                    )
                    return result
        except Exception as e:
            print(f"DEBUG: Final diagnostic model {model_name} failed: {e}")

    return _fallback_final_diagnostic(patient_context, rag_data, xray_analysis, identity_check)
