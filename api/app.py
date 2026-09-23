from __future__ import annotations
from typing import List, Optional

import json
import uuid
import base64
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from api import models, database, schemas, auth
from api import rag as rag_pipeline
from model.gradcam import make_gradcam_heatmap, overlay_gradcam

database.migrate_schema()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "baseline_pneumonia.keras"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


app = FastAPI(title="Pneumonia Detection API", version="0.1.0")

@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
model: tf.keras.Model | None = None


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image file.")

    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image


@app.on_event("startup")
def on_startup() -> None:
    global model
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/register")
def register_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "register.html")

@app.get("/history-page")
def history_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "history.html")

@app.get("/metrics")
def get_metrics() -> dict:
    metrics_file = PROJECT_ROOT / "model" / "artifacts" / "baseline_metrics.json"
    if metrics_file.exists():
        import json
        with open(metrics_file, "r") as f:
            data = json.load(f)
            
        # Extract last metrics from history if present
        history = data.get("history", {})
        accuracy = history.get("val_accuracy", [0])[-1]
        auc = history.get("val_auc", [0])[-1]
        
        return {
            "accuracy": accuracy,
            "auc": auc
        }
    return {}

app.mount("/artifacts", StaticFiles(directory=str(PROJECT_ROOT / "model" / "artifacts")), name="artifacts")


def run_xray_analysis(image_bytes: bytes) -> tuple[dict, str | None]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and export the model first.",
        )

    x = preprocess_image_bytes(image_bytes)
    preds = model.predict(x, verbose=0)
    score = float(preds[0][0])
    predicted_label = 1 if score >= 0.5 else 0
    prediction_text = CLASS_NAMES[predicted_label]
    confidence = score if predicted_label == 1 else (1.0 - score)

    gradcam_base64 = None
    try:
        heatmap = make_gradcam_heatmap(x, model, pred_index=0)
        gradcam_img = overlay_gradcam(image_bytes, heatmap)
        _, buffer = cv2.imencode(".jpg", gradcam_img)
        gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"Failed to generate Grad-CAM: {e}")

    xray_result = {
        "prediction": prediction_text,
        "label": predicted_label,
        "confidence": confidence,
        "pneumonia_probability": score,
    }
    return xray_result, gradcam_base64

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    current_user: Optional[models.User] = Depends(auth.get_optional_current_user)
) -> dict:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and export the model first.",
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        x = preprocess_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    xray_result, gradcam_base64 = run_xray_analysis(image_bytes)
    prediction_text = xray_result["prediction"]
    confidence = xray_result["confidence"]
    score = xray_result["pneumonia_probability"]
    
    # Save to database if user is logged in
    if current_user:
        new_analysis = models.Analysis(
            user_id=current_user.id,
            filename=file.filename,
            prediction=prediction_text,
            confidence=confidence
        )
        db.add(new_analysis)
        db.commit()

    return {
        "prediction": prediction_text,
        "label": xray_result["label"],
        "confidence": confidence,
        "pneumonia_probability": score,
        "gradcam_base64": gradcam_base64
    }

@app.post("/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = auth.get_user(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = auth.get_password_hash(user.password)
    auto_medical_id = "MED-" + uuid.uuid4().hex[:8].upper()
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        first_name=user.first_name,
        last_name=user.last_name,
        age=user.age,
        medical_id=auto_medical_id,
        height=user.height,
        weight=user.weight,
        blood_type=user.blood_type,
        family_has_illness=user.family_has_illness or False,
        family_illness_history=user.family_illness_history,
        smokes=user.smokes or False,
        medical_history=user.medical_history,
        current_illness=user.current_illness,
        avg_heartbeat=user.avg_heartbeat,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = auth.get_user(db, email=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user

@app.put("/users/me", response_model=schemas.UserResponse)
def update_user_me(user_update: schemas.UserUpdate, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    update_data = user_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(current_user, key, value)
    db.commit()
    db.refresh(current_user)
    return current_user

@app.post("/analyze-report")
async def analyze_report(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    """RAG-powered medical report analysis. Requires authentication."""
    content_type = file.content_type or ""
    if "pdf" not in content_type and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = rag_pipeline.analyze_report(pdf_bytes)
        
        # Save to database
        import json
        new_report = models.ReportAnalysis(
            user_id=current_user.id,
            filename=file.filename,
            risk_level=result.get("risk_level"),
            summary=result.get("summary"),
            key_findings=", ".join(result.get("key_findings", [])),
            recommendation=result.get("recommendation")
        )
        db.add(new_report)
        db.commit()
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/process-documents", response_model=schemas.ProcessDocumentsResponse)
async def process_documents(
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    """Process uploaded medical documents with RAG and return merged JSON."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one document is required.")

    document_results = []
    raw_results = []

    for upload in files:
        content_type = upload.content_type or ""
        filename = upload.filename or "document.pdf"
        if "pdf" not in content_type and not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {filename}")

        pdf_bytes = await upload.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {filename}")

        result = rag_pipeline.analyze_report(pdf_bytes)
        raw_results.append(result)
        document_results.append(
            schemas.RagDocumentResult(
                filename=filename,
                summary=result.get("summary", ""),
                risk_level=result.get("risk_level", "Unknown"),
                key_findings=result.get("key_findings", []),
                recommendation=result.get("recommendation", ""),
            )
        )

        db.add(
            models.ReportAnalysis(
                user_id=current_user.id,
                filename=filename,
                risk_level=result.get("risk_level"),
                summary=result.get("summary"),
                key_findings=", ".join(result.get("key_findings", [])),
                recommendation=result.get("recommendation"),
            )
        )

    db.commit()
    rag_payload = rag_pipeline.build_rag_payload(raw_results, [r.filename for r in document_results])
    return schemas.ProcessDocumentsResponse(documents=document_results, merged=rag_payload)


@app.post("/final-diagnostic", response_model=schemas.FinalDiagnosticResponse)
async def final_diagnostic(
    xray: UploadFile = File(...),
    rag_json: str = Form("{}"),
    documents: List[UploadFile] = File(default=[]),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    """Combine X-ray CNN analysis, RAG JSON, and patient profile into final diagnostic."""
    content_type = xray.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="X-ray must be an image file.")

    image_bytes = await xray.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="X-ray file is empty.")

    try:
        rag_data = json.loads(rag_json) if rag_json else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid rag_json payload.") from exc

    # Process uploaded PDFs server-side (always takes priority over stale/empty rag_json)
    pdf_files = []
    for upload in documents:
        filename = upload.filename or "document.pdf"
        ct = upload.content_type or ""
        if "pdf" not in ct and not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {filename}")
        pdf_bytes = await upload.read()
        if pdf_bytes:
            pdf_files.append((filename, pdf_bytes))

    if pdf_files:
        rag_data = rag_pipeline.process_uploaded_pdfs(pdf_files)
        for filename, _ in pdf_files:
            doc_result = next(
                (d for d in rag_data.get("documents", []) if d["filename"] == filename), None
            )
            if doc_result:
                db.add(
                    models.ReportAnalysis(
                        user_id=current_user.id,
                        filename=filename,
                        risk_level=doc_result.get("risk_level"),
                        summary=doc_result.get("summary"),
                        key_findings=", ".join(doc_result.get("key_findings", [])),
                        recommendation=doc_result.get("recommendation"),
                    )
                )
        db.commit()
    elif not rag_data.get("documents") and not rag_data.get("merged"):
        # Legacy: rag_json was just the merged dict
        if rag_data.get("summary") or rag_data.get("key_findings"):
            rag_data = {"merged": rag_data, "documents": [], "documents_count": rag_data.get("documents_count", 0)}

    xray_result, gradcam_base64 = run_xray_analysis(image_bytes)
    patient_context = rag_pipeline.build_patient_context(current_user)
    identity_check = rag_pipeline.check_identity_match(patient_context, rag_data)

    processed_summary = {
        "patient": patient_context,
        "documents": rag_data,
        "xray": xray_result,
        "identity_check": identity_check,
    }

    final_result = rag_pipeline.generate_final_diagnostic(
        patient_context, rag_data, xray_result, identity_check
    )

    mismatch = identity_check.get("mismatch_detected", False)

    diagnostic = models.Diagnostic(
        user_id=current_user.id,
        xray_filename=xray.filename or "xray.jpg",
        rag_json=json.dumps(rag_data),
        xray_json=json.dumps(xray_result),
        patient_json=json.dumps(patient_context),
        final_diagnostic=json.dumps(final_result),
    )
    db.add(diagnostic)
    if not mismatch:
        db.add(
            models.Analysis(
                user_id=current_user.id,
                filename=xray.filename or "xray.jpg",
                prediction=xray_result["prediction"],
                confidence=xray_result["confidence"],
            )
        )
    db.commit()

    return schemas.FinalDiagnosticResponse(
        patient_context=patient_context,
        rag_data=rag_data,
        xray_analysis=xray_result,
        identity_check=identity_check,
        processed_summary=processed_summary,
        final_diagnostic=final_result,
        gradcam_base64=None if mismatch else gradcam_base64,
    )

@app.get("/history", response_model=List[schemas.AnalysisResponse])
def get_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    return db.query(models.Analysis).filter(models.Analysis.user_id == current_user.id).order_by(models.Analysis.timestamp.desc()).all()

@app.get("/history/reports", response_model=List[schemas.ReportAnalysisResponse])
def get_report_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    return db.query(models.ReportAnalysis).filter(models.ReportAnalysis.user_id == current_user.id).order_by(models.ReportAnalysis.timestamp.desc()).all()

@app.get("/history/diagnostics", response_model=List[schemas.DiagnosticHistoryResponse])
def get_diagnostic_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    records = (
        db.query(models.Diagnostic)
        .filter(models.Diagnostic.user_id == current_user.id)
        .order_by(models.Diagnostic.timestamp.desc())
        .all()
    )
    return [
        schemas.DiagnosticHistoryResponse(
            id=r.id,
            xray_filename=r.xray_filename,
            final_diagnostic=json.loads(r.final_diagnostic),
            rag_data=json.loads(r.rag_json) if r.rag_json else None,
            xray_analysis=json.loads(r.xray_json) if r.xray_json else None,
            patient_context=json.loads(r.patient_json) if r.patient_json else None,
            timestamp=r.timestamp,
        )
        for r in records
    ]
