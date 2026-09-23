from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    age: Optional[int] = None
    medical_id: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    blood_type: Optional[str] = None
    family_has_illness: Optional[bool] = False
    family_illness_history: Optional[str] = None
    smokes: Optional[bool] = False
    medical_history: Optional[str] = None
    current_illness: Optional[str] = None
    avg_heartbeat: Optional[int] = None

class UserCreate(UserBase):
    password: str = Field(min_length=6)
    age: int = Field(ge=1, le=120)
    height: float = Field(gt=0)
    weight: float = Field(gt=0)
    blood_type: str = Field(min_length=1)
    avg_heartbeat: int = Field(ge=30, le=220)
    medical_history: str = Field(min_length=1)
    current_illness: str = Field(min_length=1)

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    medical_id: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    blood_type: Optional[str] = None
    family_has_illness: Optional[bool] = None
    family_illness_history: Optional[str] = None
    smokes: Optional[bool] = None
    medical_history: Optional[str] = None
    current_illness: Optional[str] = None
    avg_heartbeat: Optional[int] = None

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class AnalysisResponse(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    timestamp: datetime

    class Config:
        from_attributes = True

class ReportAnalysisResponse(BaseModel):
    id: int
    filename: str
    risk_level: str
    summary: str
    key_findings: str
    recommendation: str
    timestamp: datetime

    class Config:
        from_attributes = True

class RagDocumentResult(BaseModel):
    filename: str
    summary: str
    risk_level: str
    key_findings: List[str]
    recommendation: str

class ProcessDocumentsResponse(BaseModel):
    documents: List[RagDocumentResult]
    merged: dict

class FinalDiagnosticResponse(BaseModel):
    patient_context: dict
    rag_data: dict
    xray_analysis: dict
    identity_check: dict
    processed_summary: dict
    final_diagnostic: dict
    gradcam_base64: Optional[str] = None

class DiagnosticHistoryResponse(BaseModel):
    id: int
    xray_filename: str
    final_diagnostic: dict
    rag_data: Optional[dict] = None
    xray_analysis: Optional[dict] = None
    patient_context: Optional[dict] = None
    timestamp: datetime

    class Config:
        from_attributes = True
