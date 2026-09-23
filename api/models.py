from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    first_name = Column(String, index=True)
    last_name = Column(String, index=True)
    age = Column(Integer, nullable=True)
    medical_id = Column(String, nullable=True)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    blood_type = Column(String, nullable=True)
    family_has_illness = Column(Boolean, default=False)
    family_illness_history = Column(Text, nullable=True)
    smokes = Column(Boolean, default=False)
    medical_history = Column(Text, nullable=True)
    current_illness = Column(Text, nullable=True)
    avg_heartbeat = Column(Integer, nullable=True)

    analyses = relationship("Analysis", back_populates="user")
    report_analyses = relationship("ReportAnalysis", back_populates="user")
    diagnostics = relationship("Diagnostic", back_populates="user")

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")

class ReportAnalysis(Base):
    __tablename__ = "report_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    risk_level = Column(String)
    summary = Column(Text)
    key_findings = Column(Text)
    recommendation = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="report_analyses")

class Diagnostic(Base):
    __tablename__ = "diagnostics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    xray_filename = Column(String)
    rag_json = Column(Text)
    xray_json = Column(Text)
    patient_json = Column(Text)
    final_diagnostic = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="diagnostics")
