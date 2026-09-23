"""
Script to generate a 3-page complex clinical dossier for RAG testing.
Includes patient history, labs, multiple imaging reports, and consult notes.
"""
import fitz
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "complex_medical_dossier.pdf"

PAGES = [
    # PAGE 1: ADMISSION & HISTORY
    """\
CLINICAL ADMISSION SUMMARY
--------------------------
Patient: Marcus Vane, 64-year-old Male
Medical ID: MED-V99210-B
Date: 2026-05-10
Status: Urgent Admission

CHIEF COMPLAINT:
Progressive shortness of breath, pleuritic chest pain (right-sided), and productive cough with rust-colored sputum.

HISTORY OF PRESENT ILLNESS:
Mr. Vane is a 64-year-old male with a significant history of COPD and hypertension. He presents with a 10-day history of worsening respiratory distress. Symptoms began with a mild "head cold" that progressed rapidly to high fevers (max 39.5 C) and rigors. He describes the chest pain as sharp, localized to the right mid-axillary line, and worsening with deep inspiration.

PAST MEDICAL HISTORY:
- COPD (Gold Stage II), managed with Tiotropium.
- Hypertension (10 years).
- Type 2 Diabetes Mellitus (HbA1c 7.2%).
- Remote history of smoking (30 pack-years, quit 5 years ago).

SOCIAL HISTORY:
Retired construction worker. Lives at home with spouse. No recent travel. No known COVID-19 exposures.

PHYSICAL EXAMINATION:
- General: Ill-appearing male in moderate respiratory distress.
- Vitals: BP 142/88, HR 112 (tachycardic), RR 26, Temp 39.2 C, SpO2 88% on room air.
- HEENT: Mucous membranes dry. No lymphadenopathy.
- Respiratory: Decreased breath sounds at the right base. Dullness to percussion. Inspiratory crackles and coarse rhonchi heard in the right middle and lower zones. Increased vocal fremitus noted.
- Cardiovascular: S1, S2 regular. No murmurs. JVP not elevated.
- Abdomen: Soft, non-tender.
    """,

    # PAGE 2: LABORATORY & IMAGING
    """\
DIAGNOSTIC WORKUP - RESULTS
---------------------------
REPORT ID: DX-4491-VANE

LABORATORY MEDICINE (BLOOD PANEL):
- WBC Count: 18,900 /mcL (Critical High)
- Neutrophils: 88% (Left shift)
- Hemoglobin: 13.1 g/dL
- Platelets: 210,000 /mcL
- Creatinine: 1.1 mg/dL
- BUN: 24 mg/dL (Elevated)
- C-Reactive Protein (CRP): 142 mg/L (Severe Inflammation)
- Procalcitonin: 4.5 ng/mL (Strongly indicative of bacterial infection)
- Arterial Blood Gas (ABG): pH 7.32, pCO2 48, pO2 58 (Type 1 Respiratory Failure)

IMAGING REPORT 1: CHEST X-RAY (PA/LAT)
Date: 2026-05-10 14:00
Findings: Dense opacification involving the entirety of the right middle lobe and the superior segment of the right lower lobe. 'Air bronchogram' sign is prominent. No hilar mass. Small-to-moderate right pleural effusion.

IMAGING REPORT 2: CT CHEST (NON-CONTRAST)
Date: 2026-05-11 09:30
Detailed Findings: Multifocal consolidation with ground-glass opacities. The most significant finding is a necrotizing pattern within the right middle lobe consolidation, raising concerns for possible abscess formation. Trace pericardial effusion. No pulmonary embolism.
    """,

    # PAGE 3: ASSESSMENT & PLAN
    """\
ASSESSMENT AND CLINICAL PLAN
----------------------------
Physician: Dr. Helena Vask, Pulmonologist

ASSESSMENT:
1. SEVERE LOBAR PNEUMONIA (Right-sided): Presentation is classic for Streptococcus pneumoniae or possibly Staphylococcus aureus given the necrotizing features on CT. High risk for sepsis.
2. ACUTE RESPIRATORY FAILURE: Hypoxemic and mildly hypercapnic.
3. PARAPNEUMONIC EFFUSION: Right-sided.
4. COPD EXACERBATION: Likely triggered by primary infection.

PLAN:
- ADMISSION: Transfer to ICU for close monitoring and high-flow nasal cannula (HFNC).
- ANTIBIOTICS: Start IV Ceftriaxone 2g daily + Azithromycin 500mg daily. Consider adding Vancomycin if MRSA risk increases.
- RESPIRATORY: Salbutamol/Ipratropium nebulizers Q4H.
- FLUIDS: Aggressive hydration with 0.9% Normal Saline.
- PROCEDURES: Thoracentesis requested for 2026-05-12 to sample pleural fluid and rule out empyema.
- MONITORING: Repeat inflammatory markers and CXR in 24 hours.

DISCHARGE CRITERIA (Projected):
- Afebrile for 24 hours.
- Weaned from supplemental oxygen to baseline.
- Improvement in radiological opacities.
- Oral antibiotic transition plan in place.

Electronically Signed: Dr. Helena Vask
Department of Critical Care Medicine
    """
]

def create_complex_pdf(pages: list, output_path: Path):
    doc = fitz.open()
    for i, content in enumerate(pages):
        page = doc.new_page(width=595, height=842)
        # Header
        page.draw_rect(fitz.Rect(0, 0, 595, 50), color=None, fill=(0.12, 0.2, 0.4))
        page.insert_text((30, 32), f"PneumoScan Hospital - Clinical Record (Page {i+1}/3)", 
                         fontsize=12, color=(1,1,1), fontname="helv")
        
        # Body
        text_rect = fitz.Rect(40, 70, 555, 800)
        page.insert_textbox(text_rect, content, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
        
        # Confidential tag
        page.insert_text((40, 825), f"PATIENT: VANE, Marcus | ID: MED-V99210-B | CONFIDENTIAL", 
                         fontsize=8, color=(0.6, 0.6, 0.6), fontname="helv")

    doc.save(str(output_path))
    doc.close()
    print(f"✅ Complex dossier saved to: {output_path}")

if __name__ == "__main__":
    create_complex_pdf(PAGES, OUTPUT)
