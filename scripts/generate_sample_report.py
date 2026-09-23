"""
Script to generate a sample chest X-ray radiology report PDF for testing the RAG feature.
Run with: .\.venv\Scripts\python.exe scripts\generate_sample_report.py
"""
import fitz  # PyMuPDF
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "sample_medical_report.pdf"

REPORT_TEXT = """\
RADIOLOGY REPORT
================

Patient Name:     John Doe
Date of Birth:    1978-03-14
Medical ID:       MED-7F3A9C12
Examination Date: 2026-05-13
Ordering Physician: Dr. Sarah Lemaire, MD
Department:       Pulmonology

EXAMINATION: Chest X-Ray (PA and Lateral Views)
CLINICAL INDICATION: Persistent cough, fever (38.9°C), and dyspnea for 5 days.

FINDINGS:
---------
PA VIEW:
  - There is a focal area of increased opacity in the right lower lobe, consistent
    with lobar consolidation. The opacity measures approximately 5 x 4 cm.
  - Air bronchograms are visible within the area of consolidation, suggesting
    airspace disease.
  - Mild blunting of the right costophrenic angle is noted, indicating a small
    pleural effusion.
  - The left lung fields are clear with no focal airspace opacities.
  - The cardiac silhouette is within normal limits (cardiothoracic ratio < 0.5).
  - The mediastinal contours are unremarkable.
  - No pneumothorax is identified.

LATERAL VIEW:
  - The posterior right lower lobe consolidation is confirmed.
  - No retrosternal mass or mediastinal widening.
  - The diaphragmatic contours are mildly elevated on the right.

LABORATORY CONTEXT (provided by ordering physician):
  - WBC: 14,200 cells/mcL (elevated, normal: 4,500–11,000)
  - CRP: 87 mg/L (elevated, normal: < 10 mg/L)
  - Procalcitonin: 1.2 ng/mL (moderately elevated)
  - O2 Saturation: 93% on room air
  - Sputum culture: Pending

IMPRESSION:
-----------
1. RIGHT LOWER LOBE PNEUMONIA: Radiographic findings are consistent with community-
   acquired pneumonia affecting the right lower lobe. The consolidation pattern,
   combined with clinical symptoms and elevated inflammatory markers, strongly
   supports bacterial etiology.

2. SMALL RIGHT-SIDED PLEURAL EFFUSION: A small parapneumonic pleural effusion is
   noted. This is a common complication of lobar pneumonia and does not appear
   hemodynamically significant at this stage.

3. NO PNEUMOTHORAX OR MEDIASTINAL SHIFT.

RECOMMENDATION:
---------------
- Initiate empiric antibiotic therapy per community-acquired pneumonia guidelines
  (e.g., amoxicillin-clavulanate or a respiratory fluoroquinolone).
- Repeat chest X-ray in 4–6 weeks after completion of antibiotic therapy to confirm
  radiographic resolution.
- Consider CT chest if symptoms persist or worsen despite treatment.
- Monitor oxygen saturation; supplemental oxygen if SpO2 drops below 92%.
- Follow-up with pulmonology if effusion increases or patient deteriorates.

Reported by: Dr. Ahmed Karim, MD, FRCR
Radiologist — Department of Diagnostic Imaging
PneumoScan Medical Center
Date: 2026-05-13
"""

def create_pdf(text: str, output_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Header bar
    page.draw_rect(fitz.Rect(0, 0, 595, 60), color=None, fill=(0.15, 0.36, 0.92))
    page.insert_text((30, 38), "PneumoScan Medical Center — Radiology Report",
                     fontsize=14, color=(1, 1, 1), fontname="helv")

    # Body text
    text_rect = fitz.Rect(40, 75, 555, 820)
    page.insert_textbox(text_rect, text,
                        fontsize=9.5,
                        fontname="cour",
                        color=(0.1, 0.1, 0.1),
                        align=0)

    # Footer
    page.draw_line(fitz.Point(40, 820), fitz.Point(555, 820), color=(0.7, 0.7, 0.7))
    page.insert_text((40, 834), "CONFIDENTIAL — For clinical use only. PneumoScan AI Platform.",
                     fontsize=7.5, color=(0.5, 0.5, 0.5))

    doc.save(str(output_path))
    doc.close()
    print(f"✅ Sample report saved to: {output_path}")

if __name__ == "__main__":
    create_pdf(REPORT_TEXT, OUTPUT)
