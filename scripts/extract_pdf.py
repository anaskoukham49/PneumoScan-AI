import fitz

doc = fitz.open(r'c:\Users\anask.ANAS\Desktop\New folder\complex_medical_dossier.pdf')
for i, page in enumerate(doc):
    print(f"=== PAGE {i+1} ===")
    print(page.get_text())
