# PneumoScan AI

Final year project (PFA) — I wanted to build something actually useful, so I made an AI that helps detect pneumonia from chest X-rays and reads medical reports.

This started as a school project but I tried to make it like a real product: clean data pipeline, a trained model, and a web app you can actually use.

> Not a medical tool — just a school project. Don't use it for real diagnosis.

### What it does

- You upload a chest X-ray, it tells you NORMAL or PNEUMONIA with a confidence score and shows a heatmap (Grad-CAM) of where it looked
- You can also upload PDF medical reports — it pulls the text, does RAG with embeddings, and asks Gemini to summarize the findings
- If you're logged in, it merges everything (your profile + report + X-ray) into one final diagnostic and checks that the report actually belongs to you
- Everything is saved in history so you can go back and see old predictions

<img width="1919" height="871" alt="image" src="https://github.com/user-attachments/assets/7800d6a2-6176-4b5d-bdf8-b6d993e7eeb9" />
<img width="1903" height="869" alt="image" src="https://github.com/user-attachments/assets/9fcaf17c-d971-4771-9c04-6aaaa0f02b38" />

<img width="1406" height="1337" alt="image" src="https://github.com/user-attachments/assets/a7bfcaa2-8c7c-4638-bdf5-43f3a22ebbeb" />




### How I built it

- **ETL**: Python script that cleans and resizes X-rays to 224x224, splits train/val/test, and writes a manifest
- **Model**: Simple CNN baseline on TensorFlow/Keras. Got around 96.4% accuracy and 0.99 AUC on validation — not bad for a baseline. Plots are in `model/artifacts/`
- **API**: FastAPI, does the preprocessing exactly like training, returns JSON
- **RAG**: PyMuPDF to extract text -> chunk -> ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`) -> Gemini for the final summary. Falls back to keywords if there's no API key
- **Frontend**: Plain HTML/CSS/JS, no framework. Served directly by FastAPI

### Project structure

```
api/        -> FastAPI, auth, RAG pipeline
frontend/   -> web UI
etl/        -> data cleaning
model/      -> training, evaluation, gradcam + artifacts
scripts/    -> .ps1 / .bat helpers (including generate_sample_report.py to make a fake PDF for testing)
data/       -> raw X-rays go here (ignored by git)
```

### Run it locally

You need Python 3.11. I used a venv.

```bash
git clone https://github.com/anaskoukham49/PneumoScan-AI.git
cd PneumoScan-AI

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# open .env and put your GEMINI_API_KEY if you have one
# if you don't, it still works — it will just use the fallback

# put your X-ray images in:
#   data/raw/NORMAL/
#   data/raw/PNEUMONIA/
# (I used the Kaggle chest X-ray dataset)

# then:
powershell -ExecutionPolicy Bypass -File scripts/run_etl.ps1
powershell -ExecutionPolicy Bypass -File scripts/train_baseline.ps1

# start the app
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
# or: scripts\run_api.bat
# then open http://127.0.0.1:8000
```

Endpoints if you want to test with curl/Postman:
- `GET /health` — is the model loaded?
- `POST /predict` — just an X-ray
- `POST /final-diagnostic` — X-ray + PDFs + your profile (needs login)

You can also do `Terminal > Run Task... > Run API` in VS Code.

### Model files

The trained weights `baseline_pneumonia.keras` is 128MB so I didn't push it (GitHub blocks >100MB). It's gitignored. You can retrain with `train_baseline.ps1` or download it from Releases if I add it there. The metrics and plots are in `model/artifacts/`.

### Things I'd improve next

- Try transfer learning (MobileNet/ResNet) instead of baseline CNN
- Add Docker + proper tests
- Better error handling and a nicer UI
- Host the model properly

### Notes

- `.env` and `pneumoscan.db` are not tracked — don't push real patient data
- To test the RAG part, generate a fake report locally with `python scripts/generate_sample_report.py` (not tracked in git)

Made by Anas Koukham — feel free to open an issue or reach out if you want to try it.
