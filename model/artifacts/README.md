# Model Artifacts

This directory holds training outputs. The large weight file is **not** tracked in git (GitHub 100MB limit) — see `.gitignore`.

To reproduce:

```powershell
# 1. Prepare data: data/raw/NORMAL + data/raw/PNEUMONIA
powershell -ExecutionPolicy Bypass -File scripts/run_etl.ps1
# 2. Train baseline CNN
powershell -ExecutionPolicy Bypass -File scripts/train_baseline.ps1
# → generates baseline_pneumonia.keras + baseline_metrics.json + plots
```

For releases, upload `baseline_pneumonia.keras` as a GitHub Release asset and link it here.

Tracked in git:
- `baseline_metrics.json` — training metrics
- `*.png` — confusion matrix, ROC, history

Ignored (generate locally or download from Releases):
- `baseline_pneumonia.keras` (128 MB)
