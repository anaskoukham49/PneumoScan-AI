# Phase Tracker - PFA Pneumonie

## Etat global

- Phase 0 - Cadrage et initialisation: `DONE`
- Phase 1 - ETL pipeline: `DOING`
- Phase 2 - Modelisation IA: `DOING`
- Phase 3 - Backend API: `DONE`
- Phase 4 - Frontend: `DONE`
- Phase 5 - Validation et documentation finale: `TODO`

## Regle de suivi

Chaque phase est annoncee ici avec:
- date de debut
- date de fin
- livrables produits
- blocages eventuels

## Historique

### Phase 0 - Cadrage et initialisation
- Debut: 2026-04-29
- Fin: 2026-04-29
- Livrables:
  - structure initiale des dossiers
  - README de projet
  - backlog initial
  - premiere feuille de route technique

### Phase 1 - ETL pipeline
- Debut: 2026-04-29
- Fin: en cours
- Livrables en cours:
  - script ETL executable (`etl/run_etl.py`)
  - configuration ETL (`etl/config.py`)
  - scripts de lancement (`scripts/setup.ps1`, `scripts/run_etl.ps1`)
- Etat: code ETL complet, en attente d'alimentation dataset brut (`data/raw`)

### Phase 2 - Modelisation IA
- Debut: 2026-04-29
- Fin: en cours
- Livrables en cours:
  - script baseline CNN (`model/train_baseline.py`)
  - script de lancement (`scripts/train_baseline.ps1`)

### Phase 3 - Backend API
- Debut: 2026-04-29
- Fin: 2026-04-29
- Livrables:
  - API FastAPI (`api/app.py`)
  - script de lancement (`scripts/run_api.ps1`)
  - test smoke API (`/health` et `/`)

### Phase 4 - Frontend
- Debut: 2026-04-29
- Fin: 2026-04-29
- Livrables:
  - interface web (`frontend/index.html`, `frontend/app.js`, `frontend/styles.css`)
  - integration directe via l'API (route `/` + `/predict`)
