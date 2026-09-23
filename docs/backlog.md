# Backlog de travail - PFA Pneumonie

## Statuts
- `TODO`
- `DOING`
- `DONE`
- `BLOCKED`

## Phase 1 - ETL

- ETL-01 | Selection et telechargement du dataset principal | DOING
- ETL-02 | Verification d'integrite des images et labels | DONE
- ETL-03 | Script de nettoyage des donnees invalides/dupliquees | DONE
- ETL-04 | Script de preprocessing (resize, normalization) | DONE
- ETL-05 | Encodage des labels (Normal=0, Pneumonie=1) | DONE
- ETL-06 | Data augmentation train only | DONE
- ETL-07 | Split train/val/test reproductible | DONE
- ETL-08 | Export des datasets transformes | DONE
- ETL-09 | Rapport statistique des classes | DONE

## Phase 2 - Modelisation

- MOD-01 | Baseline CNN | DONE
- MOD-02 | Entrainement baseline + metriques | DOING
- MOD-03 | Transfer learning (MobileNetV2/ResNet) | TODO
- MOD-04 | Tuning hyperparametres | TODO
- MOD-05 | Regularisation et anti-overfitting | TODO
- MOD-06 | Export meilleur modele | TODO

## Phase 3 - API

- API-01 | Initialisation backend FastAPI | DONE
- API-02 | Endpoint /health | DONE
- API-03 | Endpoint /predict (upload image) | DONE
- API-04 | Preprocessing coherent avec training | DONE
- API-05 | Retour JSON avec confidence score | DONE
- API-06 | Tests de l'API | DONE

## Phase 4 - Frontend

- UI-01 | Initialisation app web | DONE
- UI-02 | Page upload image | DONE
- UI-03 | Affichage prediction + score | DONE
- UI-04 | Gestion des erreurs utilisateur | DONE
- UI-05 | Integration frontend/backend | DONE

## Phase 5 - Finalisation

- FIN-01 | Tests end-to-end | TODO
- FIN-02 | Evaluation finale sur test set | TODO
- FIN-03 | Documentation technique | TODO
- FIN-04 | Manuel utilisateur | TODO
- FIN-05 | Slides de soutenance | TODO
