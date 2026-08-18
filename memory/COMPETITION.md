# COMPETITION

Active contest only. When you switch competitions, replace this file from `config/competitions/<id>.yaml`.

| Field | Value |
|-------|--------|
| id | rsna_knee |
| slug | rsna-knee-abnormality-detection |
| deadline | 2026-10-22 |
| metric | macro_auc (max) |
| url | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection |

Labels (12): ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture

Submit header: `StudyInstanceUID` + 12 probs. Keep `Baker's` apostrophe.

Data: multimodal knee MRI DICOM. Train on Kaggle only. Workspace: `competitions/rsna_knee/`.

## Ground truth

- Only 58 of 4,407 train studies have official gold labels. Radiology reports supply the other targets through mounted report-label datasets.
- The current 0.526 public best comes from the metadata ranker in `pipeline/ranker.py`. No imaging model has scored yet. Constant output scores about 0.500.
- Studies contain axial, sagittal, and coronal series with fluid-sensitive and fat-suppressed sequences. Drop report text at inference.
- Use grouped study or site folds. Random folds can leak related studies.
