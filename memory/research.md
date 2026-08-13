# research

Distilled from Kaggle API (not browser). Keep short.

## Competition `rsna-knee-abnormality-detection`

### Submission limits

- today: 3
- total: 6
- allowed_now: 2
- limited_by_total: False
- can_submit: True

### Meta files (root CSV/JSON)

- `sample_submission.csv` (470 bytes)
- `test.csv` (212 bytes)
- `test_series.csv` (2213 bytes)

### Leaderboard (public top)

1. Brandon Low — 0.946
2. MKhlystun — 0.946
3. CloseAI — 0.946
4. Lukas Nissen Molvær — 0.943
5. qkrtkddnjs — 0.942
6. Pizza Boy — 0.941
7. JOLEE — 0.938
8. Sida Zuo — 0.937
9. Siontist — 0.936
10. Aastik Rajan15 — 0.936

### Top public kernels

- [RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) votes=306
- [RSNA Knee: read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) votes=205
- [RSNA Knee Abnormalities - Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) votes=138
- [RSNA Knee | Data structure, EDA, baseline🔥](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline) votes=93
- [RSNA Knee DINOv2 at meniscus resolution](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) votes=83
- [[0.899] Let me Cook](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook) votes=79
- [RSNA Knee | DINOsaur V2 🦖 ](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2) votes=77
- [rsna-knee-enhanced-ensemble](https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble) votes=75

### Our recent submissions

- 2026-08-13 14:14:20.853000: status=SubmissionStatus.COMPLETE score=0.526 file=submission.csv
- 2026-08-13 12:45:34.943000: status=SubmissionStatus.COMPLETE score=0.500 file=submission.csv
- 2026-08-13 11:33:30.443000: status=SubmissionStatus.COMPLETE score=0.500 file=submission.csv
- 2026-08-11 18:47:24.517000: status=SubmissionStatus.COMPLETE score=0.500 file=submission.csv
- 2026-08-11 15:17:43.230000: status=SubmissionStatus.COMPLETE score=n/a file=submission.csv
- 2026-08-11 13:15:56.533000: status=SubmissionStatus.COMPLETE score=n/a file=submission.csv

## Browser (read-only)

Source pages for `rsna-knee-abnormality-detection` (not for submit).

### overview

- menu Skip to content Create explore Home emoji_events Competitions leaderboard Benchmarks smart_toy Game Arena code Data Hub expand_more format_list_bulleted More expand_more search ​ Sign In Register Kaggle uses cookies…
- RADIOLOGICAL SOCIETY OF NORTH AMERICA · RESEARCH CODE COMPETITION · 2 MONTHS TO GO Join Competition more_horiz RSNA Knee Abnormality Detection Create a model that can detect knee abnormalities based on multimodal imaging…
- In this competition, you are tasked to build machine learning models that detect a defined set of clinically important abnormalities on knee MRI examinations.
- Start 8 days ago Close 2 months to go Merger & Entry Description link keyboard_arrow_up The knee is the most commonly injured and imaged joint in the body.
- Osteoarthritis alone affects an estimated 654 million people worldwide, while acute knee injuries account for 15 to 40 percent of all sports-related trauma.
- MRIs show clinicians ligaments, cartilage, menisci, and bone in detail, without exposing patients to radiation.
- Reading those scans isn’t always straightforward.
- ACL and MCL tears, meniscal damage, cartilage loss, fractures, and other abnormalities can be subtle, and radiologists don’t always interpret them the same way.
- Access to musculoskeletal radiologists is also limited, especially outside major medical centers, leading to delays and inconsistent diagnoses.
- In this competition, you will develop multimodal machine learning models to detect twelve clinically important knee abnormalities.
- You'll work with the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report, enabling your models to learn from both visual scans and written diagnostic text.
- High-performing models can act as robust decision support tools, delivering the accuracy, consistency, and speed needed to elevate expert-level knee MRI interpretation and improve care across disparate clinic settings.

### discussion

- menu Skip to content Create explore Home emoji_events Competitions leaderboard Benchmarks smart_toy Game Arena code Data Hub expand_more format_list_bulleted More expand_more search ​ Sign In Register Kaggle uses cookies…
- RADIOLOGICAL SOCIETY OF NORTH AMERICA · RESEARCH CODE COMPETITION · 2 MONTHS TO GO Join Competition more_horiz RSNA Knee Abnormality Detection Create a model that can detect knee abnormalities based on multimodal imaging…
- Po-Hao "Howard" Chen · Posted 7d ago arrow_drop_up 11 arrow_drop_down more_horiz push_pin How to get started + Competition's Official Discord María Cruz · Last comment 7d ago by Ali Ahmad arrow_drop_up 3 arrow_drop_down …
- Harsh_2035 · Last comment 2d ago by pandabi arrow_drop_up 26 arrow_drop_down 10 comments more_horiz Public/private test split — stratified by site, or entire sites held out?
- Matteo Vitali · Last comment 5h ago by paul margain arrow_drop_up 3 arrow_drop_down 1 comment more_horiz 58 labelled studies out of 4,407 — the rest of the supervision is in the reports Luka Duvanov · Last comment 21h ag…
- · Last comment 1d ago by Enai arrow_drop_up 2 arrow_drop_down 5 comments more_horiz Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels from the reports Fernando Faria · Last comment 4d…
- Tested for DICOM metadata shortcut Oleksii Zhukov · Last comment 5d ago by agr hmmm arrow_drop_up 17 arrow_drop_down 2 comments more_horiz Classifying labels into 12 abnormalities using Report in train.csv file Malav D M…
- Handudu · Last comment 2d ago by Kupid Technologies arrow_drop_up 2 arrow_drop_down 1 comment more_horiz "Not addressed" is a label too — what we learned reading 4,407 knee reports with an LLM stevenleehans · Last commen…
- Tiago Mazzutti · Posted 4d ago arrow_drop_up 2 arrow_drop_down more_horiz Strong-pipeline replication: SOFT wins 3/3 seeds (+0.014 AUC), but the 58-study CI crosses zero FHZ982 · Posted 4d ago arrow_drop_up 2 arrow_drop_…
- FHZ982 · Last comment 4d ago by Po-Hao "Howard" Chen arrow_drop_up 3 arrow_drop_down 1 comment more_horiz 1 2

## Deep research digest

Method cards for PLAN/CODE. Our public best: 0.526.

**Must implement**

1. Use public kernel methods (imaging or published weight packs), not constant scores.
2. Find test IDs from hidden study folders.
3. Rank-average members. AUC-style metrics only read order.
4. Train labels from reports / mounted label tables. Gold subsets are too small for priors.

**Sources**

- source: https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook — [0.899] Let me Cook; public 0.891
  next: Attach datasets ['input/rsna-knee-b3-v47-folds-0-3', 'input/rsna-knee-weights', 'input/rsna-knee-abnormality-detection'] and reuse their infer path. Our score=0.526.

- source: https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1 — RSNA Knee baseline v1; public 0.85
  next: Pull this kernel and copy its inference ID discovery + rank-average. Our score=0.526.

- source: https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee — RSNA Knee: read the report, then the knee; public 0.906
  next: Pull this kernel and copy its inference ID discovery + rank-average. Rank-average member scores; do not probability-mean. Our score=0.526.

- source: https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline — RSNA Knee | Data structure, EDA, baseline🔥; public unknown
  next: Pull this kernel and copy its inference ID discovery + rank-average. Our score=0.526.

- source: https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2 — RSNA Knee | DINOsaur V2 🦖; public unknown
  next: Pull this kernel and copy its inference ID discovery + rank-average. Rank-average member scores; do not probability-mean. Our score=0.526.

- source: https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution — RSNA Knee DINOv2 at meniscus resolution; public 0.156
  next: Attach datasets ['wguesdon/rsna-knee-llm-report-labels-opus', 'torch/GPU', 'mnt/data/Github/Kaggle/Competitions', 'data/kaggle_mirror/rsna-knee-abnormality-detection', 'data/raw', 'input/rsna-knee-abnormality-detection', 'kaggle_mirror/rsna
