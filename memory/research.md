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

- RADIOLOGICAL SOCIETY OF NORTH AMERICA · RESEARCH CODE COMPETITION · 2 MONTHS TO GO Submit Prediction more_horiz RSNA Knee Abnormality Detection Create a model that can detect knee abnormalities based on multimodal imagin…
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
- Evaluation link keyboard_arrow_up Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets: Final Score = 1 12 ∑ 𝑖 = 0 1…

### discussion

- RADIOLOGICAL SOCIETY OF NORTH AMERICA · RESEARCH CODE COMPETITION · 2 MONTHS TO GO Submit Prediction more_horiz RSNA Knee Abnormality Detection Create a model that can detect knee abnormalities based on multimodal imagin…
- Po-Hao "Howard" Chen · Posted 7d ago arrow_drop_up 11 more_horiz push_pin How to get started + Competition's Official Discord María Cruz · Last comment 7d ago by Ali Ahmad arrow_drop_up 3 1 comment more_horiz All other t…
- Matteo Vitali · Last comment 6h ago by paul margain arrow_drop_up 3 1 comment more_horiz 58 labelled studies out of 4,407 — the rest of the supervision is in the reports Luka Duvanov · Last comment 1d ago by Luka Duvanov…
- · Last comment 1d ago by Enai arrow_drop_up 2 5 comments more_horiz I got a perfect usecase for this competition.
- Harsh_2035 · Last comment 2d ago by pandabi arrow_drop_up 26 10 comments more_horiz YOLO used to be useful in such competition.
- Handudu · Last comment 2d ago by Kupid Technologies arrow_drop_up 2 1 comment more_horiz Your notebook hit an unhandled error while rerunning your code.
- Saanvi Belede Naga · Last comment 3d ago by PC Jimmmy arrow_drop_up -2 3 comments more_horiz reports will be unavailable for the hidden test set?
- Nicolas Pantoja · Last comment 3d ago by Po-Hao "Howard" Chen arrow_drop_up 0 1 comment more_horiz Clarification on MIRA Section 6 hangglider5 · Posted 4d ago arrow_drop_up 0 more_horiz Eligibility under 18 ValkBlox · La…
- Tiago Mazzutti · Posted 4d ago arrow_drop_up 2 more_horiz 1 2

### submissions

- RADIOLOGICAL SOCIETY OF NORTH AMERICA · RESEARCH CODE COMPETITION · 2 MONTHS TO GO Submit Prediction more_horiz RSNA Knee Abnormality Detection Create a model that can detect knee abnormalities based on multimodal imagin…
- If less than 2 are selected, Kaggle will automatically select from your best scoring submissions.
- Learn More Auto-selection candidates help 0/2 All Successful Selected Errors Recent Submission and Description Public Score info Select check_circle rsna-knee-agent 20260813-cards-fix - Version 1 Succeeded · 1h ago · age…

### thread-58-labels

- RADIOLOGICAL SOCIETY OF NORTH AMERICA · RESEARCH CODE COMPETITION · 2 MONTHS TO GO Submit Prediction more_horiz RSNA Knee Abnormality Detection Create a model that can detect knee abnormalities based on multimodal imagin…
- There is no partial labelling to exploit.
- The reports are in seven languages — English, Turkish, Spanish, German, Greek (178 of them, in Greek script), Dutch, French — plus 428 a stopword detector cannot place.
- Turkish negates after the term (efüzyon izlenmedi), so a left-only negation window silently inverts the second-largest language in the corpus.
- Fluid_Sensitive and Fat_Suppression are the same column, identical on all 24,371 series.
- Every study has all three planes and both contrast types — no fallback path needed for a fixed-shape input.
- A plain multilingual keyword matcher, vocabulary not tuned on the 58, gets 0.86 on Baker's cyst and 0.66 on effusion.
- The 58 are not a random sample: ACL prevalence is 41% there against ~20% in the corpus.
- https://www.kaggle.com/code/nekkon/knee-mri-58-labels-for-4-407-studies 1 add_reaction comment 3 Comments 1 appreciation comment Hotness undo redo format_size format_bold format_italic format_strikethrough insert_link fo…
- attach_file Post Comment Malav D Modi Posted 2 days ago arrow_drop_up 1 more_vert I need to ask that my approach i thought was to convert the report to identify labels for rest of report but it cannot happen due to large…
- What else approach i can use to convert my report into labels given in the csv file ?
- reply Reply add_reaction React Luka Duvanov TOPIC AUTHOR Posted a day ago · 1268th in this Competition arrow_drop_up 1 more_vert You don't need to translate the reports.

### thread-dino3

- RADIOLOGICAL SOCIETY OF NORTH AMERICA · RESEARCH CODE COMPETITION · 2 MONTHS TO GO Submit Prediction more_horiz RSNA Knee Abnormality Detection Create a model that can detect knee abnormalities based on multimodal imagin…
- Am I correct in assuming that we can’t use it?
- add_reaction React comment 2 Comments Hotness undo redo format_size format_bold format_italic format_strikethrough insert_link format_quote code format_list_numbered format_list_bulleted table_chart insert_photo smart_di…
- attach_file Post Comment PC Jimmmy Posted 7 days ago · 660th in this Competition arrow_drop_up 2 more_vert Dino v3 has been used in other competitions this year - so I think its good !
- reply Reply add_reaction React PC Jimmmy Posted 7 days ago · 660th in this Competition arrow_drop_up 1 more_vert The attached solution shows a write-up from a person who used Dino V3 - it was NOT one of the models in the…
- You might find code in that competition on how folks handled getting v3.
- I am using it now for this competition, but training local and including the files in my dataset so I have no need to get to the model on kaggle.
- I mention the solution as he had nice things to say about me :) reply Reply add_reaction React PC Jimmmy Posted 6 days ago · 660th in this Competition arrow_drop_up 1 more_vert My V2 model scored 0.775 while pretty much …
- reply Reply add_reaction React 8 more replies arrow_drop_down +2
## Deep research digest

Distilled from articles, papers, notebooks, repos, web.

- The supplied search-result snippets do not contain any Kaggle notebook content for `rsna-knee-abnormality-detection`, nor any `submission.csv`, `sample_submission`, public/private LB score, `macro_auc`, model architecture, fold count, image preprocessing, or ensemble details.
- All snippets are generic AUC-optimization research-paper abstracts: topics include weakly supervised AUC optimization, large-scale AUC maximization, kernelized AUC maximization machines, deep AUC maximization, and surrogate-loss consistency for AUC.
- One snippet names `WSAUC` as a unified framework for weakly supervised AUC optimization covering noisy-label learning, positive-unlabeled learning, multi-instance learning, and a truncated `semi-...` setting.
- One snippet states that Deep AUC Maximization has been applied successfully to imbalanced medical tasks such as chest X-ray classification and skin-lesion classification, but may overfit on small datasets because it aggressively separates positive and negative prediction scores.
- The provided search-result snippets do not contain any Kaggle notebook/code page for `rsna-knee-abnormality-detection`, nor any dataset files such as `sample_submission`, `submission.csv`, `test.csv`, or `test_series.csv`.
- No exact RSNA Knee Abnormality Detection competition metrics, public leaderboard scores, model architectures, CV schemes, or submission-format details are present in the supplied results.
- One unrelated result discusses Macro-AUC for multi-label learning, defining Macro-AUC as the arithmetic mean of class-wise AUCs and emphasizing its relevance under class imbalance.
- One unrelated medical-imaging result describes a chest X-ray multi-label pipeline using SE-ResNeXt101 `(32 × 4d)` fine-tuned for 14 thoracic findings with a sigmoid head and Multilabel Iterative Stratification.
- The provided snippets contain no Kaggle code pages and no RSNA Knee Abnormality Detection content; they reference unrelated topics such as LeakGuard, ECOVNet, the Siberian Radioheliograph, ensemble control systems, and Milnor fibration product maps.
- No verifiable RSNA knee CV/fold/leak/ensemble/DINOv2/EfficientNet metrics, architectures, notebook authors, or Kaggle entities can be extracted from the supplied search results.
- The provided search results do not include any Kaggle Code notebook content for "RSNA Knee Abnormality Detection" or "rsna-knee-abnormality-detection"; none mention RSNA knee data, leaderboard score, macro_auc value, fold CV, submission.csv format, or a competition-specific pipeline.
- The only EfficientNet-related snippets are generic/non-RSNA: one describes EfficientNet as CNNs balancing width, depth, and resolution; another reports EfficientNet experiments for brain tumor, breast cancer mammography, chest cancer, and skin cancer classification, but provides no metrics or Kaggle notebook identifiers.
- The only DINO/DINOv2-related snippets are non-RSNA: one describes a training-free deformable image registration method using DINO features, and another compares SE-ResNet, EfficientNet, and DINOv2 for smartphone-acquired eyelid-parameter measurement including MRD1, MRD2, and Levator Function.
- Several retrieved snippets are unrelated Kaggle/GitHub-style project notes, including web-scraping-books, Admission_prediction, all-the-news text classification, and ETL project guidelines; they provide no usable evidence for RSNA Knee Abnormality Detection methods or scores.

- source: http://arxiv.org/abs/1208.0645v4
- source: http://arxiv.org/abs/1211.5715v1
- source: http://arxiv.org/abs/1710.00760v4
- source: http://arxiv.org/abs/2009.02646v1
- source: http://arxiv.org/abs/2009.11850v2
- source: http://arxiv.org/abs/2201.01145v1
- source: http://arxiv.org/abs/2304.08715v3
- source: http://arxiv.org/abs/2305.05248v2
- source: http://arxiv.org/abs/2305.14258v2
- source: http://arxiv.org/abs/2310.11693v1
- source: http://arxiv.org/abs/2402.15687v1
- source: http://arxiv.org/abs/2407.18100v3
- source: http://arxiv.org/abs/2412.18231v1
- source: http://arxiv.org/abs/2504.00515v1
- source: http://arxiv.org/abs/2504.04422v1
