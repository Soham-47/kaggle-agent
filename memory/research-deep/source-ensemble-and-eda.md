# RSNA Knee: Final Clean 0.891 Baseline + Enhanced Rank Ensemble

- ref: https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble (fork of Tony Li; V3). Weights: `pilkwang/rsna-knee-weights`. Pipeline family: `pilkwang/rsna-knee-baseline-v1`. Local cache of that family: `competitions/rsna_knee/research-cache/rsna-knee-baseline-dual-grouped-folds.ipynb`.
- claimed_public: 0.891 (best 0.891, V3). T4×2, ~1.5 min. Our score 0.526.
- backbone / input: 20 DINOv2-small members, infer only. Decode log: 336 px × 12 slices, 130 mm crop. Laterality from DICOM tag or geometry. Merges three public notebooks: older EDA/metadata audit; the 20-ckpt MRI pipeline from `rsna-knee-baseline-v1`; report-analysis note that its scored MRI path is that same 20-member set.
- labels: 12 study-level findings. Members were trained on report / LLM labels (`pilkwang/rsna-knee-llm-labels`). Hidden test is expert gold; no report path in the scored artifact.
- CV: None in this kernel. Blend uses within-fold scalar holdout from the weight package. Log: `no per-target AUC metadata found`. Do not treat as grouped / scanner-safe CV (see dual-grouped-folds cache).
- inference: `submission.csv` default `RSNA_ENSEMBLE_MODE=hybrid`. Mix: HYBRID_BASE 0.75 + HYBRID_QUALITY 0.20 + HYBRID_WINDOW 0.05 (`QUALITY_TEMPERATURE=0.35`, `QUALITY_SHRINK=0.35`). Hybrid = public-style 20-member mean + quality (holdout scalar) + overlapping-window TTA ranks. Also writes the public-style baseline and three AUC-oriented alts. Log: 20 members, shape `(3, 13)`, nulls=0 (3 public example studies).
- copyable next step: cheap rank hybrid + window TTA on a fold-diverse DINOv2-small set; keep `Baker's`; rank-average, no threshold. Next code unit: the same 0.75/0.20/0.05 mix on our OOF members, not their package.
- do not copy: this is not a new trained model (90 s infer on public ckpts). Do not submit their CSV. Do not put reports on the test path. Do not use their scalar holdout as scanner-safe CV. Skip `ryanholbrook` efficiency LB (host scoreboard, not a model).

# RSNA Knee | Data structure, EDA, baseline

- ref: https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline (V14 best). Same slot / attention family as pilkwang; local cache: `competitions/rsna_knee/research-cache/rsna-knee-baseline-dual-grouped-folds.ipynb`.
- claimed_public: 0.894 (V14). Trail: V7 0.579, V9 0.809 (~1 h 41 m T4×2). V14 ~10 min T4×2 (likely infer / short run, not a from-scratch 4407-study train). Our score 0.526.
- backbone / input: partially unfrozen DINOv2 (ViT-S tags). Last `UNFREEZE_LAST=6` blocks + final norm trainable. Per-diagnosis attention over sequence slots. Cache: 3 groups × 3 slices = 9 per slot. Mount: `/kaggle/input/competitions/rsna-knee-abnormality-detection`. Shared family (cache): slots, laterality norm, mm-pitch resample, geometric slice order. AdamW + `Ema(..., EMA_DECAY=0.997)`.
- labels: same 12 findings; `Baker's` apostrophe (multilingual report map, e.g. Quiste). Own report extractor (measured on corpus). Reports as train-only aux signal distilled into the image encoder and dropped at infer; also discussed as study weights. Not a clean 12-col gold table.
- CV: single holdout, not k-fold: train 3283 / holdout 1124 (annotated held out: 14). Six listed changes are train/val, not new pixels. Not scanner/site grouped. Dual-grouped-folds cache is the safer split (metadata-only leak ~0.65 random vs ~0.60 grouped).
- inference: image-only after report distillation. Pairwise rank term on top of the usual loss: `RANK_LOSS_W=0.05`, `RANK_POS, RANK_NEG = 0.60, 0.40`. Metric is ranking (macro AUC); no threshold. Header must stay `StudyInstanceUID` + 12 names.
- copyable next step: last-6 unfreeze + EMA 0.997 + rank-pair 0.05 on the slot DINOv2, with grouped folds from the cache (not 3283/1124). Next unit: add EMA + rank-pair to one grouped-fold train.
- do not copy: random 3283/1124 holdout; H-flip (laterality); reports at test; dropping the `Baker's` apostrophe; treating V14's 10 min as a full train recipe; ryanholbrook efficiency notebook.
