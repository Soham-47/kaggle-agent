# RSNA Knee | DINOsaur V2 (romantamrazov)

- ref: https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2 (`romantamrazov/rsna-knee-dinosaur-v2`). Not in `competitions/rsna_knee/research-cache/`. Card is from the public page, Google-indexed cells, comments/output, and the early public fork `nursrijan/lb-0-668-fork-of-rsna-knee-dinosaur-v2` (same v6.1 outline, `scriptVersionId=340507439`). Full `.ipynb` was not pulled.

- claimed_public: 0.808 on the output page (~5 days ago; Best 0.808 V12, that run 3h 22m, T4 x2, then Version 13 of 13). Older public marks: comments 0.650 V8; early fork of the same v6.1 notebook 0.668. Latest listing (17 versions, ~8m 37s T4 x2, attached `knee mri fold weights`) did not show a new LB number in the HTML scrape. Treat 0.808 as the last verified public score, not as a CPU result.

- backbone / input:
  - Writeup title: Dual-Sequence Multi-Level DINOv2 v6.1. Upgrade over a 0.621 multi-plane baseline.
  - Frozen DINOv2 features (not a full ViT fine-tune). Code picks the hub checkpoint whose hidden size is nearest 384 → ViT-S/14. Uses 8 DINO blocks (`dino_layers = 8`). Kaggle model attach: Meta DINO v2 / dinov2 / V1 (the model card says "base"; the 384-d picker is small).
  - Six slots: `PLANES = ["Sagittal", "Coronal", "Axial"]` × `CONTRASTS = ["Fluid", "Structural"]` → `SLOT_NAMES = f"{plane}_{contrast}"`.
  - Writeup: one wide-context three-slice RGB per selected series (2.5D). Live `CFG` (Google cache of current cells): `img_size = 126`, `n_slices = 1`, `n_slots = 6`, `center_header_samples = 11`. Feature cache tag: `dual_sequence_wide2p5d_dino126_l8_multilevel_v1`. Read `n_slices = 1` as one sample station, RGB still three neighboring slices (name says `wide2p5d`).
  - Feature extract on both T4s: `feature_batch_size = 256`, `feature_loader_batch_size = 48`, `feature_workers = 8`, `rebuild_features = False`.
  - Latest inputs also attach knee mri fold weights, resnet-50-RadImageNet-Marwan, RSNA Knee LLM Report Labels, and LLM Report Labels (GPT-5.6-Sol). Those are later add-ons; the published method is frozen DINOv2 + report teacher, not a documented ResNet train.

- labels:
  - Report teacher: official exact labels + conservative multilingual rules + high-confidence rule-derived rows. Exact rows keep full weight; report-derived image labels are down-weighted. `exact_sample_boost = 22.0`.
  - Two image heads: weak-supervision vs exact-only. For OOF branch selection, the report teacher is re-fitted without the held-out fold (snippet cuts there; intent is no teacher leakage into OOF pick).
  - Rank losses: `rank_loss_weight = 0.07`, `finetune_rank_weight = 0.20`, `exact_only_rank_weight = 0.28`. `lr = 6e-4`, `finetune_lr = 1.2e-4`, `exact_only_lr = 2.2e-4`, `weight_decay = 3e-4`, `ema_decay = 0.995`.
  - Explicit `Baker's` alias map (`baker's` / `bakers` → `Baker's`). 12 competition labels.
  - Later versions can swap/stack public LLM label tables. Test-time reports do not exist.

- CV:
  - `StratifiedKFold` (split count not recovered). `seed = 2026`, `debug = False`.
  - Honest exact-label OOF compare of weak-sup vs exact-only heads.
  - Per-target OOF blend + rank ensemble.
  - Notebook TOC: v6.1 fold stability fix. Body not recovered. Do not assume scanner-grouped folds (other public kernels warn random folds leak site).

- inference:
  - Train path (V12/V13): extract frozen DINO features → train heads/EMA → blend/rank-ensemble → `submission.csv`. ~3.4 h on T4 x2.
  - Latest short run (~8.5 min T4 x2) is weight load + infer (`rebuild_features = False`, fold-weight dataset).
  - Dual-GPU only. No P100 on this contest; this kernel is not a CPU recipe.

- copyable next step:
  - Do not port DINOv2, 126 px tokens, or T4 feature cache. Our public is 0.526 metadata+report ranker; contest rejects P100; stay CPU.
  - Smallest slice: keep `pipeline/reports.py`, add dinosaur's teacher weights (exact rows weight 1.0, report-pseudo down-weighted, `exact_sample_boost`-style upsample of the small exact set), then per-target blend of (a) exact-only head/prior and (b) weak report head, chosen on fold-local exact-label AUC with the teacher re-fit inside the train fold only.
  - Optional same-slice metadata: assign one series per (plane × fluid|structural) slot from `test_series` (we already count fluid/plane in `ranker.py`; we do not yet fill six named slots). Still no ViT.

- do not copy:
  - Frozen DINOv2 / 8-block 126 px / dual T4 / attached fold weights.
  - `n_slices=1` + tiny FOV as a "meniscus resolution" idea (wguesdon is 336 px / 150 mm; fleongg's 130 mm crop is a fork, not this notebook).
  - Blind 5-fold copy without a recovered `n_splits`, and without grouped/site folds.
  - Test-time radiology reports.
  - RadImageNet ResNet-50 or extra LLM label datasets until a later card names a CPU use.
  - Submit this kernel or any new LB file.
