# RSNA Knee DINOv2 at meniscus resolution
- ref: wguesdon/rsna-knee-dinov2-at-meniscus-resolution (https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution)
- claimed_public: 0.156
- backbone / input: DINOv2 / ViT mentioned in source
- labels: see notebook (prefer mounted LLM/report tables over gold-only)
- CV: prefer grouped splits (report or site); avoid random folds
- inference: discover hidden test IDs from study folders, not only sample test.csv
- infer_hints: discover_test_ids_from_folders, train_on_report_or_llm_labels
- copyable next step: Attach datasets ['wguesdon/rsna-knee-llm-report-labels-opus', 'torch/GPU', 'mnt/data/Github/Kaggle/Competitions', 'data/kaggle_mirror/rsna-knee-abnormality-detection', 'data/raw', 'input/rsna-knee-abnormality-detection', 'kaggle_mirror/rsna-knee-abnormality-detection'] and reuse their infer path. Our score=0.526.
- do not copy: H-flip on laterality labels; probability-mean ensembles; P100 if host forbids it.

votes: 83
datasets_mentioned: wguesdon/rsna-knee-llm-report-labels-opus, torch/GPU, mnt/data/Github/Kaggle/Competitions, data/kaggle_mirror/rsna-knee-abnormality-detection, data/raw, input/rsna-knee-abnormality-detection, kaggle_mirror/rsna-knee-abnormality-detection
