# [0.899] Let me Cook
- ref: aadigupta7686/0-899-let-me-cook (https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook)
- claimed_public: 0.891
- backbone / input: DINOv2 / ViT mentioned in source
- labels: see notebook (prefer mounted LLM/report tables over gold-only)
- CV: prefer grouped splits (report or site); avoid random folds
- inference: discover hidden test IDs from study folders, not only sample test.csv
- infer_hints: discover_test_ids_from_folders
- copyable next step: Attach datasets ['input/rsna-knee-b3-v47-folds-0-3', 'input/rsna-knee-weights', 'input/rsna-knee-abnormality-detection'] and reuse their infer path. Our score=0.526.
- do not copy: H-flip on laterality labels; probability-mean ensembles; P100 if host forbids it.

votes: 79
datasets_mentioned: input/rsna-knee-b3-v47-folds-0-3, input/rsna-knee-weights, input/rsna-knee-abnormality-detection
