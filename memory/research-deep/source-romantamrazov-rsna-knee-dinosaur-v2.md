# RSNA Knee | DINOsaur V2 🦖 
- ref: romantamrazov/rsna-knee-dinosaur-v2 (https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2)
- claimed_public: unknown
- backbone / input: DINOv2 / ViT mentioned in source
- labels: see notebook (prefer mounted LLM/report tables over gold-only)
- CV: prefer grouped splits (report or site); avoid random folds
- inference: discover hidden test IDs from study folders, not only sample test.csv
- infer_hints: rank_mean_ensemble, train_on_report_or_llm_labels, grouped_cv
- copyable next step: Pull this kernel and copy its inference ID discovery + rank-average. Rank-average member scores; do not probability-mean. Our score=0.526.
- do not copy: H-flip on laterality labels; probability-mean ensembles; P100 if host forbids it.

votes: 77
datasets_mentioned: none
