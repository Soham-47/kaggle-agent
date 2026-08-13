# RSNA Knee | Data structure, EDA, baseline🔥
- ref: romanrozen/rsna-knee-data-structure-eda-baseline (https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline)
- claimed_public: unknown
- backbone / input: see notebook
- labels: see notebook (prefer mounted LLM/report tables over gold-only)
- CV: prefer grouped splits (report or site); avoid random folds
- inference: discover hidden test IDs from study folders, not only sample test.csv
- infer_hints: discover_test_ids_from_folders, train_on_report_or_llm_labels, grouped_cv
- copyable next step: Pull this kernel and copy its inference ID discovery + rank-average. Our score=0.526.
- do not copy: H-flip on laterality labels; probability-mean ensembles; P100 if host forbids it.

votes: 93
datasets_mentioned: none
