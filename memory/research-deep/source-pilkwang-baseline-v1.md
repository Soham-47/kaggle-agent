# RSNA Knee baseline v1 — DINOv2 slots + report/LLM labels
- ref: pilkwang/rsna-knee-baseline-v1 (https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
- claimed_public: **0.891** (Kaggle page: Best Score 0.891 **V15**; V14 was 0.824 / ~1h28; V9 0.770). V15 metadata: `currentVersionNumber` 15, GPU T4, internet **off**, Apache-2.0. Last run ~2026-08-10. Inputs: competition `rsna-knee-abnormality-detection`, datasets `pilkwang/rsna-knee-llm-labels`, `pilkwang/rsna-knee-weights`, `pilkwang/pilkwang-public-dataset-for-notebooks-figures`, model `metaresearch/dinov2/PyTorch/small/1`.
- backbone / input: **DINOv2** (Kaggle mount `metaresearch/dinov2/PyTorch/small/1`). V15 log: `backbone: 12 blocks, last 6 trainable (10.7M params), feature dim 768`. Study → **6 series slots** (3 planes × two *recovered* acquisition axes; CSV `Fluid_Sensitive` and `Fat_Suppression` are identical, so TR/TE/`ScanOptions`/`SeriesDescription` are re-read from DICOM). Physical-scale resample + **130.0 mm** centre crop → **336 px**. V15 infer: `decode … 336px x 12 slices, crop 130.0 mm`; test cache `(3, 6, 12, 336, 336)`; laterality from DICOM tag or geometry; slice order by **geometry**, not filename. Ensemble: **20 members** from `/kaggle/input/rsna-knee-weights`, **10 windows** each, `submission.csv = rank mean of 20 member(s)`. Source §1: average **ranks**, not raw probs (AUC only reads order).
- labels: **LLM table first, report-rules fallback — not a full gold train table.** Source §2: only a small subset of train studies have the twelve expert columns; every train study has `Report`. `train.csv` has `Report`, `test.csv` does not → no text branch at infer. Two readers emit the same 12 graded scores + confidence: (1) multilingual **rule extractor** (nine-language lexicon, clause polarity, silence ≠ confident-negative); (2) **language-model table** attached as `pilkwang/rsna-knee-llm-labels`. “The pipeline prefers a mounted table of model-read labels when present and runs the lexicon when not; … a partial table falls back per study.” Confidence is a **sample weight**. Graded mention (trace / unmarked / marked), not term-present ⇒ 1. Gold labels are used to **choose the reader**, not as the main train matrix.
- CV: **report-grouped, not scanner/site grouped.** Source §2/§7: identical report text ⇒ one target vector; “must be respected when splitting; §7 does.” Steven Lee Hans 101 (credits this kernel): hash the normalised report and pass it as `groups=` to **GroupKFold**. V15 checkpoints log **folds 0–4** (5-fold; 20 members). Holdout AUC binarises graded targets at the **midpoint** to pick epoch/config only — never a submitted threshold. Forks later add scanner/site grouping because this split still leaks site; **do not treat V15 CV as site-safe**.
- inference: **study directories on the competition mount, not the 3-row public CSV alone.** V15 log: `input root: /kaggle/input/competitions/rsna-knee-abnormality-detection`; `sizing for 4407 train + 1322 test studies`; `test header pass: 15 series`; `predicted 3 studies`; `submission.csv … (3, 13); nulls 0`. Public `test.csv` is three placeholder IDs; hidden scoring swaps a larger `test_series/` tree (they already size for **1322** test studies). IDs = folder names under `test_series/<StudyInstanceUID>/`. V15 is **weights-only infer** (~6 min) — it does not retrain.
- copyable next step: one kernel change we can ship on the **CPU metadata ranker** (this contest rejects P100; we should not lift the T4 20-ckpt DINOv2 loop first): **attach `pilkwang/rsna-knee-llm-labels` and train the existing 12 heads on those soft columns + `__conf` weights**, keep our regex extractor only when a study is missing from the table. Same 12 header names including `Baker's`. Expected: better than 0.526 if the ranker is label-limited; still below 0.891 without the image model.
- do not copy: the full 9-language lexicon as a rewrite of our extractor; treating CSV fluid/fat flags as two independent slot axes (they are one bit); H-flip (destroys medial/lateral); random K-fold that splits shared reports; averaging **probabilities** of the 20 public checkpoints; assuming mounted `test.csv` exists or equals the scoring set; training DINOv2 on P100 / `enable_gpu=true` in our current kernel policy.

## Source facts (notebook + V15 log)

From the pulled kernel source (`kernels/pull` V15), opening markdown:

> Each study is a set of MRI series acquired in one session, and the task is to give it twelve probabilities…

> **On the two label sources.** §2 describes two readers for one job: a rule extractor, defined in full below, and a language model reading the same reports, whose output is a public table attached as a dataset. With the table mounted it supplies the targets, without it the extractor does…

> The decisive fact is in the schemas rather than the prose: `train.csv` has a `Report` column and `test.csv` does not.

> averaging *ranks* combines the only information the metric reads.

> Derived labels are not independent across studies. A report shared verbatim by several studies yields one target vector for all of them, which must be respected when splitting; §7 does.

From the V15 kernel log (`kernels/output`):

```
input root: /kaggle/input/competitions/rsna-knee-abnormality-detection
memory: … sizing for 4407 train + 1322 test studies -> 1 group(s) of 3 = 3 slices per slot
weights package: 20 member(s) from /kaggle/input/rsna-knee-weights
test header pass: 15 series
decode group 1/1: 336px x 12 slices, crop 130.0 mm -> 20 member(s)
test laterality: 1 from the tag, 2 from geometry, 0 unresolved
test g1: cache (3, 6, 12, 336, 336) = 0.0 GB
test g1: ordered 12/12 by geometry
backbone: 12 blocks, last 6 trainable (10.7M params), feature dim 768
… fold 0..4: predicted 3 studies over 10 window(s)
submission.csv = rank mean of 20 member(s); (3, 13); nulls 0
```

## Gap vs our 0.526 metadata ranker

We fit binary report/metadata ranks. This kernel fits **soft LLM (or rule) targets**, **weights silence down**, **groups CV on report hash**, and at submit **rank-means a 20-member DINOv2 slot model** over **directory-discovered** test studies. Closest shippable slice is the label table + conf weights, not the GPU ensemble.
