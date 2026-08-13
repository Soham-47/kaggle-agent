# [0.899] Let me Cook

- ref: `aadigupta7686/0-899-let-me-cook` (https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook). Kernel id `130178726`, version 8 in the pull API; UI also shows V2 / "Version 4 of 4". Kaggle page: Copied from renta.k (+7, −2). Internal leftover title in notebook metadata: `RSNA Knee | 0.891 Baseline + Fold-Balanced Rank Ensemble`. Source notebooks listed: `rsna-knee-baseline-v1`, `rsna-knee-read-the-report-then-the-knee`, `knee-rsna-corrected-eda-improved(1)`. Parent public line: Pilkwang Kim `pilkwang/rsna-knee-baseline-v1` (public 0.891).

- claimed_public: 0.899 is a real public LB number, not CV. Kaggle UI: Public Score 0.899, Best Score 0.899 (V2). Runtime 11m 5s, GPU T4 ×2, internet off. Attached: `pilkwang/rsna-knee-weights`, `pilkwang/rsna-knee-llm-labels`, `pilkwang/pilkwang-public-dataset-for-notebooks-figures`, competition `rsna-knee-abnormality-detection`, model `metaresearch/dinov2/PyTorch/small/1`. Do not mix this with the 58-study gold-set hybrid label 0.899 from other notes. That is a label-proxy AUC, not this kernel's LB.

- backbone / input:
  - DINOv2 ViT-S/14 (small), encoder fine-tuned, not frozen. Notebook section: "Why the encoder is trained rather than frozen." Optimizer has two groups: backbone `LR_BACKBONE` and `model.head` `LR_HEAD`, plus `WEIGHT_DECAY`.
  - Study tensor: six slots = three planes × two acquisition axes (fat-suppressed fluid-sensitive vs T1-like). Code: `n_slot = N_SLOT`, "Six slots: three planes crossed with the acquisition axes." Presence mask for missing series; do not put PD/T2 in a T1 slot.
  - Cache constants in this notebook: `CACHE_IMG = 336`, `GROUP = 3` (three slices stacked as one 3-channel encoder input). Cache layout `n_study × n_slot × slices × IMG²` (five axes: study, slot, GROUP, IMG, IMG). 336 px → 0.387 mm/pixel, one 14-px patch token = 5.4 mm.
  - Physical geometry: slices ordered by patient-coordinate position (`k` signed); laterality canonicalised (missing laterality must not default to left: "every right knee among them enters the model mirrored"). No H-flip (coronal/axial would swap sides). Intensity: 1st-99th percentile over the sampled stack, not per slice.
  - Train draws one group per slot (also stack augmentation). Infer can use overlapping windows on the already-cached slices ("looking more costs no extra decoding").
  - Parent log (same cache recipe): 4407 train + 1322 test, "1 group(s) of 3 = 3 slices per slot."

- labels:
  - Twelve study-level findings (same header as the contest): ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
  - Train on report-derived labels, not expert test gold. Prefers the mounted table `pilkwang/rsna-knee-llm-labels` ("mounted table of pre-read report labels"). Soft/graded targets + per-finding confidence sample weights ("a report that never mentions …"). Multilingual lexicon extractor is in the notebook as fallback; if the label dataset is mounted but empty it refuses a silent lexicon train ("LABEL SOURCE: a label dataset is mounted but no usable table was found").
  - `train.csv` has `Report`; `test.csv` does not, so text is not an inference input. Gold table appears only as `gold = …[TARGETS]` for holdout checks (rows with all labels present).

- CV:
  - Fixed 1/5 holdout, not rotated: "One fifth is held out, and the split is fixed rather than rotated."
  - Score math in §1: unweighted mean of 12 AUCs; only rank order matters; they binarise graded labels at the midpoint only to pick an epoch, never as a submit threshold.
  - "Fold-balanced" in the leftover title is the member mix from `rsna-knee-weights`, not a new split invented in the +7/−2 fork.

- inference (this is what scores 0.899):
  - Does not train in the 0.899 run. "When a package of trained members is attached this section reads it; when none is, the sections below train one."
  - Reads weights manifest: `members = man["members"]` then `log(f"weights package: {len(members)} member(s) from {path}")` from `/kaggle/input/rsna-knee-weights`. Sibling kernels on the same package log 20 members.
  - Each member: load checkpoint, rebuild encoder, fingerprint check (`raise` if map/img_size mismatch).
  - Head: project each slot + learned slot identity; one query `q_o` per diagnosis attends over slots (absent slots masked). "The head is deliberately this small."
  - Members combined by rank ("Members are combined by rank, since §1 established order is all the metric reads"). Not a raw-probability mean.
  - Output `submission.csv` on hidden test studies (public `test.csv` is only a 3-row sample). ~11 min on T4×2.

- copyable next step:
  - Drop the metadata ranker (our 0.526). Smallest slice that can move score: same 336 / 6-slot / 3-slice cache + attach `pilkwang/rsna-knee-weights` + rank-average members, internet off, GPU kernel (this contest rejects P100; use T4). Do not retrain 20 DINOv2s on CPU. Keep `Baker's` apostrophe and discover test IDs from study folders, not only the 3-row `test.csv`.

- do not copy:
  - The title or the +7/−2 renta edit as if it were a new model. Architecture and weights are Pilkwang v1 + published weight pack.
  - Frozen DINOv2 at 126 px (wguesdon probe). This scorer is fine-tuned, 336 px.
  - H-flip, filename slice order, trusting `Fluid_Sensitive` blindly, reports at test, random (non-grouped) folds, raw-prob ensemble, or our metadata-only 0.526 path.
  - Treating 0.899 as a 58-study label-hybrid result.

## Notebook facts (citations)

| Fact | Where |
|------|--------|
| Public / best 0.899, T4×2, 11m 5s | Kaggle kernel header / comments |
| Fork of renta.k (+7, −2); leftover title 0.891 + rank ensemble | Kaggle "Copied from"; notebook `kaggle.title` |
| DINOv2 small, GPU, no internet | Kernel pull metadata + model datasource |
| Weights + LLM labels attached | `datasetDataSources`; `preferred_weights_path=/kaggle/input/rsna-knee-weights` |
| Infer if members present, else train one | Notebook prose |
| `CACHE_IMG = 336`, `GROUP = 3`, six slots, rank mix | Notebook code / markdown |
| 0.387 mm/px, 5.4 mm/token; 1-99% stack norm | Notebook markdown |
| Encoder trained; `LR_BACKBONE` / `LR_HEAD` | Notebook markdown + optimizer cell |
| Report labels + confidence weights; no test text | Notebook §2 |
| Fixed 1/5 holdout | Notebook CV section |

Our 0.526 vs their 0.899 is metadata ranks vs MRI DINOv2 members. The gap is the image pipeline and the weight pack, not a trick in seven lines of renta edits.
