# Method card: discussions + DINOv2 paper

Contest: `rsna-knee-abnormality-detection` (macro AUC, 12 labels).
Our public score: 0.526 (metadata ranker, exp `20260813-recipe-cpu`). Not a submit note.
Ask: what we must implement vs what is a leak that may not transfer to private LB.

Kaggle discussion HTML is JS-only. Facts below are from search-indexed post text on the three named threads (plus the same-author companion thread/notebooks, labeled as such). Do not treat unrecovered sentences as quotes.

DINOv2 paper is included only because public kernels cite it (`wguesdon`, `pilkwang`, `romanrozen`, and our recipe). Architecture facts only.

## Verdict (implement vs leak)

| Path | Public signal | Transfer? | Action |
|------|---------------|-----------|--------|
| Multilingual report → weak labels on ~4,349 unlabeled studies | Duvanov: almost all supervision is in reports; 58 gold rows are the exception | Must implement for train. Hidden test reports are not confirmed, so do not depend on report text at infer | Extractor + train on weak labels |
| Pixels / MRI model (DINOv2 family in public kernels) | Zhukov: public 0.8-0.93 "seem to reflect image reading"; metadata does not explain 0.932 | Must implement to leave 0.526 | Frozen or lightly tuned encoder + study aggregation |
| Host positive definitions + plane (sagittal/coronal/axial, fluid vs structure) | Chen: official what-counts-as-1 | Must implement (label map + slot/crop design) | Align extractor and heads to host wording |
| 58 gold as train prior / class weight / "CV" | Duvanov: do not set priors from them; 58 is a sicker draw | Leak / bad CV. Will not match private mix | Calibrate only, or ignore for ranking |
| DICOM metadata / series-count ranker (our 0.526 family) | Zhukov: 0.6516 random vs 0.5981 scanner-grouped; series composition alone 0.5954 | Leak. +0.053 is site/scanner memory. Does not become 0.932. May drop on private scanners | Do not chase public LB with more header features |
| Random-fold CV on report labels | Same +0.05 site gap; report style tracks site | Inflated CV, not private-predictive | Group by scanner/fingerprint |

Our 0.526 is below Zhukov's grouped metadata (0.598). That is a weak/broken ranker, not a ceiling. The ceiling of this shortcut is ~0.60 on unseen scanners, not 0.93.

## A) Luka Duvanov: 58 labelled studies of 4407

https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106

### Recovered from 734106 itself

- `train.csv` has 4,407 studies and 58 filled label rows. The other 4,349 have reports only.
- All 12 label columns are filled on the same 58 studies (no partial-label trick). A study is fully annotated or not.
- Do not set priors from the 58.
- Weak labels from reports: seconds on CPU, no GPU hours. Reports are formulaic, which is why string matching works.
- Seven languages: English, Turkish, Spanish, German, Greek (178 reports, Greek script), Dutch, French, plus 428 that a stopword language detector cannot place.
- Example synonym row: Effusion → effusion / efüzyon / derrame / Erguss / …
- Method: build a synonym table per finding, then match.
- Two skip-costs called out as worse than model choice. First recovered in full:
  - Negation is language-specific. Turkish negates after the term: `efüzyon izlenmedi` = no effusion. An English left-only negation window silently inverts the second-largest language.
- Second skip-cost was not recovered in index text. Do not invent it.

### Same-author companion (not 734106 body; do not merge blindly)

Thread [733876](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733876) and notebooks `nekkon/knee-mri-58-labels-for-4-407-studies`, `nekkon/58-studies-cannot-see-a-0-01-gain`:

- The 58 are not a random draw. Enriched ~2× for abnormality. One fixed extractor fires fracture on 19.0% of the 58 vs 6.2% of all 4,407 reports (3.1×).
- Priors / thresholds / class weights from the 58 are set for a sicker population than the (likely) test mix → optimistic.
- 58 studies cannot see a 0.01 gain. One flipped study moves a positive rate by ~1.7 points. Do not rank models on gold-58 CV.
- Practical rule he settled on: rank on weak labels over all 4,407 reports; keep the 58 for calibration only.
- Macro-AUC noise: the three rarest findings are 25% of the weight and 37.6% of the noise.

### Implement vs leak (this source)

| Implement | Leak / will not transfer |
|-----------|--------------------------|
| Multilingual synonym tables for all 12 findings (not English-only, not `OA` as the OA token) | Using the 58 for class weights, prevalence, or "did we gain 0.01?" |
| Right-side and left-side negation (Turkish post-term) | English left-window only → inverted Turkish labels at scale |
| Train imaging heads on weak report labels | Treating report text as a test-time feature unless the host later proves hidden test includes reports |
| Hold 58 out as a tiny calibration / error-analysis set | Fitting a 12-head model only on 58 gold rows |

## A) Oleksii Zhukov: 0.932 LB metadata shortcut

https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517  
Companion kernel: https://www.kaggle.com/code/zhukovoleksiy/rsna-metadata-probe

### What the title is not

0.932 is not a metadata-only score. Title: "0.932 LB within one day. Tested for DICOM metadata shortcut."

TL;DR (his words): DICOM metadata alone reaches 0.6515 macro AUC under random folds but only ~0.598 across unseen scanners. No meaningful shortcuts found. Leaderboard scores seem to reflect image reading. Public LB passed 0.8 and 0.9 within about a day of launch.

### Method (two probes; thresholds written down first)

Probe A, site identifiability. Cluster studies on Manufacturer + … (fingerprint). Result: 265 distinct fingerprints; top 20 cover 45.5% of studies. Finer than institution: he splits scanners and software revisions inside sites, so grouped folds are stricter than true site holdout.

Probe B, metadata → targets. `HistGradientBoosting` on study-level header features. Targets = report-derived labels, not the 58 expert rows (58 is "too few to fit or validate against"). Scored with random 5-fold vs GroupKFold on the fingerprint.

Series composition alone (no DICOM pixel or header reads; the four columns already in `train_series.csv`) → 0.5954. Full header pass adds +0.056 (→ ~0.651). So most of the "metadata model" is how the exam was assembled, which tracks site.

Part of the +0.053 gap "may be metadata predicting reporting style rather than disease."

### Per-label table (random vs grouped; from indexed cells)

Columns: random-fold AUC, scanner-grouped AUC, gap.

| Label | Random | Grouped | Gap |
|-------|--------|---------|-----|
| ACL | 0.705 | 0.670 | 0.035 |
| MCL | 0.683 | 0.648 | 0.035 |
| Medial Meniscus | 0.590 | 0.548 | 0.042 |
| Lateral Meniscus | 0.595 | 0.565 | 0.030 |
| Medial OA | 0.652 | 0.578 | 0.074 |
| Lateral OA | 0.637 | 0.563 | 0.074 |
| PF OA | 0.680 | 0.599 | 0.081 |
| Effusion | 0.628 | 0.582 | 0.046 |
| Synovitis | 0.644 | 0.602 | 0.042 |
| Baker's | 0.765 | 0.717 | 0.048 |
| Contusion | 0.633 | 0.587 | 0.046 |
| Fracture | 0.605 | 0.519 | 0.086 |
| Macro | 0.6516 | 0.5981 | 0.0534 |

OA heads and Fracture leak the most. Baker's is high even grouped (0.717). Protocol/site still helps a fluid collection more than a 1-3 mm tear.

### Implement vs leak (this source)

| Implement | Leak / will not transfer |
|-----------|--------------------------|
| Treat 0.93 public as an imaging problem. Metadata does not get you there | Shipping another header/series ranker to climb public LB (our 0.526 is this family, and it is worse than his grouped 0.598) |
| GroupKFold by scanner fingerprint when ranking experiments | Random / report-only folds (+~0.05 fake AUC) |
| Use metadata only as a leakage diagnostic, not as the submit model | Fitting Manufacturer / field strength / series counts to public test scanners |
| Expect OA + Fracture CV to lie the most under random splits | Believing Baker's 0.76 from metadata is a real cyst detector |

Private-LB risk: public test can share train scanners (265 fingerprints, top 20 = 45.5%). A metadata blend can look fine on public and fall on private sites. Zhukov's grouped 0.598 is the honest upper bound of this shortcut, not a floor for pixels.

## A) Host Chen: Challenge Overview

https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343  
Author: Po-Hao "Howard" Chen, Competition Host. Anatomy credit: Dr. Jacob Kazam.

### Task (must match)

- Detect twelve clinically important knee abnormalities from multimodal imaging.
- First RSNA AI Challenge to pair every imaging study with the original radiology report (multilingual corpus).
- "Models may learn from both the images and the report text." That is train design. The post does not say the hidden test ships reports. Do not assume test-time report input.
- Labels are whole-examination, single knee (study-level, not slice-level).
- Scoring / reference set: each study independently labeled by two subspecialty MSK radiologists; disagreements adjudicated by a third → one consensus truth. That is expert gold on the eval set, not the 58 train rows.

### Planes (implement in slots / crops)

- Cartilage: medial/lateral compartments on sagittal and coronal; patellofemoral on axial.
- Fluid-sensitive (PD / T2, often fat-sat) for edema, effusion, bone-marrow signal.
- Challenge focuses on ACL and MCL tears (LCL is anatomy context, not a target).

### Official positive definitions (align extractor + heads)

Quoted from host text (index fragments; wording is the spec):

| Submit column | Host positive (short) |
|---------------|------------------------|
| ACL | High-grade partial or full-thickness ACL tear: complete discontinuity, or >50% of fibers disrupted, ± secondary signs |
| MCL | High-grade partial or complete acute MCL tear: disrupted fibers + edema in/adjacent to the ligament |
| Medial Meniscus | Abnormal signal that definitely contacts the meniscal surface on ≥2 images, or a morphologic abnormality (tear) |
| Lateral Meniscus | Same criteria on the lateral meniscus |
| Medial OA | Moderate or large (~≥1 cm) high-grade cartilage loss = >50% of cartilage thickness in the medial compartment |
| Lateral OA | Same criteria, lateral compartment |
| PF OA | Same criteria, patellofemoral compartment |
| Effusion | Moderate or large fluid distending the joint (not a trace) |
| Synovitis | Inflammation / thickening of the synovial lining |
| Baker's | Moderate or large fluid collection in the characteristic location behind the knee (popliteal) |
| Contusion | Bone-marrow edema-like signal from impact, without a discrete fracture line |
| Fracture | Acute cortical break or fracture line |

Implement: report rules and imaging heads should match high-grade / moderate-or-large, not "any mention." A report that says "tiny effusion" can be a true negative under host rules.

Leak / mismatch: training on "any keyword hit" (trace fluid, mild chondrosis, chronic MCL sprain) will not match private expert consensus.

## B) DINOv2 paper: architecture only (notebooks cite it)

https://arxiv.org/abs/2304.07193. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (v2 2024).

Cited by public RSNA kernels (`wguesdon` meniscus-resolution, `pilkwang` baseline, `romanrozen` EDA/train, Kaggle model `metaresearch/dinov2/PyTorch/small/1`). Not a contest paper.

### Facts that change our implementation

- Self-supervised ViT. No class labels in pretrain. Features are meant to work without task finetune (linear probe). Frozen-encoder kernels are on-spec, not a hack.
- Patch size 14×14 for the released family (ViT-S/14, B/14, L/14, g/14). Token grid = `(H/14) × (W/14)`.
  - 224 px → 16×16 = 256 patch tokens.
  - 336 px → 24×24 = 576 patch tokens (what the public 336 kernels use).
  - 126 px → 9×9 = 81 tokens. Anatomically coarse for a 1-3 mm meniscus tear (notebook thesis, not the paper).
- ViT-S/14 (the Kaggle "small" weights): ~21M params, width 384, 12 blocks (standard ViT-S). CLS + patch tokens, both 384-d.
- Student / teacher: student is trained; teacher is EMA of the student (not a second backprop net).
- Two losses, untied heads:
  - DINO: image-level, CLS vs CLS on different multi-crops of the same image.
  - iBOT: patch-level, student predicts masked patch tokens vs teacher patches.
- Collapse control: Sinkhorn-Knopp centering (not old DINO softmax centering) + KoLeo batch diversity.
- Data: curated LVD-142M (~142M natural images), not medical MRI. A ~1B ViT-g/14 is distilled into the smaller checkpoints.
- Output we can use: CLS (global) and dense patch map (mean/max pool in `wguesdon`). Natural-image RGB pretrain → MRI slices are usually 3× stacked or repeated channels.

### Not in the paper (do not cargo-cult)

- No knee, no DICOM, no 12-label head, no 336-vs-448 contest result.
- Pretrain is natural photos. Transfer is an empirical bet the kernels already made; the paper does not prove meniscus AUC.

### Implement vs leak (this source)

| Implement | Leak / waste |
|-----------|----------------|
| ViT-S/14, patch 14, input multiple of 14; prefer ~336 if we want public-kernel parity | 126 px "DINOv2" that cannot resolve host's ≥1 cm OA well, let alone 1-3 mm meniscus |
| Use CLS + patch tokens; study-level pool after slices | Retraining ViT-g on 142M (out of scope; paper compute is huge) |
| Laterality-safe augs (no H-flip). Contest constraint, not paper | Assuming ImageNet-style flip is fine because DINOv2 used multi-crop |

## What we must implement next (from these sources only)

1. Report weak labels for 4,349 studies. 7 languages + 428 unknown; per-finding synonyms; bidirectional / post-term negation; host moderate-or-large / high-grade thresholds. CPU is enough.
2. Do not set priors, weights, or model-rank CV from the 58. Optional: calibration only.
3. Imaging model that actually reads pixels (public 0.8-0.93). Dominant cited backbone: frozen DINOv2 ViT-S/14, 14-px patches, ~336 input, slot/plane aggregation. Our 0.526 metadata path is the thing Zhukov already falsified as a 0.93 shortcut.
4. Slots from host anatomy: ACL/MCL + menisci on sag/cor; PF OA on axial; fluid-sensitive series for effusion / synovitis / contusion / Baker.
5. CV = GroupKFold on scanner fingerprint (or at least site). Quote the metadata probe as a canary: if a change only lifts random-fold AUC, drop it.

## What must not be treated as a private-LB plan

1. Metadata / series-count ranker (0.65 random / 0.60 grouped / 0.595 series-only). Public test may share scanners; private may not. This is the leak. It is also our current 0.526.
2. Gold-58 prevalence and "+0.01 on 58". Enriched, underpowered, optimistic vs private mix.
3. Report text at inference until a host/data file proves hidden test reports exist.
4. Keyword-any-mention labels that ignore host severity (trace effusion ≠ 1).
5. Copying 0.932 as if it were a metadata trick. Zhukov measured that and said it is not.

## Sources

- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106 (Duvanov)
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733876 (Duvanov companion)
- https://www.kaggle.com/code/nekkon/knee-mri-58-labels-for-4-407-studies
- https://www.kaggle.com/code/nekkon/58-studies-cannot-see-a-0-01-gain
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517 (Zhukov)
- https://www.kaggle.com/code/zhukovoleksiy/rsna-metadata-probe
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343 (Chen, host)
- https://arxiv.org/abs/2304.07193 and https://arxiv.org/html/2304.07193v2 (DINOv2; cited by public kernels)
- Public kernels that justify the paper:
  https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution  
  https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1  
  https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline
