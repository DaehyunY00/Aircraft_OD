# Where Should a Fixed Synthetic-Image Budget Go? Class-Allocation Signals for Diffusion-Based Background Augmentation in Military Aircraft Detection

**Authors:** Daehyun Yoo [+ co-authors/affiliation TBD]

> IEEE Access 투고용 초안 v0.1 (2026-08-01).
> 모든 수치는 RESULTS.md에서 검증된 값. [TODO] 표시는 투고 전 채울 것.
> 그림은 placeholder — 목록은 §Figures 참조.

---

## Abstract

Synthetic data generation with diffusion models has become a popular remedy for class imbalance in object detection, and the prevailing practice is to direct the generation budget toward *rare* classes. We show that on a 43-class military aircraft detection benchmark, class frequency is a poor proxy for class difficulty: per-class average precision correlates *negatively* with instance count (Pearson r = −0.375, p = 0.013; Spearman ρ = −0.408, p = 0.007), and the frequency-defined tail outperforms the head under a standard YOLOv8 baseline. Motivated by this, we treat the target-class *allocation signal* as the experimental variable. Holding the generation budget (1,000 images), the number of augmented classes (13), and the generation pipeline (bounding-box-protected background inpainting with per-image verification) fixed, we compare three allocation policies: uniform over the frequency-defined tail, rarity-and-weakness-weighted over the same tail, and a weakness policy that re-ranks *all* classes by measured baseline AP — which selects a class set entirely disjoint from the frequency tail. The outcome is a double dissociation: the frequency-based policy significantly improves only the frequency tail (+0.056 mAP50-95, Wilcoxon p = 0.0017) and the weakness-based policy significantly improves only its own weak set (+0.037, p = 0.040), while reweighting *within* a fixed class set yields no measurable benefit over uniform allocation (+0.003, p = 0.685). These results indicate that for synthetic augmentation, *which classes* receive the budget matters far more than *how the budget is shaped* within a class set, and that measured per-class performance — not frequency — should drive that choice. We release code and generation logs for reproducibility.

**Index Terms** — object detection, data augmentation, diffusion models, image inpainting, long-tailed learning, military aircraft recognition, YOLO

---

## I. Introduction

Object detectors degrade on under-represented classes, and a growing line of work responds by synthesizing additional training images with generative models. In these pipelines a design decision is made — usually implicitly — before a single image is generated: *which classes get the synthetic budget?* The near-universal answer is the rare classes, on the assumption that frequency ranks difficulty.

This paper examines that assumption on a realistic fine-grained benchmark, the 43-class Military Aircraft Detection dataset, and finds it inverted. Under a strong YOLOv8 baseline with default augmentation, per-class AP correlates negatively with instance count (r = −0.375). The rarest class in the frequency-defined tail (YF23, 97 instances) is the *easiest* class in the entire dataset (0.93 AP50), while the weakest classes — Rafale (0.31), F14 (0.37), JAS39 and F18 (0.49) — sit in the medium and head groups. The reason is visible in the confusion structure: distinctive airframes are easy regardless of frequency, whereas visually similar multirole fighters are hard regardless of abundance, and the dominant error mode is missed detection rather than inter-class confusion.

If frequency does not rank difficulty, then a generation budget aimed at the frequency tail is aimed at the wrong target. We turn this observation into a controlled experiment. Using a fixed budget of 1,000 synthetic images produced by one generation pipeline — background inpainting that repaints the scene around protected ground-truth boxes, so labels transfer unchanged — we vary only the *allocation signal*:

1. **Uniform-tail**: equal allocation over the 13 frequency-tail classes;
2. **Selective-tail**: rarity-and-weakness-weighted allocation over the same 13 classes;
3. **Weakness**: all 43 classes re-ranked by measured baseline AP, budget allocated to the bottom 13.

On this dataset the weakness policy selects a class set *disjoint* from the frequency tail (7 head + 6 medium classes), which makes the comparison unusually clean: same budget, same class count, same generator, different signal.

**Contributions.**

- We document, with significance testing, a benchmark where class frequency is a *negative* predictor of class difficulty, and analyze why (distinctive-airframe rarity vs. confusable-fighter abundance; miss-dominated errors).
- We isolate the allocation signal as the experimental variable and find a **double dissociation**: each policy significantly improves exactly the class set it targets (frequency policy: tail +0.056 mAP50-95, p = 0.0017; weakness policy: weak set +0.037, p = 0.040) and not the other.
- We show that **within-set reweighting is ineffectual**: weighting by rarity and weakness inside a fixed class set is statistically indistinguishable from uniform allocation (p = 0.685). The choice *of* the set dominates the shaping *within* it.
- We report an honest baseline comparison in which repeat-factor sampling — at zero generation cost — outperforms all diffusion-based arms in absolute terms, and we delineate when generation-based augmentation is and is not the appropriate tool.

## II. Related Work

**Resampling for long-tailed detection.** Repeat-factor sampling (RFS) [Gupta et al., LVIS 2019] and class-balanced oversampling remain strong, nearly free baselines for imbalanced detection. Our results reinforce this: RFS is the strongest method in our study in absolute mAP. Our contribution is orthogonal — we study how to spend a *generation* budget when one is used.

**Copy-paste and compositional augmentation.** Simple copy-paste [Ghiasi et al., 2021] and its scaled variants (X-Paste [TODO cite], Gen2Det [TODO cite]) compose new scenes from object crops. These change object context but can produce boundary artifacts and require paste-placement heuristics; in our experiments copy-paste is the weakest augmentation arm.

**Diffusion-based augmentation.** Recent work generates training images with text-to-image or inpainting diffusion models [Trabucco et al., DA-Fusion, TODO verify; He et al., TODO verify]. Background inpainting around protected ground-truth boxes [Li et al., ECCV 2024, TODO verify exact citation] is attractive for detection because annotations transfer without relabeling. We adopt this generator and hold it fixed; our variable is not the generator but the allocation of its budget.

**Class-difficulty-aware training.** Loss reweighting and curriculum methods use per-class performance signals during training [TODO cite 1-2]. We use the same signal one level earlier — to decide *where synthetic data goes* — and show it changes where the gains land.

## III. Benchmark Analysis: Frequency Is Not Difficulty

### A. Dataset

We use the public Military Aircraft Detection dataset in YOLO format (43 classes; 11,788 images; 18,832 boxes), split 80/10/10 into train/val/test (9,430 / 1,179 / 1,179 images). The train-split imbalance ratio (max/min instance count) is 10.49, with the smallest class at 86 instances — imbalanced, but far from LVIS-scale (§VII).

Classes are grouped by frequency into head (13) / medium (17) / tail (13) using the bottom-30% rule on instance counts [TODO: state exact rule from config: bottom_percent=0.3, min_val_instances=5].

### B. Frequency–difficulty correlation

Table 1 and Fig. 2 summarize the relationship between train-split instance count and per-class test AP of the `basic_aug` baseline (YOLOv8n, Ultralytics default augmentation, averaged over seeds 42–44):

| split | metric | Pearson r (p) | Spearman ρ (p) |
|---|---|---|---|
| val | mAP50-95 | −0.377 (0.013) | −0.460 (0.002) |
| test | mAP50-95 | **−0.375 (0.013)** | **−0.408 (0.007)** |

The correlation is significantly *negative*: rarer classes are, on average, easier. Group means point the same way: the frequency tail scores 0.6205 test mAP50-95 versus 0.5755 over all classes. The five weakest classes (AP50: Rafale 0.31, F14 0.37, JAS39 0.49, F18 0.49, F22 0.52) are all medium- or head-group fighters with 350–900 instances; the easiest class (YF23, 0.93) has 97.

### C. Error structure

In the normalized confusion matrix of the baseline (Fig. 6), the dominant off-diagonal mass is the *background* row — missed detections — across nearly all classes, consistent with recall (0.52–0.59) trailing precision (0.62–0.71). Inter-class confusion is secondary and concentrated in the visually similar fighter cluster (F15/F16/F18/F22/F35/JAS39). Two implications follow: (i) an augmentation that diversifies backgrounds attacks the dominant (miss) error mode; (ii) for the confusable-fighter cluster, background diversity can recover the recall component but not the discrimination component — an expectation our per-class results bear out (§VI-D).

## IV. Method

### A. Bounding-box-protected background inpainting

Fig. 1 shows the pipeline. For each source image we build an inpainting mask that *protects* every ground-truth box (padding ratio 0.10, Gaussian-blurred boundary, radius 12) and repaints only the complement with Stable Diffusion inpainting [Rombach et al., 2022; runwayml/stable-diffusion-inpainting], 20 denoising steps, guidance 7.5, strength 0.85, at 512 px. Prompts are drawn from a fixed set of five aviation-scene descriptions (clear sky, cloudy sky, runway, desert airfield, generic sky) with a negative prompt forbidding aircraft modification. Because objects are untouched, YOLO labels transfer verbatim — no relabeling and no paste-placement heuristics.

### B. Per-image verification

Every generated image passes three gates or is rejected and regenerated from a different source/seed (up to a 2× attempt multiplier so each class realizes its full budget):

1. **Background changed**: mean absolute pixel difference outside boxes ≥ 10 (rejects no-op generations);
2. **Objects protected**: mean absolute difference inside padded boxes ≤ 5 (rejects object corruption; JPEG-tolerant);
3. **Editable area**: sources with < 5% editable background are excluded up front.

A run aborts if the failure rate exceeds 50%; in practice acceptance rates were 79–82% for all three arms (§V-C), so verification pass-rate differences cannot explain performance differences.

Fig. 5 shows representative outputs. Backgrounds are replaced wholesale — urban skyline to airport runway (a), foliage to mountain cloudscape (b), airfield buildings to storm clouds (c) — while the protected aircraft, its pose, and its label are untouched. Fig. 5(d) documents the pipeline's characteristic failure mode: the gates verify that protected regions are unchanged and that the background *did* change, but they cannot detect a *new* aircraft hallucinated into the repainted background despite the negative prompt. Such objects enter training unlabeled and act as label noise; we quantify the exposure as bounded by the rejection statistics [TODO: manual audit of a 100-image sample to estimate the hallucination rate] and discuss the implication in §VII.

### C. Allocation policies

All policies share budget B = 1,000, class count K = 13, per-class bounds [5, 200], and largest-remainder rounding.

- **Uniform-tail** (`aug_uniform_inpaint`): equal weights over the frequency-tail classes.
- **Selective-tail** (`aug_selective_inpaint`): weights `α·rarity + (1−α)·weakness` over the same tail classes, α = 0.6, where rarity = 1 − normalized log instance count and weakness = 1 − normalized baseline AP (basic_aug, val split).
- **Weakness** (`aug_weakness_inpaint`): *all 43* classes ranked by baseline val AP (basic_aug); the bottom K = 13 receive the budget with weakness weights.

On this dataset the weakness set (C17, EF2000, F14, F15, F16, F18, F22, F35, F4, JAS39, Mirage2000, Rafale, Tornado — 7 head + 6 medium) is **disjoint** from the frequency tail (AG600, Be200, E7, Mig31, P3, RQ4, SR71, Su34, Tu160, Tu95, U2, XB70, YF23). The allocation signal is therefore the only variable separating the three arms.

## V. Experimental Setup

### A. Variants

| variant | synthetic | description |
|---|---|---|
| `real_only` | 0 | Ultralytics augmentation disabled (reference lower bound) |
| `basic_aug` | 0 | Ultralytics default augmentation (**primary baseline**) |
| `aug_oversample` | 1,000 | tail oversampling (duplication) with the selective plan's budget |
| `aug_rfs` | 0 | repeat-factor sampling, t = 0.1 |
| `aug_copy_paste` | 1,000 | tail copy-paste with scale jitter and IoU-constrained placement |
| `aug_uniform_inpaint` | 1,000 | §IV-C uniform-tail |
| `aug_selective_inpaint` | 1,000 | §IV-C selective-tail |
| `aug_weakness_inpaint` | 1,000 | §IV-C weakness |

All variants except `real_only` train on top of the default-augmentation baseline, following the marginal-gain reporting practice of [X-Paste / Gen2Det, TODO cite]. Synthetic images are added to the train split only.

### B. Training and evaluation

YOLOv8n, 640 px, 50 epochs, patience 15, auto batch. Baselines `real_only` and `basic_aug` were trained with seeds 42/43/44 and are reported as seed means; augmentation variants were trained with seed 42. Planning signals (weakness scores, class ranking) use the **val** split only; all reported results are on the held-out **test** split. Total training compute for the augmentation arms was 9.8 h on a single NVIDIA L4; synthetic generation was 3 × ~3.5 h on a T4.

### C. Statistics

We report scope-restricted macro mAP50-95 over three class sets — *all* (43), *tail* (13, frequency-defined), *weak* (13, weakness-plan-defined) — and Wilcoxon signed-rank tests on class-paired per-class AP (seed-averaged), the appropriate paired test given per-class difficulty heterogeneity. Realized synthetic counts equal the plan budget for all arms (1,000/1,000 accepted), with acceptance rates 81.8% (uniform), 80.8% (selective), 79.4% (weakness) over 1,223–1,260 attempts.

## VI. Results

### A. Main results

Test mAP50-95 by evaluation scope (baselines: 3-seed mean):

| variant | synthetic | all (43) | tail (13) | weak (13) |
|---|---|---|---|---|
| real_only | 0 | 0.2744 | 0.3258 | 0.1597 |
| basic_aug | 0 | 0.5755 | 0.6205 | 0.4561 |
| aug_oversample | 1,000 | 0.6008 | 0.6776 | 0.4667 |
| aug_copy_paste | 1,000 | 0.5902 | 0.6588 | 0.4513 |
| aug_uniform_inpaint | 1,000 | 0.5969 | 0.6737 | 0.4756 |
| aug_selective_inpaint | 1,000 | 0.6073 | 0.6767 | 0.4742 |
| aug_weakness_inpaint | 1,000 | 0.5938 | 0.6370 | **0.4932** |
| aug_rfs | 0 | **0.6576** | **0.7116** | **0.5694** |

### B. Double dissociation: the allocation signal decides where gains land

Wilcoxon signed-rank vs. `basic_aug` on class-paired mAP50-95 (Fig. 3):

| arm | tail scope (n=13) | weak scope (n=13) |
|---|---|---|
| selective-tail | **+0.0562, p = 0.0017** | +0.0181, p = 0.273 |
| weakness | +0.0165, p = 0.273 | **+0.0372, p = 0.0398** |

Each policy produces a significant improvement *exactly and only* on the class set it targeted. Because budget, class count, generator, and verification are identical and the two class sets are disjoint, this is a controlled demonstration that the allocation signal determines where synthetic-data gains land. Notably, the two arms are statistically indistinguishable on the *all* scope (−0.0135, p = 0.154): neither policy is globally superior — they move different classes.

### C. Within-set reweighting is a null

Selective-tail vs. uniform-tail on the tail scope: **+0.0030, p = 0.685**. Weighting the same 13 classes by rarity and weakness — the natural refinement most pipelines would reach for — is indistinguishable from equal allocation. Combined with §VI-B, the design guidance is: *the choice of class set dominates; shaping within the set is second-order at this budget.*

### D. Where background inpainting works and where it saturates

Per-class results (Fig. 4) align with the error analysis of §III-C. In the weakness set, classes whose failure was recall-dominated improve markedly under the weakness arm (EF2000 0.520→0.606 AP50; F16 0.574→0.661; C17 0.521→0.625), while the tight fighter cluster improves less — background diversity restores detection but not fine-grained discrimination (e.g., F14 0.365→0.324). This is consistent with the mechanism: inpainting varies context, not object appearance.

### E. Honest baseline: resampling wins on absolute performance

RFS, at zero generation cost, is the best method in every scope (+0.082 all, +0.091 tail, +0.113 weak over basic_aug), and interestingly its largest gain is on the *weak* set — resampling by frequency still helps the confusable fighters because several of them are medium-frequency. Two readings follow. Practically: on datasets of this scale and imbalance, resampling should be exhausted before reaching for generation. Scientifically: RFS operates on a different axis (re-exposure of real images vs. creation of new context), so it does not answer the question this paper poses — *given* a generation budget, where should it go? The two are also composable, which we leave to future work (§VII).

## VII. Discussion and Limitations

**Scope of the frequency–difficulty inversion.** Our benchmark's imbalance ratio is 10.5 with a minimum of 86 instances per class. In extreme long-tail regimes (LVIS-scale, 1000×, few-shot tail classes) frequency plausibly regains predictive power, and the weakness signal itself becomes unreliable (AP estimates on few-shot classes are noisy). Our claim is therefore conditional: *when* frequency and measured difficulty decouple — which practitioners can test with one correlation before generating anything — allocation should follow measured difficulty.

**Single dataset and detector.** Results are demonstrated on one benchmark with YOLOv8n. The controlled-comparison design (fixed budget, disjoint sets) transfers directly to other datasets and detectors; the specific effect sizes may not.

**Generation quality was gate-verified, not scored.** We verify each image against pixel-level gates and report near-identical acceptance rates across arms, but do not report FID/CLIPScore. [TODO: run `src/eval/synthetic_quality.py` and add a table — no retraining needed.]

**Verification blind spot: background hallucination.** The pixel-level gates cannot detect aircraft hallucinated into the repainted background (Fig. 5(d)); such objects enter training without labels. Because all three arms share the identical generator, prompts, and gates, this noise source is matched across arms and cannot produce the differential (double-dissociation) effects of §VI-B — but it plausibly depresses the *absolute* gains of every inpainting arm, and is one candidate explanation for the gap to RFS, which introduces no synthetic pixels at all. An object-level gate (e.g., running the baseline detector on generated backgrounds and rejecting images with confident extra detections) is a direct fix we leave to future work.

**Absolute gains are modest.** The best allocation arm adds ~0.03–0.06 mAP50-95 on its target set from 1,000 images. Whether gains scale with budget (2×, 5×), and whether allocation policies interact with budget size, is open. Composing RFS with weakness-allocated generation is the most promising follow-up suggested by our results.

## VIII. Conclusion

On a 43-class military aircraft benchmark where class frequency is a significantly *negative* predictor of class difficulty, we isolated the class-allocation signal of a fixed diffusion-augmentation budget as the experimental variable. The result is a double dissociation — frequency-based allocation helps only the frequency tail, measured-weakness allocation helps only the measured-weak classes — while reweighting within a fixed class set does nothing. The practical prescription for synthetic augmentation pipelines is one sentence: before generating, measure a baseline, test whether frequency tracks difficulty, and point the budget at the classes that are actually weak.

---

## Figures

| # | 내용 | 파일 | 상태 |
|---|---|---|---|
| Fig. 1 | 파이프라인 개요 (mask → inpaint → verify → allocate) | — | [TODO] 작도 |
| Fig. 2 | instance count vs per-class AP 산점도, r=−0.375 (p=0.013) | `figures/fig2_freq_vs_ap.{pdf,png}` | 완료 |
| Fig. 3 | 이중 해리 2×2 막대 (arm × scope, Wilcoxon 유의성) | `figures/fig3_double_dissociation.{pdf,png}` | 완료 |
| Fig. 4 | weak set 13클래스 dumbbell (basic_aug → weakness arm, AP50) | `figures/fig4_weak_class_change.{pdf,png}` | 완료 |
| Fig. 5 | 실제 생성 예시 4종 (original/mask/generated; (d)는 환각 실패 사례) | `figures/fig5_generation_examples.{pdf,png}` | 완료 |
| Fig. 6 | baseline 정규화 confusion matrix (background 행 = 미검출 지배) | `figures/fig6_confusion_matrix_baseline.png` | 완료 (run 산출물) |
| Table I | 데이터셋 통계 | `dataset_summary.csv` | 수치 확보 |

그림 재생성: `scratchpad/make_figs.py` (데이터: GCS 최신 결과. 로컬 Drive의
`outputs_full/metrics`는 GCP 학습 이전 상태이므로 사용 금지). 색은 Okabe-Ito
기반 CVD-safe 팔레트로 validator 통과 확인.

## References (초안 — 투고 전 정확한 서지 확인 필수)

1. A. Gupta, P. Dollár, R. Girshick, "LVIS: A dataset for large vocabulary instance segmentation," CVPR 2019. (RFS)
2. G. Ghiasi et al., "Simple copy-paste is a strong data augmentation method for instance segmentation," CVPR 2021.
3. R. Rombach et al., "High-resolution image synthesis with latent diffusion models," CVPR 2022.
4. Ultralytics YOLOv8, https://github.com/ultralytics/ultralytics
5. [TODO] X-Paste (ICML 2023?), Gen2Det, DA-Fusion, Li et al. ECCV 2024 background inpainting — 정확한 서지 확인
6. [TODO] 군용 항공기 인식 응용 선행 연구 2–3편 (IJASS/KIMST 계열 포함 권장)
7. F. Wilcoxon, "Individual comparisons by ranking methods," 1945.
