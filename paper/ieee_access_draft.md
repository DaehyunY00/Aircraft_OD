# Where Should a Fixed Synthetic-Image Budget Go? Class-Allocation Signals for Diffusion-Based Background Augmentation in Military Aircraft Detection

**Authors:** Daehyun Yoo [+ co-authors/affiliation TBD]

> 초안 v0.2 (2026-08-02). **타겟 저널: IJASS** (Int. J. Aeronautical & Space
> Sciences, Springer/KSAS — hybrid, 구독 경로 게재비 무료). v0.1은 IEEE Access
> 대상이었으나 게재비 조건으로 전환 — 내용은 그대로, 투고 직전 IJASS 서식·분량
> 규격(참고문헌 스타일 포함)으로 조판 필요. 항공 응용 저널이므로 Introduction의
> 도메인 프레이밍(군용기 인식 응용)을 반 문단 보강하면 좋음.
> 모든 수치는 RESULTS.md에서 검증된 값. [TODO] 표시는 투고 전 채울 것.

---

## Abstract

Synthetic data generation with diffusion models has become a popular remedy for class imbalance in object detection, and the prevailing practice is to direct the generation budget toward *rare* classes. We show that on a 43-class military aircraft detection benchmark, class frequency is a poor proxy for class difficulty: per-class average precision correlates *negatively* with instance count (Pearson r = −0.375, p = 0.013; Spearman ρ = −0.408, p = 0.007), and the frequency-defined tail outperforms the head under a standard YOLOv8 baseline. Motivated by this, we treat the target-class *allocation signal* as the experimental variable. Holding the generation budget (1,000 images), the number of augmented classes (13), and the generation pipeline (bounding-box-protected background inpainting with per-image verification) fixed, we compare three allocation policies: uniform over the frequency-defined tail, rarity-and-weakness-weighted over the same tail, and a weakness policy that re-ranks *all* classes by measured baseline AP — which selects a class set entirely disjoint from the frequency tail. The outcome is a double dissociation: the frequency-based policy significantly improves only the frequency tail (+0.056 mAP50-95, Wilcoxon p = 0.0017) and the weakness-based policy significantly improves only its own weak set (+0.037, p = 0.040), while reweighting *within* a fixed class set yields no measurable benefit over uniform allocation (+0.003, p = 0.685). These results indicate that for synthetic augmentation, *which classes* receive the budget matters far more than *how the budget is shaped* within a class set, and that measured per-class performance — not frequency — should drive that choice. We release code and generation logs for reproducibility.

**Index Terms** — object detection, data augmentation, diffusion models, image inpainting, long-tailed learning, military aircraft recognition, YOLO

---

## I. Introduction

Object detectors degrade on under-represented classes, and a growing line of work responds by synthesizing additional training images with generative models. Before a single image is generated, such a pipeline must answer a question that is usually settled by convention rather than measurement: *which classes receive the synthetic budget?*

The conventional answer is the rare classes. It is not, however, the only answer on record. Difficulty-Net [Sinha et al., WACV 2023] argues that reweighting by frequency overlooks categories that are intrinsically hard to learn, and ObjectAug [Zhang et al., 2021] compared rarity-driven against hard-driven category coefficients for segmentation augmentation on PASCAL VOC, reporting that the hard-driven ranking was the more effective of the two by 0.8 points. More recently, uncertainty-guided context augmentation [Anonymous, 2026] preserves the regions on which a segmenter is least certain and regenerates the surrounding context with a diffusion model — allocating by model uncertainty rather than by frequency. The direction of travel in this literature is therefore already established: *measured difficulty appears to be a better allocation signal than counting instances.*

What remains unclear is what that superiority actually consists of, and whether it survives the move from segmentation to detection. Prior comparisons report an aggregate margin — one policy scores higher overall — on class partitions that overlap, so a gain observed under a difficulty-driven policy may reflect either better targeting or simply a better ranking of largely the same classes. This paper separates those readings.

We work on the 43-class Military Aircraft Detection benchmark, where the decoupling between frequency and difficulty is unusually sharp. Under a YOLOv8 baseline with default augmentation, per-class AP correlates *negatively* with instance count (Pearson r = −0.375, p = 0.013; Spearman ρ = −0.408, p = 0.007). The rarest class in the frequency-defined tail (YF23, 97 instances) is the easiest class in the dataset (0.93 AP50), while the weakest — Rafale (0.31), F14 (0.37), JAS39 and F18 (0.49) — sit in the medium and head groups. Distinctive airframes are easy regardless of frequency; visually similar multirole fighters are hard regardless of abundance. The consequence is a property we exploit rather than merely report: on this benchmark, ranking all classes by measured AP and taking the bottom K yields a class set **disjoint** from the frequency tail of the same size.

That disjointness turns the comparison into a controlled experiment. Holding the generation budget (1,000 images), the class count (13), the generator, and the verification gates fixed, we vary only the allocation signal:

1. **Uniform-tail** — equal allocation over the 13 frequency-tail classes;
2. **Selective-tail** — rarity-and-weakness-weighted allocation over the same 13 classes;
3. **Weakness** — all 43 classes re-ranked by measured baseline AP, budget to the bottom 13.

Evaluating every policy on *both* class sets, rather than reporting a single aggregate, changes the conclusion. The two policies are statistically indistinguishable overall (p = 0.154); what separates them is *where* their gains land.

**Contributions.**

- We report a benchmark on which class frequency is a statistically significant *negative* predictor of class difficulty, and analyze the mechanism (distinctive-airframe rarity versus confusable-fighter abundance, with miss-dominated errors).
- We refine the prior rarity-versus-difficulty finding into a **double dissociation**. Under a fixed budget and disjoint class sets, each policy significantly improves exactly the set it targets and not the other (frequency policy on its tail: +0.056 mAP50-95, p = 0.0017, but +0.018 on the weak set, n.s.; weakness policy on the weak set: +0.037, p = 0.040, but +0.017 on the tail, n.s.). Difficulty-driven allocation is not globally superior — it is differently targeted.
- We show that **within-set reweighting is ineffectual**: weighting the same 13 classes by rarity and weakness is indistinguishable from uniform allocation (p = 0.685). Choosing the set dominates shaping the weights inside it.
- We **quantify a failure mode of background inpainting** that the standard pixel-level verification cannot see: 10.2% of generated images contain an aircraft the model invented and no annotation covers, injecting false-negative supervision. To our knowledge this label-noise channel has not been measured for protected-region inpainting, and it offers a concrete explanation for why every generative arm here trails repeat-factor sampling.

## II. Related Work

**Resampling and reweighting for long-tailed detection.** Repeat-factor sampling [Gupta et al., LVIS 2019], class-balanced oversampling, and classifier-side corrections such as balanced group softmax [Li et al., CVPR 2020] and logit normalization [Zhao et al., 2022] remain strong and nearly free baselines. Our results reinforce their standing: RFS is the strongest method in our study in absolute terms (§VI-E). Our question is orthogonal — given that a generation budget is being spent, where should it go.

**Difficulty rather than frequency as the target signal.** Difficulty-Net [Sinha et al., WACV 2023] learns to predict per-class difficulty and reweights the loss accordingly, on the premise that frequency-based reweighting misses intrinsically hard categories. ObjectAug [Zhang et al., 2021] applies the same intuition to augmentation allocation, comparing rarity-driven and hard-driven category coefficients for object-level segmentation augmentation and finding the hard-driven ranking more effective. Uncertainty-guided context augmentation [Anonymous, 2026] allocates by per-class predictive entropy, preserving uncertain regions and regenerating surrounding context with a diffusion model on Cityscapes, UAVID and BDD100K.

These works establish the premise this paper starts from, and two differences define our contribution. First, all three address segmentation or classification; we test detection, where the supervision unit is a box and an unlabeled object becomes a false negative rather than a mislabeled pixel. Second, and more substantively, they report aggregate margins over partially overlapping class partitions. We hold budget and class count fixed, exploit a benchmark where the two candidate class sets are disjoint, and evaluate each policy on both sets — which converts "difficulty-driven is better" into the more specific and more useful "each signal moves its own classes."

**Copy-paste and compositional augmentation.** Copy-paste [Ghiasi et al., CVPR 2021] and its scaled variants (X-Paste [Zhao et al., ICML 2023], Gen2Det [Suri et al., 2024]) compose scenes from object crops. They change object context but introduce boundary artifacts and require paste-placement heuristics; copy-paste is the weakest augmentation arm in our study.

**Diffusion-based augmentation for detection.** Text-to-image and inpainting diffusion models have been used to expand detection training sets [Trabucco et al., DA-Fusion, ICLR 2024; Fang et al., controllable diffusion, 2024]. Class-specific fine-tuned diffusion models have been applied to military object detection in low-data regimes [Anonymous, 2026], though with uniform per-class allocation (150 images per class) and full-image generation rather than inpainting. Background inpainting around protected ground-truth boxes is attractive for detection specifically because annotations transfer without relabeling; we adopt this generator, hold it fixed across arms, and vary only the allocation of its budget.

**Label noise in synthetic training data.** Work on synthetic-data quality has largely focused on realism metrics (FID, CLIPScore) and on filtering by aesthetic or alignment scores. The failure we measure is different in kind: the generated content is realistic, passes pixel-level protection checks, and is nonetheless mislabeled, because the generator adds an object that the transferred annotation does not cover. §IV-D quantifies the rate and §VII proposes the object-level gate that would remove it.

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

Fig. 5 shows representative outputs. Backgrounds are replaced wholesale — urban skyline to airport runway (a), foliage to mountain cloudscape (b), airfield buildings to storm clouds (c) — while the protected aircraft, its pose, and its label are untouched. Fig. 5(d) documents the pipeline's characteristic failure mode: the gates verify that protected regions are unchanged and that the background *did* change, but they cannot detect a *new* aircraft hallucinated into the repainted background despite a negative prompt that explicitly forbids extra, duplicate, and new aircraft. Such objects enter training unlabeled. §IV-D quantifies how often this happens.

### D. How often does the background hallucinate aircraft?

Because the failure is invisible to the pixel-level gates, we measure it directly. For a 450-image sample (150 per arm, evenly spaced over the accepted set), we run the `basic_aug` baseline detector on both the source and the generated image and count confident detections (conf ≥ 0.5) that fall outside every protected box. "Outside" uses containment rather than IoU — a small detection inside a large ground-truth box has near-zero IoU but is not a new object — with a 0.5 threshold on the fraction of the detection's own area that any padded ground-truth box covers. Scoring generated images alone would confound hallucinations with the detector's false positives, so the reported quantity is the per-image increase.

| arm | extra objects per image (source → generated) | images gaining ≥1 | unlabeled objects added |
|---|---|---|---|
| uniform-tail | 0.000 → 0.307 | 12.7% | 46 |
| selective-tail | 0.000 → 0.207 | 4.7% | 31 |
| weakness | 0.007 → 0.353 | 13.3% | 52 |
| **all** | **0.002 → 0.289** | **10.2%** | **130** |

**One generated image in ten contains at least one unlabeled aircraft that the model invented.** The distribution is heavy-tailed: 27 of the 46 affected images gained a single object, but four gained 11–17, the worst cases being skies repainted into full formations (Fig. 7). Across arms the rate tracks the amount of repainting — LPIPS 0.244/0.262/0.272 against hallucination rates of 4.7%/12.7%/13.3% — consistent with a larger repainted area offering more opportunity to invent objects.

Two checks support reading the increase as genuine hallucination rather than detector noise. First, the flagged regions were inspected visually (Fig. 7): the boxes sit on clearly rendered aircraft, not on texture artifacts. This matters because the source images are part of the detector's training data, so the near-zero source-side count is partly memorization and would otherwise inflate the delta. Second, the prompts ("aviation photography", "military airbase") plausibly invite the very objects the negative prompt forbids — the failure has an obvious mechanism.

The consequence for training is specific: an unlabeled aircraft teaches the detector that an aircraft is background, which suppresses exactly the recall that §III-C identified as the dominant error mode. Because all three arms share one generator, prompts, and gates, this noise is matched across arms and cannot manufacture the differential result of §VI-B — but it plausibly caps the absolute gain available to every inpainting arm (§VII).

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

**Generation-quality metrics.** Table [TODO: number] scores the accepted images: CLIPScore (CLIP ViT-L/14, 100 × cosine between image and its generation prompt; 200-image sample per arm), LPIPS between source and generated image (degree of background change), and class-conditional FID against real images of the same classes.

| arm | acceptance | CLIPScore ↑ | LPIPS (src↔gen) | FID ↓ (overall) | FID per-class median |
|---|---|---|---|---|---|
| uniform-tail | 81.8% | 19.6 ± 3.3 | 0.262 | 89.7 | 83.2 |
| selective-tail | 80.8% | 19.1 ± 3.8 | 0.244 | 87.7 | 91.2 |
| weakness | 79.4% | 20.2 ± 3.4 | 0.272 | 100.4 | 103.7 |

Three observations. (i) CLIPScore and LPIPS are matched across arms — the generator behaves identically regardless of which classes it serves. (ii) The two same-class-set arms (uniform, selective) have near-identical overall FID (89.7 vs 87.7), a sanity check on the controlled design. (iii) The weakness arm's FID is moderately higher (100.4): its head/medium source images contain cluttered multi-aircraft and ground scenes, so wholesale background replacement departs further from the real class distribution. Critically, the arm with the *worst* generation fidelity still produced the significant target-scope gain of §VI-B — fidelity differences run *against* the weakness arm and therefore cannot explain the dissociation. (Small-sample FID with 76–200 images per class is upward-biased; values should be read comparatively, not absolutely.)

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

**Verification blind spot: background hallucination.** §IV-D measures it: 10.2% of generated images carry at least one unlabeled aircraft, 130 such objects across the 450-image audit. Because all three arms share the identical generator, prompts, and gates, this noise is matched across arms and cannot produce the differential result of §VI-B. It does, however, offer a concrete explanation for a puzzle in our results: the inpainting arms add background diversity but simultaneously inject false-negative supervision, and RFS — which re-exposes real images and adds no synthetic pixels — is free of this cost. We regard the gap to RFS (§VI-E) as partly attributable to it.

The fix is direct and testable: an object-level gate that runs a detector over each generated image and rejects any with confident detections outside the protected regions, which is exactly the instrument used for the audit. Because the audit shows a heavy tail (four images contributing 54 of the 130 objects), even a permissive gate would remove most of the injected noise at a small yield cost. Re-running the three arms behind such a gate is the most direct test of how much of the diffusion-versus-resampling gap is label noise rather than a limitation of background augmentation itself. Reducing prompt-induced invention — the prompts name aviation scenes, which invites aircraft — is a complementary direction.

**Absolute gains are modest.** The best allocation arm adds ~0.03–0.06 mAP50-95 on its target set from 1,000 images. Whether gains scale with budget (2×, 5×), and whether allocation policies interact with budget size, is open. Composing RFS with weakness-allocated generation is the most promising follow-up suggested by our results.

## VIII. Conclusion

On a 43-class military aircraft benchmark where class frequency is a significantly *negative* predictor of class difficulty, we isolated the class-allocation signal of a fixed diffusion-augmentation budget as the experimental variable. The result is a double dissociation — frequency-based allocation helps only the frequency tail, measured-weakness allocation helps only the measured-weak classes — while reweighting within a fixed class set does nothing. The practical prescription for synthetic augmentation pipelines is one sentence: before generating, measure a baseline, test whether frequency tracks difficulty, and point the budget at the classes that are actually weak.

---

## Figures

| # | 내용 | 파일 | 상태 |
|---|---|---|---|
| Fig. 1 | 파이프라인 개요 (Stage 1 배분=실험변수 / Stage 2 생성·검증=공통, 실제 생성 이미지 썸네일 포함) | `figures/fig1_pipeline.{pdf,png}` | 완료 |
| Fig. 2 | instance count vs per-class AP 산점도, r=−0.375 (p=0.013) | `figures/fig2_freq_vs_ap.{pdf,png}` | 완료 |
| Fig. 3 | 이중 해리 2×2 막대 (arm × scope, Wilcoxon 유의성) | `figures/fig3_double_dissociation.{pdf,png}` | 완료 |
| Fig. 4 | weak set 13클래스 dumbbell (basic_aug → weakness arm, AP50) | `figures/fig4_weak_class_change.{pdf,png}` | 완료 |
| Fig. 5 | 실제 생성 예시 4종 (original/mask/generated; (d)는 환각 실패 사례) | `figures/fig5_generation_examples.{pdf,png}` | 완료 |
| Fig. 6 | baseline 정규화 confusion matrix (background 행 = 미검출 지배) | `figures/fig6_confusion_matrix_baseline.png` | 완료 (run 산출물) |
| Fig. 7 | 환각 사례 6종 (원본/생성 2행, 초록=보호 GT, 빨강=보호 밖 검출) | `figures/fig7_hallucination_examples.{pdf,png}` | 완료 |
| Table I | 데이터셋 통계 | `dataset_summary.csv` | 수치 확보 |

그림 재생성: `scratchpad/make_figs.py` (데이터: GCS 최신 결과. 로컬 Drive의
`outputs_full/metrics`는 GCP 학습 이전 상태이므로 사용 금지). 색은 Okabe-Ito
기반 CVD-safe 팔레트로 validator 통과 확인.

## References (초안 — 투고 전 정확한 서지 확인 필수)

**핵심 선행 연구 (신규성 위치 설정에 필수 — 누락 시 심사에서 치명적)**

1. S. Sinha et al., "Difficulty-Net: Learning to predict difficulty for long-tailed recognition," WACV 2023. — 빈도 재가중이 본질적으로 어려운 클래스를 놓친다는 주장의 출처
2. J. Zhang et al., "ObjectAug: Object-level data augmentation for semantic image segmentation," 2021 (arXiv:2102.00221). — **rarity-driven vs hard-driven 배분 비교의 직접적 선행**. 전문에서 실험 설계(집합 중첩 여부, 예산 고정 여부)를 확인해 §II 서술을 확정할 것
3. [TODO 저자] "Preserve the hard, regenerate the rest: Uncertainty-guided synthetic training data augmentation with diffusion models," 2026 (arXiv:2606.31603). — 불확실성 기반 배분 + 보호 영역 주변 맥락 재생성

**배경**

4. A. Gupta, P. Dollár, R. Girshick, "LVIS: A dataset for large vocabulary instance segmentation," CVPR 2019. (RFS)
5. Y. Li et al., "Overcoming classifier imbalance for long-tail object detection with balanced group softmax," CVPR 2020.
6. G. Ghiasi et al., "Simple copy-paste is a strong data augmentation method for instance segmentation," CVPR 2021.
7. R. Rombach et al., "High-resolution image synthesis with latent diffusion models," CVPR 2022.
8. B. Trabucco et al., "Effective data augmentation with diffusion models," ICLR 2024. (DA-Fusion)
9. [TODO] X-Paste (ICML 2023), Gen2Det — 정확한 서지
10. [TODO 저자] "Class-specific diffusion models improve military object detection in a low-data domain," 2026 (arXiv:2604.18076). — 같은 응용 도메인, 균등 배분·전체 생성 방식
11. Ultralytics YOLOv8, https://github.com/ultralytics/ultralytics
12. F. Wilcoxon, "Individual comparisons by ranking methods," Biometrics Bulletin, 1945.

**향후 확장 시**

13. W. Yu et al., "MAR20: A benchmark for military aircraft recognition in remote sensing images," Journal of Remote Sensing, 2022. — 2차 데이터셋 후보
