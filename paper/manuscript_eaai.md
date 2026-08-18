# Allocation-aware inpainting augmentation for object detection: the class set, not the per-class weighting, determines where performance gains occur

Daehyun Yoo^a^

^a^ Affiliation — TO BE COMPLETED before submission

Corresponding author: dhyoo970111@gmail.com


## Highlights

- Class-set choice, not per-class weighting, decides where inpainting gains land
- Set-by-scope interaction +0.06 (corrected p = 0.017), replicated on satellite data
- Every augmentation gain vanishes under a stronger pretrained detector
- Resampling wins on deep imbalance, fails on shallow — a cheap diagnostic probe
- Pre-registered contrasts and seed-blocked statistics guard augmentation claims

## Abstract

Synthetic images produced by diffusion inpainting can enlarge
object-detection training sets at low cost, but a practical question
remains open: given a fixed budget of generated images, which classes
should receive them? The contribution in artificial intelligence is a
controlled answer to this allocation question. We cross the allocation
signal — a frequency-defined tail set versus a difficulty-defined weak
set selected by measured per-class average precision — with the
within-set weighting (uniform versus priority-weighted), holding the
budget, class count, generator, prompts, and rule-based quality
verification fixed, and we evaluate the four resulting augmentation arms
with pre-registered, hash-frozen contrasts, seed-blocked tests, and
multiple-comparison correction. The engineering application is military
aircraft detection in two operationally distinct regimes: a 43-class
ground-level natural-view benchmark and a 20-class satellite-view
benchmark, each paired with a compact one-stage detector and a large
pretrained transformer detector. Three results follow. First, the
allocation signal determines where gains occur: each arm improves the
classes it targets (set-by-scope interaction +0.06, corrected
p = 0.017; replicated at +0.05 on the satellite benchmark), while the
weighting axis is null in all four comparisons. Second, every
augmentation gain vanishes under the stronger detector, whose
unaugmented baseline exceeds the best augmented compact model. Third,
zero-cost repeat-factor resampling dominates generation on the deeply
imbalanced benchmark but is null on the shallow one, where inpainting is
the only augmentation with significant gains. These findings yield a
three-step
engineering rule — check detector headroom, probe with free resampling,
then generate for the measured-weakest classes with a uniform split.
Code, generated image pools, and all training artifacts will be
released.

**Keywords:** Data augmentation; Diffusion inpainting; Object detection; Class imbalance; Synthetic data allocation; Military aircraft detection

# 1. Introduction


Object detection systems deployed in specialized domains face a chronic
data problem: collecting and annotating images for every class of
interest is expensive, and the resulting training sets are almost
inevitably class-imbalanced. Military aircraft recognition is a
representative case — new airframes appear rarely in public imagery,
per-class collection costs vary by orders of magnitude, and yet the
operational cost of missing a rare type is high. Diffusion-based image
generation has made synthetic training data cheap enough to be a
practical remedy, and a growing literature shows that generated images
can improve detectors. For the practicing engineer, however, the binding
question is no longer *whether* to generate but *where to spend a fixed
generation budget*: given the capacity to produce, say, one thousand
verified synthetic images, which classes should receive them?

The long-tail literature supplies a default answer: give them to the
rare classes. This prescription rests on a premise so common that it is
rarely stated — that training frequency identifies the classes that need
help. On our natural-view benchmark the premise fails. Per-class average
precision (AP) of a strong baseline correlates *negatively* with
training frequency (Pearson r = −0.375): the rarest aircraft types are
often visually distinctive and easy, while the hardest classes —
visually confusable fighter jets — sit in the middle and head of the
frequency distribution. Frequency and measured difficulty therefore
select *disjoint* sets of classes to support, and "help the rare" and
"help the weak" become competing allocation policies rather than
synonyms. Which signal should govern the budget, and does the choice
matter at all?

We answer this with a controlled factorial experiment rather than
another method. Using background inpainting as the generation mechanism
— annotated objects are geometrically protected and only their
surroundings are regenerated, so labels remain exact by construction —
we cross the *allocation signal* (frequency-defined tail set vs.
AP-defined weak set) with the *within-set weighting* (uniform vs.
priority-weighted) while holding the budget, the number of target
classes, the generator, the prompts, and the quality-verification
protocol fixed. The four resulting augmentation arms differ in exactly
one respect: how the same budget is allocated. We run this design on two
public benchmarks in operationally distinct imaging regimes — a
43-class natural-view dataset and the 20-class satellite-view MAR20 —
and under two detectors that differ by an order of magnitude in
capacity, evaluating all arms with pre-registered contrasts, seed-level
statistics, and multiple-comparison correction.

Three findings emerge. First, *the allocation signal determines where
performance gains occur*: each arm improves the class set it targets by
several points of AP while conferring only a smaller spillover
elsewhere, a double dissociation confirmed by a pre-registered
interaction test on the natural-view dataset and replicated in effect
size on MAR20. In contrast, the within-set weighting axis is null in
all four comparisons — *which classes* receive the budget matters;
*how many images each class gets* does not measurably matter at the
budgets we study. Second, every augmentation effect we measure is
conditional on detector headroom: under a large pretrained transformer
detector whose baseline already exceeds the best augmented compact
model, all gains vanish. Third, the zero-cost resampling baseline —
repeat-factor sampling — dominates every generative arm on the
deeply imbalanced dataset yet is null on the shallowly imbalanced one,
where inpainting is the only augmentation with significant gains;
resampling's outcome thus serves as a cheap diagnostic of whether
generation is worth its cost. Together these results yield a three-step
decision rule for practitioners — check headroom, probe with free
resampling, then generate for the *measured-weakest* classes with a
uniform split — and a caution for the literature: augmentation studies
that vary a single recipe on a single detector risk reporting effects
that a stronger baseline model would erase.

Our contributions are:

- **An allocation-signal finding.** Under a fixed budget, fixed class
  count, and fixed generation pipeline, the choice of class set — not
  the per-class weighting — determines where inpainting-augmentation
  gains occur (set-by-scope interaction +0.05 to +0.06 across two
  datasets; weighting null in all four pre-registered comparisons), and
  frequency is a poor default signal wherever frequency and difficulty
  decouple.
- **An evaluation protocol for augmentation claims.** A 2×2 factorial
  with budget control, pre-registered and hash-frozen contrasts,
  seed-blocked tests with Holm correction, and rule-based generation
  verification — a reusable template that blocks the confounds (budget
  differences, single-seed effects, post-hoc hypothesis selection)
  common in augmentation studies.
- **A condition map for generative augmentation.** Evidence that
  augmentation gains are moderated by detector headroom and by the
  depth of dataset imbalance, including the failure modes: all gains
  vanish under a strong pretrained detector, and resampling turns
  harmful on the weak scope of a shallowly imbalanced dataset.
- **An open, deterministic pipeline.** Bbox-protected inpainting with
  quantitative quality verification, hallucination auditing, and fully
  seed-deterministic regeneration, released with plans, logs, metrics,
  and weights for all 66 training runs.

The remainder of the paper reviews related work (Section 2), details
the method and experimental setup (Sections 3–4), reports results
(Section 5), develops engineering guidance, a mechanistic
interpretation, and limitations (Section 6), and concludes
(Section 7).


# 2. Related Work


## 2.1 Generative augmentation for object detection

Synthetic training data for detection has largely followed a
*foreground* line of work: Copy-Paste showed that pasting real object
instances into new images is a strong augmentation on its own
(Ghiasi et al., 2021); X-Paste scaled the recipe by generating the
pasted instances with text-to-image diffusion (Zhao et al., 2023);
MosaicFusion synthesizes rare-category objects together with their masks
(Xie et al., 2024); Gen2Det generates scene-centric images by grounded
inpainting and couples them with image- and instance-level filtering
(Suri et al., 2023); and DiverGen attributes much of the benefit to the
*diversity* of generated data (Fan et al., 2024). Domain-specific
variants fine-tune or condition the generator for a target domain, as in
ODGEN for specialized detection datasets (Zhu et al., 2024) and AeroGen
for layout-conditioned remote-sensing imagery (Tang et al., 2025).
Closest to our pipeline is the *background* line: Li et al. (2024)
protect annotated foregrounds and regenerate only the surroundings with
Stable Diffusion inpainting, reporting that background augmentation can
outperform object-centric augmentation for robustness and
generalization. We adopt this mechanism deliberately — it keeps labels
exact by construction — but ask a question this literature leaves open.
Existing studies establish *that* generated data can help and how to
filter it; the budget itself is typically spent uniformly or left as an
unexamined implementation choice. None isolates the *allocation signal*
— which classes receive the budget — as the experimental variable, which
is precisely the factor our design manipulates under a fixed budget,
generator, and quality-verification protocol.

## 2.2 Long-tailed detection: resampling and rebalancing

The long-tail literature defines the problem by class frequency. LVIS
introduced the large-vocabulary benchmark, the rare/common/frequent
reporting convention, and repeat-factor sampling (RFS), the standard
image-level resampling scheme (Gupta et al., 2019); instance-aware and
exponentially weighted refinements followed (Yaman et al., 2023;
Ahmed et al., 2025). Loss-side rebalancing spans balanced group softmax
(Li et al., 2020), Seesaw loss (Wang et al., 2021), and equalized focal
loss for one-stage detectors (Li et al., 2022). Two findings from this
literature shape our protocol. First, rebalancing techniques developed
on two-stage detectors transfer unevenly to modern one-stage pipelines,
where strong default augmentation (mosaic, mixup) already dominates
sampling- and loss-based corrections (Crasto, 2024);
accordingly, all of our arms are evaluated as *marginal* additions on
top of a strong augmented baseline rather than in isolation — the
reporting practice of X-Paste and Gen2Det. Second, and central to our
motivation, the entire resampling tradition presumes that frequency
identifies the classes that need help. Our data violate that premise —
per-class average precision correlates *negatively* with training
frequency on the natural-view dataset — and our factorial design treats
the frequency-defined tail and the measured-AP-defined weak set as
competing allocation signals rather than assuming either one.

## 2.3 Allocating and selecting synthetic data

A smaller body of work asks not whether to generate but *where* and
*what*. Diffusion-curriculum approaches allocate synthetic data
adaptively along an image-guidance spectrum for tail classes
(Liang et al., 2025); and generating informative samples with
class-aware generative adversarial networks for weak regions of a
classifier was explored as early as the GAN era, in medical image
classification (Bozorgtabar et al., 2019), anticipating our difficulty
signal. On the selection side, quality filtering by CLIP score or
detector feedback is by now a standard component (Suri et al., 2023;
Fan et al., 2024), and recent work cautions that retrieved real images
can match or outperform synthetic ones (Geng et al., 2024), reinforcing
the need for strong non-generative baselines. These works motivate difficulty-aware allocation but evaluate
it entangled with other design changes — curricula, generator
fine-tuning, filtering — and without budget-controlled comparisons.
Our contribution to this strand is methodological: a 2×2 factorial that
crosses the allocation signal with the within-set weighting under an
identical budget, class count, generator, and verification protocol,
evaluated against pre-registered contrasts, so that the effect of the
signal itself is identified.

## 2.4 Aircraft detection in surveillance imagery

Military aircraft recognition is served by dedicated benchmarks —
notably MAR20, the largest public military-aircraft remote-sensing
dataset (Yu et al., 2023) — and by a line of detector-architecture work,
from fine-grained one-stage designs evaluated on MAR20 (FGA-YOLO;
Wu et al., 2025) to small-object detection for aerial platforms
published in this journal (Song et al., 2024). This literature concentrates on architectural improvements
under fixed data; the complementary engineering question — how to spend
a limited synthetic-data budget when per-class collection is expensive —
has received little controlled study in the domain. We address that
question on both a ground-level natural-view benchmark and MAR20, and
use the two detector families of Section 4.2 to test whether the answer
survives a change of architecture and capacity.


# 3. Method


## 3.1 Overview

Our pipeline augments a detection training set with synthetic images in
which the *background* is regenerated by a diffusion inpainting model
while every annotated object is geometrically protected (Figure 1). Given
a fixed budget of *B* synthetic images and a fixed number of target
classes *K*, an *allocation plan* assigns a per-class image quota;
training images containing the targeted class are drawn as sources,
inpainted outside all annotated object boxes, required to pass rule-based
quality verification against quantitative criteria (Section 3.4), and
inserted into the training split only. The design isolates a single
question: holding the generator, prompts, quality verification, budget,
and class count fixed, does the *allocation signal* — the rule that
selects which classes receive the budget — change where performance gains
occur?

## 3.2 Allocation design: class set × within-set weighting

We instantiate two allocation signals and two within-set weightings,
yielding a 2×2 factorial over augmentation arms:

- **Tail set (frequency signal).** The bottom 30% of classes by training
  instance count — the conventional long-tail prescription.
- **Weak set (difficulty signal).** The *K* classes with the lowest
  baseline per-class average precision (AP), measured on the validation
  split by the `basic_aug` baseline detector and averaged over its
  training seeds. *K* is fixed to the tail-set size so that set identity,
  not set size, is the manipulated variable.
- **Uniform weighting** splits *B* equally across the *K* classes.
- **Weighted allocation** distributes *B* proportionally to a priority
  score: for the tail set, a convex combination of rarity and measured
  weakness (weight 0.6 toward rarity); for the weak set, measured
  weakness alone.

Quotas are computed by a deterministic capped largest-remainder
allocator: each class first receives a minimum quota, the remainder is
apportioned proportionally to the weighting under per-class caps, and
fractional leftovers go to the largest remainders with index-ordered tie
breaking. The allocator guarantees that every arm spends exactly *B*
images on exactly *K* classes, so budget can never confound the
comparison. Empirically, the two sets are disjoint on the natural-view
benchmark and overlap only marginally on the satellite-view benchmark
(Section 4.1) — frequency and measured difficulty select genuinely
different classes here (Section 5.1), which is what renders the
factorial informative.

## 3.3 Background inpainting generation

Synthetic images are produced with Stable Diffusion inpainting
(Rombach et al., 2022; `runwayml/stable-diffusion-inpainting`, 16-bit floating-point precision,
20 denoising steps, guidance scale 7.5, strength 0.85) at 512×512 working
resolution and resized back to the source resolution. For each source
image we build a mask that *protects* all annotated boxes with a 10%
padding margin and a blurred boundary band, so that diffusion may alter
only the background. Scene prompts are dataset-specific (five
natural-view aviation prompts for MAD; five top-down aerodrome prompts
for MAR20), as are the negative prompts, which suppress — among other
artifacts — the generation of new aircraft. After generation, the
protected regions are pasted back from the source pixels, making object
appearance and labels exact by construction; source labels are copied
unchanged. Generation is fully deterministic given the plan: each output
is a function of the source image, the prompt rotation, and a derived
per-image seed, which enables byte-identical regeneration and safe reuse
across experiments.

## 3.4 Rule-based quality verification and budget refill

Source images must first offer enough editable background for inpainting
to be meaningful (protected regions covering no more than 95% of the
frame); sources failing this pre-check are excluded before generation.
Every generated image must then pass two quantitative checks: (i) the
background changed substantially (mean absolute pixel difference ≥ 10
gray levels outside the boxes), and (ii) the protected interiors did not
change beyond a compression-noise tolerance (≤ 5 gray levels). Failed
generations are retried and, if still failing, replaced from alternative
sources within a bounded attempt budget until the class quota is met, so
all arms reach the same *realized* budget. A run aborts if more than half
of all attempts fail, flagging generator malfunction. As an additional
audit — reported as a robustness check, not a filter — a detector-based
scan counts *hallucinated* objects that diffusion occasionally paints
into backgrounds, and an object-level gate variant retrains the arms with
all affected images removed (Section 5.4).

## 3.5 Training arms and baselines

All arms share the detector, schedule, and the default
photometric/geometric augmentation of the Ultralytics training framework (Jocher et al., 2023);
they differ only in training data. `real_only` disables framework
augmentation (reference lower bound); `basic_aug` is the primary baseline
that all gains are measured against. Repeat-factor sampling (RFS) is the
zero-cost resampling baseline: with f(c) the fraction of training images
containing class c, each image I is duplicated according to
r(I) = max over classes c present in I of r(c), where
r(c) = max(1, √(t/f(c))) with t = 0.1, materialized at the dataset level.
The four inpainting arms add their *B*-image pool to the training split;
validation and test splits are never touched.

## 3.6 Statistical protocol

We separate planning from evaluation. All plan-forming measurements (weak
set selection, priority scores) use the validation split only; the test
split is reserved for final reporting, and no plan or hyperparameter was
revised after observing test metrics. For the full-factorial cells, five
contrasts were specified and frozen — with a content hash and timestamp —
before any test-split metric existed: (1) tail arms vs. baseline on the
tail scope, (2) weak arms vs. baseline on the weak scope, (3) the
set-by-scope interaction, (4–5) weighted vs. uniform within each set.
Contrasts are evaluated on per-seed macro AP (the unweighted mean of
per-class AP over the classes in a scope) with seed-blocked paired
t-tests (n = 3 independent training seeds) and Holm correction (Holm, 1979) within
each cell's five-contrast family; all tests are two-sided at α = 0.05.
Wilcoxon signed-rank tests on seed-averaged per-class AP provide
secondary, distribution-free evidence. We report evaluation over three
class scopes: *all*, the frequency-defined *tail*, and the AP-defined
*weak* set.

# 4. Experimental Setup

The engineering application is military aircraft detection in two
operationally distinct imaging regimes — ground-level natural views and
satellite views — a setting where per-class data collection is expensive
and class imbalance is unavoidable, making augmentation budget decisions
a practical engineering concern. Both benchmarks are publicly available,
supporting replication of every result.

## 4.1 Datasets

**MAD** (Military Aircraft Detection, natural side/ground views; public; rookieengg, 2023)
contains 11,788 images (9,430/1,179/1,179 train/val/test) with 18,832
boxes over 43 aircraft types; the train-split imbalance ratio is 10.5.
**MAR20** (Military Aircraft Recognition dataset, satellite view,
horizontal boxes; public; Yu et al., 2023) provides an official test split of 2,511
images, which we preserve untouched for final reporting; the official
1,331 training images are split 80/20 into train/val with
class-stratified sampling. MAR20's 20-class distribution is markedly
shallower than MAD's, which Section 5.3 exploits as a natural contrast in
imbalance structure. The tail set is a property of the dataset; the weak
set is derived from each detector's own baseline (Section 3.2) and is
therefore cell-specific. On MAD the two sets contain 13 classes each and
are disjoint for both detectors; on MAR20 they contain 6 classes each,
overlapping in one class for the compact detector and two for the
transformer.

## 4.2 Detectors and training

We use two detectors spanning an order of magnitude in capacity: YOLOv8n,
a compact one-stage detector of the You-Only-Look-Once (YOLO) family
(3.2M parameters), and RT-DETR-L, the large variant of the Real-Time
Detection Transformer (Zhao et al., 2024; 32M parameters). Both are initialized from
checkpoints pretrained on the Common Objects in Context (COCO) benchmark,
so the cells differ in capacity and architecture family, not in
pretraining corpus. Both are trained at 640×640 for 50 epochs with early
stopping (patience 15) under identical schedules across arms; the batch
size is fixed to 8 for RT-DETR-L and set automatically for YOLOv8n, with
all remaining hyperparameters at framework defaults. Every arm–cell
combination is trained with three independent seeds; all reported
statistics aggregate over seeds as described in Section 3.6.

## 4.3 Cells and budgets

Table 1 summarizes the four dataset × detector cells. The YOLOv8n cells
run the full 2×2 factorial plus baselines; the RT-DETR-L cells run a
reduced design (baseline, RFS, and the two tail-set arms) intended to
test whether gains transfer to a stronger detector rather than to
re-estimate the interaction. The uniform arm's pool depends only on the
dataset and is shared across the detector cells; the weighted arm's
difficulty component is recomputed from each detector's own baseline
validation AP, so its allocation is detector-specific. Budgets scale with
dataset size: *B* = 1,000 (*K* = 13, per-class quota 5–200) for MAD and
*B* = 500 (*K* = 6, quota 5–100) for MAR20. Realized budgets equal
planned budgets in all cells (verification failure rates: ≤ 21% of
attempts on MAD, 0.6% on MAR20, absorbed by the refill mechanism of
Section 3.4).

**Table 1.** Experimental cells. All cells share the generation,
verification, and statistical protocol.

| Dataset | Detector | Arms | Budget *B* / *K* |
|---|---|---|---|
| MAD | YOLOv8n | full 2×2 + RFS + baselines | 1,000 / 13 |
| MAD | RT-DETR-L | reduced (tail-set arms + RFS) | 1,000 / 13 |
| MAR20 | YOLOv8n | full 2×2 + RFS + baselines | 500 / 6 |
| MAR20 | RT-DETR-L | reduced (tail-set arms + RFS) | 500 / 6 |

## 4.4 Reproducibility

Generation is seed-deterministic given a plan (Section 3.3); plans,
generation logs, verification reports, contrast freezes (with content
hashes), per-class metrics, and trained weights for all 66 primary
training runs — plus the object-gate retrainings of Section 5.4 — are
archived, and the full pipeline, from dataset normalization to the
statistical reports, runs from configuration files without manual
intervention. Code and artifacts will be released upon publication.


![](/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Military_OD/paper/figures_v2/fig1_pipeline_design.png)

**Figure 1.** Overview. (a) Generation pipeline: all annotated boxes in a
source image are geometrically protected; diffusion inpainting
regenerates only the background; outputs must pass rule-based
verification (with budget refill) before insertion into the training
split. Budget *B*, class count *K*, generator, prompts, and verification
are held fixed across arms. (b) The 2×2 allocation design: class set
(frequency-defined tail vs. AP-defined weak) × within-set weighting
(uniform vs. priority-weighted).


![](/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Military_OD/paper/figures_v2/fig2_freq_vs_ap.png)

**Figure 2.** Frequency is a poor proxy for difficulty on MAD. Per-class
baseline test mAP50–95 (three-seed mean) against training instance count
(log scale). The frequency-defined tail set (blue) and the AP-defined
weak set (orange) are disjoint; the rarest class (YF-23) is the easiest,
while the weakest classes (Rafale, F-14) sit at medium-to-head
frequency.


# 5. Results


We report all results on the military aircraft detection cells of
Section 4 as test-split mean average precision over
intersection-over-union thresholds 0.50–0.95 (mAP50–95), and give the
mean over three independent training seeds with per-arm standard
deviations computed across seeds. Gains (Δ) are always relative to the
`basic_aug` baseline of the same cell. Pre-registered contrasts
(Section 3.6) are evaluated with seed-blocked paired t-tests, 95%
confidence intervals (CIs), and Holm correction within each cell's
five-contrast family; class-level Wilcoxon signed-rank tests on
seed-averaged per-class AP serve as secondary evidence. Table 2 gives the
master results for all four cells; Table 3 lists the pre-registered
contrast outcomes.

## 5.1 The allocation signal determines where gains occur

**Frequency and difficulty select different classes.** On MAD, per-class
training-instance count and baseline test AP correlate *negatively*
(mAP50–95 basis; Pearson r = −0.375, p = 0.013; Spearman ρ = −0.408,
p = 0.007). In AP50 terms, the rarest class (YF-23, 97 training
instances) attains the highest score (0.93), whereas the weakest classes
(Rafale 0.31, F-14 0.37) sit in the medium and head frequency ranges.
Empirically, the frequency-defined tail set and the AP-defined weak set
are disjoint on MAD (0/13 overlap) and overlap in a single class on
MAR20 for the compact detector (Section 4.1). This is the premise that
makes the 2×2 allocation design informative: if frequency were a good
proxy for difficulty, the two signals would select the same classes and
could not be dissociated.

**Gains land on the targeted set (MAD, confirmatory).** Figure 3 (left)
shows the result on MAD with YOLOv8n as a double dissociation: each set
family improves its own target scope substantially more than the other
family does, and vice versa. Averaged over seeds, the two tail-set arms
improved the tail scope by +0.061 over baseline while improving the weak
scope by only +0.019; the two weak-set arms showed the mirrored pattern
(+0.040 on weak, +0.022 on tail). The pre-registered set-by-scope
interaction — the difference-in-differences of the two set families
across the two scopes — was **+0.0598 (95% CI [0.045, 0.075],
Holm-adjusted p = 0.017)**, with all three seeds showing the same
direction. At the class level, gains were broad rather than
concentrated: 11 of 13 tail classes improved under the tail-uniform arm
(largest: Tu-95 +0.169, MiG-31 +0.101), and 11 of 13 weak classes
improved under the weak-uniform arm (largest: Tornado +0.123, Rafale
+0.086 — the weakest class in the dataset).

**The dissociation replicates in the satellite domain (MAR20).** Figure 3
(right) shows the same analysis for MAR20 with YOLOv8n. The interaction
estimate was **+0.0523** (95% CI [0.021, 0.083]; positive in all three
seeds), which lies inside the 95% CI of the MAD estimate; the
tail-targeted arm again gained most on the tail scope (selective: +0.044;
class-level Wilcoxon p = 0.031, the smallest attainable value at n = 6)
and the weak-targeted arms most on the weak scope (+0.030 to +0.034).
Because the MAR20 scopes contain only six classes each, the seed-blocked
tests are underpowered and the interaction does not survive Holm
correction (unadjusted p = 0.018, Holm-adjusted p = 0.091); we therefore
report MAR20 as an independent replication of the effect size rather
than a second confirmatory result.

**Within-set weighting shows no benefit.** In contrast to the choice of
class set, *how* the fixed budget is divided inside the chosen set showed
no evidence of benefit in any condition: the weighted-versus-uniform
contrasts were null in all four pre-registered comparisons, with
estimates centred near zero (+0.002, p = 0.81 and −0.012, p = 0.17 on
MAD; +0.014, p = 0.11 and +0.004, p = 0.52 on MAR20; Table 3). The CIs
do not exclude small effects in either direction, so we do not claim
strict equivalence; what the data rule out is a within-set weighting
benefit comparable to the set-choice effect. Given that the uniform and
weighted arms shared the same class sets, generator, prompts, quality
verification, and budget, we conclude that at the budgets studied the
allocation *signal* (which classes) — not the allocation *shape* (how
many per class) — is the operative variable.

**Targeted gains come with a smaller global spillover.** Augmenting one
set also produced consistent but smaller gains outside it: on MAD the
tail-uniform arm improved the weak scope by +0.031 (class-level
p = 0.002), and on MAR20 the cross-set gains ranged from +0.004 to
+0.013. Background inpainting therefore appears to combine a
set-specific targeting component with a smaller domain-wide component,
consistent with the recall-oriented mechanism discussed in Section 6.

## 5.2 A strong pretrained detector removes all augmentation gains

Replacing the compact detector with RT-DETR-L (Section 4.2) raised the
baseline from 0.575 to 0.869 on MAD and from 0.572 to 0.697 on MAR20 —
and eliminated every augmentation effect we measured (Figure 4,
Table 2). On MAD, all deltas were zero to negative: uniform −0.004,
selective −0.001, and RFS **−0.016**; the pooled
inpainting-versus-baseline contrast on the tail scope was −0.008
(p = 0.36). On MAR20 the pattern repeated (uniform −0.007, selective
−0.005, RFS +0.003; pooled tail contrast +0.004, p = 0.72). Notably, the
strong detector's *baseline* already exceeded the best augmented YOLOv8n
result by a wide margin (0.869 vs. 0.661 on MAD).

Two observations follow. First, augmentation gains in this study are
conditional on detector headroom: once a model retains little headroom
on the dataset, neither resampling nor generative augmentation adds
measurable value, and the largest single lever is the detector upgrade
itself (a cross-cell comparison; Section 6.1). Second, the only
consistently *negative* delta under the strong detector was RFS on MAD
(−0.016 on all three scopes), suggesting that increasing duplicate
exposure can be mildly harmful when capacity and pretraining already
suffice; we report this as an observation without claiming a mechanism.
Because the RT-DETR-L cells implemented a reduced design (baseline, RFS,
and the two tail-set arms only), no interaction test is available there;
the "vanishing gains" claim rests on the near-zero deltas of all
measured arms.

## 5.3 Resampling effectiveness depends on the imbalance structure

Repeat-factor sampling — the zero-cost resampling baseline that dominated
every diffusion arm on MAD (+0.086 all, +0.098 tail, +0.117 weak;
class-level p ≤ 2×10⁻⁴) — transferred poorly to MAR20. With the same
detector and protocol, RFS on MAR20 yielded +0.003 (all), +0.016 (tail),
and **−0.011 (weak; class-level p = 0.031)** — the only statistically
significant negative augmentation effect we observed under YOLOv8n. MAD
is deeply imbalanced across 43 classes (max/min instance ratio 10.5), so
re-exposing rare-class images carries additional training signal;
MAR20's 20-class distribution is comparatively shallow, and additional
exposure of already-seen images adds little, while inpainting — which
injects genuinely new background variation — remains effective
(Section 5.1). On MAR20, inpainting was in fact the only augmentation
with statistically significant gains. The practical implication is
developed in Section 6: RFS is a cheap first attempt rather than a
universal winner, and its failure is diagnostic of a regime where
generative augmentation is worth its cost.

## 5.4 Robustness checks

**Generation quality does not explain the dissociation.** The three
originally generated MAD pools passed rule-based quality verification at
similar rates (79–82%), and the fourth pool reached its quota under the
same criteria; per-pool Fréchet inception distance (FID; Heusel et al., 2017) and CLIP-based
image–text similarity scores (Radford et al., 2021) were comparable, with the *worst*-FID pool
(weakness) producing the significant weak-scope gain — quality
differences cannot account for where gains landed. On MAR20 the
verification failure rate was 0.6%, and all four pools reached the full
500-image budget.

**Hallucinated objects do not drive the RFS gap on MAD.** A
detector-based audit found unlabeled aircraft hallucinated into 10.2% of
MAD synthetic backgrounds. Removing all affected images with an
object-level gate and retraining the three original diffusion arms
changed their all-scope performance by at most |Δ| ≤ 0.017 (all nine
gated-versus-ungated comparisons non-significant, p = 0.11–0.97), while
the RFS advantage persisted essentially unchanged (Δ = −0.055 to −0.067
versus RFS, p < 10⁻⁴). Label noise from generation artifacts therefore
explains neither the dissociation nor the resampling gap.


**Table 2.** Master results: Δ mAP50–95 versus the same-cell `basic_aug`
baseline (mean over three seeds; baseline rows show absolute mAP ± SD).
Bold marks the scope targeted by each arm; scopes are cell-specific
(Section 4.1).

| Dataset × detector | Arm | all | tail | weak |
|---|---|---|---|---|
| **MAD × YOLOv8n | basic_aug | 0.575 ± 0.007 | 0.620 ± 0.009 | 0.456 ± 0.030 |
| | RFS | +0.086 | +0.098 | +0.117 |
| | tail-uniform | +0.031 | **+0.060** | +0.031 |
| | tail-weighted | +0.023 | **+0.062** | +0.007 |
| | weak-uniform | +0.016 | +0.014 | **+0.046** |
| | weak-weighted | +0.020 | +0.030 | **+0.034** |
| **MAD × RT-DETR-L | basic_aug | 0.869 ± 0.006 | 0.889 ± 0.008 | 0.842 ± 0.012 |
| | RFS | −0.016 | −0.010 | −0.016 |
| | tail-uniform | −0.004 | **−0.013** | −0.006 |
| | tail-weighted | −0.001 | **−0.004** | +0.003 |
| **MAR20 × YOLOv8n | basic_aug | 0.572 ± 0.005 | 0.532 ± 0.013 | 0.467 ± 0.017 |
| | RFS | +0.003 | +0.016 | −0.011 |
| | tail-uniform | +0.010 | **+0.030** | +0.013 |
| | tail-weighted | +0.012 | **+0.044** | +0.008 |
| | weak-uniform | +0.004 | +0.008 | **+0.030** |
| | weak-weighted | +0.003 | +0.004 | **+0.034** |
| **MAR20 × RT-DETR-L | basic_aug | 0.697 ± 0.001 | 0.676 ± 0.007 | 0.611 ± 0.010 |
| | RFS | +0.003 | +0.013 | 0.000 |
| | tail-uniform | −0.007 | **+0.007** | −0.009 |
| | tail-weighted | −0.005 | **0.000** | −0.007 |

**Table 3.** Pre-registered contrasts (seed-blocked paired t, n = 3
seeds; Holm correction within each cell's five-contrast family).

| Dataset | Contrast | Estimate | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| MAD | Tail arms vs. baseline (tail scope) | +0.061 | [+0.026, +0.096] | 0.017 | 0.069 |
| MAD | Weak arms vs. baseline (weak scope) | +0.040 | [−0.021, +0.100] | 0.106 | 0.317 |
| MAD | **Set × scope interaction** | **+0.060** | **[+0.045, +0.075]** | **0.003** | **0.017** |
| MAD | Weighted vs. uniform (tail set) | +0.002 | [−0.031, +0.035] | 0.808 | 0.808 |
| MAD | Weighted vs. uniform (weak set) | −0.012 | [−0.036, +0.013] | 0.172 | 0.345 |
| MAR20 | Tail arms vs. baseline (tail scope) | +0.037 | [+0.015, +0.058] | 0.018 | 0.091 |
| MAR20 | Weak arms vs. baseline (weak scope) | +0.032 | [−0.024, +0.087] | 0.133 | 0.336 |
| MAR20 | Set × scope interaction | +0.052 | [+0.021, +0.083] | 0.018 | 0.091 |
| MAR20 | Weighted vs. uniform (tail set) | +0.014 | [−0.008, +0.035] | 0.112 | 0.336 |
| MAR20 | Weighted vs. uniform (weak set) | +0.004 | [−0.019, +0.027] | 0.519 | 0.519 |


![](/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Military_OD/paper/figures_v2/fig3_dissociation.png)

**Figure 3.** The allocation signal determines where gains occur. Δ
mAP50–95 versus the same-cell baseline for the four inpainting arms on
the tail scope (blue) and weak scope (orange); black-edged bars mark
each arm's targeted scope; error bars span ±1 standard deviation over
three seeds. Left: MAD
(confirmatory; interaction Holm-adjusted p = 0.017). Right: MAR20
(replication of the effect size; Section 5.1).


![](/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Military_OD/paper/figures_v2/fig4_condition_map.png)

**Figure 4.** Condition map across the four cells. Best augmentation
Δ mAP50–95 (all scope) against the unaugmented baseline of each cell;
triangles denote repeat-factor sampling, circles the best inpainting
arm; color denotes the dataset. Gains shrink to zero under the
high-baseline RT-DETR-L cells (right), and resampling and inpainting
exchange ranks between MAD and MAR20 (left).


![](/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Military_OD/paper/figures_v2/fig5_generation_examples.png)

**Figure 5.** Generation examples (source, protection mask, output; mask
black = protected). Rows: two MAD and two MAR20 cases spanning
formation, airshow, apron, and airbase scenes. The top row also
illustrates the hallucination failure mode quantified in Section 5.4 —
diffusion has painted additional, unlabeled aircraft into the
regenerated sky — which motivates the detector-based audit and the
object-gate robustness check.


# 6. Discussion


## 6.1 Engineering guidance: a three-step decision rule

Read together, the four cells yield an actionable procedure for
practitioners who must decide whether — and how — to spend a
synthetic-data budget on a detection system.

**Step 1: assess detector headroom before investing in augmentation.**
The largest single improvement observed in this study came not from any
data intervention but from replacing the detector: the pretrained
transformer's *unaugmented* baseline exceeded the best augmented result
of the compact detector by 0.21 mAP50–95 on MAD (0.869 vs. 0.661), and
once the stronger detector was in place, every augmentation we measured
was ineffective and resampling was mildly harmful (Section 5.2). We
stress that this is a cross-cell engineering observation rather than a
randomized comparison — the two detectors differ simultaneously in
capacity and architecture family (both are initialized from
COCO-pretrained weights; Section 4.2) — but its practical
reading held in every condition we tested: when deployment constraints
permit a stronger pretrained detector, the model upgrade dominated every
augmentation decision we evaluated. Data-side interventions become worth
considering primarily where a compact detector is mandated, for example
by edge-compute budgets, and the model therefore retains visible
headroom.

**Step 2: use free resampling as a first attempt — and as a diagnostic.**
Repeat-factor sampling requires no generation compute and, on the deeply
imbalanced MAD (imbalance ratio 10.5), outperformed every diffusion arm
on every scope (Section 5.3). Its benefit did not transfer to the
shallowly imbalanced MAR20, where it was null overall and negative on
the weak scope. Because resampling can only re-expose existing images,
it helps precisely when under-exposure of rare classes is the binding
constraint; its failure therefore signals that the constraint lies
elsewhere — in the diversity of the training distribution rather than in
exposure frequency — which is the regime where injecting genuinely new
variation through generation remains the available lever. In practice,
RFS functions as a low-cost probe: run it first, keep it if it helps,
and read a null result as evidence that a generative budget may be worth
its price.

**Step 3: when generating, select the class set by measured difficulty;
per-class quota tuning is unnecessary.** The interaction result
(Section 5.1) shows that the allocation signal determines where gains
occur, whereas the within-set weighting axis was null in all four
pre-registered comparisons. This simplifies deployment considerably: the
only consequential choice is which classes to target. Frequency is the
conventional default for that choice, but it is a poor one whenever
frequency and difficulty decouple, as they do here (r = −0.375). If the
operational goal is to lift worst-case class performance, the per-class
validation AP of the current baseline — available at no additional cost
in any trained system — is the direct signal, and targeting the
lowest-AP classes aligns the expected gains with that goal; within the
chosen set, a uniform split of the budget suffices. The compute cost is
moderate: generating and verifying one thousand 512-pixel background
variants took a few GPU-hours on a single mid-range accelerator in our
experiments, of the same order as one detector training run at this
scale.

## 6.2 Why background inpainting behaves this way

Three observations jointly constrain the interpretation. First, the
compact detector's dominant error mode on MAD is missed detection:
baseline recall (0.52–0.59) trails precision (0.62–0.71), and the
largest off-diagonal mass in the confusion structure assigns true
objects to background. Second, gains concentrate on the targeted set yet
consistently spill over, attenuated, to non-targeted classes
(Section 5.1). Third, all gains vanish under the higher-capacity
pretrained detector (Section 5.2). These
observations are consistent with a single account: background inpainting
counteracts *background overfitting*. Re-rendering the surroundings of a
geometrically protected object weakens context cues spuriously
correlated with class identity, forcing the detector to rely on object
appearance and thereby recovering missed detections — most strongly for
classes whose contexts were re-rendered (targeting), positively but more
weakly for classes that share the diversified background statistics
(spillover), and not at all for a detector with the capacity to retain
the broad variation of its pretraining, in which the spurious
object–context coupling is already weak (headroom). The account also
rationalizes the null weighting axis as saturation: once a class
receives enough background variation to break the spurious correlation,
additional variants of the same kind contribute little. Our data contain
direct support for this reading — the weighted tail arm, whose smallest
realized per-class quota was 16 images against the uniform arm's 77,
achieved statistically indistinguishable tail-scope gains. Finally, the
MAR20 contrast between null resampling and effective inpainting argues
that re-exposure of the protected objects cannot by itself account for
the gains: the synthetic images repeat foregrounds exactly as resampling
does, yet only the arm that also renews the backgrounds improved
performance there. We advance this account as an interpretation
consistent with all of our observations, not as a demonstrated causal
chain.

## 6.3 Limitations and threats to validity

Four limitations bound our claims. (1) *Replication power.* The
satellite-domain replication reproduces the interaction estimate inside
the original confidence interval, but its six-class scopes leave the
seed-blocked tests underpowered, and the effect does not survive Holm
correction there; we accordingly claim replication of magnitude and
direction, not a second confirmatory significance. (2) *Set overlap.* On
MAR20 the frequency-defined and difficulty-defined sets share one of six
classes, slightly diluting the dissociation relative to the fully
disjoint MAD design. (3) *Strong-detector cells: coverage and
attribution.* The transformer cells omit the weak-set arms, so the
headroom finding rests on near-zero deltas of the measured arms rather
than on an interaction test; moreover, the strong-detector comparison
co-varies capacity and architecture family (the pretraining corpus is
shared), so "headroom" names their joint effect rather than isolating a
single factor. A full factorial on the strong detector, together with
capacity-matched comparisons within one architecture family, would
resolve both points. (4) *External
validity.* All evidence comes from one application domain (aircraft),
two datasets, one generator family (Stable Diffusion inpainting, at one
budget point per dataset), and two detector capacity tiers. The
condition map should therefore be read as a demonstrated pattern with
identified moderators — detector headroom and imbalance depth — not as a
parametric law. Natural next steps are budget scaling curves, a
factorial combination of resampling with targeted generation to test
whether their gains compose, stronger generators, and replication on
broader long-tailed benchmarks.


# 7. Conclusion

We asked a question that the growing literature on generative
augmentation leaves implicit: when a fixed budget of synthetic images is
available, which classes should receive it? A 2×2 factorial over the
allocation signal and the within-set weighting — run under a fixed
budget, generator, and verification protocol on two public
military-aircraft benchmarks and two detector families — gives a
consistent answer. The class set is the decision that matters: gains
land on the targeted set, confirmed by a pre-registered interaction on
the natural-view benchmark and replicated in effect size on the
satellite-view benchmark, while dividing the budget inside the chosen
set uniformly or by priority makes no measurable difference. The choice
of signal is not innocuous, because frequency and measured difficulty
select disjoint classes here; practitioners whose goal is worst-case
class performance should target the measured-weakest classes rather
than the rarest. Both findings, however, are conditional: under a large
pretrained transformer detector every augmentation effect disappears,
and the zero-cost resampling baseline that dominates generation under
deep imbalance turns null — and locally harmful — when the imbalance is
shallow. The resulting decision rule — assess headroom, probe with free
resampling, then generate for measured weakness with a uniform split —
is immediately actionable, and the evaluation protocol that produced it
(budget control, pre-registered contrasts, seed-blocked tests) is
reusable wherever augmentation claims are made. Extending the factorial
to strong detectors, tracing budget scaling curves, and testing whether
resampling and targeted generation compose are the natural next steps.

# References


- Ahmed, T., Kumar, A., Álvarez Casado, C., Zhang, A., Hänninen, T., Loven, L., Bordallo López, M., Tarkoma, S., 2025. Exponentially weighted instance-aware repeat factor sampling for long-tailed object detection model training in unmanned aerial vehicles surveillance scenarios. arXiv:2503.21893.
- Bozorgtabar, B., Mahapatra, D., von Teng, H., Pollinger, A., Ebner, L., Thiran, J.-P., Reyes, M., 2019. Informative sample generation using class aware generative adversarial networks for classification of chest X-rays. arXiv:1904.10781.
- Crasto, N., 2024. Class imbalance in object detection: An experimental diagnosis and study of mitigation strategies. arXiv:2403.07113.
- Fan, C., et al., 2024. DiverGen: Improving instance segmentation by learning wider data distribution with more diverse generative data. In: Proc. IEEE/CVF CVPR 2024.
- Geng, S., Hsieh, C.-Y., Ramanujan, V., Wallingford, M., Li, C.-L., Koh, P.W., Krishna, R., 2024. The unmet promise of synthetic training images: Using retrieved real images performs better. arXiv:2406.05184.
- Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.-Y., Cubuk, E.D., Le, Q.V., Zoph, B., 2021. Simple copy-paste is a strong data augmentation method for instance segmentation. In: Proc. IEEE/CVF CVPR 2021.
- Gupta, A., Dollár, P., Girshick, R., 2019. LVIS: A dataset for large vocabulary instance segmentation. In: Proc. IEEE/CVF CVPR 2019.
- Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S., 2017. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In: Advances in Neural Information Processing Systems 30 (NeurIPS 2017).
- Holm, S., 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6 (2), 65–70.
- Jocher, G., Chaurasia, A., Qiu, J., 2023. Ultralytics YOLOv8 (software). https://github.com/ultralytics/ultralytics.
- Li, B., et al., 2022. Equalized focal loss for dense long-tailed object detection. In: Proc. IEEE/CVF CVPR 2022.
- Li, Y., et al., 2020. Overcoming classifier imbalance for long-tail object detection with balanced group softmax. In: Proc. IEEE/CVF CVPR 2020.
- Li, Y., Dong, X., Chen, C., Zhuang, W., Lyu, L., 2024. A simple background augmentation method for object detection with diffusion model. arXiv:2408.00350.
- Liang, Y., Bhardwaj, S., Zhou, T., 2025. Diffusion curriculum: Synthetic-to-real data curriculum via image-guided diffusion. In: Proc. IEEE/CVF ICCV 2025. arXiv:2410.13674.
- Radford, A., et al., 2021. Learning transferable visual models from natural language supervision. In: Proc. ICML 2021.
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent diffusion models. In: Proc. IEEE/CVF CVPR 2022.
- rookieengg, 2023. Military Aircraft Detection Dataset (YOLO format) [dataset]. Kaggle. https://www.kaggle.com/datasets/rookieengg/military-aircraft-detection-dataset-yolo-format.
- Song, G., Du, H., Zhang, X., Bao, F., Zhang, Y., 2024. Small object detection in unmanned aerial vehicle images using multi-scale hybrid attention. Engineering Applications of Artificial Intelligence. https://www.sciencedirect.com/science/article/abs/pii/S0952197623016391.
- Suri, S., et al., 2023. Gen2Det: Generate to detect. arXiv:2312.04566.
- Tang, D., et al., 2025. AeroGen: Enhancing remote sensing object detection with diffusion-driven data generation. In: Proc. IEEE/CVF CVPR 2025.
- Wang, J., et al., 2021. Seesaw loss for long-tailed instance segmentation. In: Proc. IEEE/CVF CVPR 2021.
- Wu, J., Zhao, F., Yao, G., Jin, Z., 2025. FGA-YOLO: A one-stage and high-precision detector designed for fine-grained aircraft recognition. Neurocomputing 618, 129067.
- Xie, J., et al., 2024. MosaicFusion: Diffusion models as data augmenters for large vocabulary instance segmentation. International Journal of Computer Vision. arXiv:2309.13042.
- Yaman, B., Mahmud, T., Liu, C.-H., 2023. Instance-aware repeat factor sampling for long-tailed object detection. arXiv:2305.08069. NeurIPS Workshop on Heavy Tails in Machine Learning.
- Yu, W., Cheng, G., Wang, M., Yao, Y., Xie, X., Yao, X., Han, J., 2023. MAR20: A benchmark for military aircraft recognition in remote sensing images. National Remote Sensing Bulletin 27 (12), 2688–2696.
- Zhao, H., et al., 2023. X-Paste: Revisiting scalable copy-paste for instance segmentation using CLIP and StableDiffusion. In: Proc. ICML 2023, PMLR 202.
- Zhao, Y., et al., 2024. DETRs beat YOLOs on real-time object detection. In: Proc. IEEE/CVF CVPR 2024. arXiv:2304.08069.
- Zhu, J., et al., 2024. ODGEN: Domain-specific object detection data generation with diffusion models. In: Advances in Neural Information Processing Systems 37 (NeurIPS 2024). arXiv:2405.15199.
