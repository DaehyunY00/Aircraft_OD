# 2. Related Work

<!-- Draft v1 (2026-08-17). Citations in (Author, Year) draft form; BibTeX
     keys to be attached at assembly. Sources verified in
     research_review_prior_work.md. NOTE: §2 precedes §3 in reading order,
     so AP / RFS / YOLO-family acronyms are FIRST-DEFINED here — SKELETON
     §8 registry updated accordingly. -->

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
