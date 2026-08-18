# 1. Introduction

<!-- Draft v1 (2026-08-17). Bounded by SKELETON.md §2–§4 and §8.
     NOTE: §1 now carries the official first definitions of AP and RFS
     (reading order); SKELETON registry updated. Detector/framework names
     and all remaining acronyms are deferred to §4. -->

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
