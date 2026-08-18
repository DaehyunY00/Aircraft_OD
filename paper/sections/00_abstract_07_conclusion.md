# Abstract, Highlights, Keywords, and Conclusion

<!-- Draft v1 (2026-08-17). EAAI rules applied: abstract ~250 words,
     explicit AI-contribution / engineering-application statements
     (desk-reject criterion), zero undefined acronyms in title+abstract
     (average precision spelled out; repeat-factor resampling spelled out;
     benchmarks described, not named). Highlights ≤ 85 chars each. -->

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

*(~255 words)*

## Highlights

- Class-set choice, not per-class weighting, decides where inpainting gains land
- Set-by-scope interaction +0.06 (corrected p = 0.017), replicated on satellite data
- Every augmentation gain vanishes under a stronger pretrained detector
- Resampling wins on deep imbalance, fails on shallow — a cheap diagnostic probe
- Pre-registered contrasts and seed-blocked statistics guard augmentation claims

## Keywords

Data augmentation; Diffusion inpainting; Object detection; Class
imbalance; Synthetic data allocation; Military aircraft detection

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
