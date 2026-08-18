# Figure captions (assembly-ready)

**Figure 1.** Overview. (a) Generation pipeline: all annotated boxes in a
source image are geometrically protected; diffusion inpainting
regenerates only the background; outputs must pass rule-based
verification (with budget refill) before insertion into the training
split. Budget *B*, class count *K*, generator, prompts, and verification
are held fixed across arms. (b) The 2×2 allocation design: class set
(frequency-defined tail vs. AP-defined weak) × within-set weighting
(uniform vs. priority-weighted).

**Figure 2.** Frequency is a poor proxy for difficulty on MAD. Per-class
baseline test mAP50–95 (three-seed mean) against training instance count
(log scale). The frequency-defined tail set (blue) and the AP-defined
weak set (orange) are disjoint; the rarest class (YF-23) is the easiest,
while the weakest classes (Rafale, F-14) sit at medium-to-head
frequency.

**Figure 3.** The allocation signal determines where gains occur. Δ
mAP50–95 versus the same-cell baseline for the four inpainting arms on
the tail scope (blue) and weak scope (orange); black-edged bars mark
each arm's targeted scope; error bars span ±1 standard deviation over
three seeds. Left: MAD
(confirmatory; interaction Holm-adjusted p = 0.017). Right: MAR20
(replication of the effect size; Section 5.1).

**Figure 4.** Condition map across the four cells. Best augmentation
Δ mAP50–95 (all scope) against the unaugmented baseline of each cell;
triangles denote repeat-factor sampling, circles the best inpainting
arm; color denotes the dataset. Gains shrink to zero under the
high-baseline RT-DETR-L cells (right), and resampling and inpainting
exchange ranks between MAD and MAR20 (left).

**Figure 5.** Generation examples (source, protection mask, output; mask
black = protected). Rows: two MAD and two MAR20 cases spanning
formation, airshow, apron, and airbase scenes. The top row also
illustrates the hallucination failure mode quantified in Section 5.4 —
diffusion has painted additional, unlabeled aircraft into the
regenerated sky — which motivates the detector-based audit and the
object-gate robustness check.
