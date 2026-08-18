# 5. Results

<!-- Draft v3 (2026-08-17): paragraph-level review pass. Fixes: metric
     labeling in the correlation paragraph (AP50 examples marked), overlap
     stated as empirical fact with per-detector note, double dissociation
     defined in-line, weighting-null claim bounded to studied budgets with
     CI framing, spillover range corrected (+0.004..+0.013), 5.2 detector
     parenthetical deferred to §4.2 (both COCO-initialized), headroom
     phrasing, "three originally generated pools" in 5.4, Table 2 caption
     cell-specific scopes, zero formatting unified, Table 3 CIs filled. -->

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

<!-- ===================== TABLES ===================== -->

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
