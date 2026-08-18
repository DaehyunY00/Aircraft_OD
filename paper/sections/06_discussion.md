# 6. Discussion

<!-- Draft v2 (2026-08-17): paragraph-level revision pass — cross-cell
     observation flagged as such, RFS diagnostic logic tightened,
     worst-case-goal premise made explicit, generation-cost claim
     corrected, 16-vs-77 quota evidence and MAR20 re-exposure
     counter-argument added to §6.2, capacity/pretraining confound added
     to limitations. Bounded by SKELETON.md §4 and §8. -->

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
