"""IEEE Access 초안용 그림 생성.

데이터는 GCS에서 받은 최신 결과(scratchpad/res) 사용 — 로컬 Drive의
outputs_full/metrics 는 GCP 학습 이전 상태라 쓰면 안 된다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import pearsonr, spearmanr

RES = Path(__file__).parent / "res"  # gs://military-od-d522190f/outputs_full 의 metrics/analysis CSV를 여기로 복사
DRIVE = Path("/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Military_OD")
OUT = DRIVE / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito 기반, dataviz validator 통과 확인됨
C_HEAD, C_MED, C_TAIL = "#0072B2", "#E69F00", "#009E73"
C_SEL, C_WEAK = "#0072B2", "#D55E00"
GRAY = "#7a7a7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 120,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


groups = pd.read_csv(RES / "class_groups.csv")
ap = pd.read_csv(RES / "per_class_ap.csv")
test = ap[ap.eval_split == "test"]
weak_ids = set(pd.read_csv(RES / "augmentation_plan_weakness.csv").class_id)
tail_ids = set(groups.loc[groups.group == "tail", "class_id"])

# ---------------------------------------------------------------- Fig 2
base = (
    test[test.experiment == "basic_aug"].groupby("class_id")[["ap50", "ap50_95"]].mean()
    .join(groups.set_index("class_id")[["class_name", "group", "instance_count"]])
)
r, p = pearsonr(base.instance_count, base.ap50_95)
rho, ps = spearmanr(base.instance_count, base.ap50_95)
print(f"fig2 검증: Pearson r={r:.3f} (p={p:.3f}), Spearman rho={rho:.3f} (p={ps:.3f})")

fig, ax = plt.subplots(figsize=(3.5, 2.7))
for grp, color in (("head", C_HEAD), ("medium", C_MED), ("tail", C_TAIL)):
    d = base[base.group == grp]
    ax.scatter(d.instance_count, d.ap50_95, s=16, c=color, label=grp,
               edgecolors="white", linewidths=0.4, zorder=3)
# 시각 보조선: log10(count)에 대한 최소제곱 적합
x = np.log10(base.instance_count.to_numpy(float))
b1, b0 = np.polyfit(x, base.ap50_95.to_numpy(float), 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(10 ** xs, b0 + b1 * xs, color=GRAY, lw=1.0, ls="--", zorder=2)
ax.set_xscale("log")
ax.set_xlabel("Training instances per class (log scale)")
ax.set_ylabel("Per-class test mAP50-95")
ax.text(0.03, 0.05,
        f"Pearson r = {r:.3f} (p = {p:.3f})\nSpearman ρ = {rho:.3f} (p = {ps:.3f})",
        transform=ax.transAxes, fontsize=6.8, va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
# 주석: 반례의 얼굴들 (라벨 겹침은 오프셋으로 수동 해소)
ann = {"YF23": (8, 4), "Rafale": (7, 3), "F14": (6, 5), "F18": (5, -9),
       "EF2000": (-12, 8), "U2": (-16, -3)}
for name, (dx, dy) in ann.items():
    row = base[base.class_name == name].iloc[0]
    ax.annotate(name, (row.instance_count, row.ap50_95),
                xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                color="#333333")
ax.legend(frameon=False, loc="upper right", handletextpad=0.2,
          borderpad=0.2, labelspacing=0.3)
ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
save(fig, "fig2_freq_vs_ap")

# ---------------------------------------------------------------- Fig 3
st = pd.read_csv(RES / "statistical_tests.csv")
st = st[(st.variant_b == "basic_aug") & (st.eval_split == "test")]


def cell(arm, scope):
    row = st[(st.variant_a == arm) & (st.scope == scope)].iloc[0]
    return row.mean_diff, row.p_value


scopes = ["tail", "weak"]
arms = [("aug_selective_inpaint", "Frequency-based\n(selective-tail)", C_SEL),
        ("aug_weakness_inpaint", "Weakness-based", C_WEAK)]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
w = 0.34
xpos = np.arange(len(scopes))
for i, (arm, label, color) in enumerate(arms):
    vals, pvals = zip(*[cell(arm, s) for s in scopes])
    bars = ax.bar(xpos + (i - 0.5) * w, vals, w * 0.92, color=color, label=label, zorder=3)
    for bx, v, pv in zip(xpos + (i - 0.5) * w, vals, pvals):
        star = "**" if pv < 0.01 else ("*" if pv < 0.05 else "n.s.")
        weight = "bold" if pv < 0.05 else "normal"
        col = "#222222" if pv < 0.05 else GRAY
        ax.text(bx, v + 0.0022, star, ha="center", fontsize=7.5, weight=weight, color=col)
        ax.text(bx, v / 2, f"+{v:.3f}", ha="center", va="center", fontsize=6.3,
                color="white", zorder=4)
ax.axhline(0, color="#444444", lw=0.6)
ax.set_xticks(xpos)
ax.set_xticklabels(["Frequency-tail classes\n(n = 13)", "Measured-weak classes\n(n = 13)"])
ax.set_ylabel("Δ mAP50-95 vs. basic_aug")
ax.set_ylim(0, 0.072)
ax.set_xlabel("Evaluation scope")
ax.legend(frameon=False, loc="upper center", ncols=2, columnspacing=1.2,
          handletextpad=0.4, bbox_to_anchor=(0.5, 1.14))
ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
ax.text(0.02, 0.97, "* p<0.05   ** p<0.01\n(Wilcoxon, class-paired)",
        transform=ax.transAxes, ha="left", va="top", fontsize=6, color=GRAY)
save(fig, "fig3_double_dissociation")

# ---------------------------------------------------------------- Fig 4
piv = test.groupby(["experiment", "class_id"]).ap50.mean().unstack(0)
wk = sorted(weak_ids, key=lambda c: piv.loc[c, "basic_aug"])
names = groups.set_index("class_id").class_name
fig, ax = plt.subplots(figsize=(3.5, 3.1))
ypos = np.arange(len(wk))
for y, cid in zip(ypos, wk):
    b, a = piv.loc[cid, "basic_aug"], piv.loc[cid, "aug_weakness_inpaint"]
    ax.plot([b, a], [y, y], color="#cccccc", lw=1.1, zorder=2)
    ax.scatter([b], [y], s=17, color=GRAY, zorder=3)
    ax.scatter([a], [y], s=17, color=C_WEAK, zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels([names[c] for c in wk])
ax.set_xlabel("Per-class test AP50")
handles = [Line2D([], [], marker="o", ls="", ms=4.5, color=GRAY, label="basic_aug"),
           Line2D([], [], marker="o", ls="", ms=4.5, color=C_WEAK, label="weakness arm")]
ax.legend(handles=handles, frameon=False, loc="lower right")
ax.grid(axis="x", lw=0.4, color="#dddddd", zorder=0)
save(fig, "fig4_weak_class_change")

# ---------------------------------------------------------------- Fig 5
# review sheet 셀 좌표: (c*256, r*280+24) ~ (+256, +256)  [src/utils/image.py]
def cells(sheet, row):
    img = Image.open(sheet)
    return [img.crop((c * 256, row * 280 + 24, (c + 1) * 256, row * 280 + 280)) for c in range(3)]


SHEETS = DRIVE / "outputs_full" / "synthetic"
examples = [
    (SHEETS / "review_sheet_weakness.jpg", 5, "(a) urban → runway"),
    (SHEETS / "review_sheet_weakness.jpg", 6, "(b) foliage → mountains"),
    (SHEETS / "review_sheet_selective.jpg", 7, "(c) airfield → clouds"),
    (SHEETS / "review_sheet_selective.jpg", 1, "(d) failure:\nhallucinated aircraft"),
]
fig, axes = plt.subplots(3, 4, figsize=(7.16, 5.5))
for col, (sheet, row, title) in enumerate(examples):
    for rr, panel in enumerate(cells(sheet, row)):
        axm = axes[rr, col]
        axm.imshow(panel)
        axm.set_xticks([]); axm.set_yticks([])
        for s in axm.spines.values():
            s.set_visible(True); s.set_linewidth(0.5); s.set_color("#999999")
    axes[0, col].set_title(title, fontsize=7.5)
for rr, lab in enumerate(["Original", "Mask", "Generated"]):
    axes[rr, 0].set_ylabel(lab, fontsize=8)
fig.subplots_adjust(wspace=0.04, hspace=0.06)
save(fig, "fig5_generation_examples")

# ---------------------------------------------------------------- Fig 1
# 파이프라인 개요. Stage 1(배분 = 실험 변수) / Stage 2(생성·검증 = 모든 arm 공통).
# 썸네일은 실제 리뷰 시트의 한 행(urban → runway)을 그대로 사용.
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x0, x1, y0, y1, text, ec="#555555", fc="white", fs=6.8, lw=0.8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.35", ec=ec, fc=fc, lw=lw))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color="#222222")


def arrow(ax, p0, p1, rad=0.0, color="#555555", lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.16, 3.1))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

ax.text(1, 42.4, "Stage 1 · Budget allocation — the experimental variable",
        fontsize=7.5, weight="bold", color="#333333")
box(ax, 1.5, 18, 35, 40, "Class frequency\n(instance counts)", fs=6.3)
box(ax, 1.5, 18, 28.5, 33.5, "Measured per-class AP\n(baseline · val split)", fs=6.3)
box(ax, 26, 56, 37.5, 41.2, "Uniform-tail: equal over 13 freq-tail classes",
    ec="#8a8a8a", fc="#f2f2f2", fs=6.4)
box(ax, 26, 56, 32.4, 36.1, "Selective-tail: rarity + weakness, same 13",
    ec=C_SEL, fc="#E8F1F8", fs=6.4, lw=1.2)
box(ax, 26, 56, 27.3, 31.0, "Weakness: bottom-13 of all 43 by measured AP",
    ec=C_WEAK, fc="#FBEAE0", fs=6.4, lw=1.2)
box(ax, 64, 80, 31, 37.5, "Per-class budgets\nB = 1,000 · K = 13", weight="bold", fs=6.6)
arrow(ax, (18.7, 38.4), (25.2, 39.4), rad=-0.10)
arrow(ax, (18.7, 36.8), (25.2, 34.8), rad=0.10)
arrow(ax, (18.7, 32.4), (25.2, 33.8), rad=-0.08)
arrow(ax, (18.7, 30.4), (25.2, 29.2), rad=0.08)
arrow(ax, (56.8, 39.3), (63.2, 35.8), rad=-0.10)
arrow(ax, (56.8, 34.2), (63.2, 34.2))
arrow(ax, (56.8, 29.1), (63.2, 32.6), rad=0.10)
# 예산 → 생성 루프: elbow (chip·caption과 안 겹치는 y=23 차선 사용)
ax.plot([72, 72], [30.4, 23], color=GRAY, lw=0.8)
ax.plot([72, 7], [23, 23], color=GRAY, lw=0.8)
arrow(ax, (7, 23), (7, 20.4), color=GRAY, lw=0.8)
ax.text(40, 23.8, "per-class quota drives the loop", fontsize=6.2, color=GRAY,
        style="italic", ha="center")

thumbs = cells(SHEETS / "review_sheet_weakness.jpg", 5)
for (x0, cap), im in zip([(2, "Source + GT boxes"), (17, "Mask (pad · blur)"),
                          (32, "Inpainted (20 steps)")], thumbs):
    ax.imshow(im, extent=(x0, x0 + 10, 10, 20), aspect="auto", zorder=2)
    ax.add_patch(plt.Rectangle((x0, 10), 10, 10, fill=False, ec="#999999", lw=0.5, zorder=3))
    ax.text(x0 + 5, 8.9, cap, ha="center", va="top", fontsize=5.6, color="#222222")
arrow(ax, (12.3, 15), (16.7, 15))
arrow(ax, (27.3, 15), (31.7, 15))
arrow(ax, (42.3, 15), (46.0, 15))
box(ax, 46.5, 62, 10.5, 19.5,
    "Verification gates\nΔ background ≥ 10\nΔ protected box ≤ 5\neditable bg ≥ 5 %", fs=6.2)
arrow(ax, (62.6, 15), (68.0, 15))
ax.text(65.3, 16.2, "accept", ha="center", fontsize=5.8, color="#222222")
box(ax, 68.5, 83, 10.5, 19.5, "Synthetic train\nimages +\nunchanged labels", fs=6.4)
arrow(ax, (83.6, 15), (86.0, 15))
box(ax, 86.5, 99, 10.5, 19.5, "YOLOv8n per arm\n→ scoped eval\n(all · tail · weak)", fs=6.2)
ax.plot([54, 54], [10.0, 5], color=GRAY, lw=0.8)
ax.plot([54, 7], [5, 5], color=GRAY, lw=0.8)
arrow(ax, (7, 5), (7, 9.6), color=GRAY, lw=0.8)
ax.text(30.5, 3.2, "reject → resample source & seed (≤ 2× budget)", ha="center",
        fontsize=6.2, color=GRAY, style="italic")
ax.text(1, 0.4, "Stage 2 · Label-preserving generation & verification — identical for every arm",
        fontsize=7.5, weight="bold", color="#333333")
save(fig, "fig1_pipeline")

# ---------------------------------------------------------------- 부속 자료 복사
import shutil

cm_src = DRIVE / "outputs_full" / "runs" / "basic_aug_yolov8n_seed42_20260704_0154" / "confusion_matrix_normalized.png"
shutil.copy(cm_src, OUT / "fig6_confusion_matrix_baseline.png")
print("saved fig6 (copied confusion matrix)")
print("\n산출:", sorted(p.name for p in OUT.iterdir()))
