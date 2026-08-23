"""EAAI 원고 Figure 1–5 생성 (SKELETON §5 계획 준수).

데이터 원본: outputs_confirmatory(C1), outputs_mad_rtdetr(C2),
outputs_mar20_yolo(C3), outputs_mar20_rtdetr(C4) — 전부 로컬 검증본.
출력: paper/figures_v2/*.{pdf,png} (구 figures/는 보존).

팔레트: Okabe–Ito 부분집합, dataviz validator 통과(2026-08-17).
  BLUE  #0072B2  tail set / tail scope / MAD
  ORANGE#E69F00  weak set / weak scope / MAR20
  GREEN #009E73  (예비)
  VERM  #D55E00  RFS 강조
  GRAY  #8A8A8A  중립
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import FancyArrow, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8A8A8A"
INK, MUTED = "#1a1a1a", "#666666"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)

CELLS = {
    "C1": ("outputs_confirmatory", "MAD", "YOLOv8n"),
    "C2": ("outputs_mad_rtdetr", "MAD", "RT-DETR-L"),
    "C3": ("outputs_mar20_yolo", "MAR20", "YOLOv8n"),
    "C4": ("outputs_mar20_rtdetr", "MAR20", "RT-DETR-L"),
}
ARM_LABELS = {
    "aug_uniform_inpaint": "tail-uniform",
    "aug_selective_inpaint": "tail-priority",
    "aug_weakuniform_inpaint": "weak-uniform",
    "aug_weakness_inpaint": "weak-priority",
    "aug_rfs": "RFS",
}


def load_cell(cell: str):
    root, *_ = CELLS[cell]
    pc = pd.read_csv(ROOT / root / "metrics" / "per_class_ap.csv")
    pc = pc[pc.eval_split == "test"]
    groups = pd.read_csv(ROOT / root / "analysis" / "class_groups.csv")
    tail = set(groups[groups.group == "tail"].class_id)
    weak = set(pd.read_csv(ROOT / root / "analysis" / "augmentation_plan_weakness.csv").class_id)
    return pc, groups, tail, weak


def macro_by_seed(pc: pd.DataFrame, experiment: str, ids=None) -> pd.Series:
    d = pc[pc.experiment == experiment]
    if ids is not None:
        d = d[d.class_id.isin(ids)]
    return d.groupby("seed")["ap50_95"].mean()


def deframe(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


# ---------------------------------------------------------------- Figure 2
def fig2_freq_vs_ap():
    pc, groups, tail, weak = load_cell("C1")
    ap = (
        pc[pc.experiment == "basic_aug"].groupby("class_id", as_index=False)["ap50_95"].mean()
    )
    df = ap.merge(groups[["class_id", "class_name", "instance_count"]], on="class_id")
    pr, pp = stats.pearsonr(df.instance_count, df.ap50_95)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for ids, color, label, z in (
        (set(df.class_id) - tail - weak, GRAY, "other classes", 1),
        (tail, BLUE, "tail set (frequency)", 2),
        (weak, ORANGE, "weak set (validation AP)", 2),
    ):
        sub = df[df.class_id.isin(ids)]
        ax.scatter(
            sub.instance_count, sub.ap50_95, s=26, color=color, label=label,
            edgecolors="white", linewidths=0.5, zorder=z,
        )
    for name, dx, dy, ha in (("YF23", 0, 9, "center"), ("Tu95", 0, 9, "center"),
                             ("Rafale", 8, -3, "left"), ("F14", 0, -13, "center")):
        row = df[df.class_name == name]
        if len(row):
            ax.annotate(
                name, (row.instance_count.iloc[0], row.ap50_95.iloc[0]),
                textcoords="offset points", xytext=(dx, dy), ha=ha,
                fontsize=7, color=INK,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Training instances per class (log scale)")
    ax.set_ylabel("Baseline test mAP50–95 (3-seed mean)")
    ax.text(
        0.02, 0.04, f"Pearson r = {pr:.3f} (p = {pp:.3f})".replace("-", "−"),
        transform=ax.transAxes, fontsize=7.5, color=MUTED,
    )
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    deframe(ax)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.2, borderaxespad=0)
    save(fig, "fig2_freq_vs_ap")


# ---------------------------------------------------------------- Figure 3
def fig3_dissociation():
    arms = ["aug_uniform_inpaint", "aug_selective_inpaint",
            "aug_weakuniform_inpaint", "aug_weakness_inpaint"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, cell, title in zip(axes, ("C1", "C3"), ("MAD (43 classes)", "MAR20 (20 classes)")):
        pc, _, tail, weak = load_cell(cell)
        base_t = macro_by_seed(pc, "basic_aug", tail)
        base_w = macro_by_seed(pc, "basic_aug", weak)
        x = np.arange(len(arms))
        width = 0.36
        for off, scope_ids, base, color, label in (
            (-width / 2, tail, base_t, BLUE, "Δ on tail scope"),
            (width / 2, weak, base_w, ORANGE, "Δ on weak scope"),
        ):
            for i, arm in enumerate(arms):
                deltas = (macro_by_seed(pc, arm, scope_ids) - base).dropna()
                targeted = (arm.startswith(("aug_uniform", "aug_selective")) and scope_ids is tail) or (
                    arm.startswith("aug_weak") and scope_ids is weak
                )
                ax.bar(
                    i + off, deltas.mean(), width=width, color=color,
                    edgecolor=INK if targeted else "white",
                    linewidth=1.0 if targeted else 0.5,
                    label=label if i == 0 else None, zorder=2,
                )
                ax.errorbar(
                    i + off, deltas.mean(), yerr=deltas.std(ddof=1),
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.4,
                    capthick=0.9, zorder=3,
                )
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        deframe(ax)
    axes[0].set_ylabel("Δ mAP50–95 vs. baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.97))
    fig.text(
        0.005, 0.015,
        "Black-edged bar = the arm's targeted scope; error bars = ±1 SD over seeds.",
        fontsize=7, color=MUTED,
    )
    fig.subplots_adjust(wspace=0.08, bottom=0.24)
    save(fig, "fig3_dissociation")


# ---------------------------------------------------------------- Figure 4
def fig4_condition_map():
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ann = {
        "C1": (6, 8, "left"), "C2": (-8, -16, "right"),
        "C3": (6, 10, "left"), "C4": (6, 8, "left"),
    }
    for cell, (root, dataset, detector) in CELLS.items():
        pc, _, tail, weak = load_cell(cell)
        base = macro_by_seed(pc, "basic_aug")
        color = BLUE if dataset == "MAD" else ORANGE
        pts = {
            "RFS": ((macro_by_seed(pc, "aug_rfs") - base).mean(), "^"),
            "tail-uniform": ((macro_by_seed(pc, "aug_uniform_inpaint") - base).mean(), "o"),
        }
        bx = base.mean()
        ax.plot([bx, bx], [pts["RFS"][0], pts["tail-uniform"][0]],
                color=color, linewidth=0.8, alpha=0.45, zorder=1)
        for (label, (dy, marker)) in pts.items():
            ax.scatter(bx, dy, marker=marker, s=46, color=color,
                       edgecolors="white", linewidths=0.6, zorder=3)
        top = max(v for v, _ in pts.values())
        dx, dy, ha = ann[cell]
        ax.annotate(
            f"{dataset}\n{detector}", (bx, top), textcoords="offset points",
            xytext=(dx, dy), fontsize=7, color=INK, ha=ha, linespacing=1.1,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axhspan(-0.03, 0, color="#f2f2f2", zorder=0)
    ax.set_xlabel("Standard-augmentation baseline test mAP50–95")
    ax.set_ylabel("Δ mAP50–95 vs. standard-augmentation baseline (all scope)")
    ax.set_ylim(-0.025, 0.098)
    from matplotlib.lines import Line2D

    legend = [
        Line2D([], [], marker="^", ls="", color=INK, label="repeat-factor sampling"),
        Line2D([], [], marker="o", ls="", color=INK, label="tail-uniform arm"),
        Line2D([], [], marker="s", ls="", color=BLUE, label="MAD"),
        Line2D([], [], marker="s", ls="", color=ORANGE, label="MAR20"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper right", handletextpad=0.2)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    deframe(ax)
    save(fig, "fig4_condition_map")


# ---------------------------------------------------------------- Figure 1
def fig1_pipeline_design():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def box(ax, x, y, w, h, text, fc="#f5f7fa", ec=MUTED, fontsize=7.2, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.18",
                facecolor=fc, edgecolor=ec, linewidth=0.9,
            )
        )
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.25,
        )

    def arrow(ax, x0, y, x1):
        ax.add_patch(
            FancyArrow(x0, y, x1 - x0, 0, width=0.02, head_width=0.34,
                       head_length=0.28, length_includes_head=True, color=MUTED)
        )

    # (a) generation pipeline
    steps = [
        "Source image\n(all boxes\nprotected)",
        "Background\ninpainting\n(diffusion)",
        "Rule-based\nverification\n+ refill",
        "Insert into\ntrain split\nonly",
    ]
    xs = np.linspace(0.2, 7.8, 4)
    for x, s in zip(xs, steps):
        box(ax1, x, 3.6, 2.0, 3.4, s)
    for x0, x1 in zip(xs[:-1] + 2.0, xs[1:]):
        arrow(ax1, x0 + 0.12, 5.3, x1 - 0.12)
    box(ax1, 0.2, 0.4, 9.6, 1.9,
        "held fixed:  budget B · class count K · generator · prompts · verification",
        fc="#ffffff", ec="#bbbbbb", fontsize=7.0)
    ax1.text(0.2, 9.4, "(a) Generation pipeline", fontsize=8.2, fontweight="bold")

    # (b) four-arm crossed allocation comparison
    cells = [
        ("tail-\nuniform", BLUE), ("tail-\npriority", BLUE),
        ("weak-\nuniform", ORANGE), ("weak-\npriority", ORANGE),
    ]
    coords = [(2.6, 5.2), (6.4, 5.2), (2.6, 1.6), (6.4, 1.6)]
    for (label, color), (x, y) in zip(cells, coords):
        box(ax2, x, y, 3.2, 2.5, label, fc="#ffffff", ec=color, fontsize=7.2, bold=True)
    ax2.text(4.25 + 1.9, 8.6, "within-set quota rule", ha="center", fontsize=7.4, color=MUTED)
    ax2.text(4.25, 8.0, "uniform", ha="center", fontsize=7.2)
    ax2.text(8.05, 8.0, "set-specific priority", ha="center", fontsize=6.8)
    ax2.text(0.6, 5.0, "class set\n(allocation\nsignal)", ha="center", fontsize=7.4,
             color=MUTED, rotation=90, va="center")
    ax2.text(1.9, 6.45, "tail", ha="center", fontsize=7.2, color=BLUE, rotation=90)
    ax2.text(1.9, 2.85, "weak", ha="center", fontsize=7.2, color=ORANGE, rotation=90)
    ax2.text(0.2, 9.4, "(b) Four-arm crossed comparison", fontsize=8.2, fontweight="bold")
    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig1_pipeline_design")


# ---------------------------------------------------------------- Figure 5
def crop_row(sheet_path: Path, row: int) -> list[Image.Image]:
    """save_contact_sheet 레이아웃(256px 셀 + 24px 라벨 밴드)에서 row 추출."""
    img = Image.open(sheet_path)
    cell, band = 256, 24
    y0 = row * (cell + band) + band
    return [img.crop((c * cell, y0, (c + 1) * cell, y0 + cell)) for c in range(3)]


def fig5_generation_examples():
    rows = [
        ("MAD", ROOT / "outputs_confirmatory/synthetic/review_sheet_weakness.jpg", 0),
        ("MAD", ROOT / "outputs_confirmatory/synthetic/review_sheet_weakness.jpg", 5),
        ("MAR20", ROOT / "outputs_mar20_yolo/synthetic/review_sheet_uniform.jpg", 0),
        ("MAR20", ROOT / "outputs_mar20_yolo/synthetic/review_sheet_weakness.jpg", 1),
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(5.4, 7.3))
    col_titles = ["Source", "Protection mask", "Generated"]
    for r, (tag, sheet, row_idx) in enumerate(rows):
        for c, im in enumerate(crop_row(sheet, row_idx)):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
        axes[r, 0].set_ylabel(tag, fontsize=8, rotation=90)
    fig.subplots_adjust(wspace=0.02, hspace=0.04)
    save(fig, "fig5_generation_examples")


if __name__ == "__main__":
    fig2_freq_vs_ap()
    fig3_dissociation()
    fig4_condition_map()
    fig1_pipeline_design()
    fig5_generation_examples()
    print("done ->", OUT)


# ---------------------------------------------------------------- Figure 6
def fig6_perclass_delta():
    """클래스별 Δ 분포 — 'broad not concentrated' 주장의 시각 증거."""
    pc, groups, tail, weak = load_cell("C1")
    names = groups.set_index("class_id").class_name
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, arm, ids, color, title in (
        (axes[0], "aug_uniform_inpaint", tail, BLUE, "tail-uniform arm on tail classes"),
        (axes[1], "aug_weakuniform_inpaint", weak, ORANGE, "weak-uniform arm on weak classes"),
    ):
        base = pc[(pc.experiment == "basic_aug") & (pc.class_id.isin(ids))].groupby("class_id")["ap50_95"].mean()
        armv = pc[(pc.experiment == arm) & (pc.class_id.isin(ids))].groupby("class_id")["ap50_95"].mean()
        d = (armv - base).sort_values(ascending=False)
        labels = [names[i] for i in d.index]
        ax.bar(range(len(d)), d.values, color=color, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        deframe(ax)
        pos = int((d > 0).sum())
        ax.text(0.98, 0.92, f"{pos}/{len(d)} classes improved",
                transform=ax.transAxes, ha="right", fontsize=7, color=INK)
    axes[0].set_ylabel("Per-class Δ mAP50–95")
    fig.subplots_adjust(wspace=0.06, bottom=0.3)
    save(fig, "fig6_perclass_delta")


# ---------------------------------------------------------------- Fig A1
def figa1_gallery():
    """부록: inpainting 전후 확대 갤러리 (source | generated) 8쌍."""
    pairs = [
        ("outputs_confirmatory/synthetic/review_sheet_uniform.jpg", 0, "MAD"),
        ("outputs_confirmatory/synthetic/review_sheet_selective.jpg", 2, "MAD"),
        ("outputs_confirmatory/synthetic/review_sheet_weakness.jpg", 3, "MAD"),
        ("outputs_confirmatory/synthetic/review_sheet_weakness.jpg", 7, "MAD"),
        ("outputs_mar20_yolo/synthetic/review_sheet_selective.jpg", 0, "MAR20"),
        ("outputs_mar20_yolo/synthetic/review_sheet_uniform.jpg", 4, "MAR20"),
        ("outputs_mar20_yolo/synthetic/review_sheet_weakness_uniform.jpg", 0, "MAR20"),
        ("outputs_mar20_yolo/synthetic/review_sheet_weakness.jpg", 2, "MAR20"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(7.0, 7.2))
    for k, (sheet, row, tag) in enumerate(pairs):
        src, _, gen = crop_row(ROOT / sheet, row)
        r, c0 = divmod(k, 2)
        for j, (im, sub) in enumerate(((src, "source"), (gen, "generated"))):
            ax = axes[r, c0 * 2 + j]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(sub, fontsize=7.5)
            if j == 0 and c0 == 0:
                ax.set_ylabel(tag, fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.08)
    save(fig, "figa1_gallery")


if __name__ == "__main__":
    # These are defined below the first __main__ block; a second guard here
    # keeps a single `python make_figs_paper.py` run regenerating every figure.
    fig6_perclass_delta()
    figa1_gallery()
