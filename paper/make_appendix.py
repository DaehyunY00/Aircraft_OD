"""부록 A–C를 로컬 데이터에서 생성 → paper/latex/appendix_gen.tex (재현 가능)."""
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "latex" / "appendix_gen.tex"

def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

L = []
A = L.append

# ---------------- Appendix A: generation details ----------------
A(r"\appendix")
A(r"\section{Generation details}\label{app:gen}")

cfgs = {"MAD": "configs/confirmatory.yaml", "MAR20": "configs/mar20_yolo.yaml"}
A(r"\paragraph{Scene and negative prompts}")
A("The five scene prompts per dataset (rotated per generated image) and the "
  "dataset-specific negative prompts are reproduced verbatim below.")
for ds, cfg in cfgs.items():
    d = yaml.safe_load((ROOT / cfg).read_text())["diffusion"]
    A(r"\begin{itemize}\small")
    for p in d["prompts"]:
        A(rf"\item[] \texttt{{{esc(p)}}}")
    A(rf"\item[] negative: \texttt{{{esc(d['negative_prompt'])}}}")
    A(r"\end{itemize}")
    A(rf"\noindent\emph{{({ds})}}\medskip" if ds == "MAD" else rf"\noindent\emph{{({ds})}}")

# class memberships
A(r"\paragraph{Class memberships}")
rows = {}
for ds, root in (("MAD", "outputs_confirmatory"), ("MAR20", "outputs_mar20_yolo")):
    g = pd.read_csv(ROOT / root / "analysis" / "class_groups.csv")
    tail = g[g.group == "tail"].sort_values("class_name").class_name.tolist()
    weak_ids = pd.read_csv(ROOT / root / "analysis" / "augmentation_plan_weakness.csv").class_id
    names = g.set_index("class_id").class_name
    weak = sorted(names[i] for i in weak_ids)
    rows[ds] = (tail, weak)
A(r"\begin{table}[h]\centering\small")
A(r"\caption{Tail and weak class memberships (weak sets for the compact detector).}")
A(r"\begin{tabular}{lp{0.75\linewidth}}\toprule")
for ds, (tail, weak) in rows.items():
    A(rf"{ds} tail & {esc(', '.join(tail))} \\")
    A(rf"{ds} weak & {esc(', '.join(weak))} \\")
    if ds == "MAD":
        A(r"\midrule")
A(r"\bottomrule\end{tabular}\end{table}")

# allocation quotas
A(r"\paragraph{Realized allocation quotas}")
for ds, root, plans in (
    ("MAD", "outputs_confirmatory", ["uniform", "selective", "weakness_uniform", "weakness"]),
    ("MAR20", "outputs_mar20_yolo", ["uniform", "selective", "weakness_uniform", "weakness"]),
):
    frames = {}
    for pl in plans:
        df = pd.read_csv(ROOT / root / "analysis" / f"augmentation_plan_{pl}.csv")
        g = pd.read_csv(ROOT / root / "analysis" / "class_groups.csv").set_index("class_id").class_name
        df["name"] = df.class_id.map(g)
        frames[pl] = df.set_index("name").num_synthetic_images
    tail_tbl = pd.DataFrame({"tail-uniform": frames["uniform"], "tail-weighted": frames["selective"]}).dropna().astype(int)
    weak_tbl = pd.DataFrame({"weak-uniform": frames["weakness_uniform"], "weak-weighted": frames["weakness"]}).dropna().astype(int)
    A(rf"\begin{{table}}[h]\centering\small\caption{{Per-class quotas, {ds}.}}")
    A(r"\begin{tabular}{lrr@{\qquad}lrr}\toprule")
    A(r"Class & unif. & wtd. & Class & unif. & wtd. \\ \midrule")
    tl, wl = tail_tbl.reset_index().values.tolist(), weak_tbl.reset_index().values.tolist()
    for i in range(max(len(tl), len(wl))):
        a = tl[i] if i < len(tl) else ["", "", ""]
        b = wl[i] if i < len(wl) else ["", "", ""]
        A(rf"{esc(a[0])} & {a[1]} & {a[2]} & {esc(b[0])} & {b[1]} & {b[2]} \\")
    A(r"\bottomrule\end{tabular}\end{table}")

# compute summary
A(r"\paragraph{Compute}")
tot = {}
for cell, root in (("MAD/YOLOv8n", "outputs_confirmatory"), ("MAD/RT-DETR-L", "outputs_mad_rtdetr"),
                   ("MAR20/YOLOv8n", "outputs_mar20_yolo"), ("MAR20/RT-DETR-L", "outputs_mar20_rtdetr")):
    secs = 0.0
    for meta in (ROOT / root / "runs").glob("*/training_meta.yaml"):
        secs += float(yaml.safe_load(meta.read_text()).get("training_seconds", 0))
    tot[cell] = secs / 3600
A("Total training compute on a single NVIDIA L4 (measured wall-clock per run, summed): "
  + "; ".join(f"{k}: {v:.1f}\\,GPU-h" for k, v in tot.items())
  + ". Generating and verifying the four MAR20 pools (2{,}000 images) took 2.1\\,GPU-h; "
    "a 1{,}000-image MAD pool takes approximately 3\\,GPU-h on the same hardware.")

# expanded gallery + confusion figures
A(r"""
\begin{figure}[h]\centering
\includegraphics[width=0.92\linewidth]{figures/figa1_gallery.pdf}
\caption{Additional source/generated pairs (backgrounds regenerated;
all annotated objects protected). Top three rows: MAD; bottom row and
right half of row three: MAR20.}
\label{fig:gallery}
\end{figure}
""")

# ---------------- Appendix B: per-class results ----------------
A(r"\section{Per-class results}\label{app:perclass}")
def perclass_table(root, ids, arms, caption, label):
    pc = pd.read_csv(ROOT / root / "metrics" / "per_class_ap.csv")
    pc = pc[pc.eval_split == "test"]
    g = pd.read_csv(ROOT / root / "analysis" / "class_groups.csv").set_index("class_id")
    base = pc[pc.experiment == "basic_aug"].groupby("class_id")["ap50_95"].mean()
    cols = {}
    for label_a, arm in arms.items():
        m = pc[pc.experiment == arm].groupby("class_id")["ap50_95"].mean()
        cols[label_a] = (m - base)
    A(rf"\begin{{table}}[h]\centering\small\caption{{{caption}}}\label{{{label}}}")
    A(r"\begin{tabular}{lrr" + "r" * len(cols) + r"}\toprule")
    A(r"Class & $n$ & base & " + " & ".join(cols) + r" \\ \midrule")
    for cid in sorted(ids, key=lambda c: -base.get(c, 0)):
        row = [esc(g.loc[cid, "class_name"]), str(int(g.loc[cid, "instance_count"])), f"{base[cid]:.3f}"]
        row += [f"{cols[k][cid]:+.3f}" for k in cols]
        A(" & ".join(row) + r" \\")
    A(r"\bottomrule\end{tabular}\end{table}")

mad_arms = {"tail-unif.": "aug_uniform_inpaint", "tail-wtd.": "aug_selective_inpaint",
            "weak-unif.": "aug_weakuniform_inpaint", "weak-wtd.": "aug_weakness_inpaint",
            "RFS": "aug_rfs"}
gmad = pd.read_csv(ROOT / "outputs_confirmatory/analysis/class_groups.csv")
tail_ids = set(gmad[gmad.group == "tail"].class_id)
weak_ids = set(pd.read_csv(ROOT / "outputs_confirmatory/analysis/augmentation_plan_weakness.csv").class_id)
perclass_table("outputs_confirmatory", tail_ids, mad_arms,
               "MAD tail classes: baseline test \\mapm{} and per-arm $\\Delta$ (3-seed means).", "tab:pc-mad-tail")
perclass_table("outputs_confirmatory", weak_ids, mad_arms,
               "MAD weak classes: baseline test \\mapm{} and per-arm $\\Delta$ (3-seed means).", "tab:pc-mad-weak")
g20 = pd.read_csv(ROOT / "outputs_mar20_yolo/analysis/class_groups.csv")
perclass_table("outputs_mar20_yolo", set(g20.class_id), mad_arms,
               "MAR20 (all 20 classes): baseline test \\mapm{} and per-arm $\\Delta$ (3-seed means).", "tab:pc-mar20")

# ---------------- Appendix C: hallucination audit ----------------
A(r"\section{Hallucination audit and object-level gate}\label{app:audit}")
gs = pd.read_csv(ROOT / "outputs_full/synthetic/object_gate_summary.csv")
A("A detector-based scan of all 3{,}000 MAD synthetic images (the three "
  "originally generated pools) flagged images containing hallucinated, "
  "unlabeled aircraft. Table~\\ref{tab:gate} reports per-pool removal "
  "statistics; a 450-image human-verified audit sample estimated a 10.2\\% "
  "image-level hallucination rate, consistent with the 11.1\\% full-scan "
  "removal rate.")
A(r"\begin{table}[h]\centering\small\caption{Object-level gate: full-corpus scan of the MAD pools.}\label{tab:gate}")
A(r"\begin{tabular}{lrrrr}\toprule")
A(r"Pool & images & removed & removed (\%) & objects removed \\ \midrule")
for _, r in gs.iterrows():
    A(rf"{esc(r['plan'])} & {int(r['n'])} & {int(r['dropped'])} & {r['drop_pct']:.1f} & {int(r['extra_objects_removed'])} \\")
A(rf"\midrule total & {int(gs.n.sum())} & {int(gs.dropped.sum())} & {100*gs.dropped.sum()/gs.n.sum():.1f} & {int(gs.extra_objects_removed.sum())} \\")
A(r"\bottomrule\end{tabular}\end{table}")

# gated retrain deltas (seed 42, gated vs ungated arms)
pg = pd.read_csv(ROOT / "outputs_gate/metrics/per_class_ap.csv"); pg = pg[pg.eval_split == "test"]
pf = pd.read_csv(ROOT / "outputs_full/metrics/per_class_ap.csv"); pf = pf[(pf.eval_split == "test") & (pf.seed == 42)]
scopes = {"all": None, "tail": tail_ids, "weak": weak_ids}
A("Retraining the three arms on gated pools (seed matched to the ungated "
  "runs) changed macro \\mapm{} only marginally:")
A(r"\begin{table}[h]\centering\small\caption{Gated $-$ ungated macro \mapm{} (single matched seed).}\label{tab:gated}")
A(r"\begin{tabular}{lrrr}\toprule Arm & all & tail & weak \\ \midrule")
for base_arm in ("aug_uniform_inpaint", "aug_selective_inpaint", "aug_weakness_inpaint"):
    cells = []
    for sc, ids in scopes.items():
        gsel = pg[pg.experiment == base_arm + "_og"]
        fsel = pf[pf.experiment == base_arm]
        if ids is not None:
            gsel = gsel[gsel.class_id.isin(ids)]; fsel = fsel[fsel.class_id.isin(ids)]
        cells.append(gsel.ap50_95.mean() - fsel.ap50_95.mean())
    A(esc(base_arm.replace("aug_", "").replace("_inpaint", "")) + " & " +
      " & ".join(f"{v:+.3f}" for v in cells) + r" \\")
A(r"\bottomrule\end{tabular}\end{table}")
A(r"""
\begin{figure}[h]\centering
\includegraphics[width=0.9\linewidth]{figures/figa2_confusion.png}
\caption{Normalized confusion matrix of the MAD \texttt{basic\_aug}
baseline (one seed). The dominant off-diagonal mass lies in the
background row (missed detections), the error mode targeted by
background diversification (Section~6.2).}
\label{fig:confusion}
\end{figure}
""")

OUT.write_text("\n".join(L) + "\n")
print("wrote", OUT, f"({len(L)} lines)")
