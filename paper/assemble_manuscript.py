"""EAAI 제출용 원고 조립: sections/*.md → manuscript_eaai.md (+ pandoc docx).

- HTML 주석(<!-- -->)과 초안 메모 제거
- 순서: 제목/저자 → Highlights → Abstract → Keywords → §1–§7 → 그림 캡션 →
  References(references_verified.md)
- figure 참조는 figures_v2/ PNG를 캡션 위치에 삽입 (docx 미리보기용;
  최종 제출 시 저널 규격에 맞춰 분리 업로드)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

P = Path(__file__).resolve().parent
S = P / "sections"

TITLE = (
    "Allocation-aware inpainting augmentation for object detection: "
    "the class set, not the per-class weighting, determines where "
    "performance gains occur"
)
AUTHOR_BLOCK = (
    "Daehyun Yoo^a^\n\n"
    "^a^ Affiliation — TO BE COMPLETED before submission\n\n"
    "Corresponding author: dhyoo970111@gmail.com\n"
)


def strip_comments(text: str) -> str:
    text = re.sub(r"<!--.*?-->\n?", "", text, flags=re.S)
    return text.strip() + "\n"


def load(name: str) -> str:
    return strip_comments((S / name).read_text(encoding="utf-8"))


def split_front(text: str) -> dict[str, str]:
    """00 파일에서 Abstract/Highlights/Keywords/Conclusion 분리."""
    parts: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## Abstract"):
            current = "abstract"; buf = []; continue
        if line.startswith("## Highlights"):
            parts[current] = "\n".join(buf).strip(); current = "highlights"; buf = []; continue
        if line.startswith("## Keywords"):
            parts[current] = "\n".join(buf).strip(); current = "keywords"; buf = []; continue
        if line.startswith("# 7. Conclusion"):
            parts[current] = "\n".join(buf).strip(); current = "conclusion"; buf = []; continue
        if current:
            buf.append(line)
    parts[current] = "\n".join(buf).strip()
    parts["abstract"] = re.sub(r"\n?\*\(~\d+ words\)\*", "", parts["abstract"]).strip()
    return parts


def figure_block(num: int, stem: str, captions: dict[int, str]) -> str:
    return f"![]({(P / 'figures_v2' / (stem + '.png')).as_posix()})\n\n{captions[num]}\n"


def main() -> None:
    front = split_front(load("00_abstract_07_conclusion.md"))
    cap_text = load("FIG_CAPTIONS.md")
    captions = {
        int(m.group(1)): m.group(0).strip()
        for m in re.finditer(r"\*\*Figure (\d)\.\*\*.*?(?=\n\*\*Figure |\Z)", cap_text, flags=re.S)
    }
    figure_stems = {
        1: "fig1_pipeline_design", 2: "fig2_freq_vs_ap", 3: "fig3_dissociation",
        4: "fig4_condition_map", 5: "fig5_generation_examples",
    }

    body = []
    body.append(f"# {TITLE}\n\n{AUTHOR_BLOCK}")
    body.append("## Highlights\n\n" + front["highlights"])
    body.append("## Abstract\n\n" + front["abstract"])
    body.append("**Keywords:** " + " ".join(front["keywords"].split()))
    body.append(load("01_introduction.md"))
    body.append(load("02_related_work.md"))
    body.append(load("03_method_04_setup.md"))
    # Figure 1, 2 는 §3–4 뒤에 배치
    body.append(figure_block(1, figure_stems[1], captions))
    body.append(figure_block(2, figure_stems[2], captions))
    results = load("05_results.md")
    body.append(results)
    body.append(figure_block(3, figure_stems[3], captions))
    body.append(figure_block(4, figure_stems[4], captions))
    body.append(figure_block(5, figure_stems[5], captions))
    body.append(load("06_discussion.md"))
    body.append("# 7. Conclusion\n\n" + front["conclusion"])
    refs = strip_comments((P / "references_verified.md").read_text(encoding="utf-8"))
    refs = refs.replace("# References (verified 2026-08-18)", "# References")
    body.append(refs)

    out_md = P / "manuscript_eaai.md"
    out_md.write_text("\n\n".join(body), encoding="utf-8")
    print("wrote", out_md)

    out_docx = P / "manuscript_eaai.docx"
    subprocess.run(
        ["pandoc", str(out_md), "-o", str(out_docx), "--from", "markdown+smart",
         "--resource-path", str(P)],
        check=True,
    )
    print("wrote", out_docx)


if __name__ == "__main__":
    main()
