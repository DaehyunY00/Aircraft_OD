"""작업 초안 → IJASS 투고 원고 변환.

IJASS는 Springer/KSAS 발행이며 Word 원고를 받는다. 인용은 **번호식 순차**
(Springer Basic) — 실제 게재 논문(10.1007/s42405-023-00632-1)의 Crossref
레코드에서 참고문헌 키가 632_CR1, 632_CR2 … 로 순번을 이루는 것을 확인했다.

하는 일:
  1. 본문의 저자명 인용([Sinha and Ohashi, WACV 2023])을 [n] 번호로 치환
  2. 번호를 **본문 첫 등장 순서**로 재배정 (Springer Basic 규칙)
  3. 참고문헌을 그 순서로 재정렬하고 Springer 서식으로 변환
  4. 표지·선언문(declarations)을 붙여 투고용 Markdown 생성
  5. pandoc으로 .docx 출력

인용되지 않은 서지가 있으면 실패한다 — 투고 원고에 미인용 문헌이 남는 것은
심사에서 지적되는 결함이라 조용히 넘기지 않는다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
DRAFT = PAPER / "ieee_access_draft.md"
OUT_MD = PAPER / "ijass_manuscript.md"
OUT_DOCX = PAPER / "ijass_manuscript.docx"

# 본문 인용 문자열 → 서지 식별자. 같은 성씨가 겹치므로(Zhang/Zhao) 문자열 전체로 맞춘다.
CITE_MAP = {
    "Sinha and Ohashi, WACV 2023": "sinha2023",
    "Zhang et al., 2021": "zhang2021",
    "Röhrich et al., 2026": "rohrich2026",
    "Gupta et al., LVIS 2019": "gupta2019",
    "Li et al., CVPR 2020": "li2020",
    "Zhao et al., 2022": "zhao2022",
    "Ghiasi et al., CVPR 2021": "ghiasi2021",
    "Zhao et al., ICML 2023": "zhao2023",
    "Suri et al., 2023": "suri2023",
    "Trabucco et al., ICLR 2024": "trabucco2024",
    "Fokkinga et al., 2026": "fokkinga2026",
    "Yu et al., 2023": "yu2023",
    "Jocher et al., 2023": "jocher2023",
    "Wilcoxon, 1945": "wilcoxon1945",
    "Kim and Choi, IJASS 2019": "kim2019",
    "Lee et al., IJASS 2025": "lee2025",
    "Zhang et al., IJASS 2024": "zhangijass2024",
    "Rombach et al., 2022; runwayml/stable-diffusion-inpainting": "rombach2022",
}

# Springer Basic 서식: Author AB, Author CD (Year) Title. Journal Vol:pages. https://doi.org/...
REFERENCES = {
    "sinha2023": "Sinha S, Ohashi H (2023) Difficulty-Net: learning to predict difficulty for long-tailed recognition. In: Proc IEEE/CVF Winter Conf Appl Comput Vis (WACV), pp 6444–6453",
    "zhang2021": "Zhang J, Zhang Y, Xu X (2021) ObjectAug: object-level data augmentation for semantic image segmentation. In: Proc Int Joint Conf Neural Netw (IJCNN). arXiv:2102.00221",
    "rohrich2026": "Röhrich N, Gleißner J, Ibrahim AHA, Mertes S, Huber T (2026) Preserve the hard, regenerate the rest: uncertainty-guided synthetic training data augmentation with diffusion models. arXiv:2606.31603",
    "gupta2019": "Gupta A, Dollár P, Girshick R (2019) LVIS: a dataset for large vocabulary instance segmentation. In: Proc IEEE/CVF Conf Comput Vis Pattern Recognit (CVPR), pp 5356–5364",
    "li2020": "Li Y, Wang T, Kang B, Tang S, Wang C, Li J, Feng J (2020) Overcoming classifier imbalance for long-tail object detection with balanced group softmax. In: Proc CVPR, pp 10991–11000",
    "zhao2022": "Zhao L, Teng Y, Wang L (2022) Logit normalization for long-tail object detection. arXiv:2203.17020",
    "ghiasi2021": "Ghiasi G, Cui Y, Srinivas A, Qian R, Lin T-Y, Cubuk ED, Le QV, Zoph B (2021) Simple copy-paste is a strong data augmentation method for instance segmentation. In: Proc CVPR, pp 2918–2928",
    "zhao2023": "Zhao H, Sheng D, Bao J, Chen D, Chen D, Wen F, Yuan L, Liu C, Zhou W, Chu Q, Zhang W, Yu N (2023) X-Paste: revisiting scalable copy-paste for instance segmentation using CLIP and StableDiffusion. In: Proc Int Conf Mach Learn (ICML), PMLR 202",
    "suri2023": "Suri S, Xiao F, Sinha A, Culatana SC, Krishnamoorthi R, Zhu C, Shrivastava A (2023) Gen2Det: generate to detect. arXiv:2312.04566",
    "trabucco2024": "Trabucco B, Doherty K, Gurinas M, Salakhutdinov R (2024) Effective data augmentation with diffusion models. In: Proc Int Conf Learn Represent (ICLR)",
    "rombach2022": "Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B (2022) High-resolution image synthesis with latent diffusion models. In: Proc CVPR, pp 10684–10695",
    "fokkinga2026": "Fokkinga EP, van Woerden JE, Eker TA, Snel SP, Hofmeijer EIS, Schutte K, Heslinga FG (2026) Class-specific diffusion models improve military object detection in a low-data domain. arXiv:2604.18076",
    "yu2023": "Yu W, Cheng G, Wang M, Yao Y, Xie X, Yao X, Han J (2023) MAR20: a benchmark for military aircraft recognition in remote sensing images. Nat Remote Sens Bull 27(12):2688–2696. https://doi.org/10.11834/jrs.20222139",
    "jocher2023": "Jocher G, Chaurasia A, Qiu J (2023) Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics",
    "wilcoxon1945": "Wilcoxon F (1945) Individual comparisons by ranking methods. Biom Bull 1(6):80–83",
    "kim2019": "Kim S-H, Choi H-L (2019) Convolutional neural network-based multi-target detection and recognition method for unmanned airborne surveillance systems. Int J Aeronaut Space Sci 20(4):1038–1046. https://doi.org/10.1007/s42405-019-00182-5",
    "lee2025": "Lee H, Cho S, Shin H, Kim S, Shim DH (2025) Small airborne object recognition with image processing for feature extraction. Int J Aeronaut Space Sci 26(1):220–234. https://doi.org/10.1007/s42405-024-00765-x",
    "zhangijass2024": "Zhang H, Zhang Y, Feng Q, Zhang K (2024) Review of machine-learning approaches for object and component detection in space electro-optical satellites. Int J Aeronaut Space Sci 25(1):277–292. https://doi.org/10.1007/s42405-023-00653-w",
    "bae2024": "Bae S, Shin H, Kim H, Park M, Choi M-Y, Oh H (2024) Deep learning-based human detection using RGB and IR images from drones. Int J Aeronaut Space Sci 25(1):164–175. https://doi.org/10.1007/s42405-023-00632-1",
}

TITLE = ("Where Should a Fixed Synthetic-Image Budget Go? "
         "Class-Allocation Signals for Diffusion-Based Background Augmentation "
         "in Military Aircraft Detection")

FRONT = f"""# {TITLE}

**Daehyun Yoo**<sup>1</sup>

<sup>1</sup> [TODO: affiliation, address]

Corresponding author: [TODO: e-mail]

"""

DECLARATIONS = """
## Declarations

**Funding.** [TODO: state funding sources, or "The authors received no specific funding for this work."]

**Competing interests.** The authors declare no competing interests.

**Ethics approval.** Not applicable. This study used a publicly available image dataset and involved no human or animal subjects.

**Data availability.** The dataset analysed is publicly available on Kaggle (rookieengg/military-aircraft-detection-dataset-yolo-format). Source code, augmentation plans, generation logs, and per-class metrics supporting the findings are available at https://github.com/DaehyunY00/Aircraft_OD.

**Author contributions.** [TODO]
"""


def build() -> None:
    text = DRAFT.read_text(encoding="utf-8")
    body, refs_section = text.split("## References", 1)

    # 작업용 머리말(한국어 메모)과 그림 목록 표는 투고 원고에서 뺀다.
    body = re.sub(r"^> .*$", "", body, flags=re.M)
    body = body.split("## Figures")[0]
    # 초안의 제목·저자 줄은 투고용 표지(FRONT)가 대체한다. Abstract 앞을 잘라낸다.
    body = "## Abstract" + body.split("## Abstract", 1)[1]

    # 'Bae et al. [2024]' 형태는 별도 처리 후 일반 패턴과 합류시킨다.
    body = body.replace("Bae et al. [2024]", "Bae et al. [Bae et al., 2024]")
    CITE_MAP["Bae et al., 2024"] = "bae2024"

    order: list[str] = []

    def renumber(match: re.Match) -> str:
        inner = match.group(1)
        key = CITE_MAP.get(inner)
        if key is None:
            return match.group(0)  # 인용이 아닌 대괄호는 그대로 둔다
        if key not in order:
            order.append(key)
        return f"[{order.index(key) + 1}]"

    body = re.sub(r"\[([^\]\[]+)\]", renumber, body)

    # Springer 관례에 맞춘 표기 정리.
    body = body.replace("**Index Terms** —", "**Keywords**")
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
    # 절과 하위 절을 문서 순서대로 한 번에 처리한다. 두 번 나눠 치환하면 하위 절이
    # 마지막 절 번호를 받는다(실제로 전부 8.x 가 됐다).
    lines, current = [], 0
    for line in body.split("\n"):
        m = re.match(r"^## ([IVX]+)\. (.+)$", line)
        if m:
            current = roman[m.group(1)]
            lines.append(f"## {current} {m.group(2)}")
            continue
        m = re.match(r"^### ([A-Z])\. (.+)$", line)
        if m:
            lines.append(f"### {current}.{ord(m.group(1)) - 64} {m.group(2)}")
            continue
        lines.append(line)
    body = "\n".join(lines)
    # 본문의 절 상호참조도 같이 바꾼다 (§VI-B → Sect. 6.2).
    body = re.sub(
        r"§([IVX]+)(?:-([A-Z]))?",
        lambda m: f"Sect. {roman[m.group(1)]}" + (f".{ord(m.group(2)) - 64}" if m.group(2) else ""),
        body,
    )

    uncited = [k for k in REFERENCES if k not in order]
    if uncited:
        sys.exit(
            "[ERROR] 본문에 인용되지 않은 서지가 있습니다 (투고 전 인용하거나 제거할 것): "
            + ", ".join(uncited)
        )

    ref_lines = [f"{i + 1}. {REFERENCES[k]}" for i, k in enumerate(order)]
    manuscript = FRONT + body.rstrip() + "\n" + DECLARATIONS + "\n## References\n\n" + "\n".join(ref_lines) + "\n"
    OUT_MD.write_text(manuscript, encoding="utf-8")
    print(f"[INFO] 원고 저장: {OUT_MD}  (인용 {len(order)}건, 서지 {len(ref_lines)}건)")

    try:
        subprocess.run(
            ["pandoc", str(OUT_MD), "-o", str(OUT_DOCX), "--standalone", f"--metadata=title:{TITLE}"],
            check=True,
        )
        print(f"[INFO] Word 저장: {OUT_DOCX}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[WARN] pandoc 변환 실패({exc}). Markdown은 생성됨.")


if __name__ == "__main__":
    build()
