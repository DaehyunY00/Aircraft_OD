# EAAI LaTeX 제출 패키지

- `main.tex` — 원고 본문 (elsarticle, authoryear). 컴파일: pdflatex → bibtex → pdflatex ×2, 또는 폴더째 Overleaf 업로드(elsarticle.cls는 TeX Live/Overleaf 기본 포함).
- `refs.bib` — 검증된 참고문헌 28건 (검증 원칙은 파일 상단 주석).
- `elsarticle-harv.bst` — author-year 서지 스타일 (제출 zip에 동봉).
- `figures/` — Fig 1–5 (PDF, `paper/make_figs_paper.py`로 재생성 가능).
- `highlights.tex` — 별도 제출용 Highlights.
- `elsarticle-src/` — 사용자가 제공한 elsarticle 배포본 원본 (cls 생성: `latex elsarticle.ins`).

제출 전 TODO: main.tex의 저자 소속(`Affiliation to be completed`) 확정.
