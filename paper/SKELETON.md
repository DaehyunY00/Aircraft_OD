# EAAI 원고 골격 (0단계 — 계약 문서)

작성 2026-08-17. 이 문서가 확정되면 이후 챕터는 여기 정의된 주장·강도·구조를
벗어나지 않는다. 변경이 필요하면 이 문서를 먼저 고치고 사유를 기록한다.

- 타겟: **Engineering Applications of Artificial Intelligence** (Elsevier, 구독 트랙)
- 유형: Research paper (full length)
- 언어: 영어. 분량 목표 본문 ~9,000 단어, figure 5개 + table 3개 (+appendix)
- 근거 정본: RESULTS.md(C1 배경) / RESULTS_CONFIRMATORY.md / RESULTS_TIER2.md

## 1. 제목 (확정 2026-08-17)

**Allocation-aware inpainting augmentation for object detection: the class
set, not the per-class weighting, determines where performance gains occur**

(선언형 — 기법 + 핵심 발견을 제목에 직접 진술. 조건 지도(발견 2·3)는
abstract 첫 두 문장이 받는다. 질문형 A/B/C안은 2026-08-17 사용자 결정으로 폐기.)

## 2. 한 단락 논지 (elevator pitch)

Diffusion inpainting으로 학습 이미지를 합성해 주는 증강은 "몇 장을 만드느냐"
보다 "**누구에게 만들어 주느냐**(allocation signal)"가 성능 이득의 위치를
결정한다. 우리는 예산·클래스 수·생성기·QC를 고정한 2×2 설계(클래스 집합
{빈도 tail, 측정 weak} × 집합 내 배분 {uniform, weighted})를 두 데이터셋
(자연 시점 MAD-43, 위성 시점 MAR20-20)과 두 검출기(YOLOv8n, RT-DETR-L)에서
3-seed로 실행했다. 결과: ① 이득은 겨냥한 집합에서만 발생(사전 등록된
interaction, MAD Holm p=0.017; MAR20에서 CI 내 재현)하고 집합 내 재가중은
무효 ② COCO 사전학습 대형 검출기에서는 모든 증강(재표집 포함)이 무효 —
이득은 검출기의 headroom에 조건부 ③ 무비용 재표집(RFS)의 우세는 깊은
불균형 데이터셋에 특이적이며, 얕은 불균형에서는 inpainting이 유일하게
유효하다. 이를 실무 결정 규칙으로 종합한다.

## 3. 기여 문장 (Intro 말미 bullet, 4개 — 2026-08-17 순서 개정: 발견이 1번,
   설계는 발견을 성립시키는 수단으로 종속)

1. **배분 신호 발견 (주 기여)**: inpainting 증강의 이득 위치는 클래스 집합
   선택이 결정하며(interaction +0.052~+0.060), 집합 내 가중 배분은 어느
   조건에서도 무효. 빈도는 난이도의 나쁜 대리지표(r=−0.38)이므로 약점 보강
   목적이면 측정 AP로 집합을 골라야 한다. 이 발견은 예산·클래스 수·생성기·
   QC를 고정해 배분 신호만 변수로 남긴 통제 설계에서 성립한다.
2. **평가 프로토콜**: 사전 등록 contrast·seed-blocked 검정·Holm 보정·해리
   기반 2×2 설계 — 증강 효과 주장에 흔한 교란(예산 차이, 단일 seed, 사후
   가설)을 차단하는 재사용 가능한 틀.
3. **조건 지도**: 검출기 headroom(사전학습 대형 모델에서 이득 소멸)과
   데이터셋 불균형 깊이(RFS 유효/무효의 분기)가 증강 이득의 발생 조건을
   결정함을 2 데이터셋 × 2 검출기에서 보임.
4. **실무 결정 규칙과 공개 파이프라인**: model-upgrade-first → RFS-first →
   targeted-synthesis 3단 규칙, bbox 보호 inpainting + 자동 QC + 환각 감사
   도구 일체.

## 4. 주장 목록과 증거 강도 (검토 요청 ② — 이 표가 결과절의 상한선)

| # | 주장 | 강도 | 근거 |
|---|---|---|---|
| CL1 | interaction(set×scope) 양수 | **확정** (사전등록, Holm p=0.017) | C1 |
| CL2 | CL1이 위성 도메인에서 재현 | **재현** (추정치 C1 CI 내; Holm 0.091 경계 — "유의"라 쓰지 않음) | C3 |
| CL3 | 집합 내 재가중 무효 | **null 확정** (4 비교 전부 비유의) | C1, C3 |
| CL4 | 빈도-난이도 역상관 | **확정** (r=−0.375 p=0.013; 103cls 일반화 검정 포함) | C1, Stage0 |
| CL5 | 강한 검출기에서 증강 이득 소멸 | **관찰** (3-seed 기술통계, Δ≈0±; 검정은 보조) | C2, C4 |
| CL6 | RFS 우세는 깊은 불균형 특이적 | **관찰** (C1 +0.086 vs C3 +0.003; C3 weak −0.011 p=0.031) | C1, C3 |
| CL7 | spillover(비표적 이득 +0.01~0.03) 존재 | **관찰** | C1, C3 |
| CL8 | 환각 라벨 노이즈는 RFS 격차를 설명 못함 | **반증 실험** (gate 후 격차 유지 p<1e-4) | C1 부속 |
| 금지 | "MAR20에서 유의", "RFS는 항상 유해/무익", "모든 대형 모델에서 소멸" 등 과잉 일반화 | — | — |

## 5. Figure/Table 계획 (검토 요청 ③)

| ID | 내용 | 상태 |
|---|---|---|
| Fig 1 | 파이프라인 + 2×2 설계 모식도 (bbox 보호 inpainting, QC, 셀 구조) | 기존 fig1 확장 |
| Fig 2 | 빈도 vs AP 산점도 (r=−0.38, tail/weak 집합 색칠) — 동기 | 기존 fig2 재사용 |
| Fig 3 | 이중 해리 2패널: C1(MAD)·C3(MAR20) 나란히, arm별 표적/비표적 이득 | 기존 fig3 확장 |
| **Fig 4** | **조건 지도 (헤드라인)**: 4셀 매트릭스 — x=baseline 수준, y=최선 증강 Δ, 셀별 RFS/inpaint 마커 분리 | 신규 |
| Fig 5 | 생성 예시: MAD 3종 + MAR20 위성뷰 2종 (원본/마스크/생성) + QC 통과율 | 기존 fig5 확장 |
| Tab 1 | 셀·데이터셋·검출기·예산 설정 요약 | 신규 |
| Tab 2 | 마스터 결과: 4셀 × arm × scope Δ (3-seed mean±SD) | RESULTS_TIER2 §1-4 |
| Tab 3 | 사전 등록 contrast 결과 (C1·C3, Holm) | confirmatory_contrasts |
| App | per-class AP, 환각 감사·gate 결과, MAR20 프로토콜 상세 | 기존 자료 |

## 6. 챕터 구조와 작성 순서

| 순서 | 챕터 | 핵심 내용 | 분량 |
|---|---|---|---|
| 1 | §5 Results — **FROZEN 2026-08-17 v3(정밀 검토)** | R1 배분 신호(CL1-4, Fig3, Tab2-3) → R2 headroom(CL5, Fig4) → R3 RFS 조건성(CL6) → R4 spillover·robustness(CL7-8) | ~2,500 |
| 2 | §3 Method + §4 Setup — **FROZEN 2026-08-17 v3(정밀 검토)** | 파이프라인, 2×2, allocator, QC, 사전등록 통계 프로토콜, 셀 구성 (Fig1, Tab1) | ~2,200 |

**중대 사실 정정 (2026-08-17 정밀 검토에서 확정, 전 챕터 구속)**:
- **두 검출기 모두 COCO 사전학습 초기화** (yolov8n.pt = COCO 체크포인트).
  "ImageNet backbone" 서술 금지. 셀 간 차이는 용량+아키텍처이며 사전학습
  코퍼스는 공유 — headroom 교란 서술은 이 두 요인만.
- weighted-tail arm의 점수는 순수 rarity가 아니라 **rarity 0.6 + weakness
  0.4 볼록 결합** (configs alpha=0.6).
- QC 기준 (iii) 편집 가능 배경 ≥5%는 생성물 검사가 아니라 **소스 사전
  검사**. negative prompt는 shared가 아니라 **데이터셋별**.
- weak 집합은 검출기별 정의 — MAR20 중첩: YOLOv8n 1/6, RT-DETR-L 2/6.
  Table 2 캡션에 "scope는 셀별" 명시.
- MAR20 spillover 범위는 +0.004~+0.013 (이전 +0.008~+0.013 오기).
| 3 | §6 Discussion | 3단 결정 규칙, 기제(FN 지배 오류·배경 다양화), 한계 3+1 | ~1,300 |
| 4 | §2 Related Work — **FROZEN 2026-08-17 v1** | 2.1 생성 증강(전경/배경 계보) → 2.2 long-tail 재표집(+marginal-gain 관행 근거) → 2.3 배분·선별 전략 → 2.4 항공기 탐지 도메인. 각 절 말미에 차별화 문장 | ~1,100 |
| 5 | §1 Introduction — **FROZEN 2026-08-17 v1** | P1 공학 문제(예산 질문) → P2 전제 붕괴(r=−0.375, 경쟁하는 두 정책) → P3 설계 → P4 3발견+결정규칙 → P5 기여 4 bullet + 구성 | ~1,000 |
| 6 | Abstract + §7 Conclusion + Highlights·Keywords — **FROZEN 2026-08-17 v1** | 초록 ~255단어, AI 기여/공학 응용 명시 문장, 약어 0개(spelled out), 하이라이트 5개(≤85자) | ~600 |
| 7 | 통합 검토 — **완료 2026-08-17** (발견 6건 수정: §7 참조 누락, MAD 미정의 선사용, AP/RFS 중복 정의 정리, 초록 표현 3건) | 기호·용어 통일(arm/set/scope/headroom), 수치 대조, 흐름 | — |

**Figure 제작 완료 (2026-08-18)**: `paper/figures_v2/` fig1~5 (pdf+png,
300dpi), 생성 스크립트 `paper/make_figs_paper.py` (로컬 outputs_* 4셀
데이터에서 전량 재생성 가능), 캡션 `paper/sections/FIG_CAPTIONS.md`.
팔레트 Okabe–Ito 부분집합(dataviz validator PASS). Fig5 1행은 환각
사례를 의도적으로 포함(§5.4 연계, 캡션에 명시). 구 figures/는 보존.

**서지·조판 완료 (2026-08-18)**: 인용 23건 전수 검증(arXiv 초록 페이지
1차 확인 + 저장소 검증 문헌) — **오귀속 4건 정정**(2403.07113=Crasto 단독,
IRFS=Yaman et al., Unmet Promise=Geng et al., 1904.10781=Bozorgtabar et
al.) + Liang ICCV 2025 연도, FGA-YOLO=Wu et al., EAAI 예시=Song et al.
확정. 산출물: `references_verified.md`(전 항목 출처 표기 원칙 주석),
`assemble_manuscript.py` → `manuscript_eaai.md`(8,107단어) +
`manuscript_eaai.docx`(figure 포함). 본문↔참고문헌 대조 스크립트 통과
(미수록 0). 남은 수작업: 저자·소속 확정(placeholder), cover letter,
제출 전 최종 인적 검토.

**LaTeX 조판 완료 (2026-08-18)**: `paper/latex/` — elsarticle(authoryear)
`main.tex`(전 챕터+표 3종+그림 5종+natbib 인용) + `refs.bib`(28건) +
`highlights.tex` + `elsarticle-harv.bst` + `figures/*.pdf` + README.
검증: 환경 균형, 인용 24키↔bib 전수 일치(2-optional citep 오탐 5건은
수동 확인), figure 파일 존재, 유니코드 정리. 로컬 TeX 부재 →
**Overleaf/TeX Live에서 컴파일** 필요(첫 컴파일 확인 전 미제출).
남은 TODO: 소속 확정, Overleaf 1회 컴파일 검수, cover letter.

**약어 최종 배치 (통합 검토 확정)**: AP = §1 첫 정의(재사용 있음),
RFS = §2.2 첫 정의(§1은 spelled-out만, §3.5 재정의는 조판 시 정리),
MAD/MAR20 이름 첫 사용 = §4.1(그 전 절은 서술형 지칭), 나머지 registry
§8과 동일. **다음 단계**: Figure 4종 제작(make_figs.py 개정) → 서지
BibTeX 확정 → Elsevier 단일칼럼 조판 → cover letter.

각 챕터: 초안 → 사용자 검토 → 수정 → **freeze** → 다음. freeze된 챕터 수정은
이 골격 문서 개정을 통해서만.

## 7. 용어 통일 (전 챕터 공통)

- allocation signal (배분 신호): 클래스 집합을 고르는 기준 (frequency / measured AP)
- class set: tail set (frequency-defined) / weak set (AP-defined)
- arm: 학습 조건 하나 (e.g., tail-uniform arm)
- scope: 평가 클래스 부분집합 (all / tail / weak)
- headroom: baseline 대비 남은 개선 여지
- **rule-based quality verification** (규칙 기반 품질 검증): "automated
  verification/자동 검증" 단독 사용 금지 — 항상 정량 기준과 함께. 예산
  표현은 첫 등장 "a fixed budget of B synthetic images", 이후 "the budget".

## 8. EAAI 규정 준수 체크리스트 (2026-08-17 조사 반영)

- **약어**: 제목·초록 미정의 약어 금지. 본문 첫 등장 정의 완료 현황
  (읽기 순서 기준, 2026-08-17 §1 작성으로 최종 확정) — **AP·RFS는 §1에서
  공식 첫 정의** (§2·§3의 재정의는 조판 시 정리), YOLO·RT-DETR·COCO
  (§4.2), mAP50–95·IoU·CI(§5 도입), FID·CLIP(§5.4), 16-bit FP 풀어씀
  (§3.3). 새 약어 추가 시 여기 등록.
- **Abstract 이원 구조 (필수, desk-reject 사유)**: "AI 기여 = 배분 신호
  중심 통제 증강 프레임과 3대 발견 / 공학 응용 = 자연·위성 시점 군용
  항공기 탐지"를 명시적으로 분리 서술. 분량 ~250단어.
- **공개 데이터셋·재현성**: §4 도입부에 public 명시 + §4.4 코드/산출물
  공개. 유지할 것.
- **형식**: 단일 칼럼, highlights·keywords 별도 준비 (6단계에서 작성).
- **응용 가시화**: §4 도입 문단(군용 항공기 탐지 = 엔지니어링 의사결정
  환경) 추가됨 — Discussion 결정 규칙과 수미상관 유지.
- 검출기 표기: YOLOv8n (3.2M), RT-DETR-L (32M, COCO-pretrained)
- 지표: test mAP50-95 (주), 3-seed mean±SD
- **seed 표기 규칙**: 본문은 "three independent training seeds"까지만
  (개수는 통계의 n이라 필수). 구체 값(42/43/44)·seed별 수치 나열은 본문
  금지 — 재현성 부록/코드 저장소로. "모든 seed에서 같은 방향" 같은 방향
  일관성 서술은 허용.
- **고정 예산의 위상**: 기여가 아니라 통제 수단. "we contribute a fixed-
  budget design" 류 표현 금지 — "under a fixed budget, ..." 종속절로만.

**C1–C4 셀 ID 제거 (2026-08-18, 사용자 지적)**: 내부 실험 관리용 표기가
표 3곳에 잔존하던 것을 서술형 라벨로 교체 — Table 1은 ID 열 삭제,
Table 2 그룹 헤더·첫 열은 "Dataset × detector", Table 3은 Dataset 열
(캡션에 "둘 다 YOLOv8n 셀" 명시). 본문 산문은 원래 서술형이라 무변경.
"cell"이라는 요인설계 용어 자체는 유지. 내부 문서(RESULTS_*.md)의
C-표기는 관리용이므로 그대로 둠.

**구조 재편 v2 (2026-08-18, 사용자 요청 — arXiv:2505.21574 조직 준거)**:
`latex/main.tex`를 학회형 조직으로 전면 재편 (v1은 main_v1_backup.tex 보존).
- §1 Intro: 이탤릭 연구 질문 블록(quote) + bullet 없는 산문형 기여 + 실험
  요약 문단 (roadmap 문장 삭제)
- §2 Related work: 소절 → 굵은 run-in 문단 4개
- §3 Preliminaries 신설: 배분 문제의 형식화(allocation plan (S,q), scope,
  metric) + RFS를 기준선 정의로 이동
- §4 방법을 **TABA(TArgeted Background Augmentation)로 명명** — 초록·서론·
  결론에 일관 반영 (참조 논문 TADA와 평행). 제목·주장·수치는 전부 불변.
- §5 Experiments: 셋업(Datasets/Detectors/Cells/Reproducibility)을 run-in
  문단으로 §5 도입에 통합, 발견은 5.1~5.4 소절 유지
- 검증: 환경 균형·미정의 참조 0·인용 28키 전수 bib 일치.
- **주의**: sections/*.md 아카이브는 v1 구조 기준 — 이후 수정은
  latex/main.tex(v2)가 canonical. Abstract의 EAAI 이원 구조(AI 기여/공학
  응용)는 유지됨.

**TABA 명명 철회 (2026-08-18, 사용자 결정)**: 프레임워크 고유명 없이
"allocation-aware background-augmentation framework"로 서술. §4 제목은
"Allocation-aware background augmentation"(label sec:framework), 초록·
서론·결론·RW 연쇄 수정. v2 구조(RQ 블록, 산문 기여, Preliminaries,
run-in RW/셋업)는 유지. 검증 재통과(환경 균형, 미정의 참조 0).

**서지 확장 (2026-08-18)**: 28→44건 (+16). 전 건 arXiv 초록 1차 검증
(예외: COCO=ECCV/DDPM=NeurIPS canonical venue 병기 2건, Trabucco는
저장소 검증분). 검증 중 ID 오기억 1건 적발(2309.09777=DriveDreamer ≠
DiffusionEngine → 해당 후보 제외). 통합 지점 7곳: §1 근거 인용,
§2.1 diffusion 배경+분류측 계보+InstaGen, §2.2 서베이 2+loss 계열
3(CBL/EqLoss/decoupling), §2.3 feedback-guided(Hemmat), §2.4
DOTA/DIOR/FAIR1M, §5 COCO. 검증: 인용 44키 ↔ bib 44항목 완전 일치
(미수록·미인용 0). 본문 5,737단어.

**본문·그림 보강 (2026-08-18)**: 사용자 요청(전후 사진 추가 + 본문 보강).
- 본문 Fig 6 신설: 클래스별 Δ 정렬 막대 2패널(11/13 개선 시각화) — §5.1 연결
- 부록 신설(\input{appendix_gen}, make_appendix.py가 데이터에서 자동 생성):
  A 생성 상세(프롬프트 전문 10종+negative 2종, tail/weak 명단, 클래스별
  할당표 MAD/MAR20, 실측 컴퓨트) + Fig A1 전후 갤러리 8쌍(중복쌍·라벨
  겹침 수정), B 클래스별 AP 표 3종(MAD tail/weak 13×5arm, MAR20 20×5arm),
  C 환각 감사(전수 스캔 332/3,000=11.1%·845객체 표본치 10.2%와 병기,
  gated−ungated Δ 표) + Fig A2 confusion matrix(실제 산출물)
- 본문 연결 6곳(§4.2 갤러리, §5 Repro→A, §5.1 Fig6, §5.4 전수치+C,
  §6.2 confusion)
- 최종 규모: 산문 6,104단어 / 그림 8(본문6+부록2) / 표 11(본문3+부록8) /
  참고문헌 44건 → 예상 조판 ~19-22쪽 (EAAI 통상 범위 중앙)

**장 구조 재편 v3 (2026-08-18)**: 사용자 요청으로 3·4·5장을
Methodology–Experimental design–Results로 재편(latex/main.tex; v2는
main_v2_backup.tex로 보존). (참고: 재편 직전 상태에는 별도 편집
세션분 — Related Work 소절화·Positioning 신설·§2.2 논지 축소·§1 탐색
기원 공개 — 이 이미 반영되어 있었음.)
- §3 Methodology(sec:method ← 구 sec:framework, 참조 0건 확인 후 개명):
  구 Preliminaries 장을 3.1 Problem setup and notation(라벨 sec:prelim
  유지)으로 강등·흡수, 3.2 할당 신호/quota, 3.3 배경 생성, 3.4 규칙
  기반 검증. 서두 문단 재작성(통계 프로토콜의 §4 이동을 전방 참조).
- §4 Experimental design(sec:design ← 구 sec:exp): Datasets / Detectors
  and training / Cells and budgets(+tab:cells 선언 위치를 해당 소절
  직후로) / Pre-registered statistical protocol(구 프레임워크 §4.4에서
  이동) / Reproducibility.
- §5 Results(sec:results 신설): 리드인 1문단(~55단어) 추가, tab:master·
  tab:contrasts 선언부를 절 서두로 이동. 발견 4소절(sec:r1/r2/r3/
  robust)의 본문은 불변.
- 상호참조 \ref{sec:exp} 3건 → sec:design 전환. 라벨↔참조 전수 일치,
  중괄호 균형(332/332) 검증. 수치·주장 변경 없음.
