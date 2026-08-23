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

**Method 정밀 검토 라운드 (2026-08-19)**: 외부 편집 세션의 §3 전면 개고
(형식화 강화, RFS를 "generation-free exposure comparator"로 재규정,
t=0.1 이력·복제 수·시드 공식·재시도 정책 등 신규 사실 다수)를 코드·
설정·VM 로그와 전수 대조 검증 — **18개 검증 항목 전부 일치**. 근거:
RFS 8,930(MAD)/289(MAR20)·train 1064/val 267/test 2511은 GCS 보존
VM 로그에서 실측, 시드 공식은 실제 파일명(c29_0008_s2900050)으로 교차
확인, 우선순위 정규화의 도메인 비대칭(tail=S_t 내, weak=C 전체)은
compute_priority_scores 호출부와 일치, min-val≥5는 실제 최소 9/23으로
no-op 확인. 경미 수정 5건 반영: ① s_0 기호 정의(RFS·생성 공유 base
seed, detector.seeds[0] 배선 확인) ② AP 표기 통일(콜론 표기 제거,
\overline{AP}_c 사전 정의) ③ Fig1 캡션 refill 문장 재구성 ④ "is
claimed to be" 어법 수정 ⑤ **구축 시드 수치(42) 본문 비표기 재확정**
(사용자 결정; 부록은 시드 횟수 언급뿐이라 무변경).

**Experimental design 검토 라운드 (2026-08-21)**: 외부 리뷰 7건을 실측
대조 후 전건 채택(+파급 2건). ① "Budgets scale with dataset size" 삭제
(예산비 2× vs 학습셋 크기비 8.9×로 사실 아님) ② B/K≈77 vs ≈83 quota
유사성으로 대체 서술 ③ synthetic-to-real 비율 비일치 명시(10.6% vs
47.0%, "matched dataset-level dose 아님" 한정) ④ failure rate를 pool별
실측 범위로 교정 — MAD 18.2–20.6%(RESULTS.md pass 81.8/80.8/79.4 +
confirmatory weakness_uniform 로그 258/1,258=20.5%), MAR20 0.6–1.2%
(생성 로그 4-pool 실측; 리뷰 제안치 18.0%는 18.2%로 정정) ⑤ 용어
weighted→priority 통일(tab:master 6행, tab:contrasts 4행, §4.4·§5.1×2·
§6.2 산문, tab:cells prose — abstract/intro/method의 기존 priority와
정합) ⑥ tab:cells Arms 열 구체화(full=2×2+RFS+basic_aug+real_only,
reduced=basic_aug+RFS+tail-uniform+tail-priority; 총 66 runs 정합 확인)
+ \small ⑦ 캡션 "share the statistical protocol"→"share the generation
and verification pipeline and the seed-blocked testing framework"
(frozen 5-contrast는 four-arm cell 한정 명시). 파급: §5.4 MAR20 0.6%→
0.6–1.2%. 그림: make_figs_paper.py ARM_LABELS·fig1 패널 라벨 갱신 후
7종 전체 재생성(+fig6/figa1이 __main__ 뒤에 정의돼 실행 누락되던 버그
수정 — 2차 가드 추가), latex/figures 동기화, fig1·fig3·fig4 육안 검수.

**통계 프로토콜 검토 라운드 (2026-08-21)**: 외부 리뷰 6건 전건 채택
(+파급 4건). 핵심은 용어 무결성 — freeze JSON이 로컬 타임스탬프
(2026-08-04T14:15Z)·gitignore된 outputs에만 존재하고 config 최초 공개
커밋이 실험 종료 후(8/19)라 제3자 검증 불가 → **"pre-registered" 11곳
전부 "pre-specified"로 교체**(§4.4 제목 포함; 코드 docstring도 원래
pre-specified). §4.4 재작성: ① freeze 해시 범위 명시(payload 실측 =
baseline/arms/confidence/contrasts뿐) + "no revision" 진술을 record-
keeping으로 분리 ② reduced cell 3-contrast family 추가(4개 cell 모두
freeze 산출물 실측 확인: rtdetr 3개·mar20_yolo 5개, 훈련 전 타임스탬프)
③ "before any test-split metric" → "confirmatory-arm" 한정(MAD 탐색기
baseline test 지표 존재) ④ df=2·정규성 평가불가·추정치/CI 우선 명시
(시드 값은 비표기 방침대로 제외) ⑤ Wilcoxon "distribution-free" 철회 →
"unadjusted, exploratory sensitivity analyses"(클래스 종속성 명시)
⑥ tail=train count/weak=val AP 신호 구분 ⑦ 외부 레지스트리 미사용
공개 문장. 파급: abstract "significant gains"→"consistently positive
gains", §5.3 MAR20 재서술(전 arm·전 scope 양수는 tab:master로 지지),
§5.4 worst-FID 문장 +0.034 인용으로 대체, class-level p 인용 4곳
"unadjusted" 라벨, tab:cells 캡션에 5/3-contrast 병기, sec:repro 라벨
신설. 검증: 참조 정합·중괄호 균형·잔여 표현 0.

**§5.1 검토 라운드 (2026-08-21)**: 외부 리뷰 11건 전건 채택 + baseline
재사용 공개(사용자 승인). 수치 3건 독립 재현: ① 빈도-AP 상관을 test-
split 재계산값으로 교정 — Pearson −0.373(p=0.014)/Spearman −0.407
(p=0.007); §5.1·§6.1·fig2에 반영하고 **fig2는 하드코딩 제거, 스크립트
내 scipy 직접 계산으로 전환** ② 다중 객체 비율 실측(MAD 20.7–43.8%,
MAR20 92.0–93.2%) → spillover 문단을 기전 주장 없이 관찰 보고로 전환
("배경 매개 전이 vs 동반 비대상 객체 노출 구분 불가"), §6.2에도 대안
경로 병기 ③ MAR20 tail-priority 6/6 클래스 개선 실측 → Wilcoxon
p=0.031 강조를 기술 서술로 대체. 그 외: "determines"→"shapes"(절 제목·
fig3 캡션·abstract·§6.1; **논문 대제목은 사용자 확정 사항이라 유지**),
weak set의 val-AP 선정 명시 문장+fig2 범례 교체, Rafale→"lowest-AP
class under the YOLOv8n baseline", interaction의 baseline 소거 우위
명시, MAR20 replication을 "cross-dataset, without multiplicity-adjusted
significance"로 한정(+CI 포함 서술을 point estimate 비교로 교체 —
Limitations·결론 동기화), 검정력 원인을 seed block 3개(df=2)로 재귀속,
weighting 문단 Holm p(0.808/0.345/0.336/0.519) 사용+"operative
variable"→"dominant factor". **공개**: MAD compact cell의 basic_aug·
real_only가 탐색 단계 run 재사용(7월; arm은 8월 재훈련, 평가는 단일
하니스)임을 §4.2에 명시 — run 날짜 실측(baseline 0702–0712, arm
0804–0805) 근거. 그림 fig1–6·figa1 전체 재생성, fig2·fig3 육안 검수.

**§5.2 검토 라운드 (2026-08-21)**: 외부 리뷰 10건 전건 채택. 수치 검증:
① MAR20 pooled-tail p=0.72는 전사 오류 — 산출물(confirmatory_stats.md)
실측 +0.0035, CI [−0.019, 0.026], unadj 0.572, **Holm 0.663**으로 교정
(MAD도 CI [−0.039, 0.022]·Holm 1.00 병기, §5.1과 같이 Holm 보고로 통일)
② RFS MAD scope별 −0.016/−0.010/−0.016 정정("all three scopes −0.016"
은 tail에서 오류) + per-seed 실측(seed 1개 +0.0003 양수) 반영, duplicate-
exposure 기전 제시 철회 ③ CI 상한(+0.022/+0.026/+0.035) 근거로 등가성
미확립 문장 추가. 주장 한정: 제목 "removes all augmentation gains" →
"No measured augmentation arm improves the stronger pretrained
detector"; "all deltas zero to negative" → all-scope 한정 + 구명칭
잔존(uniform/selective) → tail-uniform/tail-priority; headroom을 관찰
수준으로 전환(MAR20 0.697 비포화 명시, "largest single lever" →
"coincided with"); null 범위를 measured arms로 한정(abstract·intro·
§6.1·결론 동기화). **fig4 오표기 수정**: 기준값이 basic_aug인데 축이
"Unaugmented baseline (higher → less headroom)"이던 것을 "Standard-
augmentation baseline"으로, 캡션도 same-cell basic_aug로 교정 + 셀별
후보 수 비대칭(4 vs 2)이던 best-inpaint 마커를 전 셀 공통 tail-uniform
arm으로 교체(차이 ≤0.003, 메시지 불변), "descriptive, not a formal
test" 명시. 전 그림 재생성·latex 동기화·fig4 육안 검수.

**셋업 절 검토 라운드 (2026-08-21)**: 외부 리뷰 8건 중 5건 전면·2건
부분 채택, 사실 오류 2건 반박 후 반영. 채택: ① "identical schedules"/
"same configuration files"/"evaluated once" 삭제 — args.yaml 실측
(baseline seed44 batch 143·workers 2 vs arm seed44 batch 41·workers 8,
7/4 run은 batch:-1 기록으로 스키마 드리프트)에 따라 §4.2를 "명목 설정
공유 vs 실현 조건 상이(오토배치 143 vs 41, 버전 미기록)"로 재작성,
limitations 신규 항목 (4) Training-period and split provenance 추가
(기존 (4) External validity → (5)) ② best.pt 선택 규칙 명문화
(collect_yolo_metrics.py:206 확증) + "up to 50 epochs" ③ Repro 완화:
"66 archived training runs"(서론 포함 2곳), "prepared source datasets
이후 스크립트화", 훈련 라이브러리 버전 미보존 명시 ④ real_only →
"no-training-augmentation reference" ⑤ Cells: RFS 병기, tail-uniform
pool 특정, quota를 (q_min, q_max) 표기 ⑥ tab:cells Arms 열 p{0.42
\linewidth} 전환(2단 조판 대비) ⑦ "fixed in"→"recorded in". 반박·수정:
18.0% 하한은 원본 verification_report.csv 재계산(18.234/19.225/20.635)
으로 기각 — 18.2–20.6% 유지; **MAD 분할은 리뷰어가 옳았음** — 외부
개정된 Datasets 절이 이미 "경로 정렬 후 반분(비층화)"을 문서화(코드
normalize_yolo_dataset.py:116 확증; 내가 본 stratified_partition은 val
부재 시 fallback) → "결정적·비층화 + 전 클래스 각 분할 존재(실측)"
명시, 근접 중복 미검증 limitation 반영. MAR20 분할의 "seed 42" 재등장
제거(→"a fixed split seed") + primary class 정의(최다 box 클래스) 추가.
시드 값 명기 요구는 재차 거부(비표기 방침).

**Conclusion 검토 라운드 (2026-08-21)**: 외부 리뷰 7건 전건 채택 +
파급 2건. 결론이 본문에서 확립한 한정보다 강하게 쓰인 내부 비일관을
해소: ① 완전 2×2는 compact detector 한정·reduced는 전이 탐침으로 분리
서술 + "cell-specific fixed budget" ② "gains land on the targeted set"
→ "larger on, and shifted toward"(off-target 관찰과 정합) ③ weighting
null → "no supported advantage at the tested budgets"(등가성 미주장
유지) ④ "disjoint classes here" → "disjoint on MAD and largely
distinct on MAR20"(1/6·2/6 중첩 반영) ⑤ "worst-case class performance"
→ "improve currently low-performing classes"(측정 대상은 weak-set
macro AP; §6.1 Step 3 동일 수정) ⑥ RFS 대조의 불균형 인과 귀속 철회
("we do not attribute this contrast to imbalance depth alone") +
"locally harmful" → "negative mean delta on its weak scope"; §5.3 제목
"depends on" → "differs with" ⑦ "immediately actionable" → "practical,
testable workflow", "wherever augmentation claims are made" →
"controlled augmentation studies, particularly for long-tailed
detection". 그림 무변경, 잔존 표현 0·중괄호 균형 검증.

**Introduction 검토 라운드 (2026-08-21)**: 외부 리뷰 8건 중 7건 채택,
1건 부분(시드 값 명기 3번째 거부, "three training seeds"만 반영).
① 탐색 유래를 tail/weak 불일치 문단으로 전진 배치("In the exploratory
analysis of the ... MAD benchmark that motivated our allocation
hypothesis ... turned out to be disjoint") + 후속 freeze 문단을 "Because
the allocation hypothesis originated in that exploratory MAD phase,
..."로 재구성(중복 제거) ② "prospective, seed-level confirmation" →
"prospectively specified, seed-level evaluation of newly trained
augmentation arms on the same benchmark" ③ "RFS strongly outperforms"
→ "produces substantially larger mean gains"(RFS-vs-inpaint는 사전
지정 contrast가 아님) ④ "reproducibility package" → "artifact package"
(버전 드리프트 공개와 정합) ⑤ 첫 사용 약어 5종 정의(MAD·AP·IoU·
MAR20·RFS) ⑥ "three independent seeds" → "three training seeds"
⑦ "direction and magnitude" → "direction and approximate magnitude",
"not confirmatory significance" → "does not reach multiplicity-adjusted
significance" — 비표준 용어 "confirmatory significance"는 §4.4(→"the
study's confirmatory conclusions")·limitations(→"a second confirmatory
result")에서도 일괄 제거 ⑧ 연구 공백을 "What has not been isolated in
a controlled object-detection experiment ... independently affect
performance"로 구체화.

**Related Work 검토 라운드 (2026-08-21)**: 외부 리뷰 10건 전건 채택.
신규 인용 3건 전수 검증 후 추가(→refs.bib 52건): ① **TADA**(ICLR 2026,
arXiv:2505.21574) — 원문 PDF로 저자·게재·detection 실험 확인; 구성
참조 문헌의 인용 누락 해소가 이번 라운드 핵심. §2.3에 사례 선택
(학습 속도, image-level) vs 본 연구(class-level, frequency/val-AP,
고정 B·K factorial) 대비 명시 ② FedEAS(arXiv:2607.06616, 실제 제목
"WHERE to Generate Matters...") — abs fetch 검증, federated 예산 배분
1문장 ③ Fokkinga et al.(arXiv:2604.18076) 군용 차량 class-specific
diffusion — abs fetch 검증, §2.4에 UAV 논문과 묶어 배치. 본문 수정:
§2.1 브리지 문장("do not isolate how a fixed generation budget should
be distributed across classes") + "We hold this generation mechanism
fixed..." 채택-기여 분리 문장, §2.2 "primary RFS"→"evaluated",
"exposure baseline"→"comparator"(§3.1과 통일), §2.3 "freezes every
allocation plan"→"pre-specifying and recording"(§4.4 hash 범위와 정합),
§2.5 Positioning "To our knowledge, no prior object-detection study
has factorially compared..." 한정 + "two contrasts of each full-design
cell" 모호성 해소, FAIR1M "fine-grained oriented object detection",
§2.4 MAD 첫 언급 \citep 추가. references_verified.md에 검증 방법 기록.

**종합 감사 라운드 (2026-08-22)**: 외부 전면 감사 ~20항목 반영. 제 판정
정정 2건 포함 — ① failure 하한: MAD RT-DETR pool 원본 verification_
report 실측(selective 18.03/uniform 18.17%)으로 리뷰어의 18.0–20.6%가
옳았음(두 차례 기각 철회) ② 체크포인트 pickle에서 Ultralytics 버전 추출
성공(8.4.87/8.4.92 baseline, 8.4.115 arm) — "버전 미기록"을 "체크포인트
에서 복원 가능, torch/CUDA는 미보존"으로 교정. **제목 변경**(사용자
승인): "...class-set selection shapes where performance gains occur
under a fixed synthetic budget" — null을 등가성처럼 싣던 "not the
per-class weighting, determines" 철회. **초록 247단어로 재작성**(<250;
four-arm compact 한정·no supported advantage·larger mean gains·MAR20
유의성 부재·provisional heuristic·"at low cost" 제거). 해석 강등: RFS
diagnostic→practical probe(§5.3·§6.1 Step2), weighting 과잉표현 정리
("rule out"→CI 명시형, "dominant factor"→"factor with detectable
effects", Step3 제목 "uniform quotas are a reasonable default"), §6.2
"exactly"→"nearly unchanged (up to JPEG re-encoding)"·"direct support"
→"consistent with", §5.4 "cannot account"→"unlikely to be the sole
explanation"·gate 결론 한정(3 arms·단일 seed 명시). 사실 교정: YF-23
"one of the rarest"(E7 86<97 실측) + \mapm 통일(0.80/0.27/0.33 실측
재기재), test-split 문구 축소+탐색 유래 재연결, "every arm"→"each
augmentation arm", "frozen arms"→"pre-specified", "identified"→
"candidate moderators", §5.4 pool명 weak-priority로. Limitations에
single-pool realization 추가. **제출 준비**: 선언 5종 삽입(경쟁이해·
funding[확인 필요]·CRediT·data availability[저장소·라이선스 확인 필요]·
생성형 AI 사용 선언[도구 표기 확인 필요]) + 익명화 주석, highlights.tex
전면 갱신(5불릿 ≤85자, 구주장 제거). 생성기: make_appendix(quota 캡션
좌/우 설명·prio. 헤더·gate/재학습 표 현행 명칭·confusion 캡션
hypothesized+\ref{sec:mechanism})·make_figs(fig4 y축 표기, figa1 4+4
재구성 — MAR20 pair가 MAD 행에 있던 오라벨 해소, 동일 소스 중복 회피
uniform r4) 수정 후 부록·그림 전체 재생성. 검증: 잔존 표현 0, 미해결
\ref 0, 중괄호 균형. 시드 값 명기는 4번째 거부.

**2차 종합 감사 라운드 (2026-08-23)**: 외부 감사 6건 전건 반영.
① **audit 정직화**: RESULTS.md 원기록 대조로 "human-verified" 무근거
확인(감사=baseline detector의 off-mask 검출 증가 측정, arm당 150장) →
부록 "detector-based paired audit"로 교체, 본문 수치 라벨 "detector-
flagged"/"flagged extra detections"화, gate의 수량 혼입(82/85/165장
차등 제거→budget 축소 미분리) 명시 ② **RT-DETR contrast 완전 보고**:
tab:contrasts에 두 reduced cell의 3-contrast 6행 추가(confirmatory_
stats.md 실측값), §4.4에 pooled/interaction 수식 명기, §5.2에 표 포인터
③ **diffusion revision 복구·이원 기술**: 외부 세션이 appendix에 직접
넣은 revision 문장이 재생성으로 소실됐던 것을 생성기에 이관 —
8/4 VM 로그에서 확인 단계의 실사용 snapshot 8a4288a… 복구(로딩 경로
실증), 탐색 단계 3개 pool은 "동일 model ID·revision 미기록"으로 정직
기술, config pin은 사후 추가임을 반영한 문안 ④ **증거 언어 통일**:
"improves"→"shows a supported gain"(초록·§5.2 제목·§6.2·highlights),
fig4 캡션 "Mean deltas cluster near zero", 초록 249단어 유지
⑤ **인과 잔재 제거**: §5.3 "dominated"→"produced larger mean deltas",
불균형 해석에 "one possible interpretation…rather than an established
mechanism", 소제목 "between the two dataset regimes"; §6.2 "one
plausible account"/"may mitigate"/"potentially encouraging"/saturation
"consistent with a reading"/re-exposure 논증에 노출량 불일치 명시
⑥ **FID·CLIP 재현 정보**: 부록 App C를 "Generation quality, …"로 확장
— 방법(torchmetrics FID·Inception-V3, 참조=pool별 클래스-매칭 실이미지
1,797/1,797/3,744로 상이함 명시, CLIPScore=clip-vit-base-patch16,
LPIPS=AlexNet, 200장/pool)+실측값 표(FID 89.7/87.7/100.4) 신설, 4번째
pool 미측정 사유(품질 분석이 확인 단계 이전) 기재, §5.4 헤더
"unlikely to be the sole explanation". 검증: 구표현 0·미해결 ref 0·
중괄호 균형·초록 249단어.
