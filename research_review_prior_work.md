# 리서치 리뷰: Tail-Class Selective Background Inpainting Augmentation

리뷰 일자: 2026-07-08 | 리뷰 범위: 선행연구 조사 기반 novelty·실험설계·저널 타겟 평가

---

## 1. 관련 선행연구 표

### A. Long-tailed object detection 방법론

| 논문 | 연도/Venue | 핵심 내용 | 이 연구와의 관계 |
|---|---|---|---|
| Gupta et al., "LVIS: A Dataset for Large Vocabulary Instance Segmentation" | CVPR 2019 | 1,203 클래스 long-tail 벤치마크. Repeat Factor Sampling(RFS) 제안. rare/common/frequent(APr/APc/APf) 그룹 보고 관행의 기원 | tail/head 그룹별 AP 보고 형식의 표준. 이 연구도 APr-style 그룹 지표 채택 필요 |
| Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition" | ICLR 2020 | representation과 classifier 학습 분리(cRT, τ-norm). 리밸런싱은 classifier 단계에만 적용하는 것이 효과적 | 데이터 리밸런싱 계열의 대표 baseline. 현재 실험군에 없음 |
| Li et al., "Balanced Group Softmax (BAGS)" | CVPR 2020 | head가 tail classifier weight를 억압하는 문제를 그룹 softmax로 해결 | loss 계열 baseline 후보 |
| Wang et al., "Seesaw Loss for Long-Tailed Instance Segmentation" | CVPR 2021 | positive/negative gradient 비율 동적 조정 | loss 계열 baseline 후보 |
| Li et al., "Equalized Focal Loss for Dense Long-Tailed Object Detection" | CVPR 2022 | one-stage(dense) detector용 long-tail loss — YOLO 계열에 직접 관련 | YOLOv8 실험에 추가 가능한 loss baseline |
| [IRFS: Instance-Aware Repeat Factor Sampling](https://arxiv.org/abs/2305.08069) | 2023 (arXiv/워크숍) | image count와 instance count를 결합한 RFS 개선. LVIS rare AP 상대 +50% | tail_oversampling(C)의 정교화 버전. rarity 정의(이미지 vs 인스턴스 수)에 참고 |
| [E-IRFS: Exponentially Weighted IRFS](https://arxiv.org/pdf/2503.21893) | 2025 | UAV 감시 시나리오(소규모·응용 데이터셋)에 IRFS 확장. YOLO 계열 사용 | 응용 도메인 + YOLO + 리샘플링 조합의 최신 사례. 저널급 논문 구성 참고 |
| [Class Imbalance in Object Detection: An Experimental Diagnosis and Study of Mitigation Strategies](https://arxiv.org/html/2403.07113v1) | 2024 | **YOLOv5에서 mosaic/mixup 증강이 sampling·loss-weighting보다 mAP 개선 효과가 큼**을 실험적으로 입증. COCO-ZIPF 벤치마크 제안 | **"basic_aug가 tail 기법을 압도" 현상의 직접적 선행 보고.** 해결책: 리밸런싱을 강증강 *위에* 얹어 조합 |
| [A Systematic Review on Long-Tailed Learning](https://arxiv.org/pdf/2408.00483) | 2024 | long-tail 학습 전반 서베이 | related work 작성 시 분류 체계 참고 |

### B. 합성 데이터 기반 tail/rare class 증강

| 논문 | 연도/Venue | 핵심 내용 | 이 연구와의 관계 |
|---|---|---|---|
| Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method" ([arXiv](https://arxiv.org/abs/2012.07177)) | CVPR 2021 | 객체를 다른 이미지에 무작위 붙여넣기만으로 LVIS rare mask AP +3.6 | 합성 없이 tail을 올리는 강력한 baseline — 현재 실험군에 없음 |
| Zhao et al., "X-Paste" ([ICML 2023](https://proceedings.mlr.press/v202/zhao23f.html)) | ICML 2023 | Stable Diffusion으로 객체 인스턴스를 생성해 Copy-Paste를 스케일업. rare 카테고리에 특히 효과 | "전경을 생성"하는 접근. 이 연구는 반대로 "배경을 생성" — 차별점이자 비교 대상 |
| Xie et al., "MosaicFusion" ([arXiv](https://arxiv.org/pdf/2309.13042)) | IJCV 2024 | training-free diffusion으로 rare 카테고리 객체+마스크 동시 생성, LVIS long-tail에 적용 | tail-특화 diffusion 생성의 대표 선행연구 |
| Suri et al., "Gen2Det: Generate to Detect" ([arXiv](https://arxiv.org/abs/2312.04566)) | 2023 | grounded inpainting으로 scene-centric 이미지 생성. **image-level/instance-level filtering + 전용 학습 레시피**로 LVIS rare AP +2.13 box | scene 단위 inpainting 생성 + 합성품질 필터링 파이프라인의 표준. QC 설계 참고 필수 |
| Fan et al., "DiverGen" ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_DiverGen_Improving_Instance_Segmentation_by_Learning_Wider_Data_Distribution_with_CVPR_2024_paper.html)) | CVPR 2024 | 생성 데이터의 '다양성'(카테고리·프롬프트·모델)이 핵심임을 규명. X-Paste 대비 rare +1.9 box AP | 프롬프트 다양성 ablation의 근거. 단일 프롬프트 inpainting의 한계 시사 |
| Trabucco et al., "DA-Fusion: Effective Data Augmentation With Diffusion Models" ([arXiv](https://arxiv.org/html/2302.07944v3)) | ICLR 2023 | textual inversion 기반 diffusion 증강, few-shot/rare 개념(FGVC Aircraft 포함)에서 일관된 이득 | diffusion 증강의 분류 태스크 대표작. 항공기 fine-grained 도메인 실험 포함 |
| [Synthetic Data Augmentation using Pre-trained Diffusion Models for Long-tailed Food Image Classification](https://arxiv.org/html/2506.01368v1) | 2025 | long-tail 분류에 사전학습 diffusion 합성 데이터 적용 | tail-선택적 합성 예산 배분과 유사한 문제의식(분류 태스크) |

### C. Diffusion inpainting 기반 detection/segmentation 증강 (가장 근접 영역)

| 논문 | 연도/Venue | 핵심 내용 | 이 연구와의 관계 |
|---|---|---|---|
| Li, Dong et al., "A Simple Background Augmentation Method for Object Detection with Diffusion Model" ([arXiv 2408.00350](https://arxiv.org/abs/2408.00350)) | **ECCV 2024** | **전경(bbox/마스크) 보존 + Stable Diffusion inpainting으로 배경만 재생성.** 배경 증강이 객체 증강보다 robustness/일반화에 효과적임을 보고 | **이 연구의 핵심 기법과 사실상 동일한 메커니즘.** 차이는 (a) tail-선택적 적용, (b) rarity+weakness 예산 배분, (c) military aircraft 도메인. 반드시 인용·비교해야 하며, novelty 주장은 이 논문과의 차별화에 달려 있음 |
| [Data Augmentation for Object Detection via Controllable Diffusion Models](https://ieeexplore.ieee.org/document/10484172/) | WACV 2024 | layout 제어 diffusion + inpaint alignment로 bbox-합성 정렬 문제 해결 | bbox 보호 inpainting의 기술적 선행. QC 논리 참고 |
| Zhu et al., "ODGEN: Domain-specific Object Detection Data Generation" ([arXiv](https://arxiv.org/abs/2405.15199)) | NeurIPS 2024 | 도메인 특화 데이터셋에 diffusion fine-tune 후 bbox 조건 생성. YOLOv5/v7에서 최대 +25.3 mAP | 도메인 특화(소규모) detection 합성의 대표작. "사전학습 SD를 그대로 쓸지 fine-tune할지" 논의에 필요 |
| Tang et al., "AeroGen" ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_AeroGen_Enhancing_Remote_Sensing_Object_Detection_with_Diffusion-Driven_Data_Generation_CVPR_2025_paper.pdf)) | CVPR 2025 | 원격탐사 detection용 layout 조건 diffusion 생성. DIOR/HRSC에서 +2.4~4.3 mAP | 항공 도메인 diffusion 생성의 최신작. related work 필수 |
| [Control Copy-Paste: Controllable Diffusion-Based Augmentation for Remote Sensing Few-Shot Object Detection](https://arxiv.org/pdf/2507.21816) | 2025 | few-shot 신규 객체를 diffusion으로 다양한 컨텍스트에 주입 | "객체 보존 + 컨텍스트 다양화"라는 동일 철학의 최신 사례 |
| [Diff-Mosaic](https://arxiv.org/pdf/2406.00632) | 2024 (IEEE TGRS 계열) | diffusion prior로 현실적인 mosaic 증강 샘플 생성(적외선 소형 표적) | 군사/원격탐사 응용 + diffusion 증강 저널 논문의 사례 |
| [Inpainting is All You Need: Diffusion-based Augmentation for Semi-supervised Medical Image Segmentation](https://arxiv.org/html/2506.23038) | 2025 | inpainting 증강으로 semi-supervised 의료 분할 개선 | inpainting 증강의 타 도메인 확장 사례 |

### D. 합성 예산 배분·선별 전략

| 논문 | 연도/Venue | 핵심 내용 | 이 연구와의 관계 |
|---|---|---|---|
| Zhu et al., "Generative Active Learning for Long-tailed Instance Segmentation (BSGAL)" ([ICML 2024](https://proceedings.mlr.press/v235/zhu24b.html)) | ICML 2024 | gradient cache 기반으로 생성 샘플의 기여도를 온라인 추정, 유용한 합성 데이터만 선별 | **"어떤 합성 데이터에 예산을 쓸 것인가"의 가장 정교한 선행연구.** priority_score는 이보다 단순한 클래스 수준 휴리스틱 — 경량 대안으로 포지셔닝 가능 |
| Liang et al., "Diffusion Curriculum (DisCL)" ([arXiv](https://arxiv.org/html/2410.13674v4)) | 2024/2025 | tail 클래스에 image-guidance 스펙트럼의 합성 데이터를 커리큘럼으로 배분. ImageNet-LT tail 정확도 4.4→23.6% | 클래스 상태(학습 난이도)에 따라 합성 데이터를 적응적으로 배분하는 아이디어의 선행 |
| [T2ID-CAS: Diffusion + Class Aware Sampling for Class Imbalance](https://arxiv.org/html/2504.21231v1) | 2025 | 의료 랜드마크 detection에서 소수 클래스에 diffusion 합성 + class-aware sampling 결합 | rarity 기반 합성 배분의 도메인 응용 사례 |
| Bhattarai et al., "Informative sample generation using class aware GAN" ([arXiv](https://arxiv.org/abs/1904.10781)) | CVIU 2019 | 분류기가 약한 영역에 GAN 샘플을 생성하는 active generation | **weakness_score(baseline AP 기반 배분)의 개념적 선조** |
| [The Unmet Promise of Synthetic Training Images](https://arxiv.org/pdf/2406.05184) | NeurIPS 2024 | 합성 이미지보다 검색된 실이미지가 나은 경우가 많음을 보고. CLIPScore 상위 50% 필터링 관행 논의 | tail_oversampling(실이미지 반복)이 합성을 이길 수 있다는 결과의 이론적 방어에 인용 가능 |

### E. Military aircraft detection 도메인

| 논문 | 연도/Venue | 핵심 내용 | 이 연구와의 관계 |
|---|---|---|---|
| Yu et al., "MAR20: A benchmark for military aircraft recognition in remote sensing images" ([Journal of Remote Sensing](https://www.ygxb.ac.cn/en/article/doi/10.11834/jrs.20222139/)) | 2023 | 최대 규모 군용기 원격탐사 데이터셋(20종, 22,341 인스턴스, HBB+OBB) | **2차 데이터셋 후보 1순위.** 단일 데이터셋 한계 해소용 |
| [Detection of Military Aircraft Using YOLO and Transformer-Based Models in Complex Environments](https://www.researchgate.net/publication/388739097_Detection_of_Military_Aircraft_Using_YOLO_and_Transformer-Based_Object_Detection_Models_in_Complex_Environments) | 2025 | **동일한 Kaggle 43-class 데이터셋**으로 YOLOv7/v8/RT-DETR 비교(YOLOv8 mAP 0.940). 클래스 불균형을 하이퍼파라미터로 대응 | 같은 데이터셋의 유일한 근접 선행. 이 논문과의 차별화(불균형의 체계적 해결) 필요. 단, 보고된 mAP 0.94는 IoU 기준·split 확인 필요 |
| [FGA-YOLO](https://www.sciencedirect.com/science/article/abs/pii/S0925231224018381) | Neurocomputing 2025 | fine-grained 항공기 인식용 one-stage 검출기 (MAR20 실험) | 도메인 SOTA 비교·저널 벤치마크 참고 |
| [Military Aircraft Recognition using Attention Mechanism](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70069) | IET Image Processing 2025 | attention 기반 군용기 인식 | 도메인 저널 논문 수준 참고 |

### F. 보고 지표·통계 관행

| 관행 | 출처 | 내용 |
|---|---|---|
| APr/APc/APf 그룹 보고 | LVIS (CVPR 2019) 이후 표준 | rare(1–10장)/common(11–100)/frequent(>100) 구분. 이 연구의 head/tail 구분도 인스턴스 수 기준 명시적 threshold 필요 |
| 다중 seed 평균±표준편차 | 최근 detection 논문 관행 (예: [Roboflow100-VL](https://arxiv.org/pdf/2505.20612) 등) | 소규모 데이터셋 fine-tuning은 seed 민감도가 커서 3~10 seeds mean±std 보고가 증가 추세. 응용 저널 심사에서도 2 seeds는 부족하다는 지적이 일반적 |
| 합성품질 지표 | Gen2Det, DiverGen, [생성 증강 서베이](https://arxiv.org/pdf/2407.04103) | FID/CLIPScore 보고 + 품질 기반 필터링(예: CLIPScore 상위 50%)이 사실상 표준 구성요소 |

---

## 2. Gap 분석

### 2.1 Novelty 평가

핵심 메커니즘(bbox 보호 + diffusion 배경 inpainting)은 **Li et al. ECCV 2024 (arXiv 2408.00350)에서 이미 일반 detection에 대해 제안·검증**되었다. 따라서 "배경 inpainting 증강" 자체를 기여로 주장할 수 없다. 이 연구가 주장할 수 있는 novelty는 다음 세 가지의 결합으로 좁혀진다.

1. **Tail-선택적 적용**: 배경 증강을 클래스 불균형 해소 도구로 재정위. Li et al.은 uniform 적용만 다뤘으므로 "selective vs uniform" 비교(RQ2)는 유효한 증분 기여.
2. **rarity+weakness 결합 priority score**: 클래스 수준의 경량 예산 배분 휴리스틱. 개념적 선조는 존재하나(BSGAL의 샘플 수준 gradient 선별, DisCL의 난이도 적응 커리큘럼, CAGAN의 약점 기반 생성), "인스턴스 희소성 × baseline 약점"의 명시적 클래스 수준 배분 공식은 문헌에서 그대로 발견되지 않음. 다만 BSGAL 대비 원리적 우월성이 아니라 **계산 효율(재학습 없는 사전 배분)**로 포지셔닝해야 함.
3. **도메인**: fine-grained 군용기 long-tail 문제. 동일 데이터셋 선행(2025 YOLO/RT-DETR 비교 논문)은 불균형을 체계적으로 다루지 않았으므로 도메인 기여 여지 있음.

종합: **top-tier CV venue급 novelty는 아니며, "기존 기법의 선택적 적용 + 배분 전략 + 도메인 검증"의 응용 논문**. Q2, 조건이 잘 갖춰지면 응용 Q1까지 현실적.

### 2.2 누락된 baseline / ablation / 지표

| 구분 | 누락 항목 | 근거 선행연구 |
|---|---|---|
| Baseline | basic_aug + tail 기법 **조합군** (현재 5개 군은 서로 배타적) | 2403.07113: 강증강과 리밸런싱은 경쟁이 아니라 보완 관계 |
| Baseline | Repeat Factor Sampling / IRFS (이미지 수준 표준 리샘플링) | LVIS, IRFS 2023 |
| Baseline | loss 계열 1개 이상 (class-balanced focal, Equalized Focal Loss — one-stage용) | EFL CVPR 2022 |
| Baseline | Copy-Paste (합성 없이 tail 인스턴스를 늘리는 강력한 대조군) | Ghiasi CVPR 2021, X-Paste ICML 2023 |
| Ablation | alpha 스윕 (rarity-only vs weakness-only vs 결합) — **priority score가 기여의 핵심이므로 필수** | BSGAL/DisCL류 선별 전략 논문의 표준 구성 |
| Ablation | 합성 budget 규모 스케일링 (x0.5/x1/x2/x4) | DiverGen: 데이터 규모-성능 곡선 보고 |
| Ablation | 프롬프트 다양성 (단일 vs 다중 배경 프롬프트) | DiverGen: 다양성이 핵심 변수 |
| 지표 | FID/CLIPScore/LPIPS 등 생성품질 정량화 + 품질 필터링 | Gen2Det의 image/instance-level filtering |
| 지표 | LVIS식 head/mid/tail 그룹 AP + 그룹 정의 기준 명시 | LVIS 관행 |
| 통계 | seeds ≥3, mean±std, 대응 검정(paired t-test 또는 Wilcoxon) | 최근 다중 seed 보고 관행 |

### 2.3 실험 규모 적정성 (SCI Q1/Q2 기준)

현재 pilot 1 seed, full 2 seeds, 단일 모델, 단일 데이터셋은 **Q2 저널에도 부족**하다. 최소선: 3 seeds mean±std + 유의성 검정, 모델 2개(YOLOv8n + YOLOv8s/m 또는 RT-DETR 계열 1개), 데이터셋 2개(Kaggle 43-class + MAR20). E-IRFS(2025)류 응용 논문도 복수 데이터셋·복수 모델 구성이 일반적이다. 특히 diffusion 증강 효과는 seed 분산보다 작게 나오는 경우가 많아(tail AP +0.01~0.02 수준이면) 통계 처리 없이는 심사 통과가 어렵다.

### 2.4 "basic_aug 압도" 현상의 선행 보고

[Class Imbalance in Object Detection (2403.07113)](https://arxiv.org/html/2403.07113v1)이 정확히 같은 현상을 보고했다: YOLOv5에서 mosaic/mixup이 sampling·loss-weighting 기법보다 크게 우수하며, two-stage에서 유효하던 리밸런싱 기법이 YOLO 계열에서는 잘 전이되지 않는다. 해결책도 제시되어 있다 — **리밸런싱/합성 기법을 기본 증강을 끈 상태에서 단독 비교하지 말고, 강증강 위에 얹어 추가 이득(marginal gain)을 측정**하는 것. Gen2Det, X-Paste, DiverGen 모두 표준 증강이 켜진 강한 baseline 위에서 합성 데이터의 추가 이득을 보고한다. 현재 설계(A~E 상호배타)는 이 관행과 어긋나며, D/E가 B를 못 이기는 것은 예상된 결과다.

---

## 3. 우선순위별 수정·보완 방향

### P0 — 논문 성립의 전제 (지금 즉시)

1. **Inpainting dry-run 버그 수정 및 생성 검증 자동화.** 원본 대비 bbox 외부 픽셀 변화율, LPIPS(원본 vs 생성), bbox 외부 SSIM을 생성 직후 자동 로깅해 "배경이 실제로 바뀌었음"을 정량 증명. 현재 pilot 결과는 전량 폐기·재실험. (근거: Gen2Det의 생성물 필터링 파이프라인 — 생성 품질 검증 없는 합성 증강 논문은 성립 불가)
2. **실험 설계를 'basic_aug 위 marginal gain' 구조로 재편.** 실험군을 B(basic_aug)를 공통 기반으로 B / B+oversampling / B+uniform_inpaint / B+selective_inpaint / B+copy-paste로 재구성. real_only(A)는 참고용으로만 유지. (근거: 2403.07113, X-Paste·Gen2Det·DiverGen의 보고 관행)

### P1 — 심사 통과의 필요조건

3. **seeds ≥3 + mean±std + paired 유의성 검정** (클래스별 AP를 대응 표본으로 Wilcoxon signed-rank 권장). (근거: 최근 다중 seed 보고 관행, 소규모 fine-tuning의 seed 민감도)
4. **표준 long-tail baseline 추가**: RFS(또는 IRFS), Equalized Focal Loss(또는 class-balanced focal), Copy-Paste. 이 셋이 있어야 "diffusion 배경 생성이 굳이 필요한가"라는 심사위원 질문에 답할 수 있음. (근거: LVIS 계열 논문의 표준 비교군)
5. **생성품질 정량 지표 도입**: 클래스별 FID(합성 vs 실제 배경 분포), CLIPScore(프롬프트 정합), 품질 기반 필터링(예: CLIPScore 하위 50% 제거) ablation. (근거: Gen2Det filtering, 생성 증강 서베이 2407.04103, Unmet Promise 2406.05184)
6. **Li et al. ECCV 2024와의 명시적 비교/차별화.** related work에서 인용하는 수준을 넘어 uniform 배경 증강(=Li et al. 방식의 재현)을 D군으로 두고 selective(E)가 같은 budget에서 우월함을 보이는 것이 논문의 생명선.

### P2 — Q1 도전을 위한 확장

7. **alpha 스윕 ablation** (α=0, 0.25, 0.5, 0.75, 1): rarity-only와 weakness-only 대비 결합의 이득을 보여야 priority score가 기여로 인정됨. budget 규모 스케일링 곡선도 함께. (근거: DiverGen의 규모-성능 분석, BSGAL의 선별 전략 비교)
8. **2차 데이터셋(MAR20) + 2차 모델(YOLOv8s 또는 RT-DETR).** MAR20은 원격탐사 시점이라 배경 inpainting의 일반화를 보이기에 오히려 좋은 대조 도메인. (근거: MAR20 2023, 응용 저널의 복수 데이터셋 관행)
9. **프롬프트 다양성 ablation**: 배경 프롬프트 풀(활주로/해안/도심/구름 등) vs 단일 프롬프트. (근거: DiverGen — 다양성이 생성 증강의 핵심 변수)

### P3 — 완성도

10. 동일 Kaggle 데이터셋 선행(2025 YOLO/RT-DETR 비교 논문)과의 프로토콜 차이(split, IoU threshold) 명시 및 성능 비교 가능성 확보.
11. 20 steps/512 해상도 고정에 대한 근거 또는 간단한 sweep. SD-1.5 inpainting 대신 SDXL-inpainting 사용 여부 검토(생성 품질 여유 확보).
12. Head AP 손상 여부(RQ3)는 클래스별 AP 변화 분포(scatter/violin)로 시각화 — Gen2Det도 rare 개선과 non-rare 유지가 동시에 성립함을 보이는 방식 사용.

---

## 4. 저널 타겟 제안

유사 성격 논문(YOLO + 응용 도메인 + 증강/불균형)의 실제 게재처: Neurocomputing(FGA-YOLO), IET Image Processing(군용기 attention), IEEE TGRS 계열(Diff-Mosaic), MDPI Remote Sensing(RSI-YOLO 등), Pattern Recognition(E2IGB).

| 후보 저널 | 등급(대략) | 적합 이유 / 조건 |
|---|---|---|
| Expert Systems with Applications | Q1 | 응용 + 방법 결합 논문 다수. P0~P2를 모두 이행하면 도전 가능 |
| Engineering Applications of Artificial Intelligence | Q1 | 도메인 응용 AI에 우호적. 위와 동일 조건 |
| Neurocomputing | Q1~Q2 | FGA-YOLO 등 항공기 detection 게재 실적. 방법론 기여(priority score) 강조 시 적합 |
| Defence Technology (Elsevier) | Q1 | 군사 도메인 특화 — 도메인 스토리가 강점이 되는 곳. 방법 novelty 부담이 상대적으로 낮음 |
| Image and Vision Computing / Machine Vision and Applications | Q2 | P2(2차 데이터셋·모델)를 생략할 경우의 현실적 타겟 |
| MDPI Remote Sensing | Q1~Q2 | MAR20을 추가해 원격탐사 프레이밍이 가능할 때만 |
| IEEE Access | Q2 | 안전판. 현재 설계+P0/P1만으로도 사정권 |

권고: **P0+P1 완료 시 Q2(IVC/MVA/IEEE Access) 안정권, P2까지 완료 시 Q1(ESWA/EAAI/Defence Technology) 도전 가능.** 현재 상태(핵심 실험군 미검증, 2 seeds, 단일 모델·데이터셋)로는 Q2도 리젝 가능성이 높음.

---

## 주요 출처

- Li et al., A Simple Background Augmentation Method for Object Detection with Diffusion Model, ECCV 2024 — https://arxiv.org/abs/2408.00350
- Suri et al., Gen2Det — https://arxiv.org/abs/2312.04566
- Zhao et al., X-Paste, ICML 2023 — https://proceedings.mlr.press/v202/zhao23f.html
- Fan et al., DiverGen, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/html/Fan_DiverGen_Improving_Instance_Segmentation_by_Learning_Wider_Data_Distribution_with_CVPR_2024_paper.html
- Zhu et al., BSGAL (Generative Active Learning for Long-tailed Instance Segmentation), ICML 2024 — https://proceedings.mlr.press/v235/zhu24b.html
- Xie et al., MosaicFusion — https://arxiv.org/pdf/2309.13042
- Ghiasi et al., Simple Copy-Paste, CVPR 2021 — https://arxiv.org/abs/2012.07177
- Trabucco et al., DA-Fusion — https://arxiv.org/html/2302.07944v3
- Class Imbalance in Object Detection: Experimental Diagnosis (2024) — https://arxiv.org/html/2403.07113v1
- IRFS — https://arxiv.org/abs/2305.08069 / E-IRFS — https://arxiv.org/pdf/2503.21893
- Kang et al., Decoupling Representation and Classifier, ICLR 2020 — https://openreview.net/pdf?id=r1gRTCVFvB
- Liang et al., Diffusion Curriculum (DisCL) — https://arxiv.org/html/2410.13674v4
- ODGEN, NeurIPS 2024 — https://arxiv.org/abs/2405.15199
- Tang et al., AeroGen, CVPR 2025 — https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_AeroGen_Enhancing_Remote_Sensing_Object_Detection_with_Diffusion-Driven_Data_Generation_CVPR_2025_paper.pdf
- Control Copy-Paste (2025) — https://arxiv.org/pdf/2507.21816
- MAR20 — https://www.ygxb.ac.cn/en/article/doi/10.11834/jrs.20222139/
- Detection of Military Aircraft Using YOLO and Transformer-Based Models (2025, 동일 Kaggle 데이터셋) — https://www.researchgate.net/publication/388739097
- FGA-YOLO, Neurocomputing 2025 — https://www.sciencedirect.com/science/article/abs/pii/S0925231224018381
- The Unmet Promise of Synthetic Training Images, NeurIPS 2024 — https://arxiv.org/pdf/2406.05184
- Advances in Diffusion Models for Image Data Augmentation (서베이) — https://arxiv.org/pdf/2407.04103
- T2ID-CAS — https://arxiv.org/html/2504.21231v1
- Informative sample generation using class-aware GAN — https://arxiv.org/abs/1904.10781
