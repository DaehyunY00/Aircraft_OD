# BUGFIX_REPORT: Inpainting 생성 검증 결함 (Phase A / P0-1)

수정 일자: 2026-07-11 | 대상 커밋: Phase 0 baseline (`4a30ff9`) → Phase A

## 1. 증상

pilot 실험(`outputs_pilot`, 현재 `outputs_pilot_deprecated/`로 보존)에서 synthetic
증강군(D/E)이 basic_aug를 이기지 못했고, 생성 로그를 보면 생성물이 유효한지 판정할
근거가 전혀 없었다.

## 2. 재현 조건 및 관측 증거

`outputs_pilot_deprecated/synthetic/`에서 직접 확인 가능:

1. **`generation_log_{uniform,selective}.csv` (각 120행)**: 전 행 `accepted=True`,
   전 행 `bbox_pixel_diff == 0.0` (정확히 0), `reject_reason` 전부 공란.
   판정 지표가 상수라는 것은 QC가 아무것도 측정하지 못했다는 뜻이다.
2. **`review_sheet_uniform.jpg` vs `review_sheet_selective.jpg`**: md5 동일
   (`801ac311463604453bcc49051e19bf80`). 두 plan의 시드 공식·source 순회·prompt 순회가
   동일해 겹치는 (class, idx)의 생성물이 픽셀 단위로 같기 때문이다(예산 통제 비교
   측면에서는 의도된 성질이며 버그는 아님 — §5 참고).
3. **review sheet 행별 원본 vs 생성 mean abs diff** (12행):
   `[6.05, 6.4, 12.88, 7.6, 8.07, 0.07, 15.97, 25.63, 6.44, 2.62, 1.54, 10.83]`.
   마지막 pilot 생성 run은 실제 diffusion이 실행됐지만(대부분 행에서 배경 변화 존재),
   **행 6은 diff 0.07로 사실상 원본 사본**이고 행 10·11도 1.5~2.6 수준으로 배경이 거의
   변하지 않았다. 이런 no-op 생성물이 전량 `accepted=True`로 통과했다.

## 3. 근본원인 (코드 위치 기준, Phase 0 baseline)

### RC-1. 판정 지표가 구조적으로 항상 0 (vacuous QC) — 핵심 원인

`src/augment/inpaint_background.py` (baseline 242–245행):

```python
generated = paste_protected_regions(original, generated, padded_boxes)   # 원본 bbox를 다시 붙임
bbox_diff = max_bbox_diff(original, generated, original_boxes)           # 붙인 뒤에 측정
accepted = valid and bbox_diff <= bbox_diff_threshold
```

`original_boxes ⊆ padded_boxes`이므로 `paste_protected_regions` 직후의 bbox 내부는
원본과 **정의상 동일**하다. 따라서 `bbox_diff`는 어떤 입력에서도 정확히 0.0이고
`accepted`는 항상 True다. 게다가 **배경이 실제로 바뀌었는지 확인하는 검사 자체가
없었다.** 결과적으로 dry-run 사본, stale 파일, diffusion no-op(아래 RC-4) 모두 "정상
생성물"로 승인됐다. pilot 로그의 `bbox_pixel_diff == 0.0` 전행 일치가 이 코드의 직접
증거다.

### RC-2. dry-run 산출물이 실제 생성물과 구분 불가능

- `src/augment/inpaint_background.py` (baseline 229–230행): `dry_run=True`면
  `generated = original.copy()`로 원본 사본을 저장하되, 로그 스키마·파일명·저장 경로가
  실제 생성물과 완전히 동일했다 (`dry_run` 컬럼조차 없음).
- `src/run_pipeline.py` (baseline 267, 277행): `--dry-run-inpaint`가 두 plan 생성에 그대로
  전달되며 아무 경고도 없었다.
- README는 "compute unit이 부족하면 먼저 --dry-run-inpaint로 구조 검증"을 권장했으므로,
  같은 `processed_data` 경로에서 dry-run → 실제 run 순서로 실행하는 시나리오가 자연스럽게
  발생한다.

### RC-3. already_exists 분기의 무검증 승인 (resume 시 stale 사본 세탁)

`src/augment/inpaint_background.py` (baseline 179–210행): 출력 파일이 이미 있으면 픽셀
검사 없이 `accepted=True, reject_reason="already_exists", bbox_pixel_diff=0.0`으로
기록하고 통과시켰다. RC-2와 결합하면 **dry-run이 만든 원본 사본이 이후 실제 run의
resume 경로에서 전량 "정상 생성물"로 승인**된다. 파이프라인 기본값이 resume enabled라
이 경로는 Colab 세션이 한 번이라도 끊기면 실행된다.

### RC-4. 기하학적 no-op: bbox(+padding)가 프레임 전체를 덮는 이미지

`create_inpainting_mask`는 bbox+padding을 보호(black)하는데, 군용기 근접 촬영처럼
bbox가 화면 대부분을 덮는 이미지는 편집 가능 배경이 거의 없어 **실제 diffusion을 돌려도
출력이 원본과 동일**하다(review sheet 행 6, diff 0.07). 이런 source를 걸러내는 로직이
없어 budget이 사본 생성에 소모됐다.

### RC-5 (부차). 마스크 resize 보간

`_run_inpaint` (baseline 105행)가 마스크를 LANCZOS로 512×512 resize했다. LANCZOS 링잉과
diffusers의 0.5 이진화가 겹치면 bbox 주변 blur 밴드가 보호 영역으로 흡수되어 편집 가능
배경이 더 줄어든다. NEAREST로 교체했다.

## 4. 수정 내용

| 위치 | 수정 |
|---|---|
| `src/utils/image.py` | `boxes_mask`, `background_mean_abs_diff`(bbox 외부 diff), `bbox_interior_mean_abs_diff`(보호 위반 감시), `editable_background_ratio` 신규 |
| `src/augment/inpaint_background.py` `verify_output_against_source` | 신규 승인 기준: **배경 diff ≥ `verification.min_background_change`(기본 10.0)** AND bbox 내부 diff ≤ `verification.max_bbox_protected_change`(기본 5.0, JPEG 노이즈 상회) AND 크기 일치 AND non-trivial |
| 〃 생성 루프 | 보호 위반(`max_bbox_diff`)을 **paste 이전의 raw diffusion 출력**에서 측정(사후 측정은 vacuous — RC-1), paste 후 배경 검증. 실패 시 `verification.max_retries_per_image` 재시도, 최종 실패 시 rejected/로 이동하고 train split의 stale 파일 삭제 |
| 〃 dry-run | 시작 시 "구조 점검 전용" 경고 출력, plan 디렉터리에 `DRY_RUN_MARKER.txt` 기록, 로그에 `dry_run=True`·`reject_reason="dry_run_copy"` 명시. 이후 비-dry-run 실행이 marker를 발견하면 already_exists를 무시하고 **전량 재생성** 후 marker 삭제 |
| 〃 already_exists 분기 | 기존 파일을 원본과 비교해 위 승인 기준으로 **재검증**. 통과 시에만 `already_exists_verified`로 재사용, 실패 시 재생성 |
| 〃 source 선정 | `editable_background_ratio < verification.min_editable_background_ratio`(기본 0.05)인 source(bbox가 화면 전체를 덮는 이미지)를 생성 대상에서 제외 (RC-4) |
| 〃 `_run_inpaint` | 마스크 resize LANCZOS → NEAREST (RC-5) |
| `src/run_pipeline.py` | `--dry-run-inpaint` 사용 시 파이프라인 수준 경고 |
| `configs/{default,smoke,pilot,full}.yaml` | `verification` 섹션 신설(파라미터 5종). `diffusion.max_retries_per_image`는 `verification.max_retries_per_image`로 이동 |
| `tests/test_inpaint_regeneration.py` | dry-run marker/플래그, dry-run 후 전량 재생성, resume 재검증, stale 사본 비재사용, no-op 생성 거부+train split 정리 — 5개 회귀 테스트 |

## 5. 수정 후 보장되는 것

- **비-dry-run 실행에서 원본 사본이 train split에 남는 경로가 없다.** (1) 신규 생성물은
  배경 diff ≥ 임계를 통과해야 저장되고, (2) 기존 파일은 재검증을 통과해야 재사용되며,
  (3) dry-run 산출물은 marker에 의해 무조건 재생성 대상이고, (4) 검증 실패 시 train
  split의 해당 파일이 삭제된다. 이는 `tests/test_inpaint_regeneration.py`가 회귀 방지한다.
- 동일 입력에서 배경 변화가 임계 미만이면 재시도 후 rejected로 격리되고, 생성 로그에
  `background_pixel_diff`·`verification_passed`·`verification_fail_reason`·`dry_run`이
  이미지 단위로 기록되어 "배경이 실제로 바뀌었음"을 로그로 증명할 수 있다.
  (Phase B에서 SSIM/LPIPS 및 `verification_report.csv`, 전체 실패율 게이트로 확장.)
- uniform/selective가 같은 (class, idx)에서 같은 시드를 쓰는 것은 유지한다. 같은 budget
  비교에서 배분 차이 이외의 생성 노이즈를 통제하는 의도된 설계이며, 논문에 명시할 것.

## 6. 기존 pilot 산출물 처리

`outputs_pilot/` → `outputs_pilot_deprecated/`로 이름만 변경해 증거 보존. 해당 결과는
전량 폐기 대상이며 논문에 사용하지 않는다(README에 명시). `configs/pilot.yaml`은 여전히
`outputs_pilot`을 가리키므로 재실험 시 깨끗한 디렉터리에서 시작된다.
