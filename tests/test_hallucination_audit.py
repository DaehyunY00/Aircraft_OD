from src.eval.audit_hallucination import _max_containment


def test_detection_outside_every_gt_box_counts_as_extra() -> None:
    # 환각 후보: GT와 전혀 겹치지 않는 검출.
    assert _max_containment((0, 0, 10, 10), [(20, 20, 30, 30)]) == 0.0


def test_small_detection_inside_large_gt_is_not_extra() -> None:
    # IoU였다면 0.02로 '바깥'으로 오판했을 케이스. containment는 1.0을 준다.
    assert _max_containment((5, 5, 8, 8), [(0, 0, 20, 20)]) == 1.0


def test_partial_overlap_reports_covered_fraction() -> None:
    assert _max_containment((0, 0, 10, 10), [(5, 0, 15, 10)]) == 0.5


def test_no_ground_truth_boxes_makes_every_detection_extra() -> None:
    assert _max_containment((0, 0, 10, 10), []) == 0.0


def test_best_matching_gt_box_wins() -> None:
    # 여러 GT 중 가장 많이 담는 쪽을 기준으로 판정해야 한다.
    gts = [(20, 20, 30, 30), (0, 0, 20, 20)]
    assert _max_containment((5, 5, 8, 8), gts) == 1.0
