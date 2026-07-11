from src.augment.masks import create_inpainting_mask


def test_mask_generation_protects_bbox_region() -> None:
    labels = [
        {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.2,
            "height": 0.2,
        }
    ]
    mask, padded_boxes, original_boxes = create_inpainting_mask((100, 100), labels, padding_ratio=0.0, blur_radius=0)
    assert mask.getpixel((50, 50)) == 0
    assert mask.getpixel((5, 5)) == 255
    assert padded_boxes == [(40, 40, 60, 60)]
    assert original_boxes == [(40, 40, 60, 60)]
