import numpy as np

from src.evaluation.gradcam import describe_attention_region


def test_attention_region_uses_display_coordinates_not_anatomy():
    heatmap = np.zeros((12, 12), dtype=float)
    heatmap[1:4, 8:11] = 1.0

    description = describe_attention_region(heatmap)

    assert description['available'] is True
    assert description['display_region'] == 'upper-right area of the displayed image'
    assert description['peak_pixel'] == {'row': 2, 'column': 9}
    assert 0 < description['highlighted_area_percent'] < 100


def test_attention_region_reports_no_signal_cleanly():
    description = describe_attention_region(np.zeros((8, 8), dtype=float))

    assert description['available'] is False
    assert 'No positive Grad-CAM signal' in description['reason']
