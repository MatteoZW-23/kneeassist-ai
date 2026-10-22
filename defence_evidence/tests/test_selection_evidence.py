from src.selection.evidence import ranking_for_target, split_summary
from src.selection.model_selector import select_models


def test_five_candidate_benchmark_is_saved_and_ranked_per_target():
    expected = {
        'abnormal': 'family_densenet121',
        'acl': 'family_resnet18',
        'meniscus': 'family_swin_t',
    }
    for target, winner in expected.items():
        ranking = ranking_for_target(target)
        assert len(ranking) == 5
        assert ranking[0]['model_id'] == winner
        assert all('metrics' in item for item in ranking)


def test_upload_time_selection_ranks_five_benchmark_candidates_before_inference():
    result = select_models(['axial', 'coronal', 'sagittal'], ['abnormal', 'acl', 'meniscus'])
    assert len(result['compatible_models']) == 5
    assert all(len(result['candidates_by_target'][target]) == 5 for target in ('abnormal', 'acl', 'meniscus'))
    assert [item['model_id'] for item in result['selections']] == [
        'family_densenet121', 'family_resnet18', 'family_swin_t'
    ]
    assert all(item['status'] == 'Selected' for item in result['selections'])


def test_fixed_split_summary_has_four_non_overlapping_roles():
    summary = split_summary()
    assert [row['Studies'] for row in summary] == [804, 226, 100, 120]
