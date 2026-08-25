from temporal_outcome.benchmark import renderers
from temporal_outcome.data import flexible


def test_runtime_query_tokens_are_model_private() -> None:
    row = {
        "benchmark_id": "case-1::clean_continuous_prefix",
        "pair_id": "case-1",
        "messages": [{"role": "user", "content": "visible prenatal records"}],
    }
    labels = {
        "case-1": {
            "outcome_targets": [275.0, 3250.0, 50.0],
            "outcome_mask": [1, 1, 1],
        }
    }
    generic = renderers.generic_vlm_example(row)
    ours = renderers.v2_regression_head_example(row, labels)
    assert all(
        token not in generic["messages"][0]["content"]
        for token in renderers.QUERY_TOKENS
    )
    assert all(
        ours["messages"][0]["content"].count(token) == 1
        for token in renderers.QUERY_TOKENS
    )


def test_target_validation_masks_invalid_optional_values() -> None:
    case = {
        "case_id": "synthetic",
        "targets": {
            "delivery_days": 275,
            "birth_weight_g": 0,
            "birth_length_cm": 50,
        },
    }
    values, mask, reasons = flexible.validated_targets(case)
    assert values == [275, 0.0, 50]
    assert mask == [1, 0, 1]
    assert "birth_weight_missing_or_invalid" in reasons
    assert "<birth_weight_g>NA</birth_weight_g>" in flexible.target_response(
        values, mask
    )
