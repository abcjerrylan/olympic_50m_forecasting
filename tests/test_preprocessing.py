import pytest

from src.preprocessing import assert_no_identity_features, tensor_feature_names


def test_model_feature_list_has_no_identity_fields():
    assert_no_identity_features(tensor_feature_names())


def test_identity_field_fails_fast():
    with pytest.raises(AssertionError):
        assert_no_identity_features(["normalized_time", "Name"])

