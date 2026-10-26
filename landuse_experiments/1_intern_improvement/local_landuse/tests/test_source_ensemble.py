import numpy as np

from local_landuse.source_ensemble import select_source_models


def test_source_ensemble_combines_complementary_models_using_validation():
    source = np.repeat([1, 2], 100)
    observed = source.copy()
    observed[:20] = 2
    observed[100:120] = 1
    a, b = source.copy(), source.copy()
    a[:20] = 2
    b[100:120] = 1
    policy, metrics, rows = select_source_models(source, observed, {"a": a, "b": b},
                                                minimum_pixels=50, minimum_changes=10)
    assert policy["source_models"] == {1: "a", 2: "b"}
    assert metrics["figure_of_merit_pct"] == 100.
    assert all(row["figure_of_merit_pct"] == 50. for row in rows)


def test_source_ensemble_keeps_global_model_for_sparse_classes():
    source = np.repeat([1, 2], 100)
    observed = source.copy()
    observed[:20] = 2
    observed[100:120] = 1
    a, b = source.copy(), source.copy()
    a[:20] = 2
    b[100:120] = 1
    policy, metrics, _ = select_source_models(source, observed, {"a": a, "b": b})
    assert set(policy["source_models"].values()) == {policy["default_model"]}
    assert metrics["figure_of_merit_pct"] == 50.
