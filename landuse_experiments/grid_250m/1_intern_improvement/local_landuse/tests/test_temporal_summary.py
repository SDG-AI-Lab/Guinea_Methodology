import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from local_landuse.models import sample_digest
from local_landuse.temporal_summary import summarize_runs


def make_report(directory, validation=(31., 30.), final=(27., 45.)):
    directory.mkdir()
    indices = np.array([0, 1], dtype=np.int64)
    np.save(directory / "train_indices.npy", indices)
    np.save(directory / "validation_indices.npy", np.array([2, 3], dtype=np.int64))
    with rasterio.open(directory / "evaluation_mask.tif", "w", driver="GTiff",
                       width=2, height=2, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(-10, 10, .01, .01)) as dst:
        dst.write(np.ones((2, 2), dtype=np.uint8), 1)
    report = dict(manifest=dict(primary_sample_sha256=sample_digest(indices),
        fit_period=[2010, 2015], spatial_validation_period=[2010, 2015],
        final_period=[2015, 2020]), evaluation_target_sha256="fixture-target",
        validation=[], results={})
    for i, name in enumerate(("chosen_by_validation", "best_final_only")):
        report["validation"].append(dict(phase="validation", name=name, figure_of_merit_pct=validation[i],
            overall_accuracy=.9, training_rows=2, features=5))
        metrics = dict(figure_of_merit_pct=final[i], pixels=4, predicted_change_pixels=1)
        report["results"][name] = dict(summary=metrics)
        report["results"][name+"_fixed_demand"] = dict(summary=metrics)
    (directory / "report.json").write_text(json.dumps(report))


def test_summary_does_not_choose_by_final_target_score(tmp_path):
    run = tmp_path / "run"
    make_report(run)
    result = summarize_runs([run], tmp_path / "summary")
    assert result["selected_model"] == "chosen_by_validation"
    assert result["selected_final_fom"] == 27.
    assert not result["target_met"]


def test_summary_rejects_changed_training_or_validation_rows(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_report(a)
    make_report(b)
    np.save(b / "validation_indices.npy", np.array([1, 2]))
    with pytest.raises(ValueError, match="Runs must share"):
        summarize_runs([a, b], tmp_path / "out")
    np.save(a / "train_indices.npy", np.array([2, 3]))
    with pytest.raises(ValueError, match="training sample"):
        summarize_runs([a], tmp_path / "out")
