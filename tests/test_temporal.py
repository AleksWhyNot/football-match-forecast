import numpy as np
import pandas as pd
import pytest

from match_forecast.temporal import split_training, choose_calibration
from match_forecast.temporal_methods import (
    apply_calibrator, fit_calibrator, pav, history_before,
    fit_interval, apply_interval, interval_metrics, joint_probabilities,
)


def test_xg_boundary_excludes_last_day_and_whole_current_month():
    frame = pd.DataFrame({"period": ["2015-08", "2015-09", "2015-10", "2015-10", "2015-11"],
                          "date": ["2015-08-15", "2015-09-20", "2015-10-20", "2015-10-31", "2015-11-01"],
                          "split": ["train"] * 5})
    train, stop, past, future = split_training(frame, "2015-11", {"kind": "xg", "embargo_days": 1})
    assert len(past) == 3
    assert past.date.max() == "2015-10-20"
    assert set(train.period) == {"2015-08", "2015-09"}
    assert set(stop.period) == {"2015-10"}
    assert len(future) == 1


def test_recent_window_uses_completed_periods_only():
    frame = pd.DataFrame({"period": ["2017", "2018", "2019", "2020"]})
    assert history_before(frame, "2020", 2).period.tolist() == ["2018", "2019"]
    assert history_before(frame, "2019").period.tolist() == ["2017", "2018"]


@pytest.mark.parametrize("method", ["none", "temperature", "temperature_shrunk", "affine", "isotonic"])
def test_calibrators_return_probabilities_and_do_not_change_input(method):
    rng = np.random.default_rng(92)
    p = rng.dirichlet([2, 1, 3], size=120)
    y = np.array([rng.choice(3, p=row) for row in p])
    original = p.copy()
    state = fit_calibrator(p, y, method)
    corrected = apply_calibrator(p, state)
    assert np.allclose(corrected.sum(axis=1), 1)
    assert np.isfinite(corrected).all()
    assert (corrected > 0).all()
    assert np.array_equal(original, p)
    if method == "none":
        assert np.array_equal(corrected, p)


def test_text_checksums_do_not_depend_on_line_endings(tmp_path):
    from match_forecast.temporal import digest
    first, second = tmp_path / "windows.csv", tmp_path / "unix.csv"
    first.write_bytes(b"x,y\r\n1,2\r\n")
    second.write_bytes(b"x,y\n1,2\n")
    assert digest(first) == digest(second)


def test_isotonic_pools_violations_and_tied_scores():
    state = pav(np.array([.1, .2, .2, .8, .9]), np.array([0, 1, 0, 0, 1]))
    assert np.allclose(state["y"], [0, 1/3, 1/3, 1])
    assert np.all(np.diff(state["y"]) >= 0)


def test_calibration_selection_uses_later_folds_not_fit_loss():
    rng = np.random.default_rng(14)
    frames = []
    for period in ("2016", "2017", "2018"):
        y = rng.binomial(1, .3, 150)
        frames.append(pd.DataFrame({"period": period, "target": y, "p_raw_0": .55, "p_raw_1": .45}))
    frame = pd.concat(frames)
    plan = {"kind": "xg", "classes": 2, "embargo_days": 0, "recent_periods": 1}
    selected, detail, rank = choose_calibration(frame, plan)
    assert set(detail.period) == {"2017", "2018"}
    assert detail.n_calibration.min() == 150
    assert np.isfinite(rank.log_loss).all()
    assert selected["periods"] == ["2017", "2018"]


def test_interval_calibration_does_not_change_point_forecasts():
    history = pd.DataFrame({"pred_minutes": np.full(40, 1000.), "target_minutes": np.arange(40.) + 980})
    point = np.array([500., 1500.])
    original = point.copy()
    state = fit_interval(history, "minutes", "absolute")
    lo, hi = apply_interval(point, state)
    assert np.array_equal(point, original)
    assert np.all(lo <= point) and np.all(point <= hi)
    measured = interval_metrics(point, lo, hi)
    assert measured["coverage"] == 1


def test_season_probabilities_preserve_eligibility_order():
    frame = pd.DataFrame({"pred_observed": [.8, .4], "pred_eligible": [.6, .1]})
    p = joint_probabilities(frame)
    assert np.allclose(p, [[.2, .2, .6], [.6, .3, .1]])
    with pytest.raises(ValueError, match="exceeds"):
        joint_probabilities(pd.DataFrame({"pred_observed": [.2], "pred_eligible": [.3]}))

