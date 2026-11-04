from pathlib import Path
import json

import numpy as np
import pandas as pd

from match_forecast.data import build_features, poisson_probabilities


CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "config/project.json").read_text()
)


def sample():
    return pd.DataFrame(
        [
            {
                "match_id": i,
                "date": pd.Timestamp(date),
                "league": "EPL",
                "season": 2015,
                "home_id": h,
                "away_id": a,
                "home": str(h),
                "away": str(a),
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": hx,
                "away_xg": ax,
                "target": 0 if hg > ag else 1 if hg == ag else 2,
            }
            for i, date, h, a, hg, ag, hx, ax in [
                (1, "2015-08-01", 1, 2, 2, 1, 1.5, 0.8),
                (2, "2015-08-08", 1, 3, 0, 0, 0.3, 0.4),
                (3, "2015-08-08", 2, 4, 1, 3, 1.2, 2.5),
                (4, "2015-08-15", 1, 2, 2, 2, 1.1, 1.7),
            ]
        ]
    )


def test_current_and_future_results_cannot_change_prematch_features():
    data = sample()
    reference = build_features(data, CONFIG)
    changed = data.copy()
    changed.loc[
        changed.date.ge("2015-08-08"),
        ["home_goals", "away_goals", "home_xg", "away_xg", "target"],
    ] = [9, 0, 9.0, 0.0, 0]
    actual = build_features(changed, CONFIG)
    columns = [c for c in actual if c.startswith(("f_", "b_", "c_"))]
    mask = reference.date.le("2015-08-08")
    pd.testing.assert_frame_equal(
        reference.loc[mask, columns], actual.loc[mask, columns]
    )
    assert actual.loc[3, "f_home_goals_for_5"] != reference.loc[3, "f_home_goals_for_5"]


def test_same_day_input_order_does_not_change_features():
    data = sample()
    pd.testing.assert_frame_equal(
        build_features(data, CONFIG), build_features(data.iloc[::-1], CONFIG)
    )


def test_rest_and_form_use_previous_dates():
    data = build_features(sample(), CONFIG)
    assert data.loc[1, "f_home_rest"] == 7
    assert data.loc[1, "f_home_goals_for_5"] == 2
    assert data.loc[3, "f_home_history"] == 2
    assert data.loc[3, "f_home_goals_for_5"] == 1


def test_new_team_has_missing_history_not_zero_attack():
    data = build_features(sample(), CONFIG)
    assert np.isnan(data.loc[1, "f_away_goals_for_5"])
    assert data.loc[1, "f_away_history"] == 0


def test_team_history_is_separate_between_leagues():
    data = sample()
    data.loc[3, "league"] = "La_Liga"
    out = build_features(data, CONFIG)
    assert out.loc[3, "f_home_history"] == 0
    assert out.loc[3, "f_away_history"] == 0


def test_poisson_symmetric_rates_and_probability_mass():
    p = poisson_probabilities(1.4, 1.4)
    assert np.isclose(p.sum(), 1)
    assert np.isclose(p[0], p[2])
    assert poisson_probabilities(2.5, 0.5)[0] > poisson_probabilities(0.5, 2.5)[0]


def test_split_is_by_season_and_never_random():
    data = sample()
    data["season"] = [2016, 2017, 2018, 2019]
    out = build_features(data, CONFIG)
    assert out.split.tolist() == ["train", "validation", "calibration", "test"]


def test_goals_and_current_xg_are_not_input_columns():
    data = build_features(sample(), CONFIG)
    inputs = [c for c in data if c.startswith(("f_", "c_"))]
    assert not set(inputs) & {
        "match_id",
        "target",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_id",
        "away_id",
    }
