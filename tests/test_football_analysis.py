import numpy as np
import pandas as pd

from match_forecast.football_analysis import outcome_tables


def test_nonzero_draw_probability_can_have_no_draw_argmax():
    frame = pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "date": ["2020-01-01"] * 3,
            "league": ["Test"] * 3,
            "home": ["A"] * 3,
            "away": ["B"] * 3,
            "home_goals": [1, 1, 0],
            "away_goals": [0, 1, 1],
            "target": [0, 1, 2],
            "p_neural_0": [0.45, 0.35, 0.3],
            "p_neural_1": [0.25, 0.3, 0.25],
            "p_neural_2": [0.3, 0.35, 0.45],
        }
    )
    outcomes, bins, groups, top, detail = outcome_tables(frame)
    draw = outcomes.loc[outcomes.outcome.eq("Ничья")].iloc[0]
    assert draw.actual == 1
    assert draw.selected == 0
    assert draw.recall == 0
    assert np.isclose(draw.expected, 0.8)
    assert bins.matches.sum() == 3
    assert groups.matches.sum() == 3
    assert outcomes.selected.sum() == 3
    assert top.match_id.tolist() == [2]
    assert detail.correct.sum() == 2
