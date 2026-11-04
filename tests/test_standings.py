from pathlib import Path

import pandas as pd
import pytest

from match_forecast.standings import team_points, summarize


def example():
    frame = pd.DataFrame(
        {
            "season": [2019, 2019],
            "league": ["Test", "Test"],
            "match_id": [1, 2],
            "home": ["A", "B"],
            "away": ["B", "A"],
            "target": [0, 1],
            "p_neural_0": [0.5, 0.2],
            "p_neural_1": [0.3, 0.4],
            "p_neural_2": [0.2, 0.4],
        }
    )
    reference = pd.DataFrame(
        {
            "season": [2019, 2019],
            "league": ["Test", "Test"],
            "team": ["A", "B"],
            "actual_position": [1, 2],
            "reference_matches": [2, 2],
            "reference_points": [4, 1],
            "ranking_basis": ["points", "points"],
        }
    )
    return frame, reference


def test_points_are_summed_for_both_sides_and_match_totals():
    frame, reference = example()
    teams = team_points(frame, reference).set_index("team")
    assert teams.loc["A", "points"] == 4
    assert teams.loc["A", "expected"] == pytest.approx(3.4)
    assert teams.loc["B", "expected"] == pytest.approx(1.9)
    assert teams.points.sum() == 3 * len(frame) - frame.target.eq(1).sum()
    assert teams.expected.sum() == pytest.approx(
        3 * len(frame) - frame.p_neural_1.sum()
    )
    assert teams.loc["A", "difference"] == pytest.approx(0.6)
    assert teams.loc["B", "difference"] == pytest.approx(-0.9)


def test_tied_expectations_receive_average_rank():
    frame, reference = example()
    frame[["p_neural_0", "p_neural_1", "p_neural_2"]] = [0.4, 0.2, 0.4]
    teams = team_points(frame.sample(frac=1, random_state=3), reference)
    assert teams.predicted_position.tolist() == [1.5, 1.5]
    assert teams.position_difference.tolist() == [0.5, -0.5]


@pytest.mark.parametrize(
    "change", ["missing_match", "duplicate", "reference_points", "reference_team"]
)
def test_incomplete_or_mismatched_season_is_rejected(change):
    frame, reference = example()
    if change == "missing_match":
        frame = frame.iloc[:1]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif change == "reference_points":
        reference.loc[0, "reference_points"] += 1
    else:
        reference.loc[0, "team"] = "C"
    with pytest.raises(ValueError):
        team_points(frame, reference)


def test_descriptive_variance_and_error_sign():
    teams = pd.DataFrame(
        {
            "league": ["Test"] * 3,
            "difference": [-2.0, 0.0, 4.0],
            "absolute_position_error": [2, 0, 1],
            "points_per_match": [1, 1, 1],
            "expected_per_match": [2, 1, 0.5],
        }
    )
    result = summarize(teams).iloc[0]
    assert result.bias == pytest.approx(2 / 3)
    assert result.mae == 2
    assert result.variance == pytest.approx(56 / 9)
    assert result.rmse**2 == pytest.approx(result.variance + result.bias**2)
    assert result.position_mae == 1


def test_french_season_reference_uses_points_per_match_and_final_tiebreaks():
    root = Path(__file__).resolve().parents[1]
    reference = pd.read_csv(root / "config/standings_2019_20.csv")
    france = reference.loc[reference.league.eq("Ligue_1")].set_index("team")
    assert france.ranking_basis.eq("points_per_match").all()
    assert france.loc["Strasbourg", "reference_matches"] == 27
    assert (
        france.loc["Strasbourg", "reference_points"]
        < france.loc["Angers", "reference_points"]
    )
    assert (
        france.loc["Strasbourg", "actual_position"]
        < france.loc["Angers", "actual_position"]
    )
    assert (
        france.loc["Nice", "reference_points"]
        == france.loc["Reims", "reference_points"]
    )
    assert france.loc["Nice", "actual_position"] == 5
    assert france.loc["Reims", "actual_position"] == 6


def test_unequal_fixture_counts_rank_expected_points_per_match():
    frame, reference = example()
    frame.loc[2] = [2019, "Test", 3, "A", "C", 1, 0.1, 0.1, 0.8]
    reference.loc[0, ["reference_matches", "reference_points"]] = [3, 5]
    reference.loc[1, "actual_position"] = 3
    reference.loc[2] = [2019, "Test", "C", 2, 1, 1, "points_per_match"]
    reference["ranking_basis"] = "points_per_match"
    teams = team_points(frame, reference).set_index("team")
    assert teams.loc["C", "expected"] < teams.loc["A", "expected"]
    assert teams.loc["C", "predicted_position"] == 1
    assert teams.loc["A", "predicted_position"] == 2
