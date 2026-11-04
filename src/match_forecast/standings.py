"""Очки команд и итоговые места по предматчевым вероятностям."""

import numpy as np
import pandas as pd


KEY = ["season", "league", "team"]


def team_points(frame, reference, model="neural"):
    if frame.duplicated(["season", "league", "match_id"]).any():
        raise ValueError("Повторный матч в прогнозах")
    probabilities = frame[[f"p_{model}_{i}" for i in range(3)]].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Некорректные вероятности исходов")
    rows = []
    for side, win in [("home", 0), ("away", 2)]:
        rows.append(
            pd.DataFrame(
                {
                    "season": frame.season,
                    "league": frame.league,
                    "team": frame[side],
                    "actual_points": np.where(
                        frame.target.eq(win), 3, np.where(frame.target.eq(1), 1, 0)
                    ),
                    "expected_points": 3 * frame[f"p_{model}_{win}"]
                    + frame[f"p_{model}_1"],
                }
            )
        )
    teams = (
        pd.concat(rows)
        .groupby(KEY)
        .agg(
            matches=("actual_points", "size"),
            points=("actual_points", "sum"),
            expected=("expected_points", "sum"),
        )
        .reset_index()
    )
    teams = teams.merge(
        reference, on=KEY, how="outer", validate="one_to_one", indicator=True
    )
    if not teams["_merge"].eq("both").all():
        raise ValueError("Состав команд не совпадает с итоговыми таблицами")
    if (
        not teams.matches.eq(teams.reference_matches).all()
        or not teams.points.eq(teams.reference_points).all()
    ):
        raise ValueError("Матчи или очки не совпадают с итоговыми таблицами")
    for _, group in teams.groupby(["season", "league"]):
        if sorted(group.actual_position) != list(range(1, len(group) + 1)):
            raise ValueError("Некорректные фактические места")
    if not teams.ranking_basis.isin(["points", "points_per_match"]).all():
        raise ValueError("Неизвестное правило ранжирования")
    teams["difference"] = teams.points - teams.expected
    teams["absolute_error"] = teams.difference.abs()
    teams["points_per_match"] = teams.points / teams.matches
    teams["expected_per_match"] = teams.expected / teams.matches
    teams["ranking_score"] = np.where(
        teams.ranking_basis.eq("points_per_match"),
        teams.expected_per_match,
        teams.expected,
    )
    # Одинаковые ожидания получают среднее место, без подстановки фактического результата.
    teams["ranking_score"] = teams.ranking_score.round(12)
    teams["predicted_position"] = teams.groupby(
        ["season", "league"]
    ).ranking_score.rank(
        method="average",
        ascending=False,
    )
    teams["position_difference"] = teams.predicted_position - teams.actual_position
    teams["absolute_position_error"] = teams.position_difference.abs()
    columns = [
        "league",
        "team",
        "matches",
        "points",
        "expected",
        "difference",
        "season",
        "absolute_error",
        "actual_position",
        "predicted_position",
        "position_difference",
        "absolute_position_error",
        "points_per_match",
        "expected_per_match",
        "ranking_basis",
    ]
    return teams.sort_values(["season", "league", "actual_position"])[
        columns
    ].reset_index(drop=True)


def summarize(teams):
    rows = []
    for league, group in [("all", teams), *teams.groupby("league", sort=True)]:
        error = group.difference.to_numpy(float)
        absolute = np.abs(error)
        ranks = group.absolute_position_error.to_numpy(float)
        rows.append(
            {
                "league": league,
                "teams": len(group),
                "bias": float(error.mean()),
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "variance": float(error.var(ddof=0)),
                "std": float(error.std(ddof=0)),
                "median_absolute_error": float(np.median(absolute)),
                "max_absolute_error": float(absolute.max()),
                "within_5_points": int((absolute <= 5).sum()),
                "within_10_points": int((absolute <= 10).sum()),
                "mae_per_match": float(
                    np.abs(group.points_per_match - group.expected_per_match).mean()
                ),
                "position_mae": float(ranks.mean()),
                "position_rmse": float(np.sqrt(np.mean(ranks**2))),
                "max_position_error": float(ranks.max()),
                "exact_positions": int((ranks == 0).sum()),
                "within_2_positions": int((ranks <= 2).sum()),
            }
        )
    return pd.DataFrame(rows)


def league_outcomes(teams):
    rows = []
    for (season, league), group in teams.groupby(["season", "league"]):
        actual = group.sort_values("actual_position")
        predicted = group.sort_values("predicted_position")
        rows.append(
            {
                "season": season,
                "league": league,
                "actual_winner": actual.iloc[0].team,
                "predicted_winner": predicted.iloc[0].team,
                "winner_correct": bool(actual.iloc[0].team == predicted.iloc[0].team),
                "top4_overlap": len(
                    set(actual.head(4).team) & set(predicted.head(4).team)
                ),
                "bottom3_overlap": len(
                    set(actual.tail(3).team) & set(predicted.tail(3).team)
                ),
            }
        )
    return pd.DataFrame(rows)
