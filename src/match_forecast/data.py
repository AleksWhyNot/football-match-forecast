from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .common import download, dump


def load_matches(root):
    root = Path(root)
    sources = json.loads((root / "config/sources.json").read_text(encoding="utf-8"))
    frames, excluded = [], 0
    for item in sources["files"]:
        path = root / "data/raw" / item["file"]
        download(item["url"], path, digest=item["sha256"])
        data = pd.read_csv(path)
        excluded += int((~data.isResult.eq(True)).sum())
        data = data[data.isResult.eq(True)].copy()
        data["league"], data["season"] = item["league"], item["season"]
        frames.append(data)
    raw = pd.concat(frames, ignore_index=True)
    data = raw.rename(
        columns={
            "id": "match_id",
            "h.id": "home_id",
            "a.id": "away_id",
            "h.title": "home",
            "a.title": "away",
            "goals.h": "home_goals",
            "goals.a": "away_goals",
            "xG.h": "home_xg",
            "xG.a": "away_xg",
        }
    )
    columns = [
        "match_id",
        "league",
        "season",
        "datetime",
        "home_id",
        "away_id",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
    ]
    data = data[columns].copy()
    data["date"] = pd.to_datetime(data.datetime).dt.normalize()
    for column in [
        "match_id",
        "home_id",
        "away_id",
        "home_goals",
        "away_goals",
        "season",
    ]:
        data[column] = data[column].astype(int)
    if data.isna().any().any() or data.match_id.duplicated().any():
        raise ValueError("Missing fields or duplicate match IDs")
    if (
        data.home_id.eq(data.away_id).any()
        or (data[["home_goals", "away_goals", "home_xg", "away_xg"]] < 0).any().any()
    ):
        raise ValueError("Invalid match totals")
    data["target"] = np.select(
        [data.home_goals.gt(data.away_goals), data.home_goals.eq(data.away_goals)],
        [0, 1],
        default=2,
    )
    data = data.sort_values(["date", "match_id"]).reset_index(drop=True)
    audit = {
        "matches": len(data),
        "excluded_unplayed": excluded,
        "leagues": sorted(data.league.unique()),
        "first_date": str(data.date.min().date()),
        "last_date": str(data.date.max().date()),
        "league_seasons": data.groupby(["league", "season"])
        .size()
        .reset_index(name="matches")
        .to_dict("records"),
        "excluded_columns": ["forecast.w", "forecast.d", "forecast.l"],
        "reason": "These source probabilities are not documented as available before kickoff.",
    }
    return data, audit


def poisson_probabilities(home_rate, away_rate):
    grid = np.arange(31)
    joint = (
        poisson.pmf(grid, home_rate)[:, None] * poisson.pmf(grid, away_rate)[None, :]
    )
    p = np.array([np.tril(joint, -1).sum(), np.trace(joint), np.triu(joint, 1).sum()])
    return p / p.sum()


def build_features(matches, config):
    history = defaultdict(list)
    ratings, seen_season = {}, {}
    league_totals = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))
    rows = []
    for date, day in matches.sort_values(["date", "match_id"]).groupby(
        "date", sort=True
    ):
        updates = []
        for match in day.itertuples(index=False):
            keys = [(match.league, match.home_id), (match.league, match.away_id)]
            for key in keys:
                if key not in ratings:
                    ratings[key] = 1500.0
                if key in seen_season and seen_season[key] != match.season:
                    ratings[key] = 1500 + config["elo_season_retention"] * (
                        ratings[key] - 1500
                    )
                seen_season[key] = match.season
            total_h, total_a, count = league_totals[match.league]
            league_h = (total_h + 20 * 1.4) / (count + 20)
            league_a = (total_a + 20 * 1.1) / (count + 20)
            row = {
                "match_id": match.match_id,
                "date": str(date.date()),
                "season": match.season,
                "league": match.league,
                "home": match.home,
                "away": match.away,
                "target": match.target,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "c_league": match.league,
                "f_league_home_xg": league_h,
                "f_league_away_xg": league_a,
            }
            venue = {}
            for side, key, location in [("home", keys[0], "h"), ("away", keys[1], "a")]:
                past = history[key]
                row[f"f_{side}_history"] = min(len(past), 100)
                row[f"f_{side}_rest"] = (
                    min((date - past[-1]["date"]).days, 60) if past else np.nan
                )
                row[f"f_{side}_matches_14d"] = sum(
                    (date - p["date"]).days <= 14 for p in past[-8:]
                )
                for window in config["history_windows"]:
                    recent = past[-window:]
                    for metric in [
                        "goals_for",
                        "goals_against",
                        "xg_for",
                        "xg_against",
                        "points",
                    ]:
                        row[f"f_{side}_{metric}_{window}"] = (
                            np.mean([p[metric] for p in recent]) if recent else np.nan
                        )
                recent = [p for p in past if p["venue"] == location][-10:]
                prior_for, prior_against = (
                    (league_h, league_a) if side == "home" else (league_a, league_h)
                )
                venue[side] = [
                    (sum(p[metric] for p in recent) + 5 * prior) / (len(recent) + 5)
                    for metric, prior in [
                        ("xg_for", prior_for),
                        ("xg_against", prior_against),
                    ]
                ]
                row[f"f_{side}_venue_xgf"], row[f"f_{side}_venue_xga"] = venue[side]
            row["f_elo_difference"] = (
                ratings[keys[0]] - ratings[keys[1]] + config["elo_home_advantage"]
            )
            row["f_poisson_home"] = float(
                np.clip(np.sqrt(venue["home"][0] * venue["away"][1]), 0.1, 6)
            )
            row["f_poisson_away"] = float(
                np.clip(np.sqrt(venue["away"][0] * venue["home"][1]), 0.1, 6)
            )
            p = poisson_probabilities(row["f_poisson_home"], row["f_poisson_away"])
            row.update({f"b_poisson_{i}": float(v) for i, v in enumerate(p)})
            row["split"] = (
                "train"
                if match.season <= config["train_last_season"]
                else (
                    "validation"
                    if match.season == config["validation_season"]
                    else (
                        "calibration"
                        if match.season == config["calibration_season"]
                        else (
                            "test"
                            if match.season == config["test_season"]
                            else "excluded"
                        )
                    )
                )
            )
            rows.append(row)
            expected = 1 / (1 + 10 ** (-row["f_elo_difference"] / 400))
            actual = 1.0 if match.target == 0 else 0.5 if match.target == 1 else 0.0
            updates.append((match, keys, config["elo_k"] * (actual - expected)))
        for match, keys, change in updates:
            ratings[keys[0]] += change
            ratings[keys[1]] -= change
            for key, venue_name, gf, ga, xf, xa in [
                (
                    keys[0],
                    "h",
                    match.home_goals,
                    match.away_goals,
                    match.home_xg,
                    match.away_xg,
                ),
                (
                    keys[1],
                    "a",
                    match.away_goals,
                    match.home_goals,
                    match.away_xg,
                    match.home_xg,
                ),
            ]:
                history[key].append(
                    {
                        "date": date,
                        "venue": venue_name,
                        "goals_for": gf,
                        "goals_against": ga,
                        "xg_for": xf,
                        "xg_against": xa,
                        "points": 3 if gf > ga else 1 if gf == ga else 0,
                    }
                )
            league_totals[match.league] += [match.home_xg, match.away_xg, 1]
    return pd.DataFrame(rows)


def prepare(root):
    root = Path(root)
    config = json.loads((root / "config/project.json").read_text(encoding="utf-8"))
    matches, audit = load_matches(root)
    features = build_features(matches, config)
    audit["splits"] = features.groupby("split").size().to_dict()
    audit["cold_start_matches"] = int(
        ((features.f_home_history < 5) | (features.f_away_history < 5)).sum()
    )
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    features.to_csv(root / "data/processed/features.csv", index=False)
    dump(root / "reports/data_quality.json", audit)
    return matches, features, audit
