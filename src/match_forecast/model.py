import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .common import (
    dump,
    losses,
    probabilities,
    train_candidates,
    predict_saved,
    check_seal,
    seal,
)


def elo_probabilities(difference, log_draw):
    advantage = np.asarray(difference) * np.log(10) / 800
    return probabilities(
        np.column_stack([advantage, np.full(len(advantage), log_draw), -advantage])
    )


def run_models(root, frame, train=True):
    root = Path(root)
    config = json.loads((root / "config/project.json").read_text(encoding="utf-8"))
    folder = root / "models"
    numeric = [c for c in frame if c.startswith("f_")]
    if train:
        neural, uncalibrated, info = train_candidates(
            frame, numeric, ["c_league"], config, folder, 3
        )
        training = frame[frame.split.eq("train")]
        constant = np.bincount(training.target, minlength=3) / len(training)
        result = minimize_scalar(
            lambda a: losses(
                training.target.to_numpy(),
                elo_probabilities(training.f_elo_difference, a),
            ).mean(),
            bounds=(-3, 3),
            method="bounded",
        )
        if not result.success:
            raise RuntimeError("Elo draw parameter failed")
        baselines = {"constant": constant.tolist(), "elo_log_draw": float(result.x)}
        dump(folder / "baselines.json", baselines)
        seal(
            root,
            [
                "config/project.json",
                "config/sources.json",
                "models/selected.keras",
                "models/preprocess.json",
                "models/baselines.json",
                "models/training.json",
                "src/match_forecast/common.py",
                "src/match_forecast/data.py",
            ],
        )
    else:
        check_seal(root)
        neural, uncalibrated = predict_saved(frame, folder)
        info = json.loads((folder / "training.json").read_text(encoding="utf-8"))
        baselines = json.loads((folder / "baselines.json").read_text(encoding="utf-8"))
    output = frame.copy()
    predictions = {
        "neural": neural,
        "uncalibrated": uncalibrated,
        "constant": np.tile(baselines["constant"], (len(frame), 1)),
        "elo": elo_probabilities(frame.f_elo_difference, baselines["elo_log_draw"]),
        "poisson": frame[[f"b_poisson_{i}" for i in range(3)]].to_numpy(),
    }
    for model, values in predictions.items():
        for i in range(3):
            output[f"p_{model}_{i}"] = values[:, i]
    output.to_csv(root / "reports/predictions.csv", index=False)
    return output, info
