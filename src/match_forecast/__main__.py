import argparse
import json
from pathlib import Path

from .common import runtime


def run(root, train=False):
    root = runtime(root)
    from .data import prepare
    from .model import run_models
    from .analysis import evaluate, make_report

    matches, frame, audit = prepare(root)
    predictions, training = run_models(root, frame, train=train)
    metrics = evaluate(root, predictions)
    make_report(root, matches, predictions, audit, training, metrics)
    return matches, predictions, audit, training, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Historical prematch probabilities with TensorFlow"
    )
    parser.add_argument("command", choices=["train", "analyze", "predict"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--match-id", type=int)
    args = parser.parse_args()
    if args.command == "predict":
        root = runtime(args.root)
        import pandas as pd
        if not (root / "reports/final_predictions.csv").exists():
            run(root)
        frame = pd.read_csv(root / "reports/final_predictions.csv")
        selected = (
            frame[frame.match_id.eq(args.match_id)]
            if args.match_id is not None
            else frame[frame.split.eq("test")].tail(5)
        )
        if selected.empty:
            raise ValueError("Match ID not found in the evaluated 2019/20 season")
        result = selected[["match_id", "date", "home", "away"]].copy()
        result[["P_home", "P_draw", "P_away"]] = selected[["p_neural_0", "p_neural_1", "p_neural_2"]].to_numpy()
        print(result.to_string(index=False))
    else:
        run(args.root, train=args.command == "train")
        print((args.root / "reports/report.md").resolve())


if __name__ == "__main__":
    main()
