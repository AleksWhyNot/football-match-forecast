"""Проверка сети с последовательным расширением обучающей выборки."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .temporal_methods import (
    CALIBRATORS, log_loss, normalize, fit_calibrator, apply_calibrator, history_before,
    joint_probabilities, joint_targets, fit_interval, apply_interval, interval_metrics,
)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    path = Path(path)
    content = (path.read_text(encoding="utf-8-sig").encode("utf-8")
               if path.suffix in {".py", ".json", ".csv", ".md", ".txt"} else path.read_bytes())
    return hashlib.sha256(content).hexdigest()


def temporal_frame(frame, plan):
    result = frame.copy()
    result["period"] = (pd.to_datetime(result.date).dt.strftime("%Y-%m")
                        if plan["kind"] == "xg" else result[plan["period_column"]].astype(int).astype(str))
    if plan["kind"] == "season":
        result["key"] = result.player_id.astype(str) + "_" + result.target_year.astype(str)
    else:
        result["key"] = result[plan["id_column"]].astype(str)
    if result.key.duplicated().any():
        raise ValueError("Duplicate observation key")
    return result


def split_training(frame, period, plan):
    past = history_before(frame, period, embargo_days=plan["embargo_days"])
    if plan["kind"] == "season":
        past = past.loc[past.split != "forecast"]
    periods = sorted(past.period.unique())
    if len(periods) < 2:
        raise ValueError("At least two complete historical periods are required")
    stop = past.loc[past.period == periods[-1]]
    train = past.loc[past.period < periods[-1]]
    future = frame.loc[frame.period == period].copy()
    if train.empty or stop.empty or future.empty:
        raise ValueError("Empty temporal partition")
    if not (train.period.max() < stop.period.min() <= past.period.max() < future.period.min()):
        raise ValueError("Temporal leakage")
    return train, stop, past, future


def fit_classification(train, stop, past, future, profile, seed, folder, plan):
    import tensorflow as tf
    from . import common
    numeric = [c for c in past if c.startswith("f_")]
    categorical = [c for c in past if c.startswith("c_")]
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1

    def dataset(x, y, shuffle=False):
        data = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            data = data.shuffle(len(x), seed=seed)
        return data.batch(256).with_options(options)

    def build(n):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        layers = [tf.keras.layers.Input((n,))]
        for width in profile["widths"]:
            layers += [tf.keras.layers.Dense(width, activation="relu",
                       kernel_regularizer=tf.keras.regularizers.L2(profile["l2"])),
                       tf.keras.layers.Dropout(profile["dropout"])]
        layers.append(tf.keras.layers.Dense(plan["classes"]))
        net = tf.keras.Sequential(layers)
        net.compile(optimizer=tf.keras.optimizers.Adam(.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        return net

    state = common.fit_preprocess(train, numeric, categorical)
    x, v = common.transform(train, state), common.transform(stop, state)
    y, vy = train.target.to_numpy(int), stop.target.to_numpy(int)
    net = build(x.shape[1])

    class Validation(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs["val_score"] = float(common.losses(vy, common.probabilities(self.model(v, training=False).numpy())).mean())

    history = net.fit(dataset(x, y, True), epochs=plan["epochs"], verbose=0,
                      callbacks=[Validation(), tf.keras.callbacks.EarlyStopping(
                          monitor="val_score", mode="min", patience=plan["patience"], restore_best_weights=True)])
    epochs = int(np.argmin(history.history["val_score"]) + 1)
    state = common.fit_preprocess(past, numeric, categorical)
    x = common.transform(past, state)
    net = build(x.shape[1])
    net.fit(dataset(x, past.target.to_numpy(int), True), epochs=epochs, verbose=0)
    prediction = common.probabilities(net(common.transform(future, state), training=False).numpy())
    net.save(folder / "model.keras")
    dump(folder / "preprocess.json", state)
    dump(folder / "epochs.json", {"epochs": epochs, "inner_validation_scores": history.history["val_score"]})
    return prediction, state, epochs


def fit_season(train, stop, past, future, profile, seed, folder, plan):
    from . import model as m
    tf, keras = m.tf, m.keras
    settings = {**profile, "learning_rate": .001}

    def build(state):
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        return m.build_model(m.transform_features(past.iloc[:1], state).shape[1], profile["widths"], settings)

    state = m.fit_preprocessor(train)
    x, v = m.transform_features(train, state), m.transform_features(stop, state)
    y, vy = m.encode_targets(train, state), m.encode_targets(stop, state)
    net = build(state)
    history = net.fit(m._dataset(x, y, 256, seed, True), epochs=plan["epochs"], verbose=0,
                      callbacks=[m.ValidationScore(v, vy), keras.callbacks.EarlyStopping(
                          monitor="val_score", mode="min", patience=plan["patience"], restore_best_weights=True)])
    epochs = int(np.argmin(history.history["val_score"]) + 1)
    state = m.fit_preprocessor(past)
    state["target_min_minutes"] = 450
    net = build(state)
    net.fit(m._dataset(m.transform_features(past, state), m.encode_targets(past, state), 256, seed, True),
            epochs=epochs, verbose=0)
    prediction = m.decode_predictions(future, m._raw_predictions(net, m.transform_features(future, state)), state)
    net.save(folder / "model.keras")
    dump(folder / "preprocess.json", state)
    dump(folder / "epochs.json", {"epochs": epochs, "inner_validation_scores": history.history["val_score"]})
    return prediction, state, epochs


def season_score(frame, state):
    values = []
    for target in ("observed", "eligible"):
        y = frame[f"target_{target}"].to_numpy(int)
        p = frame[f"pred_{target}"].to_numpy(float)
        values.append(float(log_loss(y, np.column_stack([1-p, p])).mean()))
    for target in ("minutes", "npxg90", "xa90"):
        y, p = frame[f"target_{target}"].to_numpy(float), frame[f"pred_{target}"].to_numpy(float)
        mask = np.isfinite(y)
        if target == "minutes":
            y, p = np.log1p(y), np.log1p(p)
        values.append(float(np.mean(((y[mask] - p[mask]) / state["targets"][target]["scale"]) ** 2)))
    return float(np.mean(values))


def training_signature():
    import inspect
    backend = importlib.import_module(__package__ + (".model" if __package__ == "football_forecast" else ".common"))
    code = "\n".join(inspect.getsource(function) for function in
                     (fit_classification, fit_season, split_training, temporal_frame, history_before))
    return hashlib.sha256((code + digest(backend.__file__)).encode()).hexdigest()


def predict_fold(root, frame, period, profile, plan):
    import tensorflow as tf
    future = frame.loc[frame.period == period].copy()
    folder = Path(root) / "models/temporal" / profile["name"] / period
    predictions = []
    for seed in plan["seeds"]:
        location = folder / f"seed_{seed}"
        state = json.loads((location / "preprocess.json").read_text(encoding="utf-8"))
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(location / "model.keras", compile=False)
        if plan["kind"] == "season":
            from . import model as m
            raw = m._raw_predictions(model, m.transform_features(future, state))
            predictions.append(m.decode_predictions(future, raw, state))
        else:
            from . import common
            predictions.append(common.probabilities(model(common.transform(future, state), training=False).numpy()))
    if plan["kind"] == "season":
        result = predictions[0].copy()
        for target in ("observed", "eligible", "minutes", "npxg90", "xa90"):
            column = "pred_" + target
            result[column] = np.mean([part[column].to_numpy(float) for part in predictions], axis=0)
    else:
        columns = [c for c in ["key", "match_id", "shot_id", "date", "season", "league", "target", "period", "split"] if c in future]
        result = future[columns].copy()
        probabilities = np.mean(predictions, axis=0)
        for k in range(plan["classes"]):
            result[f"p_raw_{k}"] = probabilities[:, k]
    result["profile"] = profile["name"]
    return result


def verify(root):
    root = Path(root).resolve()
    if __package__ == "football_forecast":
        from .environment import local_runtime
        local_runtime(root)
    else:
        from .common import runtime
        runtime(root)
    plan = json.loads((root / "config/temporal.json").read_text(encoding="utf-8"))
    frame = temporal_frame(pd.read_csv(root / plan["input_file"]), plan)
    fingerprint = hashlib.sha256((digest(root / plan["input_file"]) + json.dumps(plan, sort_keys=True)).encode()).hexdigest()
    results = []
    for file in sorted((root / "models/temporal").glob("*/*/complete.json")):
        info = json.loads(file.read_text(encoding="utf-8"))
        profile = next(p for p in plan["profiles"] if p["name"] == info["profile"])
        saved, _ = forecast_fold(root, frame, info["period"], profile, plan, fingerprint)
        rebuilt = predict_fold(root, frame, info["period"], profile, plan)
        assert saved.key.astype(str).tolist() == rebuilt.key.astype(str).tolist()
        columns = [c for c in saved if c.startswith(("p_raw_", "pred_"))]
        error = np.abs(saved[columns].to_numpy(float) - rebuilt[columns].to_numpy(float))
        if not np.allclose(saved[columns], rebuilt[columns], rtol=2e-6, atol=2e-7):
            raise AssertionError(f"Saved predictions differ: {profile['name']} {info['period']}")
        results.append({"profile": profile["name"], "period": info["period"], "max_absolute_difference": float(error.max())})
    dump(root / "reports/temporal/verification.json", {
        "folds": results, "models": len(results)*len(plan["seeds"]),
        "input_sha256": digest(root / plan["input_file"]), "training_signature": training_signature(),
    })
    print(f"Verified {len(results)} folds and {len(results)*len(plan['seeds'])} models", flush=True)


def forecast_fold(root, frame, period, profile, plan, fingerprint):
    folder = root / "models/temporal" / profile["name"] / period
    ready = folder / "complete.json"
    if ready.exists():
        info = json.loads(ready.read_text(encoding="utf-8"))
        if info["fingerprint"] != fingerprint:
            raise ValueError(f"Changed inputs for cached fold {period}")
        if info.get("training_signature") != training_signature():
            raise ValueError(f"Training implementation changed for cached fold {period}")
        for name, checksum in info["artifacts"].items():
            if name == "predictions.csv" and not (folder / name).exists():
                continue
            if digest(folder / name) != checksum:
                raise ValueError(f"Changed cached artifact {period}/{name}")
        if not (folder / "predictions.csv").exists():
            rebuilt = predict_fold(root, frame, period, profile, plan)
            rebuilt.to_csv(folder / "predictions.csv", index=False)
            info["artifacts"]["predictions.csv"] = digest(folder / "predictions.csv")
            dump(ready, info)
        return pd.read_csv(folder / "predictions.csv", dtype={"period": str, "key": str}), info

    train, stop, past, future = split_training(frame, period, plan)
    predictions, epochs = [], []
    started = time.monotonic()
    for seed in plan["seeds"]:
        output = folder / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        fit = fit_season if plan["kind"] == "season" else fit_classification
        predicted, state, best_epoch = fit(train, stop, past, future, profile, seed, output, plan)
        predictions.append(predicted)
        epochs.append(best_epoch)
    if plan["kind"] == "season":
        result = predictions[0].copy()
        for target in ("observed", "eligible", "minutes", "npxg90", "xa90"):
            column = "pred_" + target
            result[column] = np.mean([p[column].to_numpy(float) for p in predictions], axis=0)
        score = None if future.split.eq("forecast").all() else season_score(result, state)
    else:
        columns = [c for c in ["key", "match_id", "shot_id", "date", "season", "league", "target", "period", "split"] if c in future]
        result = future[columns].copy()
        averaged = np.mean(predictions, axis=0)
        for k in range(plan["classes"]):
            result[f"p_raw_{k}"] = averaged[:, k]
        score = float(log_loss(result.target, averaged).mean())
    result["profile"] = profile["name"]
    result.to_csv(folder / "predictions.csv", index=False)
    info = {
        "fingerprint": fingerprint, "training_signature": training_signature(), "period": period, "profile": profile["name"],
        "inner_train_periods": sorted(train.period.unique().tolist()),
        "inner_stop_period": str(stop.period.iloc[0]), "refit_periods": sorted(past.period.unique().tolist()),
        "n_inner_train": len(train), "n_inner_stop": len(stop), "n_refit": len(past), "n_forecast": len(future),
        "seeds": plan["seeds"], "best_epochs": epochs, "selection_score": score,
        "artifacts": {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*")
                      if p.is_file() and p.name != "complete.json"},
    }
    dump(ready, info)
    print(f"{profile['name']} {period}: train={len(past)}, predict={len(future)}, epochs={epochs}, {time.monotonic()-started:.1f}s", flush=True)
    return result, info


def get_probabilities(frame, plan):
    if plan["kind"] == "season":
        return joint_probabilities(frame), joint_targets(frame)
    return frame[[f"p_raw_{k}" for k in range(plan["classes"])]].to_numpy(float), frame.target.to_numpy(int)


def choose_calibration(frame, plan):
    records = []
    periods = sorted(frame.period.unique())
    windows = [("expanding", None), ("recent", plan["recent_periods"])]
    for period in periods[1:]:
        future = frame.loc[frame.period == period]
        p, y = get_probabilities(future, plan)
        for window, count in windows:
            past = history_before(frame, period, count, plan["embargo_days"])
            pp, yy = get_probabilities(past, plan)
            for method in CALIBRATORS:
                state = fit_calibrator(pp, yy, method)
                score = float(log_loss(y, apply_calibrator(p, state)).mean())
                records.append({"period": period, "window": window, "method": method,
                                "n": len(future), "n_calibration": len(past), "log_loss": score})
    results = pd.DataFrame(records)
    ranked = []
    for (window, method), part in results.groupby(["window", "method"], sort=False):
        ranked.append({"window": window, "method": method,
                       "log_loss": float(np.average(part.log_loss, weights=part.n)),
                       "periods": part.period.tolist()})
    order = {name: i for i, name in enumerate(CALIBRATORS)}
    selected = min(ranked, key=lambda r: (r["log_loss"], order[r["method"]], r["window"] != "expanding"))
    return selected, results, pd.DataFrame(ranked)


def choose_intervals(frame, plan):
    records = []
    for target in ("minutes", "npxg90", "xa90"):
        for period in sorted(frame.period.unique())[1:]:
            future = frame.loc[frame.period == period]
            for window, count in [("expanding", None), ("recent", plan["recent_periods"])]:
                past = history_before(frame, period, count)
                for method in ("absolute", "relative"):
                    state = fit_interval(past, target, method)
                    lo, hi = apply_interval(future[f"pred_{target}"], state)
                    records.append({"target": target, "period": period, "window": window, "method": method,
                                    **interval_metrics(future[f"target_{target}"], lo, hi)})
    detail = pd.DataFrame(records)
    selected, rankings = {}, []
    for target in ("minutes", "npxg90", "xa90"):
        choices = []
        for (window, method), part in detail.loc[detail.target == target].groupby(["window", "method"], sort=False):
            row = {"target": target, "window": window, "method": method}
            for key in ("coverage", "mean_width", "interval_score"):
                row[key] = float(np.average(part[key], weights=part.n))
            choices.append(row)
        # При достаточном покрытии выбираем меньший interval score.
        good = [row for row in choices if row["coverage"] >= .9]
        selected[target] = min(good, key=lambda r: r["interval_score"]) if good else max(
            choices, key=lambda r: (r["coverage"], -r["interval_score"]))
        rankings.extend(choices)
    return selected, detail, pd.DataFrame(rankings)


def correct_future(pool, future, selection, plan):
    policy = selection["calibration"]
    count = None if policy["window"] == "expanding" else plan["recent_periods"]
    period = str(future.period.iloc[0])
    history = history_before(pool, period, count, plan["embargo_days"])
    pp, yy = get_probabilities(history, plan)
    p, _ = get_probabilities(future, plan)
    state = fit_calibrator(pp, yy, policy["method"])
    adjusted = apply_calibrator(p, state)
    result = future.copy()
    info = {"period": period, "calibration_periods": sorted(history.period.unique().tolist()),
            "calibrator": state, "intervals": {}}
    if plan["kind"] == "season":
        result["cal_observed"] = adjusted[:, 1] + adjusted[:, 2]
        result["cal_eligible"] = adjusted[:, 2]
        for target, policy in selection["intervals"].items():
            count = None if policy["window"] == "expanding" else plan["recent_periods"]
            history = history_before(pool, period, count)
            state = fit_interval(history, target, policy["method"])
            result["lo90_" + target], result["hi90_" + target] = apply_interval(result["pred_" + target], state)
            info["intervals"][target] = {**state, "periods": sorted(history.period.unique().tolist())}
    else:
        for k in range(plan["classes"]):
            result[f"p_selected_{k}"] = adjusted[:, k]
    return result, info


def run(root):
    root = Path(root).resolve()
    if __package__ == "football_forecast":
        from .environment import local_runtime
        local_runtime(root)
    else:
        from .common import runtime
        runtime(root)
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    plan = json.loads((root / "config/temporal.json").read_text(encoding="utf-8"))
    input_path = root / plan["input_file"]
    frame = temporal_frame(pd.read_csv(input_path), plan)
    fingerprint = hashlib.sha256((digest(input_path) + json.dumps(plan, sort_keys=True)).encode()).hexdigest()
    output = root / "reports/temporal"
    output.mkdir(parents=True, exist_ok=True)
    periods = sorted(p for p in frame.period.unique() if p >= plan["first_oof"])
    development = [p for p in periods if p < plan["first_test"]]
    testing = [p for p in periods if p >= plan["first_test"]]
    all_oof, folds = {}, []
    for profile in plan["profiles"]:
        parts = []
        for period in development:
            forecast, info = forecast_fold(root, frame, period, profile, plan, fingerprint)
            parts.append(forecast)
            folds.append(info)
        all_oof[profile["name"]] = pd.concat(parts, ignore_index=True)
    cv = pd.DataFrame([{k:v for k,v in r.items() if k not in ("artifacts","fingerprint")} for r in folds])
    eligible = cv.loc[cv.period != development[0]]
    rank = eligible.groupby("profile").selection_score.mean()
    profile_name = str(rank.idxmin())
    profile = next(p for p in plan["profiles"] if p["name"] == profile_name)
    pool = all_oof[profile_name]
    calibration, details, ranking = choose_calibration(pool, plan)
    selection = {"profile": profile, "profile_validation_scores": rank.to_dict(), "calibration": calibration,
                 "validation_periods": development[1:], "first_test": plan["first_test"],
                 "fingerprint": fingerprint, "test_status": "previously_inspected_retrospective_evaluation"}
    if plan["kind"] == "season":
        policies, interval_detail, interval_ranking = choose_intervals(pool, plan)
        selection["intervals"] = policies
        interval_detail.to_csv(output / "interval_validation_folds.csv", index=False)
        interval_ranking.to_csv(output / "interval_validation.csv", index=False)
    dump(root / "models/temporal/selection.json", selection)
    print("Selected before test: " + json.dumps(selection, ensure_ascii=False), flush=True)
    details.to_csv(output / "calibration_validation_folds.csv", index=False)
    ranking.to_csv(output / "calibration_validation.csv", index=False)
    for name, part in all_oof.items():
        part.to_csv(output / f"oof_{name}.csv", index=False)
    final, calibration_states = [], []
    for period in testing:
        future, info = forecast_fold(root, frame, period, profile, plan, fingerprint)
        corrected, state = correct_future(pool, future, selection, plan)
        corrected["evaluation"] = "forecast" if future.split.eq("forecast").all() else "test"
        final.append(corrected)
        folds.append(info)
        calibration_states.append(state)
        if not future.split.eq("forecast").all():
            pool = pd.concat([pool, future], ignore_index=True)
    predictions = pd.concat(final, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    dump(root / "models/temporal/calibration.json", calibration_states)
    dump(output / "folds.json", folds)
    dump(output / "selection.json", selection)
    from .temporal_report import make_report
    make_report(root, plan, predictions, selection, folds)
    return predictions, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify(args.root) if args.verify else run(args.root)


if __name__ == "__main__":
    main()

