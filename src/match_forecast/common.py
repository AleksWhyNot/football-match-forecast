import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import rankdata


def runtime(root):
    root = Path(root).resolve()
    for key, folder in {
        "TEMP": "tmp",
        "TMP": "tmp",
        "TMPDIR": "tmp",
        "MPLCONFIGDIR": "matplotlib",
        "KERAS_HOME": "keras",
        "IPYTHONDIR": "ipython",
        "JUPYTER_DATA_DIR": "jupyter",
        "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
    }.items():
        path = root / ".runtime" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for key, value in {
        "KERAS_BACKEND": "tensorflow",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_DETERMINISTIC_OPS": "1",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }.items():
        os.environ[key] = value
    return root


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, path, digest=None, blob=None, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def valid(raw):
        if digest and hashlib.sha256(raw).hexdigest() != digest:
            return False
        return (
            not blob
            or hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
        )

    if path.exists():
        try:
            raw = (
                gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
            )
            if valid(raw):
                return raw
        except (OSError, EOFError):
            pass
    error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "football-research/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            if not valid(raw):
                raise ValueError(f"Source checksum mismatch: {url}")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(gzip.compress(raw, mtime=0) if compressed else raw)
            temporary.replace(path)
            return raw
        except Exception as exc:
            error = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"Cannot download {url}: {error}") from error


def fit_preprocess(frame, numeric, categorical):
    values = frame[numeric].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Infinite input feature")
    medians = np.array(
        [np.nanmedian(v) if np.isfinite(v).any() else 0 for v in values.T]
    )
    filled = np.where(np.isnan(values), medians, values)
    scale = filled.std(axis=0)
    scale[scale < 1e-8] = 1
    return {
        "numeric": numeric,
        "categorical": categorical,
        "median": medians.tolist(),
        "mean": filled.mean(axis=0).tolist(),
        "scale": scale.tolist(),
        "vocab": {
            c: sorted(frame[c].fillna("unknown").astype(str).unique().tolist())
            for c in categorical
        },
    }


def transform(frame, state):
    values = frame[state["numeric"]].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Infinite input feature")
    missing = np.isnan(values)
    filled = np.where(missing, state["median"], values)
    blocks = [(filled - state["mean"]) / state["scale"], missing.astype(float)]
    for column, vocabulary in state["vocab"].items():
        lookup = {value: i + 1 for i, value in enumerate(vocabulary)}
        indices = (
            frame[column]
            .fillna("unknown")
            .astype(str)
            .map(lookup)
            .fillna(0)
            .to_numpy(int)
        )
        blocks.append(np.eye(len(vocabulary) + 1)[indices])
    result = np.concatenate(blocks, axis=1).astype("float32")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite transformed input")
    return result


def probabilities(logits, temperature=1.0):
    return softmax(np.asarray(logits, float) / temperature, axis=1)


def losses(y, p):
    return -np.log(np.clip(p[np.arange(len(y)), np.asarray(y, int)], 1e-10, 1))


def scores(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if (
        not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid probability vector")
    onehot = np.eye(p.shape[1])[y]
    result = {
        "n": len(y),
        "log_loss": float(losses(y, p).mean()),
        "brier": float(np.square(p - onehot).sum(axis=1).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }
    if p.shape[1] == 2:
        positive = y == 1
        n1, n0 = positive.sum(), (~positive).sum()
        ranks = rankdata(p[:, 1])
        result["roc_auc"] = (
            float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            if n1 and n0
            else None
        )
        result["brier"] /= 2
        result["expected_goals"] = float(p[:, 1].sum())
        result["goals"] = int(y.sum())
    return result


def temperature_fit(logits, y):
    result = minimize_scalar(
        lambda logt: losses(y, probabilities(logits, np.exp(logt))).mean(),
        bounds=(-1.5, 1.5),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


def train_candidates(
    frame, numeric, categorical, config, folder, n_classes, widths=None
):
    import tensorflow as tf

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    train = frame.split.eq("train").to_numpy()
    validation = frame.split.eq("validation").to_numpy()
    calibration = frame.split.eq("calibration").to_numpy()
    if not all(mask.any() for mask in [train, validation, calibration]):
        raise ValueError("Empty model partition")
    state = fit_preprocess(frame.loc[train], numeric, categorical)
    x, y = transform(frame, state), frame.target.to_numpy(int)
    candidates, histories = [], []
    widths = widths or config["widths"]
    for index, shape in enumerate(widths):
        tf.keras.backend.clear_session()
        seed = config["seed"] + index
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        layers = [tf.keras.layers.Input((x.shape[1],))]
        for width in shape:
            layers += [
                tf.keras.layers.Dense(
                    width,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.001),
                ),
                tf.keras.layers.Dropout(0.1),
            ]
        layers.append(tf.keras.layers.Dense(n_classes))
        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        training = (
            tf.data.Dataset.from_tensor_slices((x[train], y[train]))
            .shuffle(train.sum(), seed=seed)
            .batch(256)
            .with_options(options)
        )
        val = (
            tf.data.Dataset.from_tensor_slices((x[validation], y[validation]))
            .batch(512)
            .with_options(options)
        )

        class ValidationLoss(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs["val_log_loss"] = float(
                    losses(
                        y[validation],
                        probabilities(
                            self.model(x[validation], training=False).numpy()
                        ),
                    ).mean()
                )

        history = model.fit(
            training,
            validation_data=val,
            epochs=config["epochs"],
            verbose=0,
            callbacks=[
                ValidationLoss(),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_log_loss",
                    patience=config["patience"],
                    mode="min",
                    restore_best_weights=True,
                ),
            ],
        )
        name = "dense_" + "_".join(map(str, shape)) if shape else "linear"
        model.save(folder / f"{name}.keras")
        val_loss = scores(
            y[validation], probabilities(model(x[validation], training=False).numpy())
        )["log_loss"]
        candidates.append(
            {
                "name": name,
                "widths": shape,
                "seed": seed,
                "parameters": model.count_params(),
                "validation_log_loss": val_loss,
                "best_epoch": int(np.argmin(history.history["val_log_loss"]) + 1),
            }
        )
        histories.extend(
            {
                "candidate": name,
                "epoch": i + 1,
                **{k: float(v[i]) for k, v in history.history.items()},
            }
            for i in range(len(history.history["loss"]))
        )
        print(f"{name}: validation log loss {val_loss:.5f}", flush=True)
    selected = min(candidates, key=lambda row: row["validation_log_loss"])
    model = tf.keras.models.load_model(
        folder / f"{selected['name']}.keras", compile=False
    )
    model.save(folder / "selected.keras")
    logits = model(x, training=False).numpy()
    state["temperature"] = temperature_fit(logits[calibration], y[calibration])
    state["n_classes"] = n_classes
    dump(folder / "preprocess.json", state)
    pd.DataFrame(histories).to_csv(folder / "history.csv", index=False)
    info = {
        "selected": selected,
        "candidates": candidates,
        "temperature": state["temperature"],
        "features": x.shape[1],
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
    }
    dump(folder / "training.json", info)
    return probabilities(logits, state["temperature"]), probabilities(logits), info


def predict_saved(frame, folder):
    import tensorflow as tf

    folder = Path(folder)
    state = json.loads((folder / "preprocess.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(folder / "selected.keras", compile=False)
    logits = model(transform(frame, state), training=False).numpy()
    return probabilities(logits, state["temperature"]), probabilities(logits)


def calibration_table(y, p, label):
    rows = []
    for klass in range(p.shape[1]):
        bins = np.minimum((p[:, klass] * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            if mask.any():
                rows.append(
                    {
                        "model": label,
                        "class": klass,
                        "bin": b,
                        "n": int(mask.sum()),
                        "mean_probability": float(p[mask, klass].mean()),
                        "frequency": float((np.asarray(y)[mask] == klass).mean()),
                    }
                )
    return pd.DataFrame(rows)


def paired_bootstrap(y, neural, baseline, groups, seed=20261102, repeats=1000):
    delta = losses(y, neural) - losses(y, baseline)
    table = (
        pd.DataFrame({"group": groups, "delta": delta})
        .groupby("group")
        .delta.agg(["sum", "count"])
    )
    rng = np.random.default_rng(seed)
    sums, counts = table["sum"].to_numpy(), table["count"].to_numpy()
    values = np.empty(repeats)
    for i in range(repeats):
        draw = rng.integers(0, len(table), len(table))
        values[i] = sums[draw].sum() / counts[draw].sum()
    return {
        "difference": float(delta.mean()),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "groups": len(table),
        "repeats": repeats,
    }


def seal(root, files):
    dump(
        Path(root) / "models/manifest.json",
        {str(name): sha256(Path(root) / name) for name in files},
    )


def check_seal(root):
    manifest = json.loads(
        (Path(root) / "models/manifest.json").read_text(encoding="utf-8")
    )
    for name, digest in manifest.items():
        if sha256(Path(root) / name) != digest:
            raise ValueError(f"Model/data/configuration mismatch: {name}")


def markdown_table(frame, digits=4):
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rows = [header, "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                (
                    f"{v:.{digits}f}"
                    if isinstance(v, (float, np.floating))
                    else str(v).replace("|", "/")
                )
                for v in row
            )
            + " |"
        )
    return "\n".join(rows)


def update_readme(root, table, conclusion):
    path = Path(root) / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        result = "\n" + markdown_table(table) + "\n\n" + conclusion + "\n"
        text = text.split(start)[0] + start + result + end + text.split(end)[1]
        path.write_text(text, encoding="utf-8", newline="\n")
