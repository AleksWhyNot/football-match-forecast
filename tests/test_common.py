import numpy as np
import pandas as pd
import pytest

from match_forecast.common import (
    fit_preprocess,
    transform,
    probabilities,
    scores,
    losses,
    temperature_fit,
    paired_bootstrap,
)


def test_preprocessor_uses_only_training_and_handles_unknowns():
    training = pd.DataFrame(
        {"f_value": [1.0, 3.0, np.nan], "c_league": ["A", "A", "B"]}
    )
    state = fit_preprocess(training, ["f_value"], ["c_league"])
    actual = transform(
        pd.DataFrame({"f_value": [1000.0, np.nan], "c_league": ["new", "B"]}), state
    )
    assert state["median"] == [2.0]
    assert state["mean"] == [2.0]
    assert actual[0, 2] == 1
    assert actual[1, 1] == 1
    assert np.isfinite(actual).all()


def test_softmax_stable_for_large_logits():
    p = probabilities(np.array([[10000.0, 10001.0, 9999.0]]))
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1)


def test_binary_metrics_agree_with_manual_calculation():
    y = np.array([1, 0, 1, 0])
    p = np.array([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
    out = scores(y, p)
    assert np.isclose(out["brier"], np.mean((p[:, 1] - y) ** 2))
    assert np.isclose(out["log_loss"], -np.log([0.8, 0.9, 0.6, 0.7]).mean())
    assert out["roc_auc"] == 1


def test_invalid_probabilities_fail():
    with pytest.raises(ValueError):
        scores(np.array([0]), np.array([[0.6, 0.6]]))


def test_temperature_reduces_calibration_loss():
    logits = np.array([[0.0, 3.0], [3.0, 0.0], [0.0, 3.0], [3.0, 0.0]] * 8)
    y = np.array([1, 0, 0, 1] * 8)
    t = temperature_fit(logits, y)
    assert t > 1
    assert (
        losses(y, probabilities(logits, t)).mean()
        < losses(y, probabilities(logits)).mean()
    )


def test_paired_bootstrap_identical_predictions_has_zero_interval():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]])
    result = paired_bootstrap(y, p, p, np.array([1, 1, 2, 2]), repeats=100)
    assert result["difference"] == result["low"] == result["high"] == 0


def test_all_missing_feature_can_be_encoded():
    data = pd.DataFrame({"f_value": [np.nan, np.nan]})
    state = fit_preprocess(data, ["f_value"], [])
    assert np.isfinite(transform(data, state)).all()


def test_saved_training_model_repeats_probabilities(tmp_path):
    from match_forecast.common import train_candidates, predict_saved

    rng = np.random.default_rng(5)
    data = pd.DataFrame(
        {
            "f_value": rng.normal(size=100),
            "c_league": ["A", "B"] * 50,
            "target": [0, 1] * 50,
            "split": ["train"] * 60
            + ["validation"] * 20
            + ["calibration"] * 10
            + ["test"] * 10,
        }
    )
    p, raw, info = train_candidates(
        data,
        ["f_value"],
        ["c_league"],
        {"seed": 5, "epochs": 2, "patience": 2, "widths": [[4]]},
        tmp_path,
        2,
    )
    again, again_raw = predict_saved(data, tmp_path)
    assert np.allclose(p, again, atol=1e-7)
    assert np.allclose(raw, again_raw, atol=1e-7)
    assert b"\r\n" not in (tmp_path / "preprocess.json").read_bytes()


def test_downloader_repairs_corrupt_gzip_cache(tmp_path, monkeypatch):
    import hashlib, io, gzip
    from match_forecast.common import download

    raw = b'{"verified": true}'
    path = tmp_path / "events.json.gz"
    path.write_bytes(b"broken cache")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(raw))
    actual = download(
        "https://example.org/events",
        path,
        digest=hashlib.sha256(raw).hexdigest(),
        compressed=True,
    )
    assert actual == raw and gzip.decompress(path.read_bytes()) == raw


def test_downloader_rejects_source_checksum_mismatch(tmp_path, monkeypatch):
    import io
    from match_forecast.common import download

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"wrong source")
    )
    monkeypatch.setattr("time.sleep", lambda *a: None)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download("https://example.org/events", tmp_path / "data.json", digest="0" * 64)
