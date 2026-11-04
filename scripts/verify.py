import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "match_forecast"
sys.path.insert(0, str(ROOT / "src"))
common = importlib.import_module(PACKAGE + ".common")
common.runtime(ROOT)


def main():
    import numpy as np

    common.check_seal(ROOT)
    data = importlib.import_module(PACKAGE + ".data")
    prepared = data.prepare(ROOT)
    frame = prepared[1] if PACKAGE == "match_forecast" else prepared[0]
    test = frame[frame.split.eq("test")]
    actual, _ = common.predict_saved(test, ROOT / "models")
    metrics = common.scores(test.target.to_numpy(), actual)
    saved = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    expected = next(
        row
        for row in saved["metrics"]
        if row["model"] == "neural" and row["group"] == "all"
    )
    for name, value in metrics.items():
        if value is not None:
            assert np.isclose(value, expected[name], rtol=1e-5, atol=1e-6), (
                name,
                value,
                expected[name],
            )
    print(json.dumps({"verified_test_rows": len(test), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
