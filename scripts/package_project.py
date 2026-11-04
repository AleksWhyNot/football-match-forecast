from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    files = [
        root / name
        for name in [
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "requirements-lock.txt",
            ".gitignore",
            ".gitattributes",
            "run.ps1",
        ]
    ]
    for folder in [
        "config",
        "src",
        "reports",
        "models",
        "notebooks",
        "scripts",
        "tests",
        "assets",
        ".github",
    ]:
        for path in (root / folder).rglob("*"):
            if not path.is_file() or set(path.parts) & {
                "__pycache__",
                ".ipynb_checkpoints",
            }:
                continue
            if path.name in {"predictions.csv", "final_predictions.csv"} or path.name.startswith("oof_"):
                continue
            if path.suffix == ".keras" and path.name != "selected.keras" and "temporal" not in path.parts:
                continue
            files.append(path)
    target = root.parent / (root.name + ".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(
            "/data/" in n
            or "/.runtime/" in n
            or "/.venv/" in n
            or n.endswith(("/predictions.csv", "/final_predictions.csv"))
            for n in archive.namelist()
        )
    print(f"{target}: {len(files)} files, {target.stat().st_size/1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
