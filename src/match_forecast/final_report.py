"""Расчёт таблиц и графиков для итоговой модели проекта."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEAGUES = {
    "EPL": "АПЛ",
    "La_Liga": "Ла Лига",
    "Bundesliga": "Бундеслига",
    "Serie_A": "Серия А",
    "Ligue_1": "Лига 1",
    "RFPL": "РПЛ",
}
TARGETS = {"minutes": "Минуты", "npxg90": "npxG / 90", "xa90": "xA / 90"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(frame, digits=4):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}" if np.isfinite(v) else "—"
        return str(v).replace("|", "/").replace("\n", " ")

    rows = [
        list(frame.columns),
        ["---"] * len(frame.columns),
        *frame.itertuples(index=False, name=None),
    ]
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows)


def navigation(content):
    links = []

    def heading(m):
        key = f"section-{len(links)+1}"
        links.append(f"- [{m[1]}](#{key})")
        return f'<a id="{key}"></a>\n\n## {m[1]}'

    content = re.sub(r"(?m)^## (.+)$", heading, content)
    index = content.index("<a id=")
    return (
        content[:index]
        + "<!-- CONTENTS:START -->\n## Содержание\n\n"
        + "\n".join(links)
        + "\n<!-- CONTENTS:END -->\n\n"
        + content[index:]
    )


def bootstrap(delta, groups, seed=20261104):
    _, index = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    counts = np.bincount(index)
    sums = np.bincount(index, weights=np.asarray(delta, float))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(30):
        sample = rng.integers(0, len(counts), size=(100, len(counts)))
        draws.extend(sums[sample].sum(axis=1) / counts[sample].sum(axis=1))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "difference": float(np.mean(delta)),
        "low": float(lo),
        "high": float(hi),
        "clusters": len(counts),
    }


def figure(root, fig, name):
    if KIND == "xg":
        fig.subplots_adjust(bottom=0.2)
        fig.text(
            0.015, 0.02, "StatsBomb Open Data · 2015/16", fontsize=8, color="#52616b"
        )
        ax = fig.add_axes([0.8, 0.01, 0.18, 0.09])
        ax.imshow(plt.imread(root / "assets/statsbomb-logo.png"))
        ax.axis("off")
    fig.savefig(
        root / "reports/figures" / ("final_" + name + ".png"),
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def chosen_predictions(root):
    config = load(root / "config/report_model.json")
    if config["probabilities"] != "raw" or config["kind"] != KIND:
        raise ValueError("Unsupported report model configuration")
    if KIND == "xg":
        frame = pd.read_csv(root / "reports/predictions.csv")
        frame = frame.loc[frame.split.eq("test")].copy()
        frame["p_neural"] = frame.p_uncalibrated
        return frame
    current = root / "reports/temporal/predictions.csv"
    if not current.exists():
        from .temporal import run

        run(root)
    temporal = pd.read_csv(current, dtype={"key": str, "period": str})
    if KIND == "season":
        frame = temporal.loc[temporal.evaluation.isin(["test", "forecast"])].copy()
        assert frame.profile.eq(config["profile"]).all()
        assert not frame.duplicated(["player_id", "target_year"]).any()
        return frame
    original = pd.read_csv(root / "reports/predictions.csv")
    frame = original.loc[original.split.eq("test")].merge(
        temporal.loc[
            temporal.evaluation.eq("test"),
            ["match_id", "target", "profile", "p_raw_0", "p_raw_1", "p_raw_2"],
        ],
        on="match_id",
        suffixes=("", "_check"),
        validate="one_to_one",
    )
    assert len(frame) == int(original.split.eq("test").sum())
    assert (
        frame.target.eq(frame.target_check).all()
        and frame.profile.eq(config["profile"]).all()
    )
    for k in range(3):
        frame[f"p_neural_{k}"] = frame[f"p_raw_{k}"]
    return frame


def class_results(root, frame):
    from .common import scores, calibration_table, losses

    names = (
        {
            "constant": "Частота голов",
            "geometry": "Геометрия",
            "neural": "Нейросеть",
            "provider": "StatsBomb xG",
        }
        if KIND == "xg"
        else {
            "constant": "Частоты исходов",
            "elo": "Elo",
            "poisson": "Пуассон по xG",
            "neural": "Нейросеть",
        }
    )
    y = frame.target.to_numpy(int)
    probabilities = {
        name: (
            np.column_stack([1 - frame["p_" + name], frame["p_" + name]])
            if KIND == "xg"
            else frame[[f"p_{name}_{k}" for k in range(3)]].to_numpy(float)
        )
        for name in names
    }
    rows = []
    reliability = []
    comparisons = []
    clusters = (
        frame.match_id
        if KIND == "xg"
        else pd.to_datetime(frame.date).dt.to_period("W").astype(str)
    )
    for name, p in probabilities.items():
        rows.append({"model": name, "group": "all", **scores(y, p)})
        for column in ["league", "c_body_part"] if KIND == "xg" else ["league"]:
            for group in sorted(frame[column].unique()):
                mask = frame[column].eq(group).to_numpy()
                rows.append(
                    {"model": name, "group": str(group), **scores(y[mask], p[mask])}
                )
        if KIND == "match":
            for group, mask in [
                ("До перерыва", frame.date.lt("2020-03-15")),
                ("После перерыва", frame.date.ge("2020-05-15")),
            ]:
                rows.append({"model": name, "group": group, **scores(y[mask], p[mask])})
        reliability.append(calibration_table(y, p, name))
        if name != "neural":
            delta = losses(y, probabilities["neural"]) - losses(y, p)
            comparisons.append({"baseline": name, **bootstrap(delta, clusters)})
    metrics = pd.DataFrame(rows)
    reliability = pd.concat(reliability, ignore_index=True)
    comparisons = pd.DataFrame(comparisons)
    for name, data in [
        ("model_metrics", metrics),
        ("probability_check", reliability),
        ("model_comparison", comparisons),
    ]:
        data.to_csv(root / f"reports/tables/final_{name}.csv", index=False)
    overall = metrics.loc[metrics.group.eq("all")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ["log_loss", "brier"]):
        bars = ax.barh(
            [names[k] for k in overall.model],
            overall[metric],
            color=["#187d82" if k == "neural" else "#8b9dab" for k in overall.model],
        )
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set(xlabel=metric, xlim=(0, overall[metric].max() * 1.22))
        ax.invert_yaxis()
    fig.tight_layout()
    figure(root, fig, "model_quality")
    classes = [1] if KIND == "xg" else [0, 1, 2]
    fig, axes = plt.subplots(
        1, len(classes), figsize=(6 if KIND == "xg" else 12, 4.2), squeeze=False
    )
    for k, ax in zip(classes, axes[0]):
        for name in ["neural", "provider"] if KIND == "xg" else ["neural", "poisson"]:
            r = reliability.loc[reliability.model.eq(name) & reliability["class"].eq(k)]
            dense = r.loc[r.n.ge(30)]
            sparse = r.loc[r.n.lt(30)]
            ax.plot(dense.mean_probability, dense.frequency, "o-", label=names[name])
            ax.scatter(
                sparse.mean_probability,
                sparse.frequency,
                facecolors="none",
                edgecolors="#62717b",
            )
        ax.plot([0, 1], [0, 1], "--", color="#aaa")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Прогноз вероятности",
            ylabel="Частота исхода",
            title="Гол" if KIND == "xg" else ["П1", "Ничья", "П2"][k],
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    figure(root, fig, "probability_check")
    if KIND == "match":
        confusion = np.zeros((3, 3), int)
        np.add.at(confusion, (y, probabilities["neural"].argmax(axis=1)), 1)
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        ax.imshow(
            confusion / confusion.sum(axis=1, keepdims=True),
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(confusion[i, j]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if confusion[i, j] / confusion[i].sum() > 0.55
                        else "#22333b"
                    ),
                )
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=["П1", "Х", "П2"],
            yticklabels=["П1", "Х", "П2"],
            xlabel="Прогноз",
            ylabel="Факт",
        )
        fig.tight_layout()
        figure(root, fig, "confusion")
    result = {
        "metrics": metrics.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
    }
    return result, names, metrics, comparisons


def training_window(root):
    folds = load(root / "reports/temporal/folds.json")
    rows = [
        {
            "Период": f["period"],
            "Обучение": f["n_refit"],
            "Прогноз": f["n_forecast"],
            "Эпохи двух сетей": " / ".join(map(str, f["best_epochs"])),
        }
        for f in folds
        if f["profile"] == "regularized"
    ]
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(frame["Период"], frame["Обучение"], "o-", color="#187d82")
    ax.set(xlabel="Год начала прогнозируемого сезона", ylabel="Наблюдений в обучении")
    fig.tight_layout()
    figure(root, fig, "training_window")
    return frame


def write_outputs(root, body, summary, frame, short):
    dictionary = DICTIONARY
    dictionary = dictionary.replace("после калибровки", "итоговой модели")
    body += "\n\n## Словарь данных\n\n" + dictionary
    body += (
        "\n\nИтоговые прогнозы хранятся в `reports/final_predictions.csv`. Метрики и параметры вывода — в `reports/final_metrics.json`, таблицы — в файлах с префиксом `final_` папки `reports/tables`.\n\n## Источники\n\n"
        + SOURCES
    )
    body += f"""\n\n## Воспроизведение

Python 3.12. Команды выполняются из корня проекта.

```bash
python -m pip install -e ".[dev]"
python -m {__package__} analyze
python -m {__package__}.final_report
```

Ноутбук `notebooks/analysis.ipynb` повторяет подготовку данных, расчёт итоговых прогнозов, метрик и графиков. Выбранная схема записана в `config/report_model.json`. Сохранённые веса и параметры преобразования находятся в `models`; версии и контрольные суммы источников — в `config/sources.json`.
"""
    (root / "reports/report.md").write_text(
        navigation(body.strip() + "\n"), encoding="utf-8"
    )
    (root / "reports/final_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    unused = [
        c
        for c in frame
        if c.startswith(("p_uncalibrated", "cal_", "p_selected_", "p_raw_"))
        or c in {"target_check", "evaluation", "profile"}
    ]
    frame.drop(columns=unused).to_csv(
        root / "reports/final_predictions.csv", index=False
    )
    intro = body.split("\n## ", 1)[0]
    readme = (
        intro
        + "\n\n"
        + short
        + f"""\n\n[Отчёт](reports/report.md) · [PDF](reports/report.pdf) · [Ноутбук](notebooks/analysis.ipynb)

## Запуск

```bash
python -m pip install -e ".[dev]"
python -m {__package__} analyze
```

Итоговая модель, данные, методика и результаты описаны в отчёте. Его таблицы и графики пересчитываются командой `python -m {__package__}.final_report`.
"""
    )
    (root / "README.md").write_text(readme, encoding="utf-8")
    return summary


def build(root):
    root = Path(root).resolve()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    (root / "reports/tables").mkdir(parents=True, exist_ok=True)
    (root / "reports/figures").mkdir(exist_ok=True)
    frame = chosen_predictions(root)
    if KIND == "season":
        return season_report(root, frame)
    result, names, metrics, comparisons = class_results(root, frame)
    return (
        xg_report(root, frame, result, names, metrics, comparisons)
        if KIND == "xg"
        else match_report(root, frame, result, names, metrics, comparisons)
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    build(args.root)


KIND = "match"

DICTIONARY = r"""| Поле | Значение и доступность |
|---|---|
| match_id, date, league, season | Ключ, дата, лига, год начала сезона |
| home_id, away_id, home, away | Команды; ключи для истории, не входы сети |
| home_goals, away_goals, home_xg, away_xg | Итог текущего матча; используются только после его даты |
| target | 0=П1, 1=Х, 2=П2 |
| f_home/away_*_5, *_10 | Средние последних 5/10 предыдущих матчей |
| f_home/away_history | Число предыдущих матчей, ограниченное 100 |
| f_home/away_rest | Дни отдыха, ограниченные 60; неизвестно при отсутствии истории |
| f_home/away_matches_14d | Число прошлых игр за 14 дней |
| f_home/away_venue_xgf/xga | Сглаженный прошлый xG дома/в гостях |
| f_elo_difference | Разность рейтингов с домашней поправкой |
| f_poisson_home/away | Интенсивности голов простого ориентира |
| c_league | Категория лиги |
| p_neural_0/1/2 | Вероятности П1/Х/П2 после калибровки |
| split | train, validation, calibration, test |

`forecast.w/d/l` из источника исключены. Префиксы `f_` и `c_` определяют список входов;
`b_` и `p_` содержат результаты ориентиров и не попадают в подготовку входов автоматически."""

SOURCES = r"""- [Understat](https://understat.com/) — первоначальный поставщик xG и результатов.
- [Markjwilkins/Understat](https://github.com/Markjwilkins/Understat), коммит `4b42df35599f574c08ac65caacaf7975ed5fa138`:
  30 таблиц в папке Matches, пять лиг за 2014–2019 годы начала сезона. Данные загружаются из публичного зеркала.
- [Keras: сохранение модели](https://keras.io/guides/serialization_and_saving/).
- [TensorFlow Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

Получено 2 ноября 2026 года. Адреса и SHA-256 каждого CSV закреплены в `config/sources.json`.
В зеркале не обнаружена отдельная лицензия на распространение данных: в Git и ZIP включён загрузчик,
а исходные CSV и построчные промежуточные данные остаются локальными. Выгрузка содержит поздний снимок данных; историю исправлений поставщика проверить по нему невозможно.

- [Hyndman и Athanasopoulos. Временная кросс-валидация](https://otexts.com/fpp3/tscv.html)."""


def match_report(root, frame, result, names, metrics, comparisons):
    audit = load(root / "reports/data_quality.json")
    window = training_window(root)
    overall = metrics.loc[
        metrics.group.eq("all"), ["model", "n", "log_loss", "brier", "accuracy"]
    ].replace({"model": names})
    overall.columns = ["Модель", "Матчи", "Log loss", "Brier", "Accuracy"]
    comp = comparisons[["baseline", "difference", "low", "high"]].replace(
        {"baseline": names}
    )
    comp.columns = ["Ориентир", "Разница log loss", "Нижняя 95%", "Верхняя 95%"]
    league = metrics.loc[
        metrics.model.eq("neural") & metrics.group.isin(audit["leagues"]),
        ["group", "n", "log_loss", "brier", "accuracy"],
    ].replace({"group": LEAGUES})
    league.columns = ["Лига", "Матчи", "Log loss", "Brier", "Accuracy"]
    periods = metrics.loc[
        metrics.group.isin(["До перерыва", "После перерыва"])
        & metrics.model.isin(["neural", "elo", "poisson"]),
        ["model", "group", "n", "log_loss", "brier"],
    ].replace({"model": names})
    periods.columns = ["Модель", "Период", "Матчи", "Log loss", "Brier"]
    examples = frame.sort_values(["date", "match_id"]).tail(8)[
        ["date", "home", "away", "p_neural_0", "p_neural_1", "p_neural_2"]
    ]
    examples.columns = ["Дата", "Хозяева", "Гости", "П1", "Ничья", "П2"]
    examples.to_csv(root / "reports/tables/final_examples.csv", index=False)
    teamrows = []
    for side, win in [("home", 0), ("away", 2)]:
        teamrows.append(
            pd.DataFrame(
                {
                    "team": frame[side],
                    "league": frame.league,
                    "actual_points": np.where(
                        frame.target.eq(win), 3, np.where(frame.target.eq(1), 1, 0)
                    ),
                    "expected_points": 3 * frame[f"p_neural_{win}"] + frame.p_neural_1,
                }
            )
        )
    teams = (
        pd.concat(teamrows)
        .groupby(["league", "team"])
        .agg(
            matches=("actual_points", "size"),
            points=("actual_points", "sum"),
            expected=("expected_points", "sum"),
        )
        .reset_index()
    )
    teams["difference"] = teams.points - teams.expected
    teams.to_csv(root / "reports/tables/final_team_points.csv", index=False)
    teamview = teams.loc[
        teams.difference.abs().nlargest(8).index,
        ["team", "matches", "points", "expected", "difference"],
    ]
    teamview.columns = ["Команда", "Матчи", "Очки", "Ожидание", "Разница"]
    chosen = metrics.loc[metrics.model.eq("neural") & metrics.group.eq("all")].iloc[0]
    draw_count = int(
        (
            frame[["p_neural_0", "p_neural_1", "p_neural_2"]].to_numpy().argmax(axis=1)
            == 1
        ).sum()
    )
    short = f"Log loss {chosen.log_loss:.5f}, Brier {chosen.brier:.5f}, accuracy {chosen.accuracy:.1%} на {len(frame):,} матчах сезона 2019/20."
    body = f"""# Вероятности исходов футбольных матчей

Нейросеть TensorFlow/Keras прогнозирует победу хозяев, ничью и победу гостей до начала матча. Используется архив Understat по пяти лигам за 2014/15–2019/20. Модель ежегодно обучается заново на всей завершённой истории.

## Задача и время прогноза

Для каждой встречи нужны три вероятности, сумма которых равна единице. Они позволяют учитывать неопределённость исхода и рассчитывать ожидаемые очки команд. Прогноз строится на начало календарного дня. Результаты встреч этого дня добавляются в историю со следующей даты.

Единица наблюдения — сыгранный матч. Целевая переменная равна 0 при победе хозяев, 1 при ничьей и 2 при победе гостей. Все матчи одного дня используют одинаковую границу доступной информации.

## Источник и подготовка данных

Первичный поставщик — Understat; таблицы получены из закреплённой версии публичного зеркала Mark Wilkins. В анализ входят {audit['matches']:,} сыгранных матчей АПЛ, Ла Лиги, Бундеслиги, Серии А и Лиги 1. Период простирается с {audit['first_date']} по {audit['last_date']}.

Исключена {audit['excluded_unplayed']} строка без сыгранного результата, преимущественно отменённые встречи Лиги 1. Проверяются уникальность ID, конечные неотрицательные голы и xG, даты и состав лиг. В архиве {audit['cold_start_matches']} матчей хотя бы с одной командой, имеющей менее пяти предыдущих игр в доступной истории.

Текущие голы и xG используются после завершения матча. Готовые `forecast.w/d/l` поставщика исключены из входов, поскольку их доступность до игры не подтверждена. Даты читаются в календарной шкале источника. Архив является поздним снимком; историю исправлений xG и дат по нему восстановить нельзя.

## Лиги и домашнее преимущество

![Домашнее преимущество](figures/home_advantage.png)

Сравниваются доля домашних побед и разница голов хозяев и гостей на матч. Показатели рассчитываются отдельно для лиги и сезона. Такое представление учитывает разные размеры чемпионатов. Сезон Лиги 1 2019/20 завершён досрочно; для него учитываются сыгранные встречи.

Графики описывают результативность дома и в гостях. На межсезонные различия влияют состав клубов, календарь и условия проведения матчей. Все агрегаты доступны в [таблице лиг и сезонов](tables/league_seasons.csv).

## Признаки до начала матча

Для хозяев и гостей используются средние голов, xG за и против и очков за последние 5 и 10 встреч. Дополняют их число дней отдыха, число матчей за 14 дней, домашняя и гостевая история и рейтинг Elo. Число предыдущих встреч ограничено 100, отдых — 60 днями. Отдельно сохраняется доступность истории.

История привязана к лиге и ID клуба. Названия и идентификаторы служат соединению записей; лига входит в сеть как категория. В матрице нет будущих итогов сезона и результата прогнозируемой игры. Тесты изменяют текущие и будущие результаты и проверяют неизменность более ранних признаков.

Пустая история заполняется медианами обучения и получает индикатор пропуска. Масштабы числовых признаков и категории лиг определяются заново внутри каждого обучающего окна. Для нового клуба Elo начинается с 1500.

## Нейросеть и расширяющееся окно

Итоговая модель усредняет вероятности двух сетей с одинаковыми скрытыми слоями 32/16 и разными seed — 20261104 и 20261121. Используются ReLU, L2=0.005 и Dropout=0.2. Три выходных логита преобразуются softmax в П1/Х/П2. Adam работает с learning rate 0.001, размер пакета 256, функция потерь — cross-entropy.

Перед началом сезона последний завершённый сезон временно выделяется для ранней остановки. На более ранних сезонах оцениваются преобразования и веса; остановка отслеживает log loss, максимум 100 эпох, patience=10. Затем сеть и преобразования обучаются с нуля на всей доступной истории с выбранным числом эпох. Две сети получают собственное число эпох.

![История обучения](figures/final_training_window.png)

{table(window,0)}

Проверочные сезоны 2017/18 и 2018/19 используют только более ранние результаты. Для оценки 2019/20 сеть обучена на 9 130 матчах сезонов 2014/15–2018/19. Во время прогнозируемого сезона веса фиксированы, а входная история команд пополняется после завершённых дней. Вероятности берутся из среднего softmax двух сетей.

## Простые ориентиры

Постоянный прогноз использует частоты П1/Х/П2. Elo обновляется после игрового дня с K=20 и домашней поправкой 70. На границе сезона сохраняется 75% отклонения рейтинга от 1500. Частоты исходов и параметр ничьей Elo оценены на сезонах 2014/15–2016/17; сам рейтинг продолжает обновляться по завершённым матчам.

Пуассоновский ориентир берёт геометрическое среднее прошлого xG атаки команды и xG обороны соперника с учётом дома/выезда. Командные средние сглаживаются пятью условными матчами к прошлому среднему лиги. Начальная лига содержит 20 условных матчей с 1.4 домашнего и 1.1 гостевого xG. Интенсивности ограничены диапазоном [0.1; 6], счёт перебирается от 0 до 30 голов, затем вероятности нормируются. Голы команд в этом ориентире предполагаются независимыми.

## Качество прогноза

Основная метрика — log loss. Brier равен средней сумме квадратов ошибок трёх вероятностей. Accuracy — доля матчей, в которых наиболее вероятный исход совпал с фактом. Все модели оцениваются на одинаковых {len(frame):,} матчах.

{table(overall,5)}

![Сравнение с ориентирами](figures/final_model_quality.png)

{table(league,5)}

Разрезы показывают качество отдельно по каждой лиге. При сравнении важны число матчей, баланс исходов и полнота истории новых команд.

## Вероятности и ошибки исходов

![Вероятности и частоты](figures/final_probability_check.png)

Для каждого исхода прогнозы разбиты на интервалы шириной 0.1. Средняя вероятность сопоставлена с долей фактических исходов. Полые точки обозначают интервалы с менее чем 30 матчами. Это проверка вероятностей итоговой сети.

![Распределение ошибок](figures/final_confusion.png)

Ничья оказалась наиболее вероятным исходом в {draw_count} матчах из {len(frame)}. Матрица отражает решение по максимуму вероятности; log loss использует весь вектор и чувствителен к уверенности прогноза.

## Неопределённость результата

{table(comp,5)}

Разница равна log loss нейросети минус log loss ориентира. 95% интервалы рассчитаны 3 000 повторами парного бутстрэпа по календарным неделям. Все встречи одной недели остаются в одном блоке. Связи повторных игр клубов между неделями могут сохраняться; поправка на несколько сравнений не вводилась.

## Перерыв в сезоне 2019/20

{table(periods,5)}

Первая группа включает даты до 15 марта 2020 года, вторая — с 15 мая. Лига 1 сезон не возобновила, поэтому состав лиг во второй группе отличается. Разрез показывает устойчивость ошибок в двух частях календаря; для объяснения различий потребовались бы посещаемость, составы и травмы.

## Прогнозы матчей и ожидаемые очки

{table(examples,3)}

Показаны исторические прогнозы последних матчей архива. Например, вероятность 0.6 означает оценку 60% для соответствующего исхода при доступной модели информации. Для каждой команды ожидаемые очки в матче равны 3 × P(победа) + P(ничья).

{table(teamview,2)}

В таблице восемь наибольших абсолютных отклонений фактических очков от суммы предматчевых ожиданий. Размер выборки ограничен матчами тестового сезона. Такое отклонение может отражать неожиданные результаты и ошибки прогноза. Полная [таблица команд](tables/final_team_points.csv) содержит все клубы.

## Выводы

{short} Парные интервалы показывают преимущество нейросети над частотным, пуассоновским и Elo-ориентирами на этом наборе матчей. Усреднение двух небольших сетей и последовательное расширение обучения позволяют использовать историю нескольких сезонов при сохранении временного порядка.

Оценка ретроспективная: итоговая конфигурация зафиксирована с учётом результатов на этом архиве. Независимое подтверждение требует следующего сезона. Для дальнейшего повышения качества полезны составы, травмы и условия проведения встреч; текущие признаки ограничены прошлой командной статистикой.
"""
    result.update(
        {
            "model": "dense_32_16_ensemble",
            "training": "expanding",
            "probabilities": "mean_softmax",
            "test_matches": len(frame),
            "evaluation": "retrospective",
        }
    )
    return write_outputs(root, body, result, frame, short)


if __name__ == "__main__":
    main()
