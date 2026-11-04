from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    scores,
    calibration_table,
    paired_bootstrap,
    dump,
    markdown_table,
    update_readme,
)


NAMES = {
    "constant": "Частоты обучения",
    "elo": "Elo",
    "poisson": "Пуассон по прошлому xG",
    "uncalibrated": "Сеть до калибровки",
    "neural": "Сеть после калибровки",
}
OUTCOMES = ["Победа хозяев", "Ничья", "Победа гостей"]
COLORS = ["#a4aeb5", "#718b77", "#b58a49", "#809fb0", "#294b63"]


def matrix(frame, model):
    return frame[[f"p_{model}_{i}" for i in range(3)]].to_numpy()


def evaluate(root, predictions):
    root = Path(root)
    test = predictions[predictions.split.eq("test")].copy()
    y = test.target.to_numpy(int)
    rows, calibration, comparisons = [], [], []
    dates = pd.to_datetime(test.date)
    periods = {
        "all": np.ones(len(test), bool),
        "before_break": (dates < "2020-03-15").to_numpy(),
        "after_break": (dates >= "2020-05-15").to_numpy(),
    }
    groups = dates.dt.to_period("W").astype(str).to_numpy()
    for model in NAMES:
        p = matrix(test, model)
        for period, mask in periods.items():
            if mask.any():
                rows.append(
                    {"model": model, "group": period, **scores(y[mask], p[mask])}
                )
        for league in sorted(test.league.unique()):
            mask = test.league.eq(league).to_numpy()
            rows.append({"model": model, "group": league, **scores(y[mask], p[mask])})
        calibration.append(calibration_table(y, p, model))
        if model in ["constant", "elo", "poisson"]:
            comparisons.append(
                {
                    "baseline": model,
                    **paired_bootstrap(y, matrix(test, "neural"), p, groups),
                }
            )
    metrics, reliability = pd.DataFrame(rows), pd.concat(calibration, ignore_index=True)
    (root / "reports/tables").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(root / "reports/tables/model_metrics.csv", index=False)
    reliability.to_csv(root / "reports/tables/calibration.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        root / "reports/tables/model_comparison.csv", index=False
    )
    examples = test[
        [
            "match_id",
            "date",
            "league",
            "home",
            "away",
            "target",
            "p_neural_0",
            "p_neural_1",
            "p_neural_2",
        ]
    ].tail(12)
    examples.to_csv(root / "reports/tables/historical_examples.csv", index=False)
    result = {"test_matches": len(test), "metrics": rows, "comparisons": comparisons}
    dump(root / "reports/metrics.json", result)
    return result


def make_report(root, matches, predictions, audit, training, metrics):
    root = Path(root)
    figures = root / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    league = (
        matches.groupby(["league", "season"])
        .agg(
            matches=("match_id", "size"),
            home_goals=("home_goals", "mean"),
            away_goals=("away_goals", "mean"),
            home_xg=("home_xg", "mean"),
            away_xg=("away_xg", "mean"),
            home_win=("target", lambda y: y.eq(0).mean()),
            draw=("target", lambda y: y.eq(1).mean()),
        )
        .reset_index()
    )
    league.to_csv(root / "reports/tables/league_seasons.csv", index=False)
    table = pd.DataFrame(metrics["metrics"])
    overall = table[table.group.eq("all")].copy()
    test = predictions[predictions.split.eq("test")].copy()
    reliability = pd.read_csv(root / "reports/tables/calibration.csv")
    history = pd.read_csv(root / "models/history.csv")
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": "#e1e6e9",
        }
    ):

        def save(fig, name):
            fig.savefig(figures / name, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for name, data in league.groupby("league"):
            axes[0].plot(data.season, data.home_win, marker="o", label=name)
            axes[1].plot(
                data.season, data.home_goals - data.away_goals, marker="o", label=name
            )
        axes[0].set(
            title="Доля побед хозяев",
            xlabel="Год начала сезона",
            ylabel="Доля матчей",
            ylim=(0.25, 0.65),
        )
        axes[1].set(
            title="Разница голов хозяев и гостей",
            xlabel="Год начала сезона",
            ylabel="Голы на матч",
        )
        for ax in axes:
            ax.grid(alpha=0.6)
        axes[1].legend(fontsize=8, frameon=True)
        fig.tight_layout()
        save(fig, "home_advantage.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, metric, title in zip(
            axes, ["log_loss", "brier"], ["Log loss", "Brier: сумма по трём исходам"]
        ):
            bars = ax.barh(
                [NAMES[m] for m in overall.model], overall[metric], color=COLORS
            )
            ax.bar_label(bars, fmt="%.3f", padding=4)
            ax.set(
                xlabel="Меньше — лучше",
                title=title,
                xlim=(0, overall[metric].max() * 1.2),
            )
        fig.suptitle("Отложенный сезон 2019/20", fontsize=15)
        fig.tight_layout()
        save(fig, "model_comparison.png")
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
        for klass, ax in enumerate(axes):
            ax.plot([0, 1], [0, 1], "--", color="#a4aeb5")
            for model, color in [("neural", "#294b63"), ("poisson", "#b58a49")]:
                r = reliability[
                    reliability.model.eq(model) & reliability["class"].eq(klass)
                ]
                reliable = r[r.n.ge(30)]
                sparse = r[r.n.lt(30)]
                ax.plot(
                    reliable.mean_probability,
                    reliable.frequency,
                    "o-",
                    label=NAMES[model],
                    color=color,
                )
                ax.scatter(
                    sparse.mean_probability,
                    sparse.frequency,
                    facecolors="none",
                    edgecolors=color,
                )
            ax.set(
                xlabel="Средняя вероятность",
                ylabel="Фактическая частота",
                title=OUTCOMES[klass],
                xlim=(0, 1),
                ylim=(0, 1),
            )
            ax.grid(alpha=0.5)
        axes[0].legend(fontsize=8)
        fig.text(
            0.5,
            0.01,
            "Полые точки: менее 30 матчей в интервале вероятностей",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        save(fig, "calibration.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, data in history.groupby("candidate"):
            ax.plot(data.epoch, data.val_log_loss, label=name)
        ax.set(title="Настройка по сезону 2017/18", xlabel="Эпоха", ylabel="Log loss")
        ax.legend()
        ax.grid(alpha=0.5)
        save(fig, "training.png")
        confusion = np.zeros((3, 3), int)
        for actual, predicted in zip(
            test.target, matrix(test, "neural").argmax(axis=1)
        ):
            confusion[actual, predicted] += 1
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.imshow(
            confusion / confusion.sum(axis=1, keepdims=True),
            cmap="Blues",
            vmin=0,
            vmax=1,
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
                        if confusion[i, j] / confusion[i].sum() > 0.6
                        else "black"
                    ),
                )
        ax.set(
            xticks=np.arange(3),
            yticks=np.arange(3),
            xticklabels=["П1", "Х", "П2"],
            yticklabels=["П1", "Х", "П2"],
            xlabel="Наиболее вероятный исход",
            ylabel="Фактический исход",
            title="Число матчей по исходам",
        )
        save(fig, "confusion.png")
    comparisons = pd.DataFrame(metrics["comparisons"])
    supported = comparisons.loc[comparisons.high.lt(0), "baseline"].map(NAMES).tolist()
    conclusion = (
        "95% интервалы разницы log loss целиком ниже нуля при сравнении с: "
        + ", ".join(supported)
        + "."
        if supported
        else "Ни один 95% интервал разницы log loss с ориентирами не находится целиком ниже нуля. Устойчивое преимущество сети по этому тесту не установлено."
    )
    unsupported = (
        comparisons.loc[comparisons.high.ge(0), "baseline"].map(NAMES).tolist()
    )
    if unsupported and supported:
        conclusion += (
            " Убедительного преимущества перед "
            + ", ".join(unsupported)
            + " не установлено."
        )
    report_table = overall[["model", "n", "log_loss", "brier", "accuracy"]].replace(
        {"model": NAMES}
    )
    comparison_table = comparisons[["baseline", "difference", "low", "high"]].replace(
        {"baseline": NAMES}
    )
    periods = table[
        table.group.isin(["before_break", "after_break"])
        & table.model.isin(["neural", "poisson", "elo"])
    ]
    examples = test[
        ["date", "home", "away", "p_neural_0", "p_neural_1", "p_neural_2"]
    ].tail(8)
    report = f"""# Вероятности исходов футбольных матчей

Задача — оценить П1, ничью и П2 до начала матча. В архиве Understat {len(matches):,} сыгранных матчей пяти лиг за 2014/15–2019/20. Источник — [зеркало Mark Wilkins](https://github.com/Markjwilkins/Understat); версия закреплена в конфигурации.

## Результат

{markdown_table(report_table)}

{conclusion} Отрицательная разница означает меньшую ошибку сети. 

![Сравнение моделей](figures/model_comparison.png)

## Подготовка данных

Исключена {audit['excluded_unplayed']} строка без сыгранного результата: отменённые встречи Лиги 1 не превращены в ничьи. Идентификаторы матчей уникальны, голы и xG проверены на пропуски и отрицательные значения. Дата в исходнике используется без предположений о часовом поясе: результаты всех встреч дня становятся известны модели только на следующую дату.

Сведения `forecast.w/d/l` из источника исключены: их доступность до начала матча не подтверждена. Текущие голы, xG и целевой исход также не входят в признаки. Поздняя выгрузка не позволяет проверить исторические исправления дат или xG поставщиком.

## Различия лиг

![Домашнее поле](figures/home_advantage.png)

График описывает домашнее преимущество в этом архиве. На различия между сезонами могут влиять состав клубов и условия проведения встреч. Полные числа находятся в [таблице](tables/league_seasons.csv).

## Что известно до матча

Используются последние 5 и 10 матчей каждой команды: голы, xG за и против, очки, число дней отдыха и матчей за 14 дней. Отдельно оцениваются показатели дома или в гостях. Командные ID и названия нужны для истории и вывода, но не подаются сети как категории. Лига кодируется по словарю обучающей части.

Elo обновляется после каждого игрового дня, с K=20, домашней поправкой 70 и сохранением 75% отклонения от 1500 при новом сезоне. Для новых клубов рейтинг равен 1500. Отсутствующая история остаётся пропущенной; в сети она заполняется медианой обучения с отдельным индикатором пропуска. В архиве {audit['cold_start_matches']} матчей хотя бы с одной командой, сыгравшей менее пяти предыдущих встреч в охвате.

Базовая модель Пуассона использует прошлый xG команды и соперника в соответствующем месте проведения, сглаженный пятью условными матчами к прошлому среднему лиги. Ориентир предполагает независимые голы команд. Elo сравнивается отдельно; параметр ничьей оценивается на обучении.

## Обучение и проверка

Обучение: 2014/15–2016/17. Выбор ширины сети и остановка: 2017/18. Калибровка температуры: 2018/19. Итоговый тест: 2019/20. Во время теста веса сети и преобразования не меняются; история команд обновляется после уже завершённых дней. Перед каждой встречей признаки пересчитываются по завершённым дням.

Выбрана сеть {training['selected']['name']}, {training['selected']['parameters']} параметров, {training['features']} входных признаков после преобразования. TensorFlow {training['tensorflow']}, Keras {training['keras']}. Температура: {training['temperature']:.3f}. Классы не перевзвешиваются: нужны вероятности в естественной частоте исходов.

![Обучение](figures/training.png)
![Калибровка](figures/calibration.png)
![Исходы](figures/confusion.png)

Brier здесь — средняя сумма квадратов ошибок трёх вероятностей. Accuracy использует самый вероятный исход и приводится вместе с log loss.
Ничья оказалась самым вероятным исходом только в {int((matrix(test,'neural').argmax(axis=1)==1).sum())} матчах из {len(test)}. По правилу максимума вероятности модель редко выбирает ничью. На калибровочном графике полые точки обозначают интервалы с менее чем 30 матчами: их частоты особенно нестабильны.

## Неопределённость сравнения

{markdown_table(comparison_table)}

Использовано 1 000 повторов парного бутстрэпа по календарным неделям с сохранением всех матчей недели. Интервалы не скорректированы на множественные сравнения и не учитывают полностью зависимость клубов между неделями.

## Перерыв в сезоне 2019/20

{markdown_table(periods[['model','group','n','log_loss','brier']].replace({'model':NAMES}))}

`before_break` — даты до 15 марта 2020 года; `after_break` — начиная с 15 мая. Лига 1 не возобновила сезон, поэтому состав второй группы отличается. Разрез описывает качество на двух частях сезона с разным составом лиг. Для объяснения различий нужны посещаемость, составы и травмы.

## Примеры исторического прогноза

{markdown_table(examples)}

Вероятности относятся к историческим матчам и рассчитаны по сведениям предыдущих дат. [Ноутбук](../notebooks/analysis.ipynb) воспроизводит расчёты.
"""
    (root / "reports/report.md").write_text(report, encoding="utf-8", newline="\n")
    dump(
        root / "reports/result_summary.json",
        {
            "test_matches": len(test),
            "neural": scores(test.target.to_numpy(), matrix(test, "neural")),
            "conclusion": conclusion,
            "selected": training["selected"]["name"],
        },
    )
    update_readme(root, report_table, conclusion)
    from .temporal_report import refresh
    refresh(root)
    return report_table
