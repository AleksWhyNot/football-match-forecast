"""Интерпретация предматчевых вероятностей."""

import numpy as np
import pandas as pd


def outcome_tables(frame):
    p = frame[[f"p_neural_{i}" for i in range(3)]].to_numpy(float)
    y = frame.target.to_numpy(int)
    selected = p.argmax(axis=1)
    outcomes = []
    for k, name in enumerate(["Победа хозяев", "Ничья", "Победа гостей"]):
        actual, chosen = y == k, selected == k
        correct = actual & chosen
        outcomes.append(
            {
                "outcome": name,
                "actual": int(actual.sum()),
                "expected": float(p[:, k].sum()),
                "selected": int(chosen.sum()),
                "correct": int(correct.sum()),
                "recall": float(correct.sum() / actual.sum()),
                "precision": (
                    float(correct.sum() / chosen.sum()) if chosen.any() else np.nan
                ),
            }
        )
    bins = pd.cut(
        p[:, 1],
        [0, 0.2, 0.25, 0.275, 0.3, 1],
        right=False,
        labels=["Менее 20%", "20–25%", "25–27.5%", "27.5–30%", "От 30%"],
    )
    draws = (
        pd.DataFrame({"range": bins, "probability": p[:, 1], "draw": y == 1})
        .groupby("range", observed=True)
        .agg(
            matches=("draw", "size"),
            draws=("draw", "sum"),
            probability=("probability", "mean"),
            frequency=("draw", "mean"),
        )
        .reset_index()
    )
    groups = np.where(
        p.max(axis=1) >= 0.6,
        "Фаворит от 60%",
        np.where(p.max(axis=1) >= 0.45, "Фаворит 45–60%", "Фаворит менее 45%"),
    )
    matches = (
        pd.DataFrame(
            {
                "group": groups,
                "correct": selected == y,
                "draw": y == 1,
                "probability": p.max(axis=1),
            }
        )
        .groupby("group")
        .agg(
            matches=("correct", "size"),
            correct=("correct", "sum"),
            win_rate=("correct", "mean"),
            draw_rate=("draw", "mean"),
            probability=("probability", "mean"),
        )
        .reset_index()
    )
    top = frame.sort_values(["p_neural_1", "match_id"], ascending=[False, True]).head(
        int(np.ceil(0.2 * len(frame)))
    )
    detail = frame[
        [
            "match_id",
            "date",
            "league",
            "home",
            "away",
            "home_goals",
            "away_goals",
            "target",
            "p_neural_0",
            "p_neural_1",
            "p_neural_2",
        ]
    ].copy()
    detail["selected_outcome"] = selected
    detail["correct"] = selected == y
    detail["draw_rank"] = detail.p_neural_1.rank(ascending=False, method="min").astype(
        int
    )
    return pd.DataFrame(outcomes), draws, matches, top, detail


def report(root, frame, table, figure):
    import matplotlib.pyplot as plt

    outcomes, draws, groups, top, detail = outcome_tables(frame)
    for name, data in [
        ("outcome_counts", outcomes),
        ("draw_analysis", draws),
        ("match_difficulty", groups),
        ("match_football", detail),
    ]:
        data.to_csv(root / f"reports/tables/final_{name}.csv", index=False)
    p = frame.p_neural_1
    actual = int(frame.target.eq(1).sum())
    view = outcomes[
        ["outcome", "actual", "expected", "selected", "correct", "recall"]
    ].copy()
    view.recall *= 100
    view.columns = ["Исход", "Факт", "Сумма P", "Выбран", "Угадан", "Найдено, %"]
    dv = draws.copy()
    dv[["probability", "frequency"]] *= 100
    dv.columns = ["P ничьей", "Матчи", "Ничьи", "Оценка, %", "Факт, %"]
    gv = groups[["group", "matches", "correct", "win_rate", "draw_rate"]].copy()
    gv[["win_rate", "draw_rate"]] *= 100
    gv.columns = ["Сила фаворита", "Матчи", "Победы фаворита", "Победы, %", "Ничьи, %"]
    examples = top.head(8).copy()
    examples["score"] = (
        examples.home_goals.astype(int).astype(str)
        + ":"
        + examples.away_goals.astype(int).astype(str)
    )
    for col in ["p_neural_0", "p_neural_1", "p_neural_2"]:
        examples[col] *= 100
    ex = examples[["home", "away", "score", "p_neural_0", "p_neural_1", "p_neural_2"]]
    ex.columns = ["Хозяева", "Гости", "Счёт", "П1, %", "Х, %", "П2, %"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    for shift, key, label, color in [
        (-0.25, "actual", "Факт", "#8b9dab"),
        (0, "expected", "Сумма вероятностей", "#187d82"),
        (0.25, "selected", "Самый вероятный исход", "#b88549"),
    ]:
        axes[0].bar(x + shift, outcomes[key], 0.25, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=["П1", "Ничья", "П2"], ylabel="Матчей")
    axes[0].legend(fontsize=8)
    for is_draw, label, color in [
        (False, "Победа одной из команд", "#8b9dab"),
        (True, "Ничья", "#187d82"),
    ]:
        axes[1].hist(
            p.loc[frame.target.eq(1) == is_draw],
            bins=np.linspace(0, 0.4, 21),
            density=True,
            alpha=0.55,
            label=label,
            color=color,
        )
    axes[1].set(xlabel="Вероятность ничьей", ylabel="Плотность")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figure(root, fig, "draw_diagnosis")
    draw_top = int(top.target.eq(1).sum())
    strong = groups.loc[groups.group.eq("Фаворит от 60%")].iloc[0]
    equal = groups.loc[groups.group.eq("Фаворит менее 45%")].iloc[0]
    conclusion = (
        f"При выборе одного наиболее вероятного исхода модель не выбирает ничью "
        f"ни в одном из {len(frame)} матчей. Среди {len(top)} встреч с наибольшей "
        f"вероятностью ничьей таких результатов {draw_top} ({draw_top / len(top):.1%})."
    )
    return (
        f"""## Что происходит с ничьими

Матрица ошибок показывает один исход с максимальной вероятностью. Три числа вероятностей превращаются в одну метку. В тестовом сезоне вероятность ничьей лежит между {p.min():.1%} и {p.max():.1%}. При трёх исходах максимум хотя бы одного из них достигает одной трети, поэтому ничья здесь всегда уступает победе хозяев или гостей.

{table(view, 1)}

Ожидаемое число ничьих составляет {p.sum():.1f}, фактическое — {actual}. Это близкая оценка общего количества, но по выбору конкретного исхода найдены 0 из {actual} ничьих. Для пользователя, которому нужен один ответ П1/Х/П2, это существенный недостаток модели. Общая доля угаданных матчей скрывает его.

![Оценка ничьих](figures/final_draw_diagnosis.png)

Насколько полезен сам порядок вероятностей? Возьмём пятую часть матчей с наибольшей оценкой ничьей. В ней {draw_top} ничьих из {len(top)} ({draw_top / len(top):.1%}), во всём тесте — {actual / len(frame):.1%}. Прирост составляет {100 * (draw_top / len(top) - actual / len(frame)):.1f} процентного пункта. Даже в этой группе {len(top) - draw_top} матч завершился победой одной из команд. Выделение ничейных встреч получается слабым.

{table(dv, 1)}

Диапазоны заданы заранее, левая граница включается. Таблица сопоставляет вероятность и частоту. Разделение матчей с ничьей и без неё остаётся небольшим, что видно по сильному перекрытию распределений на графике.

### Матчи с наибольшей вероятностью ничьей

{table(ex, 1)}

Показаны первые восемь матчей по оценке ничьей с их фактическим счётом. Точность выбора ничьих должна проверяться отдельно при следующем улучшении модели. Среди направлений — совместная модель голов двух команд и проверка низких счетов 0:0 и 1:1 на более ранних сезонах. Любое изменение правила выбора исхода потребует оценки числа найденных ничьих и ложных прогнозов ничьей вместе с качеством вероятностей.

## Как часто выигрывает выбранный фаворит

Фаворитом считается команда с большей вероятностью победы. Матчи сгруппированы по этой вероятности до раскрытия результата.

{table(gv, 1)}

При вероятности фаворита от 60% он выиграл {strong.correct} из {strong.matches} матчей ({strong.win_rate:.1%}). Если оценка ниже 45%, выиграны {equal.correct} из {equal.matches} ({equal.win_rate:.1%}). Для уверенно выделенных фаворитов выбор победителя заметно полезнее, чем в равных парах.

Эти частоты показывают, насколько можно полагаться на выбор победителя при разной уверенности сети. Ничьи входят в число ошибок выбора фаворита. Оценки рассчитаны на сезоне 2019/20, их перенос на следующий сезон потребует новой проверки.
""",
        conclusion,
    )
