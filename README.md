# Вероятности исходов футбольных матчей

Нейросеть TensorFlow/Keras прогнозирует победу хозяев, ничью и победу гостей до начала матча. Используется архив Understat по пяти лигам за 2014/15–2019/20. Модель ежегодно обучается заново на всей завершённой истории.


Log loss 0.99187, Brier 0.59131, accuracy 51.4% на 1,725 матчах сезона 2019/20.

[Отчёт](reports/report.md) · [PDF](reports/report.pdf) · [Ноутбук](notebooks/analysis.ipynb)

## Запуск

```bash
python -m pip install -e ".[dev]"
python -m match_forecast analyze
```

Итоговая модель, данные, методика и результаты описаны в отчёте. Его таблицы и графики пересчитываются командой `python -m match_forecast.final_report`.
