import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "match_forecast"
KIND = "match"
sys.path.insert(0, str(ROOT / "src"))


def build(execute=True):
    import nbformat
    from nbclient import NotebookClient

    if KIND == "season":
        importlib.import_module(PACKAGE + ".environment").local_runtime(ROOT)
    else:
        importlib.import_module(PACKAGE + ".common").runtime(ROOT)
    cells = []

    def md(value):
        cells.append(nbformat.v4.new_markdown_cell(value))

    def code(value):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(value).strip()))

    title = {
        "xg": "Вероятность гола в момент удара",
        "match": "Вероятности исходов матчей",
        "season": "Статистика футболиста в следующем сезоне",
    }[KIND]
    md(
        "# "
        + title
        + "\n\n[Отчёт](../reports/report.md) · [PDF](../reports/report.pdf)"
    )
    md("## Подготовка")
    code(
        f"""
        from pathlib import Path
        import json
        import sys
        import numpy as np
        import pandas as pd
        from IPython.display import Image, display

        ROOT = Path.cwd()
        if ROOT.name == 'notebooks':
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT / 'src'))
        from {PACKAGE}.__main__ import run

        _ = run(ROOT, {'reuse_model=True' if KIND=='season' else 'train=False'})
        REPORTS = ROOT / 'reports'
        FIGURES = REPORTS / 'figures'
        predictions = pd.read_csv(REPORTS / 'final_predictions.csv')
        metrics = pd.read_csv(REPORTS / 'tables/final_model_metrics.csv')
        configuration = json.loads((ROOT / 'config/report_model.json').read_text(encoding='utf-8'))
        display(pd.Series(configuration))
    """
    )
    if KIND == "xg":
        md("## Выборка")
        code(
            """
            audit = json.loads((REPORTS / 'data_quality.json').read_text(encoding='utf-8'))
            display(pd.DataFrame(audit['splits']))
            display(pd.read_csv(REPORTS / 'tables/final_league_profiles.csv'))
        """
        )
        md("## Обучение и качество")
        code(
            """
            from xg_model.common import scores

            p = np.column_stack([1-predictions.p_neural, predictions.p_neural])
            display(pd.Series(scores(predictions.target.to_numpy(int), p)))
            display(metrics.loc[metrics.group.eq('all')])
            display(Image(filename=str(FIGURES / 'final_training.png'), width=900))
            display(Image(filename=str(FIGURES / 'final_model_quality.png'), width=1050))
        """
        )
        md("## Вероятности и география ударов")
        code(
            """
            display(Image(filename=str(FIGURES / 'final_probability_check.png'), width=750))
            display(Image(filename=str(FIGURES / 'final_shot_map.png'), width=1000))
            display(metrics.loc[metrics.model.eq('neural')])
        """
        )
        md("## Сравнение моделей")
        code("display(pd.read_csv(REPORTS / 'tables/final_model_comparison.csv'))")
        md("## Игроки и команды")
        code(
            """
            players = pd.read_csv(REPORTS / 'tables/final_player_profiles.csv')
            teams = pd.read_csv(REPORTS / 'tables/final_team_profiles.csv')
            display(players.loc[players.shots.ge(25)].nlargest(10, 'xg'))
            display(teams.nlargest(10, 'xg'))
        """
        )
    elif KIND == "match":
        md("## Лиги и обучение")
        code(
            """
            display(Image(filename=str(FIGURES / 'home_advantage.png'), width=1000))
            display(Image(filename=str(FIGURES / 'final_training_window.png'), width=950))
        """
        )
        md("## Качество вероятностей")
        code(
            """
            from match_forecast.common import scores

            p = predictions[['p_neural_0', 'p_neural_1', 'p_neural_2']].to_numpy()
            display(pd.Series(scores(predictions.target.to_numpy(int), p)))
            display(metrics.loc[metrics.group.eq('all')])
            display(Image(filename=str(FIGURES / 'final_model_quality.png'), width=1050))
            display(Image(filename=str(FIGURES / 'final_probability_check.png'), width=1050))
        """
        )
        md("## Ошибки и сравнение моделей")
        code(
            """
            display(pd.read_csv(REPORTS / 'tables/final_model_comparison.csv'))
            display(metrics.loc[metrics.model.eq('neural')])
            display(Image(filename=str(FIGURES / 'final_confusion.png'), width=700))
        """
        )
        md("## Матчи и ожидаемые очки")
        code(
            """
            MATCH_ID = None
            example = predictions.loc[predictions.match_id.eq(MATCH_ID)] if MATCH_ID else predictions.sort_values('date').tail(10)
            display(example[['match_id','date','home','away','p_neural_0','p_neural_1','p_neural_2']])
        """
        )
        md("## Очки команд")
        code(
            """
            teams = pd.read_csv(REPORTS / 'tables/final_team_points.csv')
            summary = pd.read_csv(REPORTS / 'tables/final_team_error_summary.csv')
            display(summary[['league', 'teams', 'mae', 'rmse', 'bias', 'std', 'variance', 'mae_per_match']].round(3))
            display(Image(filename=str(FIGURES / 'final_team_points.png'), width=1000))
            display(pd.read_csv(REPORTS / 'tables/final_team_model_comparison.csv')[['model', 'mae', 'rmse', 'position_mae']].round(3))
        """
        )
        md("## Места в лиге")
        code(
            """
            display(summary[['league', 'teams', 'position_mae', 'exact_positions', 'within_2_positions', 'max_position_error']].round(2))
            display(pd.read_csv(REPORTS / 'tables/final_standings_outcomes.csv'))

            LEAGUE = 'EPL'
            display(teams.loc[teams.league.eq(LEAGUE), ['team', 'matches', 'points', 'expected', 'difference',
                                                      'actual_position', 'predicted_position', 'position_difference']].round(2))
        """
        )
    else:
        md("## Лиги и команды")
        code(
            """
            display(pd.read_csv(REPORTS / 'tables/league_seasons.csv').tail(6))
            display(Image(filename=str(FIGURES / 'league_trends.png'), width=1050))
            display(Image(filename=str(FIGURES / 'team_strength.png'), width=1050))
        """
        )
        md("## Игроки и устойчивость показателей")
        code(
            """
            display(Image(filename=str(FIGURES / 'player_profiles.png'), width=1050))
            display(Image(filename=str(FIGURES / 'player_persistence.png'), width=1050))
            display(Image(filename=str(FIGURES / 'team_residuals.png'), width=1050))
        """
        )
        md("## Обучение и ошибка прогноза")
        code(
            """
            test = predictions.loc[predictions.split.eq('test')]
            errors = {}
            for target in ['minutes','npxg90','xa90']:
                observed = test.loc[test['target_'+target].notna()]
                errors[target] = np.abs(observed['target_'+target]-observed['pred_'+target]).mean()
            display(pd.Series(errors, name='MAE'))
            display(metrics.loc[metrics.group.eq('all')])
            display(Image(filename=str(FIGURES / 'final_training_window.png'), width=950))
            display(Image(filename=str(FIGURES / 'final_model_quality.png'), width=1050))
            display(Image(filename=str(FIGURES / 'final_predicted_actual.png'), width=1050))
        """
        )
        md("## Сравнение моделей")
        code("display(pd.read_csv(REPORTS / 'tables/final_model_comparison.csv'))")
        md("## Вероятности и интервалы")
        code(
            """
            display(pd.read_csv(REPORTS / 'tables/final_probabilities.csv'))
            display(pd.read_csv(REPORTS / 'tables/final_intervals.csv'))
            display(Image(filename=str(FIGURES / 'final_probability_check.png'), width=1050))
            display(Image(filename=str(FIGURES / 'final_interval_coverage.png'), width=850))
        """
        )
        md("## Прогноз сезона 2025/26")
        code(
            """
            PLAYER_QUERY = ''
            forecast = pd.read_csv(REPORTS / 'tables/final_forecast_2025_26.csv')
            examples = (forecast.loc[forecast.player_name.str.contains(PLAYER_QUERY, case=False, regex=False)]
                        if PLAYER_QUERY else forecast.loc[forecast.b_last_minutes.ge(900)].nlargest(8,'current_attacking90'))
            display(examples[['player_name','c_league','pred_observed','pred_eligible','pred_minutes',
                              'pred_npxg90','lo90_npxg90','hi90_npxg90','pred_xa90','pred_change_attacking90']])
        """
        )
    md("## Футбольный результат")
    code(
        """
        display(pd.read_csv(REPORTS / 'tables/final_outcome_counts.csv').round(3))
        display(pd.read_csv(REPORTS / 'tables/final_draw_analysis.csv').round(3))
        display(pd.read_csv(REPORTS / 'tables/final_match_difficulty.csv').round(3))
        display(Image(filename=str(FIGURES / 'final_draw_diagnosis.png'), width=1050))
        """
    )
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    destination = ROOT / "notebooks/analysis.ipynb"
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    if execute:
        os.environ["JUPYTER_PATH"] = str(ROOT / ".runtime/jupyter")
        kernel = "football-report"
        folder = ROOT / ".runtime/jupyter/kernels" / kernel
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-X",
                        "utf8",
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Football report",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        NotebookClient(
            notebook,
            timeout=1800,
            kernel_name=kernel,
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nbformat.write(notebook, destination)
    print(destination, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-execute", action="store_true")
    build(execute=not parser.parse_args().no_execute)
