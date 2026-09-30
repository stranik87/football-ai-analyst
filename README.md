# Football AI Analyst

Football AI Analyst — система футбольной аналитики на Python, которая собирает данные матчей, рассчитывает статистические признаки, обучает модель CatBoost и строит прогнозы исхода футбольных матчей.

Проект включает:

* импорт футбольных данных из API-Football;
* базу данных матчей и статистики;
* Feature Engineering;
* обучение и оптимизацию CatBoost;
* временную (temporal) проверку качества модели;
* прогнозы исторических и будущих матчей;
* Explainable AI через SHAP/CatBoost;
* REST API;
* Web Dashboard;
* Telegram-бота;
* автоматические тесты.

---

## Статус проекта

**Основной ML-пайплайн v1.0.0 финализирован.**

Основные этапы завершены:

1. импорт данных матчей и статистики;
2. построение расширенного датасета;
3. обучение и оптимизация CatBoost;
4. калибровочный эксперимент;
5. финальная честная оценка модели;
6. подключение финальной модели к прогнозированию;
7. проверка прогнозов будущих матчей;
8. проверка основного pipeline.

Осталось выполнить финальную очистку и сделать финальный Git commit/push.

---

## Данные

Основной источник данных:

```text
API-Football
```

Текущая локальная база содержит данные по пяти основным европейским чемпионатам:

* Premier League;
* Ligue 1;
* Bundesliga;
* Serie A;
* La Liga.

В базе загружены матчи сезонов 2024, 2025 и 2026, включая статистику завершённых матчей.

---

## Датасет

Финальный датасет:

```text
data/datasets/matches_dataset.csv
```

Текущее состояние:

```text
Матчей: 3762
Колонок: 86
Период: 2024-08-15 — 2026-09-20
```

Распределение целевой переменной:

| Исход | Количество | Доля |
| --- | ---: | ---: |
| H — победа хозяев | 1617 | 43.0% |
| D — ничья | 950 | 25.3% |
| A — победа гостей | 1195 | 31.8% |

Целевая переменная:

```text
result
```

---

## Feature Engineering

Для построения ML-признаков используются анализаторы, которые рассчитывают характеристики команд до начала анализируемого матча.

В `FeatureBuilder` используются, в частности:

* TeamFormAnalyzer;
* TeamGoalkeeperAnalyzer;
* TeamPassAnalyzer;
* TeamPossessionAnalyzer;
* TeamRestDaysAnalyzer;
* TeamShotEfficiencyAnalyzer;
* TeamStatisticsAnalyzer;
* TeamVenueSplitAnalyzer.

Принципиальное требование проекта — не использовать результат анализируемого матча при построении его признаков.

Это необходимо для предотвращения **data leakage**.

---

## Machine Learning

Основная модель:

```text
CatBoostClassifier
```

Модель прогнозирует три класса:

```text
H — победа хозяев
D — ничья
A — победа гостей
```

### Финальная модель

```text
data/models/match_result_catboost_optimized.cbm
```

Список признаков:

```text
data/models/match_result_features_optimized.joblib
```

Финальная модель использует:

```text
50 признаков
68 итераций
Depth: 5
Learning rate: 0.05
L2 leaf regularization: 7
Random strength: 0.5
Bagging temperature: 1.0
```

---

## Честная оценка модели

Для оценки используется **temporal split**: более ранние матчи используются для обучения, а более поздние — для финального теста.

Финальный тест содержит:

```text
565 матчей
```

Период тестовой выборки:

```text
2026-04-11 16:30:00
—
2026-09-20 19:00:00
```

Финальные метрики:

| Метрика | Результат |
| --- | ---: |
| Accuracy | 0.4761 |
| LogLoss | 1.0227 |

Для сравнения, baseline «всегда победа хозяев» на этой же тестовой выборке составляет:

```text
Accuracy: 0.4283
```

### Метрики по исходам

| Исход | Precision | Recall | F1 | Матчей |
| --- | ---: | ---: | ---: | ---: |
| Победа хозяев `H` | 0.5068 | 0.7645 | 0.6096 | 242 |
| Ничья `D` | 1.0000 | 0.0138 | 0.0272 | 145 |
| Победа гостей `A` | 0.4141 | 0.4607 | 0.4362 | 178 |

### Confusion Matrix

```text
          predicted_H  predicted_D  predicted_A
actual_H           185            0           57
actual_D            84            2           59
actual_A            96            0           82
```

Важное ограничение текущей модели: **она очень редко выбирает класс ничьей**. Это видно по recall класса `D = 0.0138` на финальном тесте.

---

## Accuracy по лигам

| Лига | Матчей | Accuracy |
| --- | ---: | ---: |
| Bundesliga | 87 | 0.4713 |
| La Liga | 146 | 0.5205 |
| Ligue 1 | 100 | 0.4100 |
| Premier League | 116 | 0.4483 |
| Serie A | 116 | 0.5086 |

---

## Accuracy по уверенности

| Уверенность | Матчей | Accuracy |
| --- | ---: | ---: |
| Ниже 40% | 104 | 0.3269 |
| 40–50% | 246 | 0.4309 |
| 50–60% | 135 | 0.5185 |
| 60% и выше | 80 | 0.7375 |

Эти значения относятся именно к финальной временной тестовой выборке.

---

## Калибровка вероятностей

Для финальной модели был проведён отдельный эксперимент temperature scaling.

Лучшее значение на validation:

```text
Temperature = 0.70
```

Однако на независимом финальном test применение `T=0.70` ухудшило результаты:

```text
Без калибровки:
LogLoss = 1.022737
Brier    = 0.612341

T=0.70:
LogLoss = 1.034855
Brier    = 0.620139
```

Поэтому **temperature scaling T=0.70 не используется в production-прогнозах**. Финальная система работает с исходными вероятностями CatBoost.

Результаты эксперимента сохранены в:

```text
data/reports/calibration_final/
```

Файлы:

```text
calibration_v1_3_summary.csv
calibration_v1_3_test_predictions.csv
temperature_search_v1_3.csv
```

---

## Оценка модели

Запуск полной оценки:

```bash
python -m scripts.evaluate_model
```

Отчёт:

```text
data/reports/model_evaluation.json
```

---

## Прогноз будущих матчей

Основной скрипт:

```bash
python -m scripts.predict_upcoming
```

Для отдельных матчей:

```bash
python -m scripts.predict_match --fixture-id <ID>
```

Оба варианта подключены к финальной оптимизированной модели:

```text
data/models/match_result_catboost_optimized.cbm
```

---

## Проверка pipeline

Финальный pipeline проверен:

1. датасет построен на расширенной выборке;
2. финальная CatBoost-модель загружается корректно;
3. список из 50 признаков совпадает с моделью;
4. классы модели соответствуют `A / D / H`;
5. `evaluate_model` успешно выполняется;
6. `predict_upcoming` успешно строит прогнозы будущих матчей;
7. `predict_match` успешно строит прогноз отдельных матчей;
8. прогнозы отдельных матчей соответствуют прогнозам `predict_upcoming`.

Пример:

```text
Inter - Parma
H: 76.90%
D: 13.75%
A: 9.35%
Prediction: H
```

---

## Explainable AI

Для объяснения прогнозов используются SHAP-значения CatBoost.

Сервис:

```text
app/services/prediction_explanation_service.py
```

---

## REST API

Запуск:

```bash
uvicorn app.web_api:app --reload
```

Основные endpoints:

```text
GET /fixtures/latest
GET /fixtures/upcoming
GET /fixtures/search
GET /fixtures/{fixture_id}
GET /predict/{fixture_id}
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Web Dashboard

```text
http://127.0.0.1:8000/dashboard
```

Основные страницы:

```text
/dashboard
/dashboard/fixtures
/dashboard/upcoming
/dashboard/teams
/dashboard/leagues
/dashboard/standings
/dashboard/predict/{fixture_id}
/dashboard/model-evaluation
```

---

## Telegram-бот

Запуск:

```bash
python -m scripts.run_bot
```

Команды:

```text
/start
/help
/predict <fixture_id>
/next
/cancel
```

---

## Обучение и обновление модели

Экспорт датасета:

```bash
python -m scripts.export_dataset
```

Обучение:

```bash
python -m scripts.train_model
```

Оптимизация:

```bash
python -m scripts.optimize_model
```

Сравнение:

```bash
python -m scripts.compare_models
```

Отчёт:

```text
data/reports/catboost_model_comparison.csv
```

`compare_models.py` используется для экспериментальных сравнений и может обучать модели заново; его результаты не следует автоматически считать сравнением сохранённой финальной production-модели.

---

## Тесты

Запуск:

```bash
python -m pytest -v
```

Проверка синтаксиса:

```bash
python -m compileall app scripts tests
```

---

## Безопасность

API-ключи и Telegram Bot Token должны храниться только в:

```text
.env
```

`.env` не должен попадать в Git.

Для настройки используется:

```text
.env.example
```

---

## API-Football

Дальнейшее автоматическое обновление данных зависит от доступности API-Football и действующего тарифа.

На момент финализации проекта необходимые исторические данные и статистика для текущего датасета загружены.

---

## Репозиторий

```text
https://github.com/stranik87/football-ai-analyst
```

---

## Финальный статус

Финальный датасет:

```text
3762 матча / 86 колонок
```

Финальная модель:

```text
data/models/match_result_catboost_optimized.cbm
```

Production-прогнозирование:

```text
predict_upcoming
predict_match
PredictionService
```

Перед окончательным релизом необходимо:

```text
1. проверить Git status;
2. удалить README.md.backup после проверки;
3. сделать финальный Git commit;
4. выполнить git push.
```

---

## Версия

```text
v1.0.0
```
