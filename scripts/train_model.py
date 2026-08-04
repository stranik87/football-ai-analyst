from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "data" / "datasets" / "matches_dataset.csv"
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_PATH = MODEL_DIR / "match_result_catboost.cbm"
FEATURES_PATH = MODEL_DIR / "match_result_features.joblib"

TARGET_COLUMN = "result"

EXCLUDED_COLUMNS = [
    "fixture_id",
    "kickoff",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    TARGET_COLUMN,
]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}\n"
            "Сначала запусти: python -m scripts.export_dataset"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    if dataframe.empty:
        raise ValueError("Датасет пустой.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"В датасете отсутствует целевая колонка: {TARGET_COLUMN}"
        )

    return dataframe


def prepare_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("Не найдены признаки для обучения модели.")

    x = dataframe[feature_columns].copy()
    y = dataframe[TARGET_COLUMN].copy()

    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([float("inf"), float("-inf")], pd.NA)
    x = x.fillna(0)

    y = y.astype(str).str.strip().str.upper()

    valid_results = {"H", "D", "A"}
    valid_mask = y.isin(valid_results)

    removed_rows = int((~valid_mask).sum())

    if removed_rows:
        logger.warning(
            "Удалено строк с неизвестным результатом: {}",
            removed_rows,
        )

    x = x.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if x.empty:
        raise ValueError("После очистки не осталось данных для обучения.")

    return x, y, feature_columns


def train_model() -> None:
    logger.info("Загрузка датасета: {}", DATASET_PATH)

    dataframe = load_dataset()

    logger.info("Всего строк в датасете: {}", len(dataframe))
    logger.info("Всего колонок: {}", len(dataframe.columns))

    x, y, feature_columns = prepare_data(dataframe)

    logger.info("Количество признаков: {}", len(feature_columns))
    logger.info("Распределение результатов:")
    logger.info("\n{}", y.value_counts().to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logger.info("Обучающая выборка: {}", len(x_train))
    logger.info("Тестовая выборка: {}", len(x_test))

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
    )

    logger.info("Начало обучения CatBoost...")

    model.fit(
        x_train,
        y_train,
        eval_set=(x_test, y_test),
        early_stopping_rounds=50,
    )

    predictions = model.predict(x_test)
    predictions = predictions.reshape(-1)

    accuracy = accuracy_score(y_test, predictions)

    logger.info("Accuracy: {:.4f}", accuracy)

    logger.info(
        "Classification report:\n{}",
        classification_report(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            zero_division=0,
        ),
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["H", "D", "A"],
    )

    logger.info(
        "Confusion matrix, порядок классов H, D, A:\n{}",
        matrix,
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    logger.info(
        "Топ-20 важных признаков:\n{}",
        feature_importance.head(20).to_string(index=False),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_PATH)

    joblib.dump(
        feature_columns,
        FEATURES_PATH,
    )

    logger.success("Модель сохранена: {}", MODEL_PATH)
    logger.success("Список признаков сохранён: {}", FEATURES_PATH)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as error:
        logger.exception("Ошибка обучения модели: {}", error)
        raise