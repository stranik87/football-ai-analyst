from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


DATASET_PATH = Path(
    "data/datasets/matches_dataset.csv"
)

MODEL_DIR = Path(
    "data/models"
)

MODEL_PATH = MODEL_DIR / "match_result_model.cbm"

TARGET_COLUMN = "result"

EXCLUDED_COLUMNS = (
    "fixture_id",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    TARGET_COLUMN,
)

CLASS_NAMES = (
    "A",
    "D",
    "H",
)

TRAIN_RATIO = 0.80

RANDOM_SEED = 42


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    if dataframe.empty:
        raise ValueError(
            "Датасет пуст"
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            "В датасете отсутствует "
            f"целевая колонка: {TARGET_COLUMN}"
        )

    return dataframe


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "В датасете нет признаков для обучения"
        )

    features = dataframe[
        feature_columns
    ].copy()

    target = dataframe[
        TARGET_COLUMN
    ].copy()

    features = features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    features = features.fillna(0.0)

    target = target.astype(str).str.strip()

    valid_mask = target.isin(
        CLASS_NAMES
    )

    features = features.loc[
        valid_mask
    ].reset_index(drop=True)

    target = target.loc[
        valid_mask
    ].reset_index(drop=True)

    if features.empty:
        raise ValueError(
            "После очистки не осталось данных"
        )

    return features, target


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    split_index = int(
        len(features) * TRAIN_RATIO
    )

    if split_index <= 0:
        raise ValueError(
            "Недостаточно данных для обучения"
        )

    if split_index >= len(features):
        raise ValueError(
            "Недостаточно данных для тестирования"
        )

    x_train = features.iloc[
        :split_index
    ].copy()

    x_test = features.iloc[
        split_index:
    ].copy()

    y_train = target.iloc[
        :split_index
    ].copy()

    y_test = target.iloc[
        split_index:
    ].copy()

    return (
        x_train,
        x_test,
        y_train,
        y_test,
    )


def create_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=RANDOM_SEED,
        l2_leaf_reg=5.0,
        auto_class_weights="Balanced",
        verbose=50,
        allow_writing_files=False,
    )


def print_class_distribution(
    name: str,
    target: pd.Series,
) -> None:
    print()
    print(name)

    distribution = (
        target.value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0,
        )
    )

    for class_name, count in distribution.items():
        percentage = (
            count / len(target) * 100
            if len(target) > 0
            else 0.0
        )

        print(
            f"{class_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


def main() -> None:
    dataframe = load_dataset()

    features, target = prepare_dataset(
        dataframe
    )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = split_dataset(
        features,
        target,
    )

    print(
        f"Всего матчей: {len(features)}"
    )

    print(
        f"Количество признаков: "
        f"{len(features.columns)}"
    )

    print(
        f"Обучающая выборка: {len(x_train)}"
    )

    print(
        f"Тестовая выборка: {len(x_test)}"
    )

    print_class_distribution(
        name="Распределение классов в обучении:",
        target=y_train,
    )

    print_class_distribution(
        name="Распределение классов в тесте:",
        target=y_test,
    )

    model = create_model()

    print()
    print("Начинается обучение модели...")

    model.fit(
        x_train,
        y_train,
        eval_set=(
            x_test,
            y_test,
        ),
        use_best_model=True,
        early_stopping_rounds=75,
    )

    predictions = model.predict(
        x_test
    ).reshape(-1)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_NAMES,
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"Факт_{label}"
            for label in CLASS_NAMES
        ],
        columns=[
            f"Прогноз_{label}"
            for label in CLASS_NAMES
        ],
    )

    print("Confusion matrix:")

    print(
        matrix_dataframe.to_string()
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_PATH
    )

    feature_importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": (
                model.get_feature_importance()
            ),
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    importance_path = (
        MODEL_DIR
        / "feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_path,
        index=False,
    )

    print()
    print(
        f"Модель сохранена: {MODEL_PATH}"
    )

    print(
        "Важность признаков сохранена: "
        f"{importance_path}"
    )

    print()
    print("Топ-15 признаков:")

    print(
        feature_importance.head(
            15
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()