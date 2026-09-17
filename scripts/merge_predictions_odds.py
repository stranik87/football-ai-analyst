from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "reports"
    / "live_predictions.csv"
)

ODDS_PATH = (
    BASE_DIR
    / "data"
    / "reports"
    / "odds"
    / "odds_snapshots.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "reports"
    / "live_predictions_with_odds.csv"
)


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    odds = pd.read_csv(ODDS_PATH)

    # Оставляем только строки с полноценными odds.
    odds = odds[
        odds["fixture_id"].notna()
        & odds["home_odds"].notna()
        & odds["draw_odds"].notna()
        & odds["away_odds"].notna()
    ].copy()

    # Последний снимок по каждому матчу и букмекеру.
    odds["snapshot_at"] = pd.to_datetime(
        odds["snapshot_at"],
        errors="coerce",
        utc=True,
    )

    odds = (
        odds.sort_values("snapshot_at")
        .groupby(["fixture_id", "bookmaker_id"], as_index=False)
        .tail(1)
    )

    # Лучший доступный коэффициент среди букмекеров.
    best_odds = (
        odds.groupby("fixture_id")
        .agg(
            best_home_odds=("home_odds", "max"),
            best_draw_odds=("draw_odds", "max"),
            best_away_odds=("away_odds", "max"),
            bookmakers=("bookmaker_id", "nunique"),
        )
        .reset_index()
    )

    result = predictions.merge(
        best_odds,
        on="fixture_id",
        how="left",
    )

    # Fair odds.
    result["fair_home"] = 1 / result["p_home"]
    result["fair_draw"] = 1 / result["p_draw"]
    result["fair_away"] = 1 / result["p_away"]

    # Expected edge относительно лучшего доступного коэффициента.
    result["edge_home"] = (
        result["p_home"] * result["best_home_odds"] - 1
    )

    result["edge_draw"] = (
        result["p_draw"] * result["best_draw_odds"] - 1
    )

    result["edge_away"] = (
        result["p_away"] * result["best_away_odds"] - 1
    )

    result["best_edge"] = result[
        ["edge_home", "edge_draw", "edge_away"]
    ].max(axis=1)

    result = result.sort_values("kickoff").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 100)
    print("PREDICTIONS + ODDS")
    print("=" * 100)

    print(
        result[
            [
                "fixture_id",
                "home_team",
                "away_team",
                "p_home",
                "p_draw",
                "p_away",
                "best_home_odds",
                "best_draw_odds",
                "best_away_odds",
                "edge_home",
                "edge_draw",
                "edge_away",
                "best_edge",
            ]
        ].to_string(index=False)
    )

    print("")
    print(f"Матчей: {len(result)}")
    print(f"Файл: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
