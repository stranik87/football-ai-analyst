import csv
import json
from datetime import datetime
from pathlib import Path

from app.database.database import SessionLocal
from app.models.fixture import Fixture
from app.services.prediction_service import PredictionService


OUTPUT = Path("data/reports/real_match_predictions.csv")
MODEL_VERSION = "v1.2"


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def main():
    session = SessionLocal()

    try:
        now = datetime.now()

        fixtures = (
            session.query(Fixture)
            .filter(Fixture.kickoff > now)
            .order_by(Fixture.kickoff)
            .limit(100)
            .all()
        )

        if not fixtures:
            print("Будущих матчей не найдено.")
            return

        service = PredictionService(session)

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        rows = []

        for fixture in fixtures:
            try:
                prediction = service.predict(fixture.id)
            except Exception as exc:
                print(
                    f"ERROR fixture_id={fixture.id}: {exc}"
                )
                continue

            probabilities = prediction.get("probabilities", {})
            stats = prediction.get("statistics_predictions", {})
            goals = stats.get("goals", {})
            over_under = goals.get("over_under", {})

            row = {
                "prediction_time": now.isoformat(timespec="seconds"),
                "model_version": MODEL_VERSION,
                "fixture_id": fixture.id,
                "kickoff": fixture.kickoff.isoformat(
                    timespec="minutes"
                ),
                "home_team": fixture.home_team.name,
                "away_team": fixture.away_team.name,

                "predicted_result": prediction.get(
                    "prediction_name"
                ),
                "result_code": prediction.get(
                    "predicted_result"
                ),
                "home_probability": clean(
                    probabilities.get("home_win")
                ),
                "draw_probability": clean(
                    probabilities.get("draw")
                ),
                "away_probability": clean(
                    probabilities.get("away_win")
                ),
                "confidence": clean(
                    prediction.get("confidence")
                ),

                "goals_home": clean(
                    stats.get("goals_home")
                ),
                "goals_away": clean(
                    stats.get("goals_away")
                ),
                "goals_total": clean(
                    stats.get("goals_total")
                ),

                "goals_1h_home": clean(
                    stats.get("goals_1h_home")
                ),
                "goals_1h_away": clean(
                    stats.get("goals_1h_away")
                ),
                "goals_1h_total": clean(
                    stats.get("goals_1h_total")
                ),

                "goals_2h_home": clean(
                    stats.get("goals_2h_home")
                ),
                "goals_2h_away": clean(
                    stats.get("goals_2h_away")
                ),
                "goals_2h_total": clean(
                    stats.get("goals_2h_total")
                ),

                "goals_over_under": json.dumps(
                    over_under,
                    ensure_ascii=False,
                ),

                "corners_total": clean(
                    stats.get("corners_total")
                ),
                "corners_home": clean(
                    stats.get("corners_home")
                ),
                "corners_away": clean(
                    stats.get("corners_away")
                ),

                "yellow_cards_total": clean(
                    stats.get("yellow_cards_total")
                ),
                "yellow_cards_home": clean(
                    stats.get("yellow_cards_home")
                ),
                "yellow_cards_away": clean(
                    stats.get("yellow_cards_away")
                ),

                "shots_total": clean(
                    stats.get("shots_total")
                ),
                "shots_home": clean(
                    stats.get("shots_home")
                ),
                "shots_away": clean(
                    stats.get("shots_away")
                ),

                "shots_on_target_total": clean(
                    stats.get("shots_on_target_total")
                ),
                "shots_on_target_home": clean(
                    stats.get("shots_on_target_home")
                ),
                "shots_on_target_away": clean(
                    stats.get("shots_on_target_away")
                ),

                "offsides_total": clean(
                    stats.get("offsides_total")
                ),
                "offsides_home": clean(
                    stats.get("offsides_home")
                ),
                "offsides_away": clean(
                    stats.get("offsides_away")
                ),

                "fouls_total": clean(
                    stats.get("fouls_total")
                ),
                "fouls_home": clean(
                    stats.get("fouls_home")
                ),
                "fouls_away": clean(
                    stats.get("fouls_away")
                ),

                "full_prediction_json": json.dumps(
                    prediction,
                    ensure_ascii=False,
                    default=str,
                ),
            }

            rows.append(row)

            print(
                f"{fixture.id} | "
                f"{fixture.home_team.name} — "
                f"{fixture.away_team.name} | "
                f"{prediction.get('prediction_name')}"
            )

        if not rows:
            print("Не удалось получить ни одного прогноза.")
            return

        fieldnames = list(rows[0].keys())

        with OUTPUT.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

        print()
        print(f"Сохранено прогнозов: {len(rows)}")
        print(f"Файл: {OUTPUT}")


    finally:
        session.close()


if __name__ == "__main__":
    main()
