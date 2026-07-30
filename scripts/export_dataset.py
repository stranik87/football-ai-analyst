from pathlib import Path

import pandas as pd

from app.database.session import SessionLocal
from app.ml.dataset_builder import DatasetBuilder
from app.models.fixture import Fixture


OUTPUT_DIR = Path("data/datasets")
OUTPUT_FILE = OUTPUT_DIR / "matches_dataset.csv"


def main():
    session = SessionLocal()

    try:
        builder = DatasetBuilder(session)

        fixtures = (
            session.query(Fixture)
            .filter(
                Fixture.home_goals.isnot(None),
                Fixture.away_goals.isnot(None),
            )
            .order_by(Fixture.kickoff)
            .all()
        )

        rows = []

        for fixture in fixtures:
            features = builder.build_match(fixture)

            features["fixture_id"] = fixture.id
            features["home_team_id"] = fixture.home_team_id
            features["away_team_id"] = fixture.away_team_id

            features["home_goals"] = fixture.home_goals
            features["away_goals"] = fixture.away_goals

            if fixture.home_goals > fixture.away_goals:
                features["result"] = "H"
            elif fixture.home_goals < fixture.away_goals:
                features["result"] = "A"
            else:
                features["result"] = "D"

            rows.append(features)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print(f"Всего матчей: {len(df)}")
        print(f"Количество признаков: {len(df.columns)}")
        print(f"Файл сохранен: {OUTPUT_FILE}")

    finally:
        session.close()


if __name__ == "__main__":
    main()