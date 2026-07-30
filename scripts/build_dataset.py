from app.database.session import SessionLocal
from app.ml.dataset_builder import DatasetBuilder
from app.models.fixture import Fixture


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
            .limit(10)
            .all()
        )

        print(f"Fixtures: {len(fixtures)}")

        for fixture in fixtures:
            features = builder.build_match(fixture)

            print(
                fixture.id,
                len(features),
                features,
            )

    finally:
        session.close()


if __name__ == "__main__":
    main()