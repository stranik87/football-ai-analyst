from app.ml.feature_builder import FeatureBuilder


class DatasetBuilder:
    """
    Построение датасета для обучения модели.
    """

    def __init__(self, session):
        self.session = session
        self.feature_builder = FeatureBuilder(session)

    def build_match(
        self,
        fixture,
    ) -> dict:
        return self.feature_builder.build(
            home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id,
            fixture_id=fixture.id,
        )