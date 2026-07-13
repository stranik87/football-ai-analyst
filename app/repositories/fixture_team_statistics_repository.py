from app.models.fixture_team_statistics import FixtureTeamStatistics


class FixtureTeamStatisticsRepository:
    """
    Репозиторий статистики команды в матче.
    """

    def __init__(self, db):
        self.db = db

    def get(self, fixture_id: int, team_id: int):
        return (
            self.db.query(FixtureTeamStatistics)
            .filter_by(
                fixture_id=fixture_id,
                team_id=team_id,
            )
            .first()
        )

    def add(self, statistics: FixtureTeamStatistics):
        self.db.add(statistics)

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()