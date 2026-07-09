from app.models.league_season import LeagueSeason


class LeagueSeasonRepository:

    def __init__(self, db):
        self.db = db

    def get(self, league_id: int, season: int):
        return (
            self.db.query(LeagueSeason)
            .filter_by(
                league_id=league_id,
                season=season,
            )
            .first()
        )

    def add(self, season: LeagueSeason):
        self.db.add(season)

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()