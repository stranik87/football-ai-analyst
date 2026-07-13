from sqlalchemy.orm import Session

from app.models.standing import Standing


class StandingRepository:
    """
    Репозиторий турнирных таблиц.
    """

    def __init__(self, session: Session):
        self.session = session

    def get(
        self,
        league_season_id: int,
        team_id: int,
    ) -> Standing | None:
        return (
            self.session.query(Standing)
            .filter(
                Standing.league_season_id == league_season_id,
                Standing.team_id == team_id,
            )
            .first()
        )

    def add(self, standing: Standing) -> Standing:
        self.session.add(standing)
        return standing

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()