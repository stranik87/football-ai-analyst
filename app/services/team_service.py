from sqlalchemy.orm import joinedload

from app.models.team import Team


class TeamService:
    """
    Сервис для получения и поиска футбольных команд.
    """

    def __init__(self, session) -> None:
        self.session = session

    def get_by_id(
        self,
        team_id: int,
    ) -> Team | None:
        """
        Получить команду по локальному ID.
        """

        return (
            self.session.query(Team)
            .options(
                joinedload(Team.league),
                joinedload(Team.venue),
            )
            .filter(Team.id == team_id)
            .first()
        )

    def get_all(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Team]:
        """
        Получить список команд.
        """

        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)

        return (
            self.session.query(Team)
            .options(
                joinedload(Team.league),
                joinedload(Team.venue),
            )
            .order_by(Team.name.asc())
            .offset(safe_offset)
            .limit(safe_limit)
            .all()
        )

    def search(
        self,
        name: str,
        limit: int = 20,
    ) -> list[Team]:
        """
        Найти команды по части названия.
        """

        normalized_name = name.strip()

        if not normalized_name:
            return []

        safe_limit = max(1, min(limit, 50))

        return (
            self.session.query(Team)
            .options(
                joinedload(Team.league),
                joinedload(Team.venue),
            )
            .filter(
                Team.name.ilike(
                    f"%{normalized_name}%"
                )
            )
            .order_by(Team.name.asc())
            .limit(safe_limit)
            .all()
        )

    @staticmethod
    def serialize(
        team: Team,
    ) -> dict:
        """
        Преобразовать команду в словарь.
        """

        return {
            "team_id": team.id,
            "api_id": team.api_id,
            "name": team.name,
            "code": team.code,
            "country": team.country,
            "founded": team.founded,
            "logo": team.logo,
            "league_id": team.league_id,
            "league_name": (
                team.league.name
                if team.league is not None
                else None
            ),
            "venue_id": team.venue_id,
            "venue_name": (
                team.venue.name
                if team.venue is not None
                else None
            ),
        }