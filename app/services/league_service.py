from sqlalchemy.orm import joinedload

from app.models.league import League


class LeagueService:
    """
    Сервис для работы с футбольными лигами.
    """

    def __init__(self, session) -> None:
        self.session = session

    def get_by_id(
        self,
        league_id: int,
    ) -> League | None:
        """
        Получить лигу по локальному ID.
        """

        return (
            self.session.query(League)
            .options(
                joinedload(League.seasons),
            )
            .filter(
                League.id == league_id
            )
            .first()
        )

    def get_all(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[League]:
        """
        Получить список лиг.
        """

        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)

        return (
            self.session.query(League)
            .order_by(League.name.asc())
            .offset(safe_offset)
            .limit(safe_limit)
            .all()
        )

    def search(
        self,
        name: str,
        limit: int = 20,
    ) -> list[League]:
        """
        Найти лиги по части названия.
        """

        normalized_name = name.strip()

        if not normalized_name:
            return []

        safe_limit = max(1, min(limit, 50))

        return (
            self.session.query(League)
            .filter(
                League.name.ilike(
                    f"%{normalized_name}%"
                )
            )
            .order_by(League.name.asc())
            .limit(safe_limit)
            .all()
        )

    @staticmethod
    def serialize(
        league: League,
    ) -> dict:
        """
        Преобразовать лигу в словарь.
        """

        return {
            "league_id": league.id,
            "api_id": league.api_id,
            "name": league.name,
            "type": league.type,
            "country": league.country,
            "logo": league.logo,
        }