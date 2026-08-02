from sqlalchemy.orm import joinedload

from app.models.league import League
from app.models.league_season import LeagueSeason
from app.models.standing import Standing


class StandingService:
    """
    Сервис для получения турнирных таблиц.
    """

    def __init__(self, session) -> None:
        self.session = session

    def get_by_league_and_season(
        self,
        league_id: int,
        season: int,
    ) -> list[Standing]:
        """
        Получить турнирную таблицу по локальному ID лиги
        и номеру сезона.
        """

        league_season = (
            self.session.query(LeagueSeason)
            .filter(
                LeagueSeason.league_id == league_id,
                LeagueSeason.season == season,
            )
            .first()
        )

        if league_season is None:
            return []

        return (
            self.session.query(Standing)
            .options(
                joinedload(Standing.team),
                joinedload(
                    Standing.league_season
                ).joinedload(
                    LeagueSeason.league
                ),
            )
            .filter(
                Standing.league_season_id
                == league_season.id
            )
            .order_by(
                Standing.rank.asc()
            )
            .all()
        )

    def get_by_league_api_id_and_season(
        self,
        league_api_id: int,
        season: int,
    ) -> list[Standing]:
        """
        Получить турнирную таблицу по API ID лиги
        и номеру сезона.
        """

        league_season = (
            self.session.query(LeagueSeason)
            .join(
                League,
                League.id == LeagueSeason.league_id,
            )
            .filter(
                League.api_id == league_api_id,
                LeagueSeason.season == season,
            )
            .first()
        )

        if league_season is None:
            return []

        return (
            self.session.query(Standing)
            .options(
                joinedload(Standing.team),
                joinedload(
                    Standing.league_season
                ).joinedload(
                    LeagueSeason.league
                ),
            )
            .filter(
                Standing.league_season_id
                == league_season.id
            )
            .order_by(
                Standing.rank.asc()
            )
            .all()
        )

    @staticmethod
    def serialize(
        standing: Standing,
    ) -> dict:
        """
        Преобразовать позицию команды в словарь.
        """

        league_season = standing.league_season
        league = (
            league_season.league
            if league_season is not None
            else None
        )

        return {
            "standing_id": standing.id,
            "league_season_id": (
                standing.league_season_id
            ),
            "league_id": (
                league.id
                if league is not None
                else None
            ),
            "league_api_id": (
                league.api_id
                if league is not None
                else None
            ),
            "league_name": (
                league.name
                if league is not None
                else None
            ),
            "season": (
                league_season.season
                if league_season is not None
                else None
            ),
            "team_id": standing.team_id,
            "team_name": (
                standing.team.name
                if standing.team is not None
                else None
            ),
            "rank": standing.rank,
            "points": standing.points,
            "goals_diff": standing.goals_diff,
            "group_name": standing.group_name,
            "form": standing.form,
            "status": standing.status,
            "description": standing.description,
            "played": standing.played,
            "wins": standing.wins,
            "draws": standing.draws,
            "losses": standing.losses,
            "goals_for": standing.goals_for,
            "goals_against": standing.goals_against,
            "home_played": standing.home_played,
            "home_wins": standing.home_wins,
            "home_draws": standing.home_draws,
            "home_losses": standing.home_losses,
            "home_goals_for": standing.home_goals_for,
            "home_goals_against": (
                standing.home_goals_against
            ),
            "away_played": standing.away_played,
            "away_wins": standing.away_wins,
            "away_draws": standing.away_draws,
            "away_losses": standing.away_losses,
            "away_goals_for": standing.away_goals_for,
            "away_goals_against": (
                standing.away_goals_against
            ),
        }