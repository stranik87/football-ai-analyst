from pydantic import BaseModel


class StandingResponse(BaseModel):
    """
    Ответ API с позицией команды в таблице.
    """

    standing_id: int
    league_season_id: int

    league_id: int | None
    league_api_id: int | None
    league_name: str | None
    season: int | None

    team_id: int
    team_name: str | None

    rank: int
    points: int
    goals_diff: int

    group_name: str | None
    form: str | None
    status: str | None
    description: str | None

    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int

    home_played: int
    home_wins: int
    home_draws: int
    home_losses: int
    home_goals_for: int
    home_goals_against: int

    away_played: int
    away_wins: int
    away_draws: int
    away_losses: int
    away_goals_for: int
    away_goals_against: int