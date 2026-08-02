from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FixtureResponse(BaseModel):
    """
    Ответ API с информацией о матче.
    """

    model_config = ConfigDict(
        from_attributes=True
    )

    fixture_id: int
    api_id: int
    kickoff: datetime
    status_short: str
    status_long: str
    round: str

    home_team_id: int
    away_team_id: int

    home_team: str | None
    away_team: str | None

    home_goals: int | None
    away_goals: int | None