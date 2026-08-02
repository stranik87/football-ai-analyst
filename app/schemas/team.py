from pydantic import BaseModel


class TeamResponse(BaseModel):
    """
    Ответ API с информацией о команде.
    """

    team_id: int
    api_id: int

    name: str
    code: str
    country: str
    founded: int
    logo: str

    league_id: int
    league_name: str | None

    venue_id: int | None
    venue_name: str | None