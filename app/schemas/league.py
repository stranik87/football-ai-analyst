from pydantic import BaseModel


class LeagueResponse(BaseModel):
    """
    Ответ API с информацией о лиге.
    """

    league_id: int
    api_id: int

    name: str
    type: str
    country: str
    logo: str