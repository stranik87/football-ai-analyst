from datetime import datetime

from pydantic import BaseModel


class ProbabilityResponse(BaseModel):
    home_win: float
    draw: float
    away_win: float


class ActualScoreResponse(BaseModel):
    home_goals: int | None
    away_goals: int | None


class PredictionResponse(BaseModel):
    """
    Ответ API с прогнозом матча.
    """

    fixture_id: int
    kickoff: datetime

    home_team_id: int
    away_team_id: int

    home_team: str
    away_team: str

    predicted_result: str
    predicted_result_name: str
    confidence: float

    probabilities: ProbabilityResponse
    actual_score: ActualScoreResponse