from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from app.analytics.team_form_analyzer import TeamFormAnalyzer
from app.analytics.team_statistics_analyzer import TeamStatisticsAnalyzer
from app.models.team import Team


@dataclass
class TeamComparisonResult:
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str

    requested_limit: int

    home_matches: int
    away_matches: int

    home_statistics_matches: int
    away_statistics_matches: int

    home_points_per_match: float
    away_points_per_match: float

    home_average_goals_for: float
    away_average_goals_for: float

    home_average_goals_against: float
    away_average_goals_against: float

    home_average_expected_goals: float
    away_average_expected_goals: float

    home_average_total_shots: float
    away_average_total_shots: float

    home_average_shots_on_goal: float
    away_average_shots_on_goal: float

    home_average_ball_possession: float
    away_average_ball_possession: float

    home_average_corner_kicks: float
    away_average_corner_kicks: float

    home_average_yellow_cards: float
    away_average_yellow_cards: float

    home_form: str
    away_form: str

    form_advantage: str
    attack_advantage: str
    defence_advantage: str
    xg_advantage: str

    def to_dict(self) -> dict:
        return asdict(self)


class TeamComparisonAnalyzer:
    """
    Сравнивает домашнюю и гостевую команды перед матчем.
    """

    MIN_STATISTICS_MATCHES = 3

    def __init__(self, session: Session):
        self.session = session

        self.form_analyzer = TeamFormAnalyzer(session)
        self.statistics_analyzer = TeamStatisticsAnalyzer(session)

    def analyze(
        self,
        home_team_id: int,
        away_team_id: int,
        limit: int = 10,
        before_fixture_id: int | None = None,
    ) -> TeamComparisonResult:
        if home_team_id == away_team_id:
            raise ValueError(
                "Нельзя сравнивать команду саму с собой"
            )

        if limit <= 0:
            raise ValueError(
                "Количество матчей limit должно быть больше нуля"
            )

        home_team = self._get_team(home_team_id)
        away_team = self._get_team(away_team_id)

        home_form = self.form_analyzer.analyze(
            team_id=home_team.id,
            limit=limit,
            venue="home",
            before_fixture_id=before_fixture_id,
        )

        away_form = self.form_analyzer.analyze(
            team_id=away_team.id,
            limit=limit,
            venue="away",
            before_fixture_id=before_fixture_id,
        )

        home_statistics = self.statistics_analyzer.analyze(
            team_id=home_team.id,
            limit=limit,
            venue="home",
            before_fixture_id=before_fixture_id,
        )

        away_statistics = self.statistics_analyzer.analyze(
            team_id=away_team.id,
            limit=limit,
            venue="away",
            before_fixture_id=before_fixture_id,
        )

        return TeamComparisonResult(
            home_team_id=home_team.id,
            home_team_name=home_team.name,
            away_team_id=away_team.id,
            away_team_name=away_team.name,

            requested_limit=limit,

            home_matches=home_form.matches,
            away_matches=away_form.matches,

            home_statistics_matches=home_statistics.matches,
            away_statistics_matches=away_statistics.matches,

            home_points_per_match=home_form.points_per_match,
            away_points_per_match=away_form.points_per_match,

            home_average_goals_for=home_form.average_goals_for,
            away_average_goals_for=away_form.average_goals_for,

            home_average_goals_against=(
                home_form.average_goals_against
            ),
            away_average_goals_against=(
                away_form.average_goals_against
            ),

            home_average_expected_goals=(
                home_statistics.average_expected_goals
            ),
            away_average_expected_goals=(
                away_statistics.average_expected_goals
            ),

            home_average_total_shots=(
                home_statistics.average_total_shots
            ),
            away_average_total_shots=(
                away_statistics.average_total_shots
            ),

            home_average_shots_on_goal=(
                home_statistics.average_shots_on_goal
            ),
            away_average_shots_on_goal=(
                away_statistics.average_shots_on_goal
            ),

            home_average_ball_possession=(
                home_statistics.average_ball_possession
            ),
            away_average_ball_possession=(
                away_statistics.average_ball_possession
            ),

            home_average_corner_kicks=(
                home_statistics.average_corner_kicks
            ),
            away_average_corner_kicks=(
                away_statistics.average_corner_kicks
            ),

            home_average_yellow_cards=(
                home_statistics.average_yellow_cards
            ),
            away_average_yellow_cards=(
                away_statistics.average_yellow_cards
            ),

            home_form=home_form.form,
            away_form=away_form.form,

            form_advantage=self._advantage(
                home_value=home_form.points_per_match,
                away_value=away_form.points_per_match,
                home_name=home_team.name,
                away_name=away_team.name,
            ),

            attack_advantage=self._advantage(
                home_value=home_form.average_goals_for,
                away_value=away_form.average_goals_for,
                home_name=home_team.name,
                away_name=away_team.name,
            ),

            defence_advantage=(
                self._lower_is_better_advantage(
                    home_value=(
                        home_form.average_goals_against
                    ),
                    away_value=(
                        away_form.average_goals_against
                    ),
                    home_name=home_team.name,
                    away_name=away_team.name,
                )
            ),

            xg_advantage=self._statistics_advantage(
                home_value=(
                    home_statistics.average_expected_goals
                ),
                away_value=(
                    away_statistics.average_expected_goals
                ),
                home_matches=home_statistics.matches,
                away_matches=away_statistics.matches,
                home_name=home_team.name,
                away_name=away_team.name,
            ),
        )

    def _get_team(self, team_id: int) -> Team:
        team = (
            self.session.query(Team)
            .filter(Team.id == team_id)
            .first()
        )

        if not team:
            raise ValueError(
                f"Команда не найдена: team_id={team_id}"
            )

        return team

    @staticmethod
    def _advantage(
        home_value: float,
        away_value: float,
        home_name: str,
        away_name: str,
        tolerance: float = 0.05,
    ) -> str:
        """
        Сравнивает показатели, где большее значение лучше.
        """

        difference = home_value - away_value

        if abs(difference) <= tolerance:
            return "equal"

        if difference > 0:
            return home_name

        return away_name

    @staticmethod
    def _lower_is_better_advantage(
        home_value: float,
        away_value: float,
        home_name: str,
        away_name: str,
        tolerance: float = 0.05,
    ) -> str:
        """
        Сравнивает показатели, где меньшее значение лучше.

        Например, среднее количество пропущенных голов.
        """

        difference = home_value - away_value

        if abs(difference) <= tolerance:
            return "equal"

        if home_value < away_value:
            return home_name

        return away_name

    @classmethod
    def _statistics_advantage(
        cls,
        home_value: float,
        away_value: float,
        home_matches: int,
        away_matches: int,
        home_name: str,
        away_name: str,
        tolerance: float = 0.05,
    ) -> str:
        """
        Сравнивает статистические показатели только тогда,
        когда у обеих команд есть минимум по 3 матча
        со статистикой.
        """

        if (
            home_matches < cls.MIN_STATISTICS_MATCHES
            or away_matches < cls.MIN_STATISTICS_MATCHES
        ):
            return "insufficient_data"

        difference = home_value - away_value

        if abs(difference) <= tolerance:
            return "equal"

        if difference > 0:
            return home_name

        return away_name