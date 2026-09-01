from app.analytics.team_form_analyzer import TeamFormAnalyzer
from app.analytics.team_goalkeeper_analyzer import TeamGoalkeeperAnalyzer
from app.analytics.team_pass_analyzer import TeamPassAnalyzer
from app.analytics.team_possession_analyzer import TeamPossessionAnalyzer
from app.analytics.team_rest_days_analyzer import TeamRestDaysAnalyzer
from app.analytics.team_shot_efficiency_analyzer import (
    TeamShotEfficiencyAnalyzer,
)
from app.analytics.team_statistics_analyzer import TeamStatisticsAnalyzer
from app.analytics.team_venue_split_analyzer import TeamVenueSplitAnalyzer


class FeatureBuilder:
    """
    Построение признаков для ML
    без использования будущих матчей.
    """

    def __init__(self, session):
        self.session = session

        self.form = TeamFormAnalyzer(session)
        self.statistics = TeamStatisticsAnalyzer(session)
        self.venue = TeamVenueSplitAnalyzer(session)
        self.rest = TeamRestDaysAnalyzer(session)
        self.possession = TeamPossessionAnalyzer(session)
        self.passing = TeamPassAnalyzer(session)
        self.shot_efficiency = TeamShotEfficiencyAnalyzer(session)
        self.goalkeeper = TeamGoalkeeperAnalyzer(session)

    @staticmethod
    def _difference(home, away):
        """
        Абсолютная разница между хозяевами и гостями.
        """
        return abs(
            float(home or 0)
            - float(away or 0)
        )

    @staticmethod
    def _signed_difference(home, away):
        """
        Разница хозяева - гости.
        """
        return (
            float(home or 0)
            - float(away or 0)
        )

    @staticmethod
    def _closeness(home, away):
        """
        Близость двух команд.
        1.0 = показатели одинаковые.
        0.0 = показатели сильно различаются.
        """
        home = float(home or 0)
        away = float(away or 0)

        denominator = (
            abs(home)
            + abs(away)
            + 1e-9
        )

        return 1.0 - (
            abs(home - away)
            / denominator
        )

    def build(
        self,
        home_team_id: int,
        away_team_id: int,
        fixture_id: int,
    ) -> dict:
        common = {
            "limit": 10,
            "before_fixture_id": fixture_id,
        }

        home_form = self.form.analyze(
            team_id=home_team_id,
            **common,
        )

        away_form = self.form.analyze(
            team_id=away_team_id,
            **common,
        )

        home_stats = self.statistics.analyze(
            team_id=home_team_id,
            **common,
        )

        away_stats = self.statistics.analyze(
            team_id=away_team_id,
            **common,
        )

        home_venue = self.venue.analyze(
            team_id=home_team_id,
            limit=20,
            before_fixture_id=fixture_id,
        )

        away_venue = self.venue.analyze(
            team_id=away_team_id,
            limit=20,
            before_fixture_id=fixture_id,
        )

        home_rest = self.rest.analyze(
            team_id=home_team_id,
            **common,
        )

        away_rest = self.rest.analyze(
            team_id=away_team_id,
            **common,
        )

        home_possession = self.possession.analyze(
            team_id=home_team_id,
            **common,
        )

        away_possession = self.possession.analyze(
            team_id=away_team_id,
            **common,
        )

        home_passing = self.passing.analyze(
            team_id=home_team_id,
            **common,
        )

        away_passing = self.passing.analyze(
            team_id=away_team_id,
            **common,
        )

        home_shooting = self.shot_efficiency.analyze(
            team_id=home_team_id,
            **common,
        )

        away_shooting = self.shot_efficiency.analyze(
            team_id=away_team_id,
            **common,
        )

        home_goalkeeper = self.goalkeeper.analyze(
            team_id=home_team_id,
            **common,
        )

        away_goalkeeper = self.goalkeeper.analyze(
            team_id=away_team_id,
            **common,
        )

        features = {
            # ==========================================================
            # БАЗОВЫЕ ПРИЗНАКИ
            # ==========================================================

            "home_points_per_match":
                home_form.points_per_match,

            "away_points_per_match":
                away_form.points_per_match,

            "home_average_goals_for":
                home_form.average_goals_for,

            "away_average_goals_for":
                away_form.average_goals_for,

            "home_average_goals_against":
                home_form.average_goals_against,

            "away_average_goals_against":
                away_form.average_goals_against,

            "home_shots_on_goal":
                home_stats.average_shots_on_goal,

            "away_shots_on_goal":
                away_stats.average_shots_on_goal,

            "home_ball_possession":
                home_stats.average_ball_possession,

            "away_ball_possession":
                away_stats.average_ball_possession,

            "home_pass_accuracy":
                home_stats.average_passes_percentage,

            "away_pass_accuracy":
                away_stats.average_passes_percentage,

            "home_rest_days":
                home_rest.average_rest_days,

            "away_rest_days":
                away_rest.average_rest_days,

            "home_home_points":
                home_venue.home.points_per_match,

            "away_away_points":
                away_venue.away.points_per_match,

            "home_home_goals":
                home_venue.home.average_goals_for,

            "away_away_goals":
                away_venue.away.average_goals_for,

            "home_possession_matches":
                home_possession.matches,

            "away_possession_matches":
                away_possession.matches,

            "home_average_possession":
                home_possession.average_possession,

            "away_average_possession":
                away_possession.average_possession,

            "home_highest_possession":
                home_possession.highest_possession,

            "away_highest_possession":
                away_possession.highest_possession,

            "home_possession_above_50_percentage":
                home_possession.possession_above_50_percentage,

            "away_possession_above_50_percentage":
                away_possession.possession_above_50_percentage,

            "home_pass_matches":
                home_passing.matches,

            "away_pass_matches":
                away_passing.matches,

            "home_average_total_passes":
                home_passing.average_total_passes,

            "away_average_total_passes":
                away_passing.average_total_passes,

            "home_average_accurate_passes":
                home_passing.average_accurate_passes,

            "away_average_accurate_passes":
                away_passing.average_accurate_passes,

            "home_average_pass_accuracy":
                home_passing.average_pass_accuracy,

            "away_average_pass_accuracy":
                away_passing.average_pass_accuracy,

            "home_above_85_accuracy_percentage":
                home_passing.above_85_accuracy_percentage,

            "away_above_85_accuracy_percentage":
                away_passing.above_85_accuracy_percentage,

            "home_shooting_matches":
                home_shooting.matches,

            "away_shooting_matches":
                away_shooting.matches,

            "home_average_total_shots":
                home_shooting.average_total_shots,

            "away_average_total_shots":
                away_shooting.average_total_shots,

            "home_average_shots_on_goal":
                home_shooting.average_shots_on_goal,

            "away_average_shots_on_goal":
                away_shooting.average_shots_on_goal,

            "home_shot_accuracy_percentage":
                home_shooting.shot_accuracy_percentage,

            "away_shot_accuracy_percentage":
                away_shooting.shot_accuracy_percentage,

            "home_goal_conversion_percentage":
                home_shooting.goal_conversion_percentage,

            "away_goal_conversion_percentage":
                away_shooting.goal_conversion_percentage,

            "home_shots_per_goal":
                home_shooting.shots_per_goal,

            "away_shots_per_goal":
                away_shooting.shots_per_goal,

            "home_goalkeeper_matches":
                home_goalkeeper.matches,

            "away_goalkeeper_matches":
                away_goalkeeper.matches,

            "home_average_saves":
                home_goalkeeper.average_saves,

            "away_average_saves":
                away_goalkeeper.average_saves,

            "home_average_goals_conceded_gk":
                home_goalkeeper.average_goals_conceded,

            "away_average_goals_conceded_gk":
                away_goalkeeper.average_goals_conceded,

            "home_save_percentage":
                home_goalkeeper.save_percentage,

            "away_save_percentage":
                away_goalkeeper.save_percentage,

            "home_clean_sheet_percentage":
                home_goalkeeper.clean_sheet_percentage,

            "away_clean_sheet_percentage":
                away_goalkeeper.clean_sheet_percentage,
        }

        # ==============================================================
        # НОВЫЕ ПАРНЫЕ ПРИЗНАКИ
        # ==============================================================

        features.update(
            {
                # Форма
                "points_difference":
                    self._signed_difference(
                        home_form.points_per_match,
                        away_form.points_per_match,
                    ),

                "points_closeness":
                    self._closeness(
                        home_form.points_per_match,
                        away_form.points_per_match,
                    ),

                # Атака
                "goals_for_difference":
                    self._signed_difference(
                        home_form.average_goals_for,
                        away_form.average_goals_for,
                    ),

                "goals_for_closeness":
                    self._closeness(
                        home_form.average_goals_for,
                        away_form.average_goals_for,
                    ),

                # Защита
                "goals_against_difference":
                    self._signed_difference(
                        home_form.average_goals_against,
                        away_form.average_goals_against,
                    ),

                # Владение
                "possession_difference":
                    self._signed_difference(
                        home_possession.average_possession,
                        away_possession.average_possession,
                    ),

                "possession_closeness":
                    self._closeness(
                        home_possession.average_possession,
                        away_possession.average_possession,
                    ),

                # Передачи
                "total_passes_difference":
                    self._signed_difference(
                        home_passing.average_total_passes,
                        away_passing.average_total_passes,
                    ),

                "accurate_passes_difference":
                    self._signed_difference(
                        home_passing.average_accurate_passes,
                        away_passing.average_accurate_passes,
                    ),

                "pass_accuracy_difference":
                    self._signed_difference(
                        home_passing.average_pass_accuracy,
                        away_passing.average_pass_accuracy,
                    ),

                "pass_accuracy_closeness":
                    self._closeness(
                        home_passing.average_pass_accuracy,
                        away_passing.average_pass_accuracy,
                    ),

                "high_accuracy_difference":
                    self._signed_difference(
                        home_passing.above_85_accuracy_percentage,
                        away_passing.above_85_accuracy_percentage,
                    ),

                # Удары
                "total_shots_difference":
                    self._signed_difference(
                        home_shooting.average_total_shots,
                        away_shooting.average_total_shots,
                    ),

                "shots_on_goal_difference":
                    self._signed_difference(
                        home_shooting.average_shots_on_goal,
                        away_shooting.average_shots_on_goal,
                    ),

                "shot_accuracy_difference":
                    self._signed_difference(
                        home_shooting.shot_accuracy_percentage,
                        away_shooting.shot_accuracy_percentage,
                    ),

                "goal_conversion_difference":
                    self._signed_difference(
                        home_shooting.goal_conversion_percentage,
                        away_shooting.goal_conversion_percentage,
                    ),

                "shots_per_goal_difference":
                    self._signed_difference(
                        home_shooting.shots_per_goal,
                        away_shooting.shots_per_goal,
                    ),

                # Вратари
                "saves_difference":
                    self._signed_difference(
                        home_goalkeeper.average_saves,
                        away_goalkeeper.average_saves,
                    ),

                "save_percentage_difference":
                    self._signed_difference(
                        home_goalkeeper.save_percentage,
                        away_goalkeeper.save_percentage,
                    ),

                "clean_sheet_difference":
                    self._signed_difference(
                        home_goalkeeper.clean_sheet_percentage,
                        away_goalkeeper.clean_sheet_percentage,
                    ),

                # Общая близость атаки
                "attack_closeness":
                    (
                        self._closeness(
                            home_form.average_goals_for,
                            away_form.average_goals_for,
                        )
                        + self._closeness(
                            home_shooting.goal_conversion_percentage,
                            away_shooting.goal_conversion_percentage,
                        )
                        + self._closeness(
                            home_shooting.shot_accuracy_percentage,
                            away_shooting.shot_accuracy_percentage,
                        )
                    ) / 3.0,
            }
        )

        return features