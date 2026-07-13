from sqlalchemy import Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class FixtureTeamStatistics(Base):
    """
    Статистика одной команды в конкретном матче.
    """

    __tablename__ = "fixture_team_statistics"

    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "team_id",
            name="uq_fixture_team_statistics",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id"),
        nullable=False,
        index=True,
    )

    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"),
        nullable=False,
        index=True,
    )

    shots_on_goal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shots_off_goal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blocked_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)

    shots_inside_box: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    shots_outside_box: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corner_kicks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    offsides: Mapped[int | None] = mapped_column(Integer, nullable=True)

    ball_possession: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)

    goalkeeper_saves: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    total_passes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passes_accurate: Mapped[int | None] = mapped_column(Integer, nullable=True)

    passes_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    expected_goals: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    goals_prevented: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    fixture = relationship("Fixture")
    team = relationship("Team")