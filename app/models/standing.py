from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Standing(Base):
    """
    Положение команды в турнирной таблице.
    """

    __tablename__ = "standings"

    __table_args__ = (
        UniqueConstraint(
            "league_season_id",
            "team_id",
            name="uq_standing_league_season_team",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    league_season_id: Mapped[int] = mapped_column(
        ForeignKey("league_seasons.id"),
        nullable=False,
        index=True,
    )

    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"),
        nullable=False,
        index=True,
    )

    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    points: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    goals_diff: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    group_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    form: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    status: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    played: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    wins: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    draws: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    losses: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    goals_for: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    goals_against: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    home_played: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    home_wins: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    home_draws: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    home_losses: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    home_goals_for: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    home_goals_against: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    away_played: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    away_wins: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    away_draws: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    away_losses: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    away_goals_for: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    away_goals_against: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    league_season = relationship("LeagueSeason")
    team = relationship("Team")