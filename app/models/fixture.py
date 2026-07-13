from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Fixture(Base):
    """
    Футбольный матч.
    """

    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    api_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
        index=True,
    )

    league_season_id: Mapped[int] = mapped_column(
        ForeignKey("league_seasons.id"),
        nullable=False,
        index=True,
    )

    home_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"),
        nullable=False,
        index=True,
    )

    away_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id"),
        nullable=False,
        index=True,
    )

    venue_id: Mapped[int | None] = mapped_column(
        ForeignKey("venues.id"),
        nullable=True,
    )

    kickoff: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    timestamp: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    timezone: Mapped[str] = mapped_column(
        String(50),
        default="UTC",
        nullable=False,
    )

    round: Mapped[str] = mapped_column(
        String(100),
        default="",
        nullable=False,
    )

    referee: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    status_short: Mapped[str] = mapped_column(
        String(20),
        default="",
        nullable=False,
    )

    status_long: Mapped[str] = mapped_column(
        String(100),
        default="",
        nullable=False,
    )

    elapsed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    extra_time: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    home_goals: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    away_goals: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    halftime_home: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    halftime_away: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fulltime_home: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fulltime_away: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    extratime_home: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    extratime_away: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    penalty_home: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    penalty_away: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    league_season = relationship("LeagueSeason")
    venue = relationship("Venue")

    home_team = relationship(
        "Team",
        foreign_keys=[home_team_id],
    )

    away_team = relationship(
        "Team",
        foreign_keys=[away_team_id],
    )