from datetime import date

from sqlalchemy import Boolean, Date, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class LeagueSeason(Base):
    __tablename__ = "league_seasons"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    league_id: Mapped[int] = mapped_column(
        ForeignKey("leagues.id"),
        nullable=False,
    )

    season: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    start: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    end: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    current: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    league: Mapped["League"] = relationship(
        "League",
        back_populates="seasons",
    )