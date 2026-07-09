from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Team(Base):
    """
    Модель футбольной команды.
    """

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    api_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
    )

    league_id: Mapped[int] = mapped_column(
        ForeignKey("leagues.id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    code: Mapped[str] = mapped_column(
        String(20),
        default="",
    )

    country: Mapped[str] = mapped_column(
        String(100),
        default="",
    )

    founded: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    logo: Mapped[str] = mapped_column(
        String(500),
        default="",
    )

    league: Mapped["League"] = relationship(
    "League",
    back_populates="teams",
        )

    venue_id: Mapped[int | None] = mapped_column(
    ForeignKey("venues.id"),
    nullable=True,
    )

    venue: Mapped["Venue"] = relationship(
    "Venue",
    back_populates="teams",
)