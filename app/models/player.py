from datetime import date

from sqlalchemy import Boolean, Date, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Player(Base):
    """
    Модель футбольного игрока.
    """

    __tablename__ = "players"

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

    team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id"),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    firstname: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    lastname: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    age: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    birth_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    birth_place: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    birth_country: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    nationality: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    height: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    weight: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    position: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    photo: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    is_injured: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    team: Mapped["Team | None"] = relationship(
        "Team",
        back_populates="players",
    )