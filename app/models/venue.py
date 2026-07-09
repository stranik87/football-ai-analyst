from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Venue(Base):
    """
    Модель стадиона.
    """

    __tablename__ = "venues"

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

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    address: Mapped[str] = mapped_column(
        String(255),
        default="",
    )

    city: Mapped[str] = mapped_column(
        String(100),
        default="",
    )

    capacity: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    surface: Mapped[str] = mapped_column(
        String(50),
        default="",
    )

    image: Mapped[str] = mapped_column(
        String(500),
        default="",
    )

    teams = relationship(
        "Team",
        back_populates="venue",
    )   