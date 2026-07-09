from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import relationship

from app.database.base import Base


class League(Base):
    """
    Модель футбольной лиги.
    """

    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    api_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    type: Mapped[str] = mapped_column(String(50))

    logo: Mapped[str] = mapped_column(String(500))

    country: Mapped[str] = mapped_column(String(100))

    country_code: Mapped[str] = mapped_column(String(10))

    flag: Mapped[str] = mapped_column(String(500))

    

    teams: Mapped[list["Team"]] = relationship(
    "Team",
    back_populates="league",
    cascade="all, delete-orphan",
)
    seasons: Mapped[list["LeagueSeason"]] = relationship(
    "LeagueSeason",
    back_populates="league",
    cascade="all, delete-orphan",
)