from app.database.base import Base
from app.database.database import engine

from app.models.league import League
from app.models.team import Team


def init_database():
    Base.metadata.create_all(bind=engine)