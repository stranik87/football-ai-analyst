from app.models.venue import Venue


class VenueRepository:
    """
    Репозиторий стадионов.
    """

    def __init__(self, db):
        self.db = db

    def get_by_api_id(self, api_id: int):
        return (
            self.db.query(Venue)
            .filter_by(api_id=api_id)
            .first()
        )

    def add(self, venue: Venue):
        self.db.add(venue)

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()