"""make player fields nullable

Revision ID: 1e1d79a9e3e9
Revises: aa4c5bcd19d4
Create Date: 2026-07-18 21:21:22.132842

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1e1d79a9e3e9"
down_revision: Union[str, Sequence[str], None] = "aa4c5bcd19d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Разрешить NULL в необязательных полях игроков."""

    with op.batch_alter_table("players") as batch_op:
        batch_op.alter_column(
            "firstname",
            existing_type=sa.String(length=100),
            nullable=True,
        )
        batch_op.alter_column(
            "lastname",
            existing_type=sa.String(length=100),
            nullable=True,
        )
        batch_op.alter_column(
            "age",
            existing_type=sa.Integer(),
            nullable=True,
        )
        batch_op.alter_column(
            "birth_place",
            existing_type=sa.String(length=100),
            nullable=True,
        )
        batch_op.alter_column(
            "birth_country",
            existing_type=sa.String(length=100),
            nullable=True,
        )
        batch_op.alter_column(
            "nationality",
            existing_type=sa.String(length=100),
            nullable=True,
        )
        batch_op.alter_column(
            "height",
            existing_type=sa.String(length=20),
            nullable=True,
        )
        batch_op.alter_column(
            "weight",
            existing_type=sa.String(length=20),
            nullable=True,
        )
        batch_op.alter_column(
            "position",
            existing_type=sa.String(length=50),
            nullable=True,
        )
        batch_op.alter_column(
            "photo",
            existing_type=sa.String(length=500),
            nullable=True,
        )


def downgrade() -> None:
    """Вернуть обязательность полей игроков."""

    # Перед возвратом NOT NULL заменяем существующие NULL,
    # иначе SQLite не сможет пересоздать таблицу.
    op.execute(
        """
        UPDATE players
        SET
            firstname = COALESCE(firstname, ''),
            lastname = COALESCE(lastname, ''),
            age = COALESCE(age, 0),
            birth_place = COALESCE(birth_place, ''),
            birth_country = COALESCE(birth_country, ''),
            nationality = COALESCE(nationality, ''),
            height = COALESCE(height, ''),
            weight = COALESCE(weight, ''),
            position = COALESCE(position, ''),
            photo = COALESCE(photo, '')
        """
    )

    with op.batch_alter_table("players") as batch_op:
        batch_op.alter_column(
            "firstname",
            existing_type=sa.String(length=100),
            nullable=False,
        )
        batch_op.alter_column(
            "lastname",
            existing_type=sa.String(length=100),
            nullable=False,
        )
        batch_op.alter_column(
            "age",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.alter_column(
            "birth_place",
            existing_type=sa.String(length=100),
            nullable=False,
        )
        batch_op.alter_column(
            "birth_country",
            existing_type=sa.String(length=100),
            nullable=False,
        )
        batch_op.alter_column(
            "nationality",
            existing_type=sa.String(length=100),
            nullable=False,
        )
        batch_op.alter_column(
            "height",
            existing_type=sa.String(length=20),
            nullable=False,
        )
        batch_op.alter_column(
            "weight",
            existing_type=sa.String(length=20),
            nullable=False,
        )
        batch_op.alter_column(
            "position",
            existing_type=sa.String(length=50),
            nullable=False,
        )
        batch_op.alter_column(
            "photo",
            existing_type=sa.String(length=500),
            nullable=False,
        )