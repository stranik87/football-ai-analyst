import argparse

from app.core.logger import logger
from app.importers.fixture_importer import FixtureImporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Импорт и обновление футбольных матчей."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help=(
            "Сезон для импорта. "
            "Например: 2025 или 2026."
        ),
    )

    parser.add_argument(
        "--league-id",
        type=int,
        action="append",
        dest="league_ids",
        default=None,
        help=(
            "API ID лиги. Параметр можно указать "
            "несколько раз. Например: "
            "--league-id 39 --league-id 140"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.season < 2000:
        logger.error(
            "Указан некорректный сезон."
        )
        return

    logger.info(
        "Запуск импорта матчей: "
        f"season={args.season}, "
        f"leagues={args.league_ids or 'все поддерживаемые'}"
    )

    importer = FixtureImporter(
        season=args.season,
        league_api_ids=args.league_ids,
    )

    success = importer.run()

    if success:
        logger.success(
            "Импорт матчей завершён."
        )
    else:
        logger.error(
            "Импорт матчей завершился с ошибкой."
        )


if __name__ == "__main__":
    main()