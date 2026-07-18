import argparse

from app.core.logger import logger
from app.importers.player_importer import PlayerImporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Импорт игроков из API-Football."
        )
    )

    parser.add_argument(
        "--team-id",
        type=int,
        default=None,
        help=(
            "API ID конкретной команды. "
            "Например: 33 для Manchester United."
        ),
    )

    parser.add_argument(
        "--league-id",
        type=int,
        default=39,
        help=(
            "API ID лиги. "
            "По умолчанию: 39 — Premier League."
        ),
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help=(
            "Сезон для импорта. "
            "По умолчанию: 2024."
        ),
    )

    parser.add_argument(
        "--team-limit",
        type=int,
        default=1,
        help=(
            "Количество команд для обработки. "
            "По умолчанию: 1."
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Пропускать команды, у которых "
            "в базе уже есть минимум 40 игроков."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.team_limit < 1:
        logger.error(
            "Параметр --team-limit "
            "должен быть больше нуля."
        )
        return

    if args.season < 2000:
        logger.error(
            "Указан некорректный сезон."
        )
        return

    logger.info("Запуск импорта игроков...")

    if args.team_id is not None:
        logger.info(
            f"Выбрана конкретная команда: "
            f"api_id={args.team_id}"
        )
    else:
        logger.info(
            f"Выбрана лига: "
            f"api_id={args.league_id}, "
            f"лимит команд={args.team_limit}"
        )

    if args.skip_existing:
        logger.info(
            "Уже загруженные команды "
            "будут пропущены."
        )

    importer = PlayerImporter(
        team_api_id=args.team_id,
        league_api_id=args.league_id,
        season=args.season,
        team_limit=args.team_limit,
        skip_existing=args.skip_existing,
    )

    success = importer.run()

    if success:
        logger.success(
            "Импорт игроков завершён."
        )
    else:
        logger.error(
            "Импорт игроков завершился "
            "с ошибкой."
        )


if __name__ == "__main__":
    main()