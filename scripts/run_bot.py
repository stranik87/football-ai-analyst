from app.bot.bot import FootballTelegramBot
from app.core.logger import logger


def main() -> None:
    try:
        bot = FootballTelegramBot()
        bot.run()
    except Exception:
        logger.exception(
            "Не удалось запустить Telegram-бота."
        )
        raise


if __name__ == "__main__":
    main()