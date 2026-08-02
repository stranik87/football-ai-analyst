from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import Config
from app.bot.handlers import (
    help_command,
    menu_message_handler,
    predict_command,
    start_command,
)
from app.core.logger import logger


class FootballTelegramBot:
    """
    Telegram-бот проекта Football AI Analyst.
    """

    def __init__(self) -> None:
        self.token = Config.TELEGRAM_BOT_TOKEN

        if not self.token:
            raise ValueError(
                "Переменная TELEGRAM_BOT_TOKEN "
                "не указана в файле .env."
            )

        self.application = self._build_application()

    def _build_application(self) -> Application:
        application = (
            ApplicationBuilder()
            .token(self.token)
            .build()
        )

        application.add_handler(
            CommandHandler(
                "start",
                start_command,
            )
        )

        application.add_handler(
            CommandHandler(
                "help",
                help_command,
            )
        )

        application.add_handler(
            CommandHandler(
                "predict",
                predict_command,
            )
        )

        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                menu_message_handler,
            )
        )

        application.add_error_handler(
            self._error_handler
        )

        return application

    @staticmethod
    async def _error_handler(
        update: object,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        logger.error(
            "Ошибка Telegram-бота: "
            f"{context.error}"
        )

        if isinstance(update, Update):
            message = update.effective_message

            if message is not None:
                await message.reply_text(
                    "❌ Произошла внутренняя ошибка."
                )

    def run(self) -> None:
        logger.info(
            "Запуск Telegram-бота..."
        )

        self.application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )