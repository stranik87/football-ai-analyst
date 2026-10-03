from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)

from config import Config
from app.bot.handlers import (
    WAITING_FIXTURE_ID,
    cancel_command,
    help_command,
    predict_command,
    receive_fixture_id,
    start_command,
    start_prediction_dialog,
    unknown_message_handler,
    fixture_callback_handler,
    fixture_section_callback_handler,
    show_fixture_list,
    next_matches_command,
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

        prediction_conversation = ConversationHandler(
            entry_points=[
                MessageHandler(
                    filters.Regex(
                        r"^📊 Прогноз матча$"
                    ),
                    show_fixture_list,
                ),
            ],
            states={
                WAITING_FIXTURE_ID: [
                    CallbackQueryHandler(
                        fixture_callback_handler,
                        pattern=(
                            r"^(predict_fixture:\d+|"
                            r"predict_manual)$"
            ),
        ),
                    MessageHandler(
                        filters.TEXT
                        & ~filters.COMMAND,
                        receive_fixture_id,
        ),
    ],
},
            fallbacks=[
                CommandHandler(
                    "cancel",
                    cancel_command,
                ),
                CommandHandler(
                    "start",
                    start_command,
                ),
            ],
            allow_reentry=True,
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
            CommandHandler(
                "next",
                next_matches_command,
    )
)

        application.add_handler(
            MessageHandler(
                filters.Regex(r"^\U0001f4c5 \u0411\u043b\u0438\u0436\u0430\u0439\u0448\u0438\u0435 \u043c\u0430\u0442\u0447\u0438$"),
                next_matches_command,
            )
        )

        application.add_handler(
            MessageHandler(
                filters.Regex(r"^\u2139\ufe0f \u041f\u043e\u043c\u043e\u0449\u044c$"),
                help_command,
            )
        )

        application.add_handler(
            prediction_conversation
        )

        application.add_handler(
            CallbackQueryHandler(
                fixture_callback_handler,
                pattern=r"^predict_fixture:\d+$",
            )
        )

        application.add_handler(
            CallbackQueryHandler(
                fixture_section_callback_handler,
                pattern=r"^(fixture_section:|fixture_back_to_menu$|fixture_back_to_list$|fixture_main_menu$)",
            )
        )

        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                unknown_message_handler,
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

if __name__ == "__main__":
    FootballTelegramBot().run()
