from typing import Any

from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import ContextTypes, ConversationHandler

from app.core.logger import logger
from app.database.database import SessionLocal
from app.services.prediction_service import PredictionService


WAITING_FIXTURE_ID = 1


MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        ["📊 Прогноз матча"],
        ["ℹ️ Помощь"],
    ],
    resize_keyboard=True,
    is_persistent=True,
)


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Команда /start.
    """

    if update.message is None:
        return

    context.user_data.clear()

    text = (
        "⚽ Football AI Analyst\n\n"
        "Бот анализирует футбольные матчи "
        "и рассчитывает вероятности исходов.\n\n"
        "Нажми «📊 Прогноз матча» "
        "или используй команду:\n"
        "/predict <ID матча>\n\n"
        "Пример:\n"
        "/predict 1377"
    )

    await update.message.reply_text(
        text,
        reply_markup=MAIN_MENU_KEYBOARD,
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Команда /help.
    """

    if update.message is None:
        return

    text = (
        "📖 Помощь\n\n"
        "Получить прогноз можно двумя способами:\n\n"
        "1. Нажать кнопку «📊 Прогноз матча» "
        "и отправить ID матча.\n\n"
        "2. Ввести команду:\n"
        "/predict <ID матча>\n\n"
        "Пример:\n"
        "/predict 1377\n\n"
        "Для отмены ввода используй:\n"
        "/cancel"
    )

    await update.message.reply_text(
        text,
        reply_markup=MAIN_MENU_KEYBOARD,
    )


async def start_prediction_dialog(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    """
    Начать диалог получения прогноза.
    """

    if update.message is None:
        return ConversationHandler.END

    await update.message.reply_text(
        "📊 Введи ID матча одним числом.\n\n"
        "Например:\n"
        "1377\n\n"
        "Для отмены введи /cancel",
        reply_markup=MAIN_MENU_KEYBOARD,
    )

    return WAITING_FIXTURE_ID


async def receive_fixture_id(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    """
    Получить ID матча обычным сообщением.
    """

    if update.message is None:
        return ConversationHandler.END

    fixture_id_value = (
        update.message.text or ""
    ).strip()

    try:
        fixture_id = int(fixture_id_value)
    except ValueError:
        await update.message.reply_text(
            "❌ ID матча должен быть целым числом.\n\n"
            "Попробуй ещё раз, например:\n"
            "1377\n\n"
            "Для отмены введи /cancel"
        )
        return WAITING_FIXTURE_ID

    if fixture_id <= 0:
        await update.message.reply_text(
            "❌ ID матча должен быть больше нуля.\n"
            "Попробуй ещё раз."
        )
        return WAITING_FIXTURE_ID

    await send_prediction(
        update=update,
        fixture_id=fixture_id,
    )

    return ConversationHandler.END


async def predict_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Команда /predict <fixture_id>.
    """

    if update.message is None:
        return

    if not context.args:
        await update.message.reply_text(
            "Укажи ID матча.\n\n"
            "Пример:\n"
            "/predict 1377",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return

    fixture_id_value = context.args[0]

    try:
        fixture_id = int(fixture_id_value)
    except ValueError:
        await update.message.reply_text(
            "ID матча должен быть целым числом.\n\n"
            "Пример:\n"
            "/predict 1377",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return

    if fixture_id <= 0:
        await update.message.reply_text(
            "ID матча должен быть больше нуля.",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return

    await send_prediction(
        update=update,
        fixture_id=fixture_id,
    )


async def send_prediction(
    update: Update,
    fixture_id: int,
) -> None:
    """
    Построить прогноз и отправить его пользователю.
    """

    if update.message is None:
        return

    status_message = await update.message.reply_text(
        "⏳ Анализирую матч..."
    )

    session = SessionLocal()

    try:
        service = PredictionService(session)
        prediction = service.predict(fixture_id)

        response_text = build_prediction_text(
            prediction
        )

        await status_message.edit_text(
            response_text
        )

        await update.message.reply_text(
            "Выбери следующее действие:",
            reply_markup=MAIN_MENU_KEYBOARD,
        )

        logger.info(
            "Telegram-прогноз: "
            f"fixture_id={fixture_id}, "
            f"result={prediction['predicted_result']}, "
            f"confidence="
            f"{prediction['confidence']:.4f}"
        )

    except ValueError as error:
        logger.warning(
            f"Ошибка Telegram-прогноза: {error}"
        )

        await status_message.edit_text(
            f"❌ {error}"
        )

    except FileNotFoundError as error:
        logger.error(
            f"Файл модели не найден: {error}"
        )

        await status_message.edit_text(
            "❌ Модель прогнозирования не найдена."
        )

    except Exception:
        logger.exception(
            "Непредвиденная ошибка Telegram-прогноза."
        )

        await status_message.edit_text(
            "❌ Не удалось построить прогноз.\n"
            "Ошибка записана в журнал."
        )

    finally:
        session.close()


def build_prediction_text(
    prediction: dict[str, Any],
) -> str:
    """
    Сформировать текст прогноза.
    """

    probabilities = prediction["probabilities"]

    home_probability = (
        probabilities["home_win"] * 100
    )
    draw_probability = (
        probabilities["draw"] * 100
    )
    away_probability = (
        probabilities["away_win"] * 100
    )
    confidence = prediction["confidence"] * 100

    kickoff = prediction.get("kickoff")

    if kickoff is not None:
        kickoff_text = kickoff.strftime(
            "%d.%m.%Y %H:%M"
        )
    else:
        kickoff_text = "не указана"

    actual_score = (
        prediction.get("actual_score") or {}
    )

    home_goals = actual_score.get("home_goals")
    away_goals = actual_score.get("away_goals")

    score_text = ""

    if (
        home_goals is not None
        and away_goals is not None
    ):
        score_text = (
            f"\nФактический счёт: "
            f"{home_goals}:{away_goals}\n"
        )

    return (
        "⚽ Прогноз матча\n\n"
        f"{prediction['home_team']} — "
        f"{prediction['away_team']}\n"
        f"Дата: {kickoff_text}\n"
        f"{score_text}\n"
        "Вероятности:\n"
        f"🏠 Победа хозяев: "
        f"{home_probability:.1f}%\n"
        f"🤝 Ничья: "
        f"{draw_probability:.1f}%\n"
        f"✈️ Победа гостей: "
        f"{away_probability:.1f}%\n\n"
        f"Прогноз: "
        f"{prediction['predicted_result_name']}\n"
        f"Уверенность модели: "
        f"{confidence:.1f}%"
    )


async def cancel_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    """
    Отменить текущий диалог.
    """

    if update.message is not None:
        await update.message.reply_text(
            "Ввод отменён.",
            reply_markup=MAIN_MENU_KEYBOARD,
        )

    return ConversationHandler.END


async def unknown_message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Обработка неизвестного текста.
    """

    if update.message is None:
        return

    text = update.message.text or ""

    if text == "ℹ️ Помощь":
        await help_command(update, context)
        return

    await update.message.reply_text(
        "Команда не распознана.\n"
        "Выбери действие в меню.",
        reply_markup=MAIN_MENU_KEYBOARD,
    )