from telegram import Update
from telegram.ext import ContextTypes

from app.core.logger import logger
from app.database.database import SessionLocal
from app.services.prediction_service import PredictionService


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Команда /start.
    """

    if update.message is None:
        return

    text = (
        "⚽ Football AI Analyst\n\n"
        "Бот анализирует футбольные матчи "
        "и рассчитывает вероятности исходов.\n\n"
        "Доступные команды:\n"
        "/start — запуск бота\n"
        "/help — помощь\n"
        "/predict <ID матча> — прогноз матча\n\n"
        "Пример:\n"
        "/predict 1377"
    )

    await update.message.reply_text(text)


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
        "Для получения прогноза введи:\n"
        "/predict <ID матча>\n\n"
        "Пример:\n"
        "/predict 1377\n\n"
        "Бот покажет:\n"
        "• команды;\n"
        "• дату матча;\n"
        "• вероятности исходов;\n"
        "• основной прогноз;\n"
        "• уверенность модели."
    )

    await update.message.reply_text(text)


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
            "/predict 1377"
        )
        return

    fixture_id_value = context.args[0]

    try:
        fixture_id = int(fixture_id_value)
    except ValueError:
        await update.message.reply_text(
            "ID матча должен быть целым числом.\n\n"
            "Пример:\n"
            "/predict 1377"
        )
        return

    if fixture_id <= 0:
        await update.message.reply_text(
            "ID матча должен быть больше нуля."
        )
        return

    status_message = await update.message.reply_text(
        "⏳ Анализирую матч..."
    )

    session = SessionLocal()

    try:
        service = PredictionService(session)
        prediction = service.predict(fixture_id)

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

        actual_score = prediction.get("actual_score") or {}

        home_goals = actual_score.get("home_goals")
        away_goals = actual_score.get("away_goals")

        score_text = ""

        if (
            home_goals is not None
            and away_goals is not None
        ):
            score_text = (
                "\n"
                f"Фактический счёт: "
                f"{home_goals}:{away_goals}\n"
            )

        response_text = (
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

        await status_message.edit_text(
            response_text
        )

        logger.info(
            "Telegram-прогноз: "
            f"fixture_id={fixture_id}, "
            f"result={prediction['predicted_result']}, "
            f"confidence={prediction['confidence']:.4f}"
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