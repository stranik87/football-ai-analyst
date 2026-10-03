from typing import Any

from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import ContextTypes, ConversationHandler

from app.core.logger import logger
from app.database.database import SessionLocal
from app.services.prediction_service import PredictionService

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
    
)

from app.services.fixture_service import FixtureService

WAITING_FIXTURE_ID = 1


MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        ["📊 Прогноз матча"],
        ["📅 Ближайшие матчи"],
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

def build_fixture_list_keyboard(fixtures):
    keyboard = []

    for fixture in fixtures:
        home_team = (
            fixture.home_team.name
            if fixture.home_team is not None
            else "\u041d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u0430\u044f \u043a\u043e\u043c\u0430\u043d\u0434\u0430"
        )

        away_team = (
            fixture.away_team.name
            if fixture.away_team is not None
            else "\u041d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u0430\u044f \u043a\u043e\u043c\u0430\u043d\u0434\u0430"
        )

        keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"{home_team} — {away_team}",
                    callback_data=f"predict_fixture:{fixture.id}",
                )
            ]
        )

    keyboard.append(
        [
            InlineKeyboardButton(
                text="\u2328\ufe0f \u0412\u0432\u0435\u0441\u0442\u0438 ID \u0432\u0440\u0443\u0447\u043d\u0443\u044e",
                callback_data="predict_manual",
            )
        ]
    )

    return InlineKeyboardMarkup(keyboard)


async def show_fixture_list(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    """
    Показать последние матчи для выбора.
    """

    if update.message is None:
        return ConversationHandler.END

    session = SessionLocal()

    try:
        fixture_service = FixtureService(session)
        fixtures = fixture_service.get_latest_matches(
            limit=5
        )

        if not fixtures:
            await update.message.reply_text(
                "Матчи в базе не найдены.",
                reply_markup=MAIN_MENU_KEYBOARD,
            )
            return ConversationHandler.END

        keyboard = build_fixture_list_keyboard(fixtures)

        await update.message.reply_text(
            "⚽ Выбери матч:",
            reply_markup=InlineKeyboardMarkup(
                keyboard
            ),
        )

        return WAITING_FIXTURE_ID

    finally:
        session.close()

def build_fixture_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("\U0001f4ca \u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043f\u0440\u043e\u0433\u043d\u043e\u0437", callback_data="fixture_section:main")],
        [InlineKeyboardButton("\u26bd \u0413\u043e\u043b\u044b", callback_data="fixture_section:goals")],
        [InlineKeyboardButton("\U0001f6a9 \u0423\u0433\u043b\u043e\u0432\u044b\u0435", callback_data="fixture_section:corners")],
        [InlineKeyboardButton("\U0001f7e8 \u0416\u0451\u043b\u0442\u044b\u0435 \u043a\u0430\u0440\u0442\u043e\u0447\u043a\u0438", callback_data="fixture_section:yellow_cards")],
        [InlineKeyboardButton("\U0001f3af \u0423\u0434\u0430р\u044b", callback_data="fixture_section:shots")],
        [InlineKeyboardButton("\U0001f3af \u0423\u0434\u0430р\u044b \u0432 \u0441твор", callback_data="fixture_section:shots_on_target")],
        [InlineKeyboardButton("\U0001f6a9 \u041e\u0444\u0441а\u0439д\u044b", callback_data="fixture_section:offsides")],
        [InlineKeyboardButton("\U0001f7e8 \u0424\u043e\u043b\u044b", callback_data="fixture_section:fouls")],
        [InlineKeyboardButton("\U0001f519 \u041a \u0441\u043f\u0438\u0441\u043a\u0443 \u043c\u0430\u0442\u0447\u0435\u0439", callback_data="fixture_back_to_list")],
        [InlineKeyboardButton("\U0001f3e0 \u0413\u043b\u0430\u0432\u043d\u043e\u0435 \u043c\u0435\u043d\u044e", callback_data="fixture_main_menu")],
    ]
    return InlineKeyboardMarkup(keyboard)

def build_fixture_menu_text(prediction: dict[str, Any]) -> str:
    kickoff = prediction.get("kickoff")

    if kickoff:
        kickoff_text = kickoff.strftime("%d.%m.%Y %H:%M")
    else:
        kickoff_text = "\u043d\u0435 \u0443\u043a\u0430\u0437\u0430\u043d\u043e"

    lines = [
        "\u0412\u044b\u0431\u0440\u0430\u043d\u043d\u044b\u0439 \u043c\u0430\u0442\u0447",
        "",
        f"{prediction['home_team']} - {prediction['away_team']}",
        f"\u0414\u0430\u0442\u0430: {kickoff_text}",
        "",
        "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0440\u0430\u0437\u0434\u0435\u043b \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430:",
    ]

    return "\n".join(lines)


async def fixture_callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    query = update.callback_query

    if query is None:
        return ConversationHandler.END

    await query.answer()

    callback_data = query.data or ""

    if callback_data == "predict_manual":
        await query.edit_message_text(
            "Enter fixture ID as a number.\n\n"
            "Example:\n"
            "1377\n\n"
            "Use /cancel to cancel."
        )
        return WAITING_FIXTURE_ID

    if not callback_data.startswith("predict_fixture:"):
        await query.edit_message_text(
            "Could not determine selected fixture."
        )
        return ConversationHandler.END

    fixture_id_value = callback_data.split(
        ":",
        maxsplit=1,
    )[1]

    try:
        fixture_id = int(fixture_id_value)
    except ValueError:
        await query.edit_message_text(
            "Invalid fixture ID."
        )
        return ConversationHandler.END

    await query.edit_message_text(
        "Analyzing fixture..."
    )

    session = SessionLocal()

    try:
        service = PredictionService(session)
        prediction = service.predict(fixture_id)

        context.user_data["selected_prediction"] = prediction

        response_text = build_fixture_menu_text(
            prediction
        )

        await query.edit_message_text(
            response_text,
            reply_markup=build_fixture_menu_keyboard(),
        )

        logger.info(
            "Telegram fixture prediction: "
            f"fixture_id={fixture_id}, "
            f"result={prediction['prediction']}, "
            f"confidence="
            f"{prediction['confidence']:.4f}"
        )

    except ValueError as error:
        logger.warning(
            f"Telegram prediction error: {error}"
        )
        await query.edit_message_text(
            f"Error: {error}"
        )

    except FileNotFoundError as error:
        logger.error(
            f"Model file not found: {error}"
        )
        await query.edit_message_text(
            "Prediction model was not found."
        )

    except Exception:
        logger.exception(
            "Error predicting selected fixture."
        )
        await query.edit_message_text(
            "Could not build prediction.\n"
            "The error was logged."
        )

    finally:
        session.close()

    return ConversationHandler.END



def build_section_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "\u2b05\ufe0f \u041a \u0440\u0430\u0437\u0434\u0435\u043b\u0430\u043c",
                    callback_data="fixture_back_to_menu",
                )
            ],
            [
                InlineKeyboardButton(
                    "\ud83d\udd19 \u041a \u0441\u043f\u0438\u0441\u043a\u0443 \u043c\u0430\u0442\u0447\u0435\u0439",
                    callback_data="fixture_back_to_list",
                ),
                InlineKeyboardButton(
                    "\ud83c\udfe0 \u0413\u043b\u0430\u0432\u043d\u043e\u0435 \u043c\u0435\u043d\u044e",
                    callback_data="fixture_main_menu",
                ),
            ],
        ]
    )


def build_statistics_section_text(
    prediction: dict[str, Any],
    section: str,
) -> str:
    home = prediction["home_team"]
    away = prediction["away_team"]
    stats = prediction.get("statistics_predictions", {})

    if section == "goals":
        data = stats.get("goals")

        if not data:
            return "\u26a0\ufe0f \u041f\u0440\u043e\u0433\u043d\u043e\u0437 \u0433\u043e\u043b\u043e\u0432 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d."

        lines = [
            "\u26bd \u0413\u043e\u043b\u044b",
            "",
            f"{home} - {away}",
            "",
            f"\u041e\u0436\u0438\u0434\u0430\u0435\u043c\u044b\u0435 \u0433\u043e\u043b\u044b {home}: "
            f"{data['expected_home_goals']:.2f}",
            f"\u041e\u0436\u0438\u0434\u0430\u0435\u043c\u044b\u0435 \u0433\u043e\u043b\u044b {away}: "
            f"{data['expected_away_goals']:.2f}",
            f"\u041e\u0436\u0438\u0434\u0430\u0435\u043c\u044b\u0435 \u0433\u043e\u043b\u044b \u0432\u0441\u0435\u0433\u043e: "
            f"{data['expected_total_goals']:.2f}",
            "",
            "\U0001f550 1-\u0439 \u0442\u0430\u0439\u043c:",
            f"\u0412\u0441\u0435\u0433\u043e: {data['expected_1h_total_goals']:.2f}",
            f"{home}: {data['expected_1h_home_goals']:.2f}",
            f"{away}: {data['expected_1h_away_goals']:.2f}",
            "",
            "\U0001f551 2-\u0439 \u0442\u0430\u0439\u043c:",
            f"\u0412\u0441\u0435\u0433\u043e: {data['expected_2h_total_goals']:.2f}",
            f"{home}: {data['expected_2h_home_goals']:.2f}",
            f"{away}: {data['expected_2h_away_goals']:.2f}",
            "",
            "\u041e\u0431\u0449\u0438\u0439 \u0442\u043e\u0442\u0430\u043b:",
        ]

        for line, values in data["over_under"].items():
            lines.append(
                f"{line}: \u0411\u043e\u043b\u044c\u0448\u0435 "
                f"{values['over'] * 100:.1f}% | "
                f"\u041c\u0435\u043d\u044c\u0448\u0435 "
                f"{values['under'] * 100:.1f}%"
            )

        return "\n".join(lines)

    sections = {
        "corners": {
            "label": "\U0001f6a9 \u0423\u0433\u043b\u043e\u0432\u044b\u0435",
            "total": "corners_total",
            "home": "corners_home",
            "away": "corners_away",
        },
        "yellow_cards": {
            "label": "\U0001f7e8 \u0416\u0451\u043b\u0442\u044b\u0435 \u043a\u0430\u0440\u0442\u043e\u0447\u043a\u0438",
            "total": "yellow_cards_total",
            "home": "yellow_cards_home",
            "away": "yellow_cards_away",
        },
        "shots": {
            "label": "\U0001f3af \u0423\u0434\u0430\u0440\u044b",
            "total": "shots_total",
            "home": "shots_home",
            "away": "shots_away",
        },
        "shots_on_target": {
            "label": "\U0001f3af \u0423\u0434\u0430\u0440\u044b \u0432 \u0441\u0442\u0432\u043e\u0440",
            "total": "shots_on_target_total",
            "home": "shots_on_target_home",
            "away": "shots_on_target_away",
        },
        "offsides": {
            "label": "\U0001f6a9 \u041e\u0444\u0441\u0430\u0439\u0434\u044b",
            "total": "offsides_total",
            "home": "offsides_home",
            "away": "offsides_away",
        },
        "fouls": {
            "label": "\U0001f7e8 \u0424\u043e\u043b\u044b",
            "total": "fouls_total",
            "home": "fouls_home",
            "away": "fouls_away",
        },
    }

    config = sections.get(section)

    if config is None:
        return "\u26a0\ufe0f \u041d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u044b\u0439 \u0440\u0430\u0437\u0434\u0435\u043b."

    total = stats.get(config["total"])
    home_value = stats.get(config["home"])
    away_value = stats.get(config["away"])

    if total is None or home_value is None or away_value is None:
        return (
            f"\u26a0\ufe0f {config['label']}: "
            "\u043f\u0440\u043e\u0433\u043d\u043e\u0437 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d."
        )

    return (
        f"{config['label']}\n\n"
        f"{home} - {away}\n\n"
        f"\u0412\u0441\u0435\u0433\u043e: {float(total):.2f}\n"
        f"{home}: {float(home_value):.2f}\n"
        f"{away}: {float(away_value):.2f}"
    )


async def fixture_section_callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    query = update.callback_query

    if query is None:
        return

    await query.answer()

    prediction = context.user_data.get("selected_prediction")

    if not prediction:
        await query.edit_message_text(
            "\u0414\u0430\u043d\u043d\u044b\u0435 \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430 \u0443\u0441\u0442\u0430\u0440\u0435\u043b\u0438. "
            "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043c\u0430\u0442\u0447 \u0437\u0430\u043d\u043e\u0432\u043e."
        )
        return

    callback_data = query.data or ""

    if callback_data == "fixture_back_to_menu":
        await query.edit_message_text(
            build_fixture_menu_text(prediction),
            reply_markup=build_fixture_menu_keyboard(),
        )
        return

    if callback_data == "fixture_main_menu":
        await query.edit_message_text(
            "\u0413\u043b\u0430\u0432\u043d\u043e\u0435 \u043c\u0435\u043d\u044e."
        )
        await query.message.reply_text(
            "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435:",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return

    if callback_data == "fixture_back_to_list":
        session = SessionLocal()

        try:
            fixture_service = FixtureService(session)
            fixtures = fixture_service.get_latest_matches(limit=5)

            if not fixtures:
                await query.edit_message_text(
                    "\u041c\u0430\u0442\u0447\u0438 \u0432 \u0431\u0430\u0437\u0435 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u044b.",
                    reply_markup=build_section_keyboard(),
                )
                return

            await query.edit_message_text(
                "\u26bd \u0412\u044b\u0431\u0435\u0440\u0438 \u043c\u0430\u0442\u0447:",
                reply_markup=build_fixture_list_keyboard(fixtures),
            )
        finally:
            session.close()

        return

    if callback_data.startswith("fixture_section:"):
        section = callback_data.split(":", 1)[1]

        if section == "main":
            await query.edit_message_text(
                build_prediction_text(prediction),
                reply_markup=build_section_keyboard(),
            )
            return

        text = build_statistics_section_text(
            prediction,
            section,
        )

        await query.edit_message_text(
            text,
            reply_markup=build_section_keyboard(),
        )
        return

    await query.edit_message_text(
        "\u041d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u0430\u044f \u043a\u043d\u043e\u043f\u043a\u0430."
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
            f"result={prediction['prediction']}, "
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

    actual_score = prediction.get("actual_score")

    if actual_score:
        score_text = "Фактический счёт: " + str(actual_score)
    else:
        score_text = ""


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
        f"{prediction['prediction_name']}\n"
        f"Уверенность модели: "
        f"{confidence:.1f}%"
    )

async def next_matches_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Команда /next.

    Показать ближайшие будущие матчи.
    """

    if update.message is None:
        return

    session = SessionLocal()

    try:
        fixture_service = FixtureService(session)

        fixtures = fixture_service.get_upcoming_matches(
            limit=5,
        )

        if not fixtures:
            await update.message.reply_text(
                "📅 Ближайших матчей пока нет.\n\n"
                "В текущей базе нет будущих матчей.",
                reply_markup=MAIN_MENU_KEYBOARD,
            )
            return

        keyboard = []

        for fixture in fixtures:
            home_team = (
                fixture.home_team.name
                if fixture.home_team is not None
                else "Неизвестная команда"
            )

            away_team = (
                fixture.away_team.name
                if fixture.away_team is not None
                else "Неизвестная команда"
            )

            kickoff_text = fixture.kickoff.strftime(
                "%d.%m.%Y %H:%M"
            )

            keyboard.append(
                [
                    InlineKeyboardButton(
                        text=(
                            f"{kickoff_text} | "
                            f"{home_team} — {away_team}"
                        ),
                        callback_data=(
                            f"predict_fixture:{fixture.id}"
                        ),
                    )
                ]
            )

        await update.message.reply_text(
            "📅 Ближайшие матчи:\n"
            "Выбери матч для прогноза:",
            reply_markup=InlineKeyboardMarkup(
                keyboard
            ),
        )

    finally:
        session.close()

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