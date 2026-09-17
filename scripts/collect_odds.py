from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
from loguru import logger
from sqlalchemy import select

from app.database.session import SessionLocal
from app.models.fixture import Fixture
from app.api.client import FootballAPIClient


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/reports/odds")
OUTPUT_PATH = OUTPUT_DIR / "odds_snapshots.csv"

# Сколько ближайших матчей проверять за один запуск.
LIMIT = 30

# Горизонт сбора коэффициентов.
LOOKAHEAD_DAYS = 7

# Если коэффициенты bookmaker не изменились относительно
# последнего сохранённого snapshot — новую строку не создаём.
SKIP_UNCHANGED = True


# ============================================================
# HELPERS
# ============================================================

def normalize_odds(value):
    """Приводит коэффициент к float или None."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_existing_odds():
    """Загружает существующий файл snapshots."""
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(OUTPUT_PATH)

        if df.empty:
            return df

        for column in ["fixture_id", "bookmaker_id"]:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        for column in [
            "home_odds",
            "draw_odds",
            "away_odds",
        ]:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # snapshot_at ОСТАВЛЯЕМ СТРОКОЙ.
        # Преобразование в datetime выполняется только
        # внутри odds_unchanged(), когда это необходимо.
        if "snapshot_at" in df.columns:
            df["snapshot_at"] = (
                df["snapshot_at"]
                .fillna("")
                .astype(str)
                .replace(
                    {
                        "nan": "",
                        "NaT": "",
                        "None": "",
                    }
                )
            )

        return df

    except Exception as exc:
        logger.warning(
            f"Не удалось прочитать существующий CSV: {exc}"
        )
        return pd.DataFrame()


def odds_unchanged(
    existing_df,
    fixture_id,
    bookmaker_id,
    home_odds,
    draw_odds,
    away_odds,
):
    """
    Проверяет последний ВАЛИДНЫЙ snapshot
    конкретного fixture + bookmaker.

    Старые записи с пустым snapshot_at
    не считаются актуальными.
    """

    if not SKIP_UNCHANGED:
        return False

    if existing_df.empty:
        return False

    required_columns = {
        "fixture_id",
        "bookmaker_id",
        "home_odds",
        "draw_odds",
        "away_odds",
        "snapshot_at",
    }

    if not required_columns.issubset(existing_df.columns):
        return False

    matches = existing_df[
        (existing_df["fixture_id"] == fixture_id)
        & (existing_df["bookmaker_id"] == bookmaker_id)
    ].copy()

    if matches.empty:
        return False

    # ВАЖНО:
    # старые записи без timestamp не считаем валидным snapshot.
    matches["snapshot_at"] = pd.to_datetime(
        matches["snapshot_at"],
        errors="coerce",
        utc=True,
    )

    matches = matches.dropna(
        subset=["snapshot_at"]
    ).copy()

    if matches.empty:
        return False

    matches = matches.sort_values("snapshot_at")
    last = matches.iloc[-1]

    return (
        normalize_odds(last["home_odds"])
        == normalize_odds(home_odds)
        and
        normalize_odds(last["draw_odds"])
        == normalize_odds(draw_odds)
        and
        normalize_odds(last["away_odds"])
        == normalize_odds(away_odds)
    )

def extract_match_winner_odds(bookmaker):
    """
    Извлекает 1X2 / Match Winner.

    Возвращает:
        home_odds,
        draw_odds,
        away_odds
    """

    home_odds = None
    draw_odds = None
    away_odds = None

    bets = bookmaker.get("bets", [])

    for bet in bets:

        bet_name = str(
            bet.get("name", "")
        ).strip().lower()

        bet_id = bet.get("id")

        if bet_id == 1 or bet_name in {
            "match winner",
            "1x2",
        }:

            values = bet.get("values", [])

            for value in values:

                value_name = str(
                    value.get("value", "")
                ).strip().lower()

                odd = normalize_odds(
                    value.get("odd")
                )

                if value_name in {"home", "1"}:
                    home_odds = odd

                elif value_name in {"draw", "x"}:
                    draw_odds = odd

                elif value_name in {"away", "2"}:
                    away_odds = odd

            break

    return (
        home_odds,
        draw_odds,
        away_odds,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_df = load_existing_odds()

    now = datetime.now()
    lookahead_end = now + timedelta(
        days=LOOKAHEAD_DAYS
    )

    logger.info("=" * 70)
    logger.info("ODDS COLLECTOR")
    logger.info("=" * 70)

    logger.info(f"Сейчас:       {now}")
    logger.info(f"До:           {lookahead_end}")
    logger.info(f"Лимит матчей: {LIMIT}")
    logger.info(f"CSV:          {OUTPUT_PATH}")

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # FUTURE FIXTURES
        # ----------------------------------------------------

        fixtures = db.execute(
            select(Fixture)
            .where(
                Fixture.kickoff >= now,
                Fixture.kickoff <= lookahead_end,
            )
            .order_by(Fixture.kickoff.asc())
            .limit(LIMIT)
        ).scalars().all()

        logger.info(
            f"Найдено будущих матчей: {len(fixtures)}"
        )

        if not fixtures:
            logger.info(
                "Будущих матчей в заданном диапазоне нет."
            )
            return

        api = FootballAPIClient()

        new_rows = []

        skipped_unchanged = 0
        skipped_no_odds = 0
        total_bookmakers = 0

        # ----------------------------------------------------
        # PROCESS FIXTURES
        # ----------------------------------------------------

        for index, fixture in enumerate(
            fixtures,
            1,
        ):

            home_name = (
                fixture.home_team.name
                if fixture.home_team
                else f"team_{fixture.home_team_id}"
            )

            away_name = (
                fixture.away_team.name
                if fixture.away_team
                else f"team_{fixture.away_team_id}"
            )

            logger.info("")

            logger.info(
                f"[{index}/{len(fixtures)}] "
                f"fixture={fixture.id} "
                f"api_id={fixture.api_id}"
            )

            logger.info(
                f"{home_name} - {away_name}"
            )

            logger.info(
                f"Kickoff: {fixture.kickoff}"
            )

            logger.info(
                f"Status: {fixture.status_short}"
            )

            # ------------------------------------------------
            # API
            # ------------------------------------------------

            try:
                data = api.get(
                    "odds",
                    {"fixture": fixture.api_id},
                )

            except Exception as exc:

                logger.error(
                    f"Ошибка API для fixture "
                    f"{fixture.id}: {exc}"
                )

                continue

            response = data.get(
                "response",
                [],
            )

            if not response:

                logger.warning(
                    f"NO ODDS: fixture={fixture.id}, "
                    f"api_id={fixture.api_id}"
                )

                skipped_no_odds += 1

                continue

            snapshot_at = datetime.now(
                timezone.utc
            ).isoformat()

            fixture_new_rows = 0

            # ------------------------------------------------
            # BOOKMAKERS
            # ------------------------------------------------

            for response_item in response:

                bookmakers = response_item.get(
                    "bookmakers",
                    [],
                )

                for bookmaker in bookmakers:

                    bookmaker_id = bookmaker.get(
                        "id"
                    )

                    bookmaker_name = bookmaker.get(
                        "name"
                    )

                    if bookmaker_id is None:
                        continue

                    total_bookmakers += 1

                    (
                        home_odds,
                        draw_odds,
                        away_odds,
                    ) = extract_match_winner_odds(
                        bookmaker
                    )

                    # Не сохраняем неполный рынок 1X2.
                    if (
                        home_odds is None
                        or draw_odds is None
                        or away_odds is None
                    ):
                        continue

                    # ------------------------------------------------
                    # DUPLICATE / UNCHANGED CHECK
                    # ------------------------------------------------

                    if odds_unchanged(
                        existing_df,
                        fixture.id,
                        bookmaker_id,
                        home_odds,
                        draw_odds,
                        away_odds,
                    ):

                        skipped_unchanged += 1

                        logger.debug(
                            f"UNCHANGED: "
                            f"{bookmaker_name} "
                            f"{home_odds}/"
                            f"{draw_odds}/"
                            f"{away_odds}"
                        )

                        continue

                    # ------------------------------------------------
                    # SAVE ROW
                    # ------------------------------------------------

                    new_rows.append(
                        {
                            "snapshot_at": snapshot_at,
                            "fixture_id": fixture.id,
                            "fixture_api_id": fixture.api_id,
                            "kickoff": fixture.kickoff,
                            "status_short": fixture.status_short,
                            "home_team_id": fixture.home_team_id,
                            "away_team_id": fixture.away_team_id,
                            "home_team_name": home_name,
                            "away_team_name": away_name,
                            "bookmaker_id": bookmaker_id,
                            "bookmaker": bookmaker_name,
                            "home_odds": home_odds,
                            "draw_odds": draw_odds,
                            "away_odds": away_odds,
                        }
                    )

                    fixture_new_rows += 1

            if fixture_new_rows:

                logger.success(
                    f"Новых odds: "
                    f"{fixture_new_rows}"
                )

            else:

                logger.info(
                    "Новых изменений "
                    "коэффициентов нет."
                )

        # ====================================================
        # SAVE
        # ====================================================

        logger.info("")

        if not new_rows:

            logger.info("=" * 70)
            logger.info("НОВЫХ ДАННЫХ НЕТ")
            logger.info("=" * 70)

            logger.info(
                f"Без изменений: {skipped_unchanged}"
            )

            logger.info(
                f"Без odds:       {skipped_no_odds}"
            )

            logger.info(
                f"Bookmakers:     {total_bookmakers}"
            )

            return

        new_df = pd.DataFrame(new_rows)

        # ----------------------------------------------------
        # NORMALIZE SNAPSHOT TIMESTAMP
        # ----------------------------------------------------
        #
        # ВАЖНО:
        # snapshot_at должен оставаться обычной строкой
        # ISO-формата. Не смешиваем datetime/string/NaT.
        #

        if "snapshot_at" not in new_df.columns:
            raise RuntimeError(
                "new_df does not contain snapshot_at"
            )

        new_df["snapshot_at"] = new_df["snapshot_at"].map(
            lambda value: (
                value.isoformat()
                if isinstance(value, (datetime, pd.Timestamp))
                else (
                    str(value)
                    if pd.notna(value)
                    and str(value).strip().lower() not in {"", "nat", "none"}
                    else ""
                )
            )
        )

        if existing_df.empty:

            final_df = new_df.copy()

        else:

            existing_df = existing_df.copy()

            if "snapshot_at" in existing_df.columns:

                existing_df["snapshot_at"] = (
                    existing_df["snapshot_at"]
                    .fillna("")
                    .astype(str)
                    .replace(
                        {
                            "nan": "",
                            "NaT": "",
                            "None": "",
                        }
                    )
                )

            final_df = pd.concat(
                [
                    existing_df,
                    new_df,
                ],
                ignore_index=True,
            )

        # ----------------------------------------------------
        # REMOVE INVALID SNAPSHOTS
        # ----------------------------------------------------

        final_df["snapshot_at"] = (
            final_df["snapshot_at"]
            .fillna("")
            .astype(str)
        )

        final_df = final_df[
            final_df["snapshot_at"].str.strip() != ""
        ].copy()

        # ----------------------------------------------------
        # REMOVE ABSOLUTE DUPLICATES
        # ----------------------------------------------------

        final_df = final_df.drop_duplicates()

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        final_df = final_df.sort_values(
            [
                "fixture_id",
                "bookmaker_id",
                "snapshot_at",
            ],
            kind="stable",
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        valid_snapshots = (
            pd.to_datetime(
                final_df["snapshot_at"],
                errors="coerce",
                utc=True,
            )
            .notna()
            .sum()
        )

        invalid_snapshots = (
            len(final_df) - valid_snapshots
        )

        if invalid_snapshots:
            raise RuntimeError(
                f"Invalid snapshot_at values before save: "
                f"{invalid_snapshots}"
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        final_df.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        # ====================================================
        # SUMMARY
        # ====================================================


        # ====================================================

        logger.info("=" * 70)
        logger.success("ODDS SAVED")
        logger.info("=" * 70)

        logger.info(
            f"Новых строк:       {len(new_df)}"
        )

        logger.info(
            f"Всего строк:       {len(final_df)}"
        )

        logger.info(
            f"Матчей:            "
            f"{final_df['fixture_id'].nunique()}"
        )

        logger.info(
            f"Bookmakers:        "
            f"{final_df['bookmaker'].nunique()}"
        )

        logger.info(
            f"Без изменений:     "
            f"{skipped_unchanged}"
        )

        logger.info(
            f"Без odds:           "
            f"{skipped_no_odds}"
        )

        logger.info(
            f"Файл:              {OUTPUT_PATH}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
