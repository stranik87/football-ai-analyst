import os
import sys
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# =============================================================================
# CONFIG
# =============================================================================

DATASET_PATH = Path("data/datasets/matches_dataset.csv")
DB_PATH = Path("football.db")
OUTPUT_PATH = Path("data/reports/odds/api_football_odds.csv")

# 0 = все матчи
# Для теста сейчас 100 самых свежих.
LIMIT = 100

BASE_URL = "https://v3.football.api-sports.io"


# =============================================================================
# ENV
# =============================================================================

def load_api_key():
    """
    Надёжная загрузка .env.
    Не используем find_dotenv(), чтобы python -c / Windows Git Bash
    не вызывали AssertionError.
    """

    env_path = Path(".env")

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    api_key = (
        os.getenv("API_FOOTBALL_KEY")
        or os.getenv("API_KEY")
    )

    return api_key


# =============================================================================
# DATABASE
# =============================================================================

def load_fixture_mapping():
    """
    Загружает соответствие:

    dataset fixture_id -> API-Football fixture api_id
    """

    print("Loading fixture mapping from football.db...")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)

    try:
        query = """
            SELECT
                id AS fixture_id,
                api_id AS fixture_api_id
            FROM fixtures
        """

        db = pd.read_sql_query(
            query,
            connection,
        )

    finally:
        connection.close()

    db["fixture_id"] = pd.to_numeric(
        db["fixture_id"],
        errors="coerce",
    )

    db["fixture_api_id"] = pd.to_numeric(
        db["fixture_api_id"],
        errors="coerce",
    )

    db = db.dropna(
        subset=[
            "fixture_id",
            "fixture_api_id",
        ]
    )

    db["fixture_id"] = db["fixture_id"].astype(int)
    db["fixture_api_id"] = db["fixture_api_id"].astype(int)

    db = db.drop_duplicates(
        subset=["fixture_id"]
    )

    print(
        f"Database fixtures: {len(db)}"
    )

    return db


# =============================================================================
# API
# =============================================================================

def request_odds(
    session,
    api_key,
    fixture_api_id,
):
    """
    Запрашивает odds для одного fixture.
    """

    url = f"{BASE_URL.rstrip('/')}/odds"

    headers = {
        "x-apisports-key": api_key,
        "x-rapidapi-key": api_key,
    }

    try:
        response = session.get(
            url,
            headers=headers,
            params={
                "fixture": int(fixture_api_id),
            },
            timeout=30,
        )

    except requests.RequestException as exc:
        return {
            "status": "REQUEST_ERROR",
            "error": str(exc),
            "data": None,
        }

    if response.status_code != 200:
        return {
            "status": "HTTP_ERROR",
            "error": (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ),
            "data": None,
        }

    try:
        data = response.json()

    except ValueError:
        return {
            "status": "INVALID_JSON",
            "error": response.text[:500],
            "data": None,
        }

    if not isinstance(data, dict):
        return {
            "status": "INVALID_RESPONSE",
            "error": (
                f"Expected dict, got "
                f"{type(data).__name__}"
            ),
            "data": None,
        }

    api_errors = data.get("errors")

    if api_errors:
        return {
            "status": "API_ERROR",
            "error": str(api_errors),
            "data": data,
        }

    return {
        "status": "OK",
        "error": None,
        "data": data,
    }


# =============================================================================
# ODDS EXTRACTION
# =============================================================================

def extract_match_winner(bookmakers):
    """
    Извлекает рынок Match Winner / 1X2.

    Возвращает список:

    bookmaker_id
    bookmaker
    home_odds
    draw_odds
    away_odds
    """

    rows = []

    if not isinstance(bookmakers, list):
        return rows

    for bookmaker in bookmakers:

        if not isinstance(bookmaker, dict):
            continue

        bookmaker_id = bookmaker.get("id")
        bookmaker_name = bookmaker.get(
            "name",
            "unknown",
        )

        bets = bookmaker.get(
            "bets",
            [],
        )

        if not isinstance(bets, list):
            continue

        for bet in bets:

            if not isinstance(bet, dict):
                continue

            bet_name = str(
                bet.get("name", "")
            ).strip().lower()

            bet_id = bet.get("id")

            # API-Football обычно использует:
            # 1 = Match Winner
            if bet_name not in {
                "match winner",
                "1x2",
            } and bet_id != 1:
                continue

            values = bet.get(
                "values",
                [],
            )

            if not isinstance(values, list):
                continue

            home_odds = None
            draw_odds = None
            away_odds = None

            for value in values:

                if not isinstance(value, dict):
                    continue

                value_name = str(
                    value.get("value", "")
                ).strip().lower()

                odd = value.get("odd")

                try:
                    odd = float(odd)
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if value_name in {
                    "home",
                    "1",
                }:
                    home_odds = odd

                elif value_name in {
                    "draw",
                    "x",
                }:
                    draw_odds = odd

                elif value_name in {
                    "away",
                    "2",
                }:
                    away_odds = odd

            # Сохраняем только полноценный 1X2.
            if (
                home_odds is not None
                and draw_odds is not None
                and away_odds is not None
            ):
                rows.append(
                    {
                        "bookmaker_id": bookmaker_id,
                        "bookmaker": bookmaker_name,
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                    }
                )

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("API-FOOTBALL ODDS EXPORT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # API KEY
    # -------------------------------------------------------------------------

    api_key = load_api_key()

    if not api_key:
        print()
        print("ERROR: API key not found.")
        print()
        print(
            "Check .env for:"
        )
        print(
            "API_FOOTBALL_KEY=..."
        )
        sys.exit(1)

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------

    if not DATASET_PATH.exists():

        print()
        print(
            f"ERROR: Dataset not found: "
            f"{DATASET_PATH}"
        )

        sys.exit(1)

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = [
        "fixture_id",
        "kickoff",
        "home_team_id",
        "away_team_id",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print()
        print(
            f"ERROR: Dataset missing columns: "
            f"{missing}"
        )

        sys.exit(1)

    # -------------------------------------------------------------------------
    # FIXTURE MAPPING
    # -------------------------------------------------------------------------

    mapping = load_fixture_mapping()

    df["fixture_id"] = pd.to_numeric(
        df["fixture_id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["fixture_id"]
    )

    df["fixture_id"] = df[
        "fixture_id"
    ].astype(int)

    merged = df.merge(
        mapping,
        on="fixture_id",
        how="left",
    )

    missing_api_id = merged[
        "fixture_api_id"
    ].isna()

    if missing_api_id.any():

        print()
        print(
            "WARNING: Fixtures without API ID:",
            int(missing_api_id.sum()),
        )

        merged = merged[
            ~missing_api_id
        ].copy()

    merged["fixture_api_id"] = pd.to_numeric(
        merged["fixture_api_id"],
        errors="coerce",
    )

    merged = merged.dropna(
        subset=["fixture_api_id"]
    )

    merged["fixture_api_id"] = merged[
        "fixture_api_id"
    ].astype(int)

    # -------------------------------------------------------------------------
    # SORT BY NEWEST
    # -------------------------------------------------------------------------

    merged["kickoff"] = pd.to_datetime(
        merged["kickoff"],
        errors="coerce",
    )

    merged = merged.dropna(
        subset=["kickoff"]
    )

    merged = merged.sort_values(
        "kickoff",
        ascending=False,
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # LIMIT
    # -------------------------------------------------------------------------

    if LIMIT > 0:

        work_df = merged.head(
            LIMIT
        ).copy()

    else:

        work_df = merged.copy()

    # -------------------------------------------------------------------------
    # INFO
    # -------------------------------------------------------------------------

    print()
    print(
        f"Base URL: {BASE_URL}"
    )

    print(
        f"Dataset matches: {len(df)}"
    )

    print(
        f"Matches to query: {len(work_df)}"
    )

    if len(work_df) > 0:

        print()
        print(
            "Date range selected:"
        )

        print(
            f"Newest: "
            f"{work_df['kickoff'].min()}"
            if False
            else
            f"Newest: "
            f"{work_df['kickoff'].max()}"
        )

        print(
            f"Oldest: "
            f"{work_df['kickoff'].min()}"
        )

    print()
    print(
        "IMPORTANT: API-Football may not return "
        "historical odds for old fixtures."
    )

    print(
        "NO_ODDS is therefore NOT treated as an error."
    )

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Заголовок CSV создаём заранее.
    output_columns = [
        "fixture_id",
        "fixture_api_id",
        "kickoff",
        "home_team_id",
        "away_team_id",
        "bookmaker_id",
        "bookmaker",
        "home_odds",
        "draw_odds",
        "away_odds",
        "status",
    ]

    all_rows = []

    # -------------------------------------------------------------------------
    # REQUEST LOOP
    # -------------------------------------------------------------------------

    session = requests.Session()

    total = len(work_df)

    successful_with_odds = 0
    without_odds = 0
    without_1x2 = 0
    errors = 0

    for index, (_, row) in enumerate(
        work_df.iterrows(),
        start=1,
    ):

        fixture_id = int(
            row["fixture_id"]
        )

        fixture_api_id = int(
            row["fixture_api_id"]
        )

        kickoff = row["kickoff"]

        print(
            f"[{index}/{total}] "
            f"fixture={fixture_id} "
            f"api_id={fixture_api_id} "
            f"kickoff={kickoff} ... ",
            end="",
            flush=True,
        )

        result = request_odds(
            session=session,
            api_key=api_key,
            fixture_api_id=fixture_api_id,
        )

        if result["status"] != "OK":

            errors += 1

            print(
                f"ERROR: {result['status']}"
            )

            if result["error"]:
                print(
                    f"    {result['error']}"
                )

            continue

        data = result["data"]

        results = data.get(
            "results",
            [],
        )

        # ---------------------------------------------------------------------
        # ВАЖНО:
        # API должен вернуть список.
        # Если сервер вернул int/string/dict — не падаем.
        # ---------------------------------------------------------------------

        if isinstance(results, int):

            print(
                f"INVALID RESULTS TYPE: int={results}"
            )

            errors += 1

            continue

        if isinstance(results, dict):

            print(
                "INVALID RESULTS TYPE: dict"
            )

            errors += 1

            continue

        if not isinstance(results, list):

            print(
                f"INVALID RESULTS TYPE: "
                f"{type(results).__name__}"
            )

            errors += 1

            continue

        if len(results) == 0:

            without_odds += 1

            print(
                "NO ODDS"
            )

            continue

        # ---------------------------------------------------------------------
        # BOOKMAKERS
        # ---------------------------------------------------------------------

        bookmakers = []

        for result_group in results:

            if not isinstance(
                result_group,
                dict,
            ):
                continue

            group_bookmakers = (
                result_group.get(
                    "bookmakers",
                    [],
                )
            )

            if isinstance(
                group_bookmakers,
                list,
            ):

                bookmakers.extend(
                    group_bookmakers
                )

        if not bookmakers:

            without_odds += 1

            print(
                "NO BOOKMAKERS"
            )

            continue

        # ---------------------------------------------------------------------
        # EXTRACT 1X2
        # ---------------------------------------------------------------------

        odds_rows = extract_match_winner(
            bookmakers
        )

        if not odds_rows:

            without_1x2 += 1

            print(
                f"BOOKMAKERS={len(bookmakers)}, "
                f"NO 1X2"
            )

            continue

        successful_with_odds += 1

        print(
            f"1X2 bookmakers={len(odds_rows)}"
        )

        # ---------------------------------------------------------------------
        # SAVE ROWS
        # ---------------------------------------------------------------------

        for odds in odds_rows:

            all_rows.append(
                {
                    "fixture_id": fixture_id,
                    "fixture_api_id": fixture_api_id,
                    "kickoff": kickoff,
                    "home_team_id": int(
                        row["home_team_id"]
                    ),
                    "away_team_id": int(
                        row["away_team_id"]
                    ),
                    "bookmaker_id": odds[
                        "bookmaker_id"
                    ],
                    "bookmaker": odds[
                        "bookmaker"
                    ],
                    "home_odds": odds[
                        "home_odds"
                    ],
                    "draw_odds": odds[
                        "draw_odds"
                    ],
                    "away_odds": odds[
                        "away_odds"
                    ],
                    "status": "OK",
                }
            )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    output_df = pd.DataFrame(
        all_rows,
        columns=output_columns,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("ODDS EXPORT SUMMARY")
    print("=" * 80)

    print(
        f"Fixtures checked: {total}"
    )

    print(
        f"Fixtures with 1X2 odds: "
        f"{successful_with_odds}"
    )

    print(
        f"Fixtures without odds: "
        f"{without_odds}"
    )

    print(
        f"Fixtures without 1X2: "
        f"{without_1x2}"
    )

    print(
        f"Fixtures with errors: "
        f"{errors}"
    )

    print(
        f"Odds rows saved: "
        f"{len(output_df)}"
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    # -------------------------------------------------------------------------
    # SAMPLE
    # -------------------------------------------------------------------------

    if not output_df.empty:

        print()
        print("-" * 80)
        print("FIRST ODDS ROWS")
        print("-" * 80)

        print(
            output_df[
                [
                    "fixture_id",
                    "fixture_api_id",
                    "kickoff",
                    "bookmaker",
                    "home_odds",
                    "draw_odds",
                    "away_odds",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("=" * 80)
    print(
        "API-FOOTBALL ODDS EXPORT COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
