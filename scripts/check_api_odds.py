import os
import sys
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


DATASET_PATH = Path("data/datasets/matches_dataset.csv")
DB_PATH = Path("football.db")

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY") or os.getenv("API_KEY")

BASE_URL = os.getenv(
    "API_FOOTBALL_BASE_URL",
    "https://v3.football.api-sports.io",
)


def load_fixture_mapping():
    """Load dataset fixture_id -> API-Football fixture api_id."""

    if not DB_PATH.exists():
        print()
        print(f"ERROR: Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    try:
        mapping = pd.read_sql_query(
            """
            SELECT
                id AS fixture_id,
                api_id AS fixture_api_id
            FROM fixtures
            """,
            conn,
        )
    finally:
        conn.close()

    if mapping.empty:
        print()
        print("ERROR: fixtures table is empty.")
        sys.exit(1)

    return mapping


def extract_match_winner(bookmaker):
    """Find Match Winner / 1X2 market."""

    for bet in bookmaker.get("bets", []):
        name = str(bet.get("name", "")).strip().lower()

        if name in {"match winner", "1x2"}:
            return bet

    return None


def main():
    print("=" * 80)
    print("API-FOOTBALL ODDS CHECK")
    print("=" * 80)

    # ------------------------------------------------------------------
    # API KEY
    # ------------------------------------------------------------------

    if not API_KEY:
        print()
        print("ERROR: API key not found.")
        print()
        print("Add this to .env:")
        print()
        print("API_FOOTBALL_KEY=YOUR_API_KEY")
        print()
        sys.exit(1)

    # ------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------

    if not DATASET_PATH.exists():
        print()
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)

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
        print(f"ERROR: Dataset missing columns: {missing}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # DATABASE MAPPING
    # ------------------------------------------------------------------

    print()
    print("Loading fixture mapping from football.db...")

    mapping = load_fixture_mapping()

    print(f"Database fixtures: {len(mapping)}")

    df = df.merge(
        mapping,
        on="fixture_id",
        how="left",
        validate="one_to_one",
    )

    missing_api = df["fixture_api_id"].isna()

    if missing_api.any():
        print()
        print(
            "ERROR: Fixtures without API-Football ID:"
        )

        print(
            df.loc[
                missing_api,
                [
                    "fixture_id",
                    "kickoff",
                    "home_team_id",
                    "away_team_id",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

        sys.exit(1)

    df["fixture_api_id"] = df["fixture_api_id"].astype(int)

    # ------------------------------------------------------------------
    # SAMPLE
    # ------------------------------------------------------------------

    sample_indexes = [
        0,
        len(df) // 3,
        (len(df) * 2) // 3,
        len(df) - 1,
    ]

    samples = (
        df.iloc[sample_indexes]
        .drop_duplicates(subset=["fixture_api_id"])
    )

    print()
    print(f"Base URL: {BASE_URL}")
    print(f"Dataset matches: {len(df)}")
    print(f"Testing fixtures: {len(samples)}")

    # ------------------------------------------------------------------
    # API HEADERS
    # ------------------------------------------------------------------

    headers = {
        "x-apisports-key": API_KEY,
    }

    url = f"{BASE_URL.rstrip('/')}/odds"

    successful = 0
    no_odds = 0
    errors = 0

    # ------------------------------------------------------------------
    # REQUESTS
    # ------------------------------------------------------------------

    for _, row in samples.iterrows():

        fixture_id = int(row["fixture_id"])
        fixture_api_id = int(row["fixture_api_id"])

        print()
        print("-" * 80)

        print(f"Dataset fixture_id: {fixture_id}")
        print(f"API-Football fixture ID: {fixture_api_id}")
        print(f"Kickoff: {row['kickoff']}")
        print(
            f"Teams: "
            f"{int(row['home_team_id'])} vs "
            f"{int(row['away_team_id'])}"
        )

        try:
            response = requests.get(
                url,
                headers=headers,
                params={
                    "fixture": fixture_api_id,
                },
                timeout=30,
            )

            print(f"HTTP status: {response.status_code}")

            if response.status_code != 200:
                errors += 1

                print("Response:")
                print(response.text[:1000])

                continue

            data = response.json()

            api_errors = data.get("errors")

            if api_errors:
                errors += 1

                print("API errors:")
                print(api_errors)

                continue

            # API-Football normally returns data in "response".
            # "results" is kept as fallback for compatibility.
            results = data.get(
                "response",
                data.get("results", []),
            )

            print(f"Odds result groups: {len(results)}")

            if not results:
                no_odds += 1
                print("NO ODDS RETURNED")
                continue

            successful += 1

            # ----------------------------------------------------------
            # BOOKMAKERS
            # ----------------------------------------------------------

            for result_index, result in enumerate(
                results[:3],
                start=1,
            ):

                bookmakers = result.get(
                    "bookmakers",
                    [],
                )

                if not bookmakers:
                    continue

                print()
                print(
                    f"Result group {result_index}"
                )
                print(
                    f"Bookmakers available: "
                    f"{len(bookmakers)}"
                )

                for bookmaker in bookmakers[:3]:

                    bookmaker_name = bookmaker.get(
                        "name",
                        "unknown",
                    )

                    print()
                    print(
                        f"Bookmaker: "
                        f"{bookmaker_name}"
                    )

                    bet = extract_match_winner(
                        bookmaker
                    )

                    if bet is None:
                        print(
                            "  Match Winner / 1X2 "
                            "market not found"
                        )
                        continue

                    print(
                        f"  Market: "
                        f"{bet.get('name', 'unknown')}"
                    )

                    values = bet.get(
                        "values",
                        [],
                    )

                    if not values:
                        print(
                            "  No odds values"
                        )
                        continue

                    for value in values:

                        outcome = value.get(
                            "value",
                            "unknown",
                        )

                        odd = value.get(
                            "odd",
                            "N/A",
                        )

                        print(
                            f"    {outcome}: {odd}"
                        )

        except requests.RequestException as exc:

            errors += 1

            print()
            print(
                f"REQUEST ERROR: {exc}"
            )

        except ValueError as exc:

            errors += 1

            print()
            print(
                f"JSON ERROR: {exc}"
            )

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("ODDS CHECK SUMMARY")
    print("=" * 80)

    print(
        f"Successful API responses: {successful}"
    )

    print(
        f"No odds returned: {no_odds}"
    )

    print(
        f"Errors: {errors}"
    )

    print()
    print("=" * 80)
    print("ODDS CHECK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
