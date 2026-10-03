from pathlib import Path
import sqlite3
import pandas as pd

DB = Path("football.db")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(DB)

# ---------------------------------------------------------
# LOAD FIXTURES
# ---------------------------------------------------------

fixtures = pd.read_sql_query(
    """
    SELECT
        id AS fixture_id,
        kickoff,
        home_goals,
        away_goals,
        halftime_home,
        halftime_away
    FROM fixtures
    WHERE home_goals IS NOT NULL
      AND away_goals IS NOT NULL
    """,
    con,
)

# ---------------------------------------------------------
# LOAD TEAM STATISTICS
# ---------------------------------------------------------

stats = pd.read_sql_query(
    """
    SELECT
        fixture_id,
        team_id,
        shots_on_goal,
        total_shots,
        fouls,
        corner_kicks,
        offsides,
        yellow_cards
    FROM fixture_team_statistics
    """,
    con,
)

# ---------------------------------------------------------
# CREATE HOME / AWAY STATISTICS
# ---------------------------------------------------------

fixture_teams = pd.read_sql_query(
    """
    SELECT
        id AS fixture_id,
        home_team_id,
        away_team_id
    FROM fixtures
    """,
    con,
)

stats = stats.merge(
    fixture_teams,
    on="fixture_id",
    how="left",
)

stats["side"] = None

stats.loc[
    stats["team_id"] == stats["home_team_id"],
    "side"
] = "home"

stats.loc[
    stats["team_id"] == stats["away_team_id"],
    "side"
] = "away"

stats = stats[stats["side"].isin(["home", "away"])].copy()

# Only fixtures with exactly one home and one away stats record
valid_fixture_ids = (
    stats.groupby("fixture_id")["side"]
    .agg(lambda x: set(x) == {"home", "away"})
)

valid_fixture_ids = valid_fixture_ids[
    valid_fixture_ids
].index

stats = stats[
    stats["fixture_id"].isin(valid_fixture_ids)
].copy()

# ---------------------------------------------------------
# PIVOT HOME / AWAY
# ---------------------------------------------------------

stat_names = [
    "corner_kicks",
    "yellow_cards",
    "total_shots",
    "shots_on_goal",
    "offsides",
    "fouls",
]

home_stats = (
    stats[stats["side"] == "home"]
    .set_index("fixture_id")[stat_names]
    .add_prefix("home_")
)

away_stats = (
    stats[stats["side"] == "away"]
    .set_index("fixture_id")[stat_names]
    .add_prefix("away_")
)

stats_wide = home_stats.join(
    away_stats,
    how="inner",
)

# ---------------------------------------------------------
# BUILD TARGET DATASET
# ---------------------------------------------------------

df = fixtures.set_index("fixture_id").join(
    stats_wide,
    how="left",
)

# Goals
df["goals_total"] = (
    df["home_goals"] +
    df["away_goals"]
)

df["goals_home"] = df["home_goals"]
df["goals_away"] = df["away_goals"]

df["goals_1h_total"] = (
    df["halftime_home"] +
    df["halftime_away"]
)

df["goals_1h_home"] = df["halftime_home"]
df["goals_1h_away"] = df["halftime_away"]

df["goals_2h_home"] = (
    df["home_goals"] -
    df["halftime_home"]
)

df["goals_2h_away"] = (
    df["away_goals"] -
    df["halftime_away"]
)

df["goals_2h_total"] = (
    df["goals_2h_home"] +
    df["goals_2h_away"]
)

df["btts"] = (
    (df["home_goals"] > 0) &
    (df["away_goals"] > 0)
).astype(int)

# Match statistics
for stat in stat_names:
    home_col = f"home_{stat}"
    away_col = f"away_{stat}"
    total_col = f"{stat}_total"

    df[total_col] = (
        df[home_col] +
        df[away_col]
    )

# ---------------------------------------------------------
# TARGET DEFINITIONS
# ---------------------------------------------------------

targets = [
    ("goals_total", "Goals — Total"),
    ("goals_home", "Goals — Home Team Total"),
    ("goals_away", "Goals — Away Team Total"),
    ("goals_1h_total", "Goals — 1H Total"),
    ("goals_1h_home", "Goals — 1H Home Total"),
    ("goals_1h_away", "Goals — 1H Away Total"),
    ("goals_2h_total", "Goals — 2H Total"),
    ("goals_2h_home", "Goals — 2H Home Total"),
    ("goals_2h_away", "Goals — 2H Away Total"),
    ("btts", "BTTS"),
    ("corner_kicks_total", "Corners — Total"),
    ("home_corner_kicks", "Corners — Home"),
    ("away_corner_kicks", "Corners — Away"),
    ("yellow_cards_total", "Yellow Cards — Total"),
    ("home_yellow_cards", "Yellow Cards — Home"),
    ("away_yellow_cards", "Yellow Cards — Away"),
    ("total_shots_total", "Shots — Total"),
    ("home_total_shots", "Shots — Home"),
    ("away_total_shots", "Shots — Away"),
    ("shots_on_goal_total", "Shots on Target — Total"),
    ("home_shots_on_goal", "Shots on Target — Home"),
    ("away_shots_on_goal", "Shots on Target — Away"),
    ("offsides_total", "Offsides — Total"),
    ("home_offsides", "Offsides — Home"),
    ("away_offsides", "Offsides — Away"),
    ("fouls_total", "Fouls — Total"),
    ("home_fouls", "Fouls — Home"),
    ("away_fouls", "Fouls — Away"),
]

# ---------------------------------------------------------
# AUDIT
# ---------------------------------------------------------

rows = []

for target, description in targets:

    if target not in df.columns:
        rows.append({
            "target": target,
            "description": description,
            "available_matches": 0,
            "missing_matches": len(df),
            "fill_percent": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "status": "INSUFFICIENT_DATA",
        })
        continue

    series = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    available = int(series.notna().sum())
    missing = int(series.isna().sum())

    fill = (
        available / len(df) * 100
        if len(df)
        else 0
    )

    if available == 0:
        status = "INSUFFICIENT_DATA"
    else:
        status = "AUDIT_REQUIRED"

    rows.append({
        "target": target,
        "description": description,
        "available_matches": available,
        "missing_matches": missing,
        "fill_percent": round(fill, 2),
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "status": status,
    })

audit = pd.DataFrame(rows)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

csv_path = REPORT_DIR / "v12_targets_audit.csv"
md_path = REPORT_DIR / "v12_targets_audit.md"

audit.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)

with open(md_path, "w", encoding="utf-8") as f:

    f.write("# v1.2 Targets Audit\n\n")
    f.write(
        "Аудит target для новых статистических рынков.\n\n"
    )

    f.write(
        "| Target | Available | Missing | Fill % | "
        "Min | Max | Mean | Median | Std | Status |\n"
    )
    f.write(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )

    for _, row in audit.iterrows():

        f.write(
            f"| {row['description']} "
            f"| {row['available_matches']} "
            f"| {row['missing_matches']} "
            f"| {row['fill_percent']} "
            f"| {row['min']} "
            f"| {row['max']} "
            f"| {row['mean']} "
            f"| {row['median']} "
            f"| {row['std']} "
            f"| {row['status']} |\n"
        )

print("=" * 70)
print("v1.2 TARGET AUDIT")
print("=" * 70)

print(audit.to_string(index=False))

print("\nReports:")
print(csv_path)
print(md_path)

con.close()
