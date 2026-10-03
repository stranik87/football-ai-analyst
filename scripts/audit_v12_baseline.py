from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
)

DB = Path("football.db")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(DB)

fixtures = pd.read_sql_query(
    """
    SELECT
        id AS fixture_id,
        kickoff,
        home_team_id,
        away_team_id,
        home_goals,
        away_goals,
        halftime_home,
        halftime_away
    FROM fixtures
    WHERE home_goals IS NOT NULL
      AND away_goals IS NOT NULL
      AND halftime_home IS NOT NULL
      AND halftime_away IS NOT NULL
    ORDER BY kickoff, id
    """,
    con,
)

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

stats = stats.merge(
    fixtures[
        ["fixture_id", "home_team_id", "away_team_id"]
    ],
    on="fixture_id",
    how="inner",
)

stats["side"] = np.where(
    stats["team_id"] == stats["home_team_id"],
    "home",
    np.where(
        stats["team_id"] == stats["away_team_id"],
        "away",
        None,
    ),
)

stats = stats[
    stats["side"].isin(["home", "away"])
].copy()

valid_ids = (
    stats.groupby("fixture_id")["side"]
    .agg(lambda x: set(x) == {"home", "away"})
)

valid_ids = valid_ids[valid_ids].index

stats = stats[
    stats["fixture_id"].isin(valid_ids)
].copy()

stat_names = [
    "corner_kicks",
    "yellow_cards",
    "total_shots",
    "shots_on_goal",
    "offsides",
    "fouls",
]

home = (
    stats[stats["side"] == "home"]
    .set_index("fixture_id")[stat_names]
    .add_prefix("home_")
)

away = (
    stats[stats["side"] == "away"]
    .set_index("fixture_id")[stat_names]
    .add_prefix("away_")
)

wide = home.join(away, how="inner")

df = fixtures.set_index("fixture_id").join(
    wide,
    how="left",
)

# ---------------------------------------------------------
# TARGETS
# ---------------------------------------------------------

df["goals_total"] = df["home_goals"] + df["away_goals"]
df["goals_home"] = df["home_goals"]
df["goals_away"] = df["away_goals"]

df["goals_1h_total"] = (
    df["halftime_home"] + df["halftime_away"]
)
df["goals_1h_home"] = df["halftime_home"]
df["goals_1h_away"] = df["halftime_away"]

df["goals_2h_home"] = (
    df["home_goals"] - df["halftime_home"]
)
df["goals_2h_away"] = (
    df["away_goals"] - df["halftime_away"]
)
df["goals_2h_total"] = (
    df["goals_2h_home"] + df["goals_2h_away"]
)

df["btts"] = (
    (df["home_goals"] > 0) &
    (df["away_goals"] > 0)
).astype(int)

for stat in stat_names:
    home_col = f"home_{stat}"
    away_col = f"away_{stat}"
    total_col = f"{stat}_total"

    df[total_col] = (
        df[home_col] + df[away_col]
    )

# ---------------------------------------------------------
# TEMPORAL SPLIT
# Same kickoff timestamps stay in the same split.
# ---------------------------------------------------------

df = df.sort_values(
    ["kickoff", "fixture_id"]
).copy()

unique_kickoffs = (
    df["kickoff"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

n = len(unique_kickoffs)

train_cut = int(n * 0.70)
val_cut = int(n * 0.85)

train_kickoffs = set(unique_kickoffs[:train_cut])
val_kickoffs = set(unique_kickoffs[train_cut:val_cut])
test_kickoffs = set(unique_kickoffs[val_cut:])

df["split"] = np.select(
    [
        df["kickoff"].isin(train_kickoffs),
        df["kickoff"].isin(val_kickoffs),
        df["kickoff"].isin(test_kickoffs),
    ],
    [
        "train",
        "validation",
        "test",
    ],
    default="unknown",
)

print("=" * 70)
print("v1.2 BASELINE")
print("=" * 70)

print("\nSPLIT:")
print(
    df["split"].value_counts()
    .sort_index()
)

print(
    "\nDATES:"
)
for split in ["train", "validation", "test"]:
    part = df[df["split"] == split]
    if len(part):
        print(
            split,
            part["kickoff"].min(),
            "->",
            part["kickoff"].max(),
            "matches:",
            len(part),
        )

# ---------------------------------------------------------
# TARGET LIST
# ---------------------------------------------------------

targets = [
    ("goals_total", "Goals — Total"),
    ("goals_home", "Goals — Home"),
    ("goals_away", "Goals — Away"),
    ("goals_1h_total", "Goals — 1H Total"),
    ("goals_1h_home", "Goals — 1H Home"),
    ("goals_1h_away", "Goals — 1H Away"),
    ("goals_2h_total", "Goals — 2H Total"),
    ("goals_2h_home", "Goals — 2H Home"),
    ("goals_2h_away", "Goals — 2H Away"),

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

results = []

# ---------------------------------------------------------
# REGRESSION BASELINES
# ---------------------------------------------------------

for target, description in targets:

    train = df[
        (df["split"] == "train") &
        df[target].notna()
    ][target]

    test = df[
        (df["split"] == "test") &
        df[target].notna()
    ][target]

    if len(train) == 0 or len(test) == 0:
        continue

    baseline_mean = float(train.mean())

    predictions = np.full(
        len(test),
        baseline_mean,
    )

    mae = mean_absolute_error(
        test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            predictions,
        )
    )

    results.append({
        "target": target,
        "description": description,
        "type": "regression",
        "train_matches": len(train),
        "test_matches": len(test),
        "baseline_mean": baseline_mean,
        "mae": mae,
        "rmse": rmse,
    })

# ---------------------------------------------------------
# BTTS BASELINE
# ---------------------------------------------------------

train = df[
    df["split"] == "train"
]["btts"]

test = df[
    df["split"] == "test"
]["btts"]

if len(train) and len(test):

    probability = float(train.mean())

    probabilities = np.full(
        len(test),
        probability,
    )

    classes = (
        probabilities >= 0.5
    ).astype(int)

    results.append({
        "target": "btts",
        "description": "BTTS",
        "type": "classification",
        "train_matches": len(train),
        "test_matches": len(test),
        "baseline_mean": probability,
        "accuracy": accuracy_score(
            test,
            classes,
        ),
        "precision": precision_score(
            test,
            classes,
            zero_division=0,
        ),
        "recall": recall_score(
            test,
            classes,
            zero_division=0,
        ),
        "f1": f1_score(
            test,
            classes,
            zero_division=0,
        ),
        "logloss": log_loss(
            test,
            probabilities,
            labels=[0, 1],
        ),
        "brier": brier_score_loss(
            test,
            probabilities,
        ),
    })

result_df = pd.DataFrame(results)

csv_path = REPORT_DIR / "v12_baseline.csv"
md_path = REPORT_DIR / "v12_baseline.md"

result_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)

with open(
    md_path,
    "w",
    encoding="utf-8",
) as f:

    f.write("# v1.2 Baseline\n\n")

    for _, row in result_df.iterrows():

        f.write(
            "## " +
            str(row["description"]) +
            "\n\n"
        )

        for key, value in row.items():
            f.write(
                f"- {key}: {value}\n"
            )

        f.write("\n")

print("\nRESULTS:")
print(
    result_df.to_string(
        index=False
    )
)

print("\nREPORTS:")
print(csv_path)
print(md_path)

con.close()
