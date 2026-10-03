from pathlib import Path
import joblib
import pandas as pd


REPORT_DIR = Path("data/reports")
MODEL_DIR = Path("data/models")
OUTPUT = REPORT_DIR / "v12_models_final_audit.csv"


REPORTS = [
    "v12_goals_models.csv",
    "v12_corners_catboost.csv",
    "v12_yellow_cards_models.csv",
    "v12_shots_models.csv",
    "v12_shots_on_target_models.csv",
    "v12_offsides_models.csv",
    "v12_fouls_models.csv",
    "v12_btts_v2.csv",
]


def find_model_file(target):
    candidates = [
        MODEL_DIR / f"v12_{target}_catboost.joblib",
        MODEL_DIR / f"v12_{target}_poisson.joblib",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def main():
    print("=" * 80)
    print("v1.2 FINAL MODEL AUDIT")
    print("=" * 80)

    all_results = []

    for report_name in REPORTS:
        path = REPORT_DIR / report_name

        if not path.exists():
            print(f"\nWARNING: report not found: {path}")
            continue

        df = pd.read_csv(path)

        print()
        print("-" * 80)
        print(report_name)
        print("-" * 80)

        for _, row in df.iterrows():
            model_name = str(row.get("model", ""))

            if model_name == "baseline":
                continue

            target = str(
                row.get(
                    "target",
                    row.get("model", ""),
                )
            )

            status = str(row.get("status", ""))

            feature_count = row.get(
                "feature_count",
                None,
            )

            model_path = find_model_file(target)

            file_exists = (
                model_path is not None
            )

            loaded_feature_count = None
            loaded_target = None
            loaded_status = None

            if file_exists:
                try:
                    bundle = joblib.load(model_path)

                    loaded_feature_count = len(
                        bundle.get(
                            "feature_columns",
                            [],
                        )
                    )

                    loaded_target = bundle.get(
                        "target"
                    )

                    loaded_status = bundle.get(
                        "status"
                    )

                except Exception as exc:
                    print(
                        f"LOAD ERROR: {model_path}: {exc}"
                    )

            valid_features = (
                pd.isna(feature_count)
                or int(feature_count) == 79
            )

            valid_loaded_features = (
                loaded_feature_count is None
                or loaded_feature_count == 79
            )

            valid_status = (
                status in {
                    "READY",
                    "WORSE_THAN_BASELINE",
                    "NOT_READY",
                    "INSUFFICIENT_DATA",
                }
            )

            ready_model_consistent = True

            if status == "READY":
                ready_model_consistent = (
                    file_exists
                    and valid_features
                    and valid_loaded_features
                    and (
                        loaded_status is None
                        or loaded_status == "READY"
                    )
                )

            final_check = (
                "PASS"
                if valid_status
                and valid_features
                and valid_loaded_features
                and (
                    status != "READY"
                    or ready_model_consistent
                )
                else "FAIL"
            )

            print(
                f"{target:35} "
                f"status={status:24} "
                f"features={feature_count} "
                f"file={'YES' if file_exists else 'NO'} "
                f"check={final_check}"
            )

            all_results.append(
                {
                    "target": target,
                    "source_report": report_name,
                    "status": status,
                    "report_feature_count": feature_count,
                    "model_file": (
                        str(model_path)
                        if model_path
                        else ""
                    ),
                    "model_file_exists": file_exists,
                    "loaded_feature_count": (
                        loaded_feature_count
                    ),
                    "loaded_target": loaded_target,
                    "loaded_status": loaded_status,
                    "final_check": final_check,
                }
            )

    # Throw-ins: explicitly insufficient data according to v1.2 audit.
    all_results.append(
        {
            "target": "throw_ins",
            "source_report": "v12_targets_audit.csv",
            "status": "INSUFFICIENT_DATA",
            "report_feature_count": "",
            "model_file": "",
            "model_file_exists": False,
            "loaded_feature_count": "",
            "loaded_target": "",
            "loaded_status": "",
            "final_check": "PASS",
        }
    )

    result = pd.DataFrame(all_results)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL TABLE")
    print("=" * 80)

    print(
        result[
            [
                "target",
                "status",
                "report_feature_count",
                "model_file_exists",
                "loaded_feature_count",
                "final_check",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("STATUS COUNTS")
    print("=" * 80)

    print(
        result["status"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 80)
    print("CHECK COUNTS")
    print("=" * 80)

    print(
        result["final_check"]
        .value_counts()
        .to_string()
    )

    failures = result[
        result["final_check"] != "PASS"
    ]

    print()

    if failures.empty:
        print("FINAL AUDIT: PASS")
    else:
        print("FINAL AUDIT: FAIL")
        print()
        print(
            failures.to_string(
                index=False
            )
        )

    print()
    print("REPORT:", OUTPUT)


if __name__ == "__main__":
    main()
