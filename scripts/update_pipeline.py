from loguru import logger

from scripts.collect_odds import main as collect_odds
from scripts.analyze_live_odds import main as analyze_live_odds
from scripts.settle_odds import main as settle_odds


def run_step(name, function):
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"STEP: {name}")
    logger.info("=" * 78)

    try:
        function()
    except Exception:
        logger.exception(
            f"Ошибка на этапе: {name}"
        )
        raise

    logger.info(
        f"STEP OK: {name}"
    )


def main():
    logger.info("")
    logger.info("=" * 78)
    logger.info("FOOTBALL AI ANALYTICS PIPELINE")
    logger.info("=" * 78)

    run_step(
        "COLLECT ODDS",
        collect_odds,
    )

    run_step(
        "ANALYZE LIVE ODDS",
        analyze_live_odds,
    )

    run_step(
        "SETTLE ODDS",
        settle_odds,
    )

    logger.info("")
    logger.info("=" * 78)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
