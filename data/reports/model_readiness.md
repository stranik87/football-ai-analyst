# Model Readiness Report

Сводный статус моделей после temporal test и сравнения с baseline.

## Статусы

- **goals** — `READY_CANDIDATE` — Model beats baseline
- **btts** — `NOT_READY` — Mixed result versus baseline
- **corners** — `READY_CANDIDATE` — Model beats baseline
- **yellow_cards** — `NOT_READY` — Mixed result versus baseline
- **shots** — `WORSE_THAN_BASELINE` — Model worse on MAE and RMSE
- **shots_on_target** — `READY_CANDIDATE` — Model beats baseline
- **offsides** — `READY_CANDIDATE` — Model beats baseline
- **fouls** — `READY_CANDIDATE` — Model beats baseline
- **throw_ins** — `INSUFFICIENT_DATA` — No throw-in data available

## Критерий

Для regression target модель считается кандидатом на READY, если на temporal test она лучше baseline одновременно по MAE и RMSE.

Throw-ins имеют статус INSUFFICIENT_DATA, потому что в доступных исторических данных нет этого показателя.
