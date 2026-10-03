# Targets Audit

Аудит target выполнен только на матчах, которые входят в текущий `matches_dataset.csv`.

Количество матчей dataset: **3762**.

| Target | Available | Missing | Fill rate | Min | Max | Mean | Median | Std | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| goals | 3762 | 0 | 100.00% | 0 | 10 | 2.8121 | 3.0000 | 1.6690 | CANDIDATE |
| btts | 3762 | 0 | 100.00% | 0 | 1 | 0.5478 | 1.0000 | 0.4978 | CANDIDATE |
| corners | 3627 | 135 | 96.41% | 1 | 28 | 9.6016 | 9.0000 | 3.4134 | CANDIDATE |
| yellow_cards | 3521 | 241 | 93.59% | 1.0000 | 15.0000 | 4.0846 | 4.0000 | 2.0008 | CANDIDATE |
| shots | 3627 | 135 | 96.41% | 8 | 54 | 25.2644 | 25.0000 | 5.9658 | CANDIDATE |
| shots_on_target | 3627 | 135 | 96.41% | 0 | 20 | 8.7221 | 9.0000 | 3.1084 | CANDIDATE |
| offsides | 3434 | 328 | 91.28% | 1.0000 | 14.0000 | 3.5865 | 3.0000 | 1.9735 | CANDIDATE |
| fouls | 3627 | 135 | 96.41% | 6 | 47 | 23.5274 | 23.0000 | 5.7633 | CANDIDATE |
| throw_ins | 0 | 3762 | 0.00% | - | - | - | - | - | INSUFFICIENT_DATA |

## Notes

- CANDIDATE означает только наличие достаточного количества данных для дальнейшего исследования.
- CANDIDATE не означает READY.
- Финальный статус определяется после temporal baseline и model evaluation.
- Throw-ins: INSUFFICIENT_DATA, так как соответствующего поля нет в текущей БД.
