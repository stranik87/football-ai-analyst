# Statistics Data Audit

Stage 1 — аудит данных для дополнительных прогнозов.

Всего матчей в `fixtures`: **5264**

Аудит выполнен только по уже загруженной локальной БД. Новые API-запросы не выполнялись.

| Target | Available | Missing | Fill % | Min | Max | Mean | Median | Std | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| goals | 3762 | 1502 | 71.47% | 0.0 | 10.0 | 2.8121 | 3.0 | 1.669 | CANDIDATE |
| corners | 3627 | 1637 | 68.90% | 1.0 | 28.0 | 9.6016 | 9.0 | 3.4134 | CANDIDATE |
| yellow_cards | 3521 | 1743 | 66.89% | 1.0 | 15.0 | 4.0846 | 4.0 | 2.0008 | CANDIDATE |
| shots | 3627 | 1637 | 68.90% | 8.0 | 54.0 | 25.2644 | 25.0 | 5.9658 | CANDIDATE |
| shots_on_target | 3627 | 1637 | 68.90% | 0.0 | 20.0 | 8.7221 | 9.0 | 3.1084 | CANDIDATE |
| offsides | 3434 | 1830 | 65.24% | 1.0 | 14.0 | 3.5865 | 3.0 | 1.9735 | CANDIDATE |
| fouls | 3627 | 1637 | 68.90% | 6.0 | 47.0 | 23.5274 | 23.0 | 5.7633 | CANDIDATE |
| throw_ins | 0 | 5264 | 0.00% | - | - | - | - | - | INSUFFICIENT_DATA |

## Quality comments

### goals
- Status: `CANDIDATE`
- Данных достаточно для дальнейшей проверки leakage, baseline и temporal model evaluation.

### corners
- Status: `CANDIDATE`
- Данных достаточно для дальнейшей проверки leakage, baseline и temporal model evaluation.

### yellow_cards
- Status: `CANDIDATE`
- Данных достаточно для дальнейшей проверки leakage, baseline и temporal model evaluation.

### shots
- Status: `CANDIDATE`
- Данных достаточно для дальнейшей проверки leakage, baseline и temporal model evaluation.

### shots_on_target
- Status: `CANDIDATE`
- Данных достаточно для дальнейшей проверки leakage, baseline и temporal model evaluation.

### offsides
- Status: `CANDIDATE`
- Данных достаточно для дальнейшей проверки leakage, baseline и temporal model evaluation.

### fouls
- Status: `CANDIDATE`
- Данных достаточно для дальнейшей проверки leakage, baseline и temporal model evaluation.

### throw_ins
- Status: `INSUFFICIENT_DATA`
- Поле throw-ins отсутствует в БД; обучение невозможно.

## Important

Статус `CANDIDATE` не означает READY. Перед обучением необходимо проверить temporal leakage, построить baseline и сравнить модель с baseline на временном тесте.

Throw-ins имеют статус `INSUFFICIENT_DATA`, поскольку соответствующего поля в `fixture_team_statistics` нет.
