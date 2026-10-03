# Final Model Readiness

Финальная проверка моделей после temporal test.

## Критерий READY

Для regression-модели статус READY устанавливается только тогда, когда модель одновременно лучше baseline по MAE и RMSE.

## Результаты

### goals
- Status: **READY**
- Type: `regression`
- Model MAE: `1.3360`
- Baseline MAE: `1.3695`
- MAE improvement: `0.0335`
- Model RMSE: `1.6784`
- Baseline RMSE: `1.7303`
- RMSE improvement: `0.0519`

### btts
- Status: **NOT_READY**
- Type: `classification`
- Reason: Mixed result versus baseline

### corners
- Status: **READY**
- Type: `regression`
- Model MAE: `2.8778`
- Baseline MAE: `2.9173`
- MAE improvement: `0.0395`
- Model RMSE: `3.6174`
- Baseline RMSE: `3.6414`
- RMSE improvement: `0.0240`

### yellow_cards
- Status: **NOT_READY**
- Type: `regression`
- Model MAE: `1.5576`
- Baseline MAE: `1.5538`
- MAE improvement: `-0.0038`
- Model RMSE: `1.9486`
- Baseline RMSE: `1.9592`
- RMSE improvement: `0.0106`

### shots
- Status: **WORSE_THAN_BASELINE**
- Type: `regression`
- Model MAE: `5.5004`
- Baseline MAE: `5.4272`
- MAE improvement: `-0.0732`
- Model RMSE: `7.2326`
- Baseline RMSE: `7.0358`
- RMSE improvement: `-0.1968`

### shots_on_target
- Status: **READY**
- Type: `regression`
- Model MAE: `2.6390`
- Baseline MAE: `2.6902`
- MAE improvement: `0.0512`
- Model RMSE: `3.2786`
- Baseline RMSE: `3.3292`
- RMSE improvement: `0.0506`

### offsides
- Status: **READY**
- Type: `regression`
- Model MAE: `1.5765`
- Baseline MAE: `1.6147`
- MAE improvement: `0.0382`
- Model RMSE: `2.0378`
- Baseline RMSE: `2.1162`
- RMSE improvement: `0.0784`

### fouls
- Status: **READY**
- Type: `regression`
- Model MAE: `4.5415`
- Baseline MAE: `4.6465`
- MAE improvement: `0.1050`
- Model RMSE: `5.6112`
- Baseline RMSE: `5.7807`
- RMSE improvement: `0.1695`

### throw_ins
- Status: **INSUFFICIENT_DATA**
- Type: `regression`
- Reason: No throw-in data available

## READY models

- `goals`
- `corners`
- `shots_on_target`
- `offsides`
- `fouls`

## Models not ready

- `btts` — `NOT_READY`
- `yellow_cards` — `NOT_READY`
- `shots` — `WORSE_THAN_BASELINE`
- `throw_ins` — `INSUFFICIENT_DATA`