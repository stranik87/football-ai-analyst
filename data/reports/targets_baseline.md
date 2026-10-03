# Targets Baseline Evaluation

Baseline рассчитан только на temporal test-период.

- Train: 2633
- Validation: 566
- Test: 563

| Target | Type | Test | Baseline | MAE | RMSE | Accuracy | Precision | Recall | F1 | LogLoss | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| goals | regression | 563 | 2.7942 | 1.3695 | 1.7303 | - | - | - | - | - | - |
| corner_kicks | regression | 562 | 9.5810 | 2.9182 | 3.6377 | - | - | - | - | - | - |
| yellow_cards | regression | 545 | 4.1623 | 1.5640 | 1.9616 | - | - | - | - | - | - |
| total_shots | regression | 562 | 24.9232 | 5.4051 | 7.0037 | - | - | - | - | - | - |
| shots_on_goal | regression | 562 | 8.6363 | 2.6925 | 3.3328 | - | - | - | - | - | - |
| offsides | regression | 538 | 3.5688 | 1.5803 | 2.0150 | - | - | - | - | - | - |
| fouls | regression | 562 | 23.7015 | 4.6286 | 5.7794 | - | - | - | - | - | - |
| btts | classification | 563 | 0.5427 | - | - | 0.5702 | 0.5702 | 1.0000 | 0.7262 | 0.6848 | 0.2458 |
