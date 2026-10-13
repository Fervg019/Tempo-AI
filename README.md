# Tempo

An agent that decides **when** to service each machine in a factory, not just which one looks unhealthy.

Built for the AMD Developer Hackathon ACT III (Intelligent Industry track).

## What it does

1. **Forecaster:** predicts how many days each machine has left, as a range with honest uncertainty.
2. **Planner:** replays the future thousands of times (Monte Carlo, GPU-ready) and picks the cheapest repair schedule given production demand, technicians, spare parts and costs.
3. **Explainer + validator:** an LLM served with vLLM on AMD explains the plan; a rule-based checker verifies it before a human approves.

## Status

| Part | State |
|---|---|
| Factory simulator | done |
| Baseline forecaster (quantile gradient boosting) | done, ranges slightly overconfident |
| Planner (Monte Carlo cost matrix + exact assignment) | done on CPU (NumPy); PyTorch/GPU path written, **not yet run** |
| Forecast calibration | **next, needed (see results below)** |
| Validator | next |
| LLM explainer (vLLM on AMD) | next |
| Backend + app | next |
| Final evaluation (all strategies, cost sensitivity) | next |

## Data honesty

The factory simulator is synthetic. Public datasets have sensor data but no production schedules, technician shifts, spare parts or costs, so the decision setting is simulated. All cost numbers are assumptions and will be tested across a range of values.

## How the planner works

For every machine and every possible service day, it averages the cost over thousands of simulated futures (failure times drawn from the forecast range). Because machines only compete for technician-days and spare parts, picking the best day for every machine at once is an assignment problem, solved exactly (Hungarian algorithm). "Optimal" means optimal for our forecast and our cost assumptions, not a promise about the real world.

## Setup on Windows (PowerShell)

Python 3.13 works. Library versions are pinned in `requirements.txt` because brand-new numpy/scipy releases were blocked by Windows Application Control on the developer's PC.

```powershell
cd tempo-ai
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run everything through `.\venv\Scripts\python.exe` (no need to activate the environment).

## Run

```powershell
.\venv\Scripts\python.exe -m pytest -q                      # all tests
.\venv\Scripts\python.exe -m tempo.forecast.train           # train + evaluate the forecaster (~10 s)
.\venv\Scripts\python.exe -m tempo.planner.demo             # compare strategies on 50 unseen fleets (~20 s)
```

## Results so far (simulated fleets, machines unseen in training)

Forecaster on unseen machines:

| | all rows | last 30 days before failure |
|---|---|---|
| Median error (days) | 8.2 | 4.7 |
| "Guess the average" error | 17.9 | |
| 90% range contains the truth | 85.9% | 81.1% |
| 50% range contains the truth | 42.4% | 36.1% |

Strategy comparison, 60 unseen fleets of 20 machines, 30-day horizon, average cost per fleet:

| Strategy | Cost | Failures per fleet |
|---|---|---|
| Run to failure | 1,113,397 | 15.5 |
| Best simple alarm (threshold 45 days, tuned on these same fleets) | 240,172 | 1.6 |
| **Tempo (current)** | **310,140** | **3.1** |
| Oracle (planner given the true failure times; unreachable upper bound) | 159,280 | 1.3 |

**Current honest finding: Tempo beats doing nothing by 72% but does NOT yet beat a well-tuned simple alarm.** On a healthier fleet the gap is similar (Tempo 228,058 vs alarm 191,348). Diagnosis: the forecast is too optimistic about remaining life and the sampler ignores the lowest 5% tail, so Tempo defers repairs it should not. Pretending machines fail 20% sooner than forecast brings Tempo to 238,105, level with the alarm, which confirms the cause but is not a legitimate fix (it was tuned on the test fleets). The real fix is calibrating the forecast on held-out data, then testing on harder scenarios where technicians and parts are scarce, which is where scheduling should matter most.

## Project layout

```
tempo/simulator   fake factory: machines, sensors, schedule, costs
tempo/forecast    remaining-life model
tempo/planner     cost matrix, exact scheduler, baseline strategies, comparison demo
tempo/validator   rule checker (to do)
tempo/llm         LLM explanation layer (to do)
tempo/api         backend server (to do)
app/              the supervisor's screen (to do)
eval/             final results scripts (to do)
```

## License

MIT
