# Tempo

An agent that decides **when** to service each machine in a factory, not just which one looks unhealthy.

Built for the AMD Developer Hackathon ACT III (Intelligent Industry track).

## What it does

1. **Forecaster:** predicts how many days each machine has left, as a range with honest uncertainty.
2. **Planner:** replays the future thousands of times (Monte Carlo, on the GPU) and picks the cheapest repair schedule given production demand, technicians, spare parts and costs.
3. **Explainer + validator:** an LLM served with vLLM on AMD explains the plan; a rule-based checker verifies it before a human approves.

## Status

| Part | State |
|---|---|
| Factory simulator | done |
| Baseline forecaster (quantile gradient boosting) | done |
| GPU planner | next |
| Validator | next |
| LLM explainer (vLLM on AMD) | next |
| Backend + app | next |
| Evaluation (4 strategies compared) | next |

## Data honesty

The factory simulator is synthetic. Public datasets have sensor data but no production schedules, technician shifts, spare parts or costs, so the decision setting is simulated. All cost numbers are assumptions and are tested across a range of values.

## Setup on Windows (PowerShell)

Use Python 3.11 or 3.12.

```powershell
cd tempo-ai
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
git init
git config user.name "Your Name"
git config user.email "you@example.com"
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

## Run

```powershell
python -m pytest -q                 # all tests
python -m tempo.forecast.train      # train + evaluate the forecaster (about 10 seconds)
```

## Baseline results (simulated machines, unseen test fleet)

| | all rows | last 30 days before failure |
|---|---|---|
| Median error (days) | 8.2 | 4.7 |
| "Guess the average" error | 17.9 | |
| 90% range contains the truth | 85.9% | 81.1% |
| 50% range contains the truth | 42.4% | 36.1% |

The ranges are slightly too narrow (overconfident). Fixing calibration is on the list.

## Project layout

```
tempo/simulator   fake factory: machines, sensors, schedule, costs
tempo/forecast    remaining-life model
tempo/planner     GPU schedule optimizer
tempo/validator   rule checker
tempo/llm         LLM explanation layer
tempo/api         backend server
app/              the supervisor's screen
eval/             tests that produce the results table
```

## License

MIT
