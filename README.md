# Tempo

An agent that decides **when** to service each machine in a factory, not just which one looks unhealthy.

Built for the AMD Developer Hackathon ACT III (Intelligent Industry track).

## What it does

1. **Forecaster:** predicts how many days each machine has left, as a calibrated range with honest uncertainty.
2. **Planner:** replays the future thousands of times (Monte Carlo, GPU-ready) and picks the cheapest repair schedule given production demand, technicians, spare parts and costs.
3. **Explainer + validator:** an LLM running on AMD GPUs (AMD Token Factory API) explains the plan; a rule-based checker verifies it before a human approves.

## Status

| Part | State |
|---|---|
| Factory simulator (with scarce technicians / late parts options) | done |
| Forecaster (quantile gradient boosting) + tail sampling + calibration on held-out machines | done |
| Planner (Monte Carlo cost matrix + exact assignment) | done on CPU (NumPy); PyTorch/GPU path written, **not yet run** |
| Strategy comparison on unseen fleets | done (results below) |
| Validator (plan rules, work-order safety checklist, unverified-number check) | done |
| LLM explainer (Token Factory API; offline stand-in model; template fallback) | done and tested offline; real-API run pending |
| Backend + app | next |
| Final evaluation (cost sensitivity) | next |

## Data honesty

The factory simulator is synthetic. Public datasets have sensor data but no production schedules, technician shifts, spare parts or costs, so the decision setting is simulated. All cost numbers are assumptions (a surprise failure costs roughly 13x a planned repair) and still need a sensitivity test.

## How the planner works

For every machine and every possible service day, it averages the cost over thousands of simulated futures (failure times drawn from the forecast range, including the risky early-failure tail). Machines only compete for technician-days and spare parts, so choosing every machine's best day at once is an assignment problem, solved exactly (Hungarian algorithm). "Optimal" means optimal for our forecast and our cost assumptions, not a promise about the real world.

## The validator (the LLM proposes, the rulebook decides)

Plain, non-AI checks that every plan must pass before a human sees it (`tempo/validator/rules.py`):

- **Plan:** no repair on a day without technicians, none before its spare part arrives, none outside the planning window, an optional company cap per day. Risky machines left unrepaired raise a warning for a human.
- **Work order (drafted by the LLM):** must match the planned day, be well formed, and cover the mandatory safety checklist (lockout/tagout, energy isolation, protective equipment, test before restart).
- **Explanation text:** every number the LLM writes (costs, chances) must exist in the real plan data, otherwise it is flagged. This check is strong for costs and percentages and weak for small whole numbers (days, machine numbers), because many small numbers are legitimately in the data.

The final evaluation will report how often the LLM's raw draft broke a rule and was caught.

## The explainer (LLM on AMD, with the rulebook in charge)

`tempo/llm` turns a computed plan into plain language and work orders, using the AMD Token Factory API (OpenAI-compatible models running on AMD GPUs). It uses only Python's standard library, so there is nothing extra to install.

1. **Facts:** every real number about the plan is collected. These are the only numbers the model may use.
2. **Explanation:** one call writes a short summary and one reason per machine. Numbers in the text are checked against the facts.
3. **Work orders:** drafted in small batches from example procedures; each draft goes through the validator (right day? full safety checklist?). A failed draft gets one repair attempt with the exact errors; if it still fails, a no-AI template order is used. A human never sees an invalid order.
4. **Fallbacks:** if the API is down or rate-limited, the app still produces a valid summary and orders from templates.

The example procedures are short demo text, not real maintenance instructions. A real plant would use its own manuals.

Token Factory limits to design around: about 20 requests a minute per account (shared by the whole team), a daily cap, no uptime promise, trial and development use only, and no reselling or proxying. The app therefore uses the model only for these fixed jobs, keeps the key on the server, and caches results.

The key goes in a `.env` file (copy `.env.example`), which git ignores. Then:

```powershell
.\venv\Scripts\python.exe -m tempo.llm.demo --check           # does the key work? lists today's models
.\venv\Scripts\python.exe -m tempo.llm.demo --fake sloppy     # offline: a pretend model that makes typical mistakes
.\venv\Scripts\python.exe -m tempo.llm.demo                   # the real model
```

Note: the "sloppy" model is scripted, so its 25% first-try rate shows the pipeline works, not how good a real model is. The real rate is measured with `explain_plan(..., tell_model_the_rules=False)`, which leaves the safety rules out of the prompt.

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
.\venv\Scripts\python.exe -m pytest -q                                  # all tests
.\venv\Scripts\python.exe -m tempo.forecast.train                       # train + evaluate the forecaster
.\venv\Scripts\python.exe -m tempo.planner.demo                         # standard fleets (2-4 minutes)
.\venv\Scripts\python.exe -m tempo.planner.demo --preset scarce         # few technicians, late parts
.\venv\Scripts\python.exe -m tempo.planner.demo --preset healthy        # machines mostly far from failure
```

## Results (simulated fleets, 60 unseen fleets each, 30-day horizon, average cost per fleet)

Nothing is graded on data it learned from: the forecaster, the range calibration, the alarm threshold and the test fleets all use different random seeds. The baseline alarm's threshold is tuned on separate practice fleets, so it is a strong, fair opponent (an untuned alarm, e.g. 14 days, does far worse).

| Fleet type | Run to failure | Best tuned alarm | **Tempo** | Oracle (knows the truth, unreachable) |
|---|---|---|---|---|
| Standard (20 machines, 2 technicians/weekday) | 1,113,397 | **236,350** | 270,556 | 159,280 |
| Healthy (machines mostly far from failure) | 713,179 | 198,788 | 195,943 | 59,118 |
| **Scarce (30 machines, 1 technician/weekday, none at weekends, late parts)** | 1,637,645 | 904,469 | **721,576** | 362,011 |

Tempo versus the tuned alarm:

- **Scarce resources: Tempo is 20% cheaper (+/- 4.6), and cheaper in 92% of fleets.** This is the case Tempo is designed for: when you cannot fix everything at once, choosing which machine goes first and when matters.
- **Healthy fleets: a tie** (+1.4% +/- 8.2).
- **Standard fleets: Tempo is 14.5% more expensive (+/- 7.6).** When technicians are plentiful and most machines are near failure, "service nearly everything immediately" is hard to beat, and the forecast's remaining errors cost Tempo more than smart scheduling gains.
- Against doing nothing, Tempo is 56-76% cheaper in every case.

How Tempo improved: the first version lost to the alarm by 31% on standard fleets. Sampling the risky early-failure tail brought that to 17%, and range calibration to 14.5%. On scarce fleets those two fixes made no measurable difference.

Caveats: the scarce setting was added after the standard result, as the case where scheduling should matter, and it is still a simulation. The remaining gap to the oracle (about 270k vs 159k on standard fleets) is forecast error, which is where further improvement would come from.

## Forecaster (before calibration), unseen machines

| | all rows | last 30 days before failure |
|---|---|---|
| Median error (days) | 8.2 | 4.7 |
| "Guess the average" error | 17.9 | |
| 90% range contains the truth | 85.9% | 81.1% |

## Project layout

```
tempo/simulator   fake factory: machines, sensors, schedule, costs
tempo/forecast    remaining-life model + calibration
tempo/planner     cost matrix, exact scheduler, baseline strategies, comparison demo
tempo/validator   rulebook: checks plans, work orders and LLM text
tempo/llm         API client, explainer, example procedures, offline stand-in model
tempo/api         backend server (to do)
app/              the supervisor's screen (to do)
eval/             final results scripts (to do)
```

## License

MIT
