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
| LLM explainer (AMD Token Factory API; offline stand-in model; template fallback) | done; tested offline and against the real model |
| Backend (planning, AI job with cache and limits, tamper-evident decision log) + supervisor screen | done, running as a public demo |
| Final evaluation: planner cost sensitivity | done (`eval/planner_eval.py`) |
| Final evaluation: does the rulebook catch the AI's mistakes | done (`eval/llm_eval.py`, results below) |

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

## The app (supervisor screen)

```powershell
.\venv\Scripts\python.exe -m tempo.api.server        # then open http://127.0.0.1:8000
.\venv\Scripts\python.exe -m tempo.api.server --warm  # once: saves the AI text for the 3 example situations
```

Standard library only, so nothing extra to install. The first start trains the forecaster (about 10 seconds) and saves it in `.cache/`.

- **Schedule board:** machines down the side, days across. The pale band is when the machine may fail, the tick is its most likely day, the block is the repair. Days with no technicians are hatched. Settings (technicians, parts, fleet size, condition) re-plan instantly.
- **Explanation and work orders:** an AI model on AMD GPUs writes them in the background. Results are cached on disk (`demo_cache/`), so repeat visits are instant and do not use the API. Limits: 40 AI runs a day and 6 per visitor per hour (`TEMPO_DAILY_AI`, `TEMPO_VISITOR_AI`); past that, or with no key, the screen shows a plain no-AI version and says so.
- **Rulebook:** the plan, each work order and every number in the text are checked. A person can only approve after reading the explanation.
- **Decision history:** every approval or rejection is a line in `data/audit.jsonl`, each linked to the previous one by a hash. Editing or deleting an old line is detected and shown. Someone who can rewrite the whole file can still replace the history; this makes quiet edits detectable, not impossible.
- **Reveal:** because the factory is simulated, the screen can show each machine's real failure day (the planner never sees it) and the cost of the plan against waiting for breakdowns and against perfect foresight.
- The key stays on the server (`.env`) and is never sent to the browser.

## Public demo

Live demo: https://tempo-ai-669c3.containers.snapdeploy.app (all AI text on it comes from AMD Token Factory, DeepSeek-V4-Flash on AMD GPUs; the three preset situations have saved AI text so they open instantly).

`Dockerfile` builds the app (the forecaster is trained while the image is built, so it starts in seconds). It runs on any host that builds from a Dockerfile; the demo runs on SnapDeploy's free tier:

- Create a container from this GitHub repo (Dockerfile build, port 8080, health check `/api/health`).
- Environment variable `RADEON_API_KEY` = your Token Factory key (a secret on the host, never in the repo). Optional: `TEMPO_MODEL`, `TEMPO_DAILY_AI`, `TEMPO_VISITOR_AI`.
- The free tier sleeps after 15 minutes without visitors (the first visit then takes about a minute to wake) and its disk resets on restart, so the decision log on the demo is not permanent.
- Visitors are limited to a few AI runs per hour and a few decisions per hour, the log is capped at 5000 entries, and at most 3 plans are computed at the same time.

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

## Results

Simulated fleets, 40 unseen fleets per row, 30-day horizon, average cost per fleet. Nothing is graded on data it learned from: the forecaster, the range calibration, the alarm threshold and the test fleets all use different random seeds. The baseline alarm (service a machine when its median forecast drops below N days) is tuned on separate practice fleets under each row's TRUE costs, a head start Tempo does not get. "Planner believes default costs (wrong)" means Tempo plans with the default cost numbers while reality uses the row's costs. The +/- is two standard errors, so differences smaller than that are not real.

Reproduce with `python -m eval.planner_eval` (writes `eval/results/`).

| Fleet | Costs | Failure vs planned repair | Planner believes | Run to failure | Tuned alarm | **Tempo** | Oracle | Tempo vs alarm | Cheaper in |
|---|---|---|---|---|---|---|---|---|---|
| standard | default costs | 13.6x | true costs | 1,143,672 | 240,355 | **279,439** | 166,483 | 16.3% dearer (+/- 9.4) | 38% of fleets |
| standard | failures cheaper | 4.4x | true costs | 368,182 | 163,678 | **158,596** | 100,115 | 3.1% cheaper (+/- 6.0) | 57% of fleets |
| standard | failures cheaper | 4.4x | default costs (wrong) | 368,182 | 163,678 | **156,039** | 100,115 | 4.7% cheaper (+/- 3.9) | 72% of fleets |
| standard | failures much costlier | 27.2x | true costs | 2,287,344 | 338,195 | **431,836** | 264,323 | 27.7% dearer (+/- 11.4) | 40% of fleets |
| standard | failures much costlier | 27.2x | default costs (wrong) | 2,287,344 | 338,195 | **461,439** | 264,323 | 36.4% dearer (+/- 13.5) | 33% of fleets |
| standard | planned repairs costlier | 5.2x | true costs | 1,143,672 | 448,897 | **443,282** | 276,104 | 1.3% cheaper (+/- 6.3) | 50% of fleets |
| standard | planned repairs costlier | 5.2x | default costs (wrong) | 1,143,672 | 448,897 | **436,578** | 276,104 | 2.7% cheaper (+/- 4.4) | 62% of fleets |
| scarce | default costs | 13.6x | true costs | 1,644,284 | 940,779 | **739,602** | 372,152 | 21.4% cheaper (+/- 5.9) | 88% of fleets |
| scarce | failures cheaper | 4.4x | true costs | 530,645 | 354,863 | **310,834** | 197,100 | 12.4% cheaper (+/- 4.1) | 85% of fleets |
| scarce | failures cheaper | 4.4x | default costs (wrong) | 530,645 | 354,863 | **310,702** | 197,100 | 12.4% cheaper (+/- 4.2) | 75% of fleets |
| scarce | failures much costlier | 27.2x | true costs | 3,288,568 | 1,805,107 | **1,363,991** | 632,392 | 24.4% cheaper (+/- 6.4) | 90% of fleets |
| scarce | failures much costlier | 27.2x | default costs (wrong) | 3,288,568 | 1,805,107 | **1,373,002** | 632,392 | 23.9% cheaper (+/- 6.4) | 88% of fleets |
| scarce | planned repairs costlier | 5.2x | true costs | 1,644,284 | 1,064,579 | **917,086** | 552,372 | 13.9% cheaper (+/- 4.4) | 82% of fleets |
| scarce | planned repairs costlier | 5.2x | default costs (wrong) | 1,644,284 | 1,064,579 | **911,165** | 552,372 | 14.4% cheaper (+/- 4.6) | 78% of fleets |
| healthy | default costs | 13.6x | true costs | 719,808 | 201,095 | **202,847** | 62,888 | 0.9% dearer (+/- 10.8) | 60% of fleets |
| healthy | failures cheaper | 4.4x | true costs | 232,142 | 133,556 | **120,018** | 51,388 | 10.1% cheaper (+/- 4.2) | 82% of fleets |
| healthy | failures cheaper | 4.4x | default costs (wrong) | 232,142 | 133,556 | **119,203** | 51,388 | 10.7% cheaper (+/- 8.0) | 57% of fleets |
| healthy | failures much costlier | 27.2x | true costs | 1,439,616 | 212,232 | **289,294** | 79,888 | 36.3% dearer (+/- 21.0) | 33% of fleets |
| healthy | failures much costlier | 27.2x | default costs (wrong) | 1,439,616 | 212,232 | **326,167** | 79,888 | 53.7% dearer (+/- 24.0) | 23% of fleets |
| healthy | planned repairs costlier | 5.2x | true costs | 719,808 | 386,481 | **340,192** | 136,075 | 12.0% cheaper (+/- 4.7) | 90% of fleets |
| healthy | planned repairs costlier | 5.2x | default costs (wrong) | 719,808 | 386,481 | **331,351** | 136,075 | 14.3% cheaper (+/- 8.5) | 62% of fleets |

What this shows, without rounding in our favour:

- **When repair capacity is scarce (30 machines, 1 technician per weekday, none at weekends, late parts), Tempo beats the tuned alarm in every cost setting:** 12% to 24% cheaper, cheaper in 75% to 90% of fleets, and almost unchanged when its cost assumptions are wrong. This is the situation Tempo is built for: when you cannot fix everything at once, choosing which machine goes first and when matters.
- **When technicians are plentiful (standard, healthy), it depends on how costly a surprise failure is.** If a failure costs about 4 to 5 times a planned repair, Tempo is level with the alarm on standard fleets (1% to 5% cheaper, within the error bars) and clearly ahead on healthy fleets (10% to 14% cheaper). At our default (13.6 times) it is 16% dearer on standard fleets and level on healthy ones. At 27 times it is 28% to 54% dearer. When failures dominate the bill and there are enough technicians, "service nearly everything early" is hard to beat, and Tempo's remaining forecast errors cost more than its smarter scheduling gains.
- **Wrong cost assumptions** barely matter on scarce fleets. They do hurt when real failure costs are much higher than Tempo assumes (standard and healthy fleets at 27 times: 9 to 17 percentage points worse).
- **Against doing nothing, Tempo is 55% to 76% cheaper** at default costs (scarce 55%, healthy 72%, standard 76%).
- The next improvement would be a risk-aversion setting that leans earlier when failures are very costly. It is not built.

How Tempo improved (earlier run, different fleet counts): the first version lost to the alarm by 31% on standard fleets. Sampling the risky early-failure tail brought that to 17%, and range calibration to 14.5%. On scarce fleets those two fixes made no measurable difference.

Caveats: this is a simulation, the cost numbers are assumptions, the scarce setting was added after the standard result as the case where scheduling should matter, and 40 fleets per row leaves wide error bars in several rows.

## Does the rulebook catch the AI's mistakes?

Reproduce with `python -m eval.llm_eval` (needs the Token Factory key; about 15 minutes per prompt; results in `eval/results/`). Six simulated fleets, 40 work orders, model DeepSeek-V4-Flash on AMD Token Factory. Every draft goes through the validator. Two prompts:

| Prompt | Work orders | Valid as written | Valid after one repair try | Replaced by a safe template |
|---|---|---|---|---|
| safety rules spelled out (how the app runs) | 40 | 100% | 100% | 0% |
| safety rules hidden (stress test) | 40 | 0% | 92% | 8% |

What this shows, and what it does not:

- **In normal use the model followed the checklist in all 40 orders.** That is a small test (6 fleets), not a guarantee, which is why the rulebook stays in charge anyway.
- **With the rules left out of the prompt on purpose, the model forgot safety steps in every order** (most often the test before restart, then energy isolation, then lockout/tagout). The validator caught all of them. One repair try, in which the model is shown its earlier draft and the full checklist, fixed 92%; the other 8% became safe template orders. No order that broke the checklist reached the human.
- This second row is a stress test of the safety net with a deliberately naive prompt. It is not how often the model breaks rules in normal use.
- The checker is keyword matching: it proves the steps are mentioned, not that they are good, and it cannot tell whether an explanation's reasoning is true. A human still approves every plan.
- Two bugs found while running this are fixed and covered by tests: the safety checker rejected valid wordings ("isolating", "functional test"), and the repair request did not show the model its previous draft, so repairs dropped other safety steps.

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
