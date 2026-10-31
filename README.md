# AI-Based Data Center Traffic Scheduler Simulator

A macOS-native project for studying how algorithmic and learned schedulers route data-center flows across paths with changing capacity. The Python experiment uses synthetic traffic and a flow-level discrete-event simulation. It needs no Linux install, Docker, network card, or external dataset.

The optional C++ executable is a compact fixed-capacity baseline from the first project layer. The integrated AI experiment is in Python; compare policies within the Python experiment, not across the separate C++ simulator.

## What the experiment models

- Each flow has an arrival time and size. Arrivals follow a seeded Poisson process; sizes use a bounded log-normal distribution.
- Three paths have different base capacities. Their effective capacities change through persistent Markov congestion states.
- Each path is a FIFO single-server queue. A flow can be dropped when the per-path buffer limit is reached.
- Schedulers see queue telemetry and delayed capacity observations. They do not see the hidden current capacity state.

The algorithmic references are round robin, shortest queue, least estimated delay, a **persistence-delay** scheduler, and an **oracle**. Persistence delay assumes the latest observed capacity state continues when the flow is served; it is a no-learning forecasting baseline. The oracle sees the hidden capacity state and exact queue completion time when it makes a decision. It is an intentionally unavailable reference for measuring how far practical schedulers are from perfect current-state information; it is not a deployable policy.

Two latency models estimate a flow's completion time on each candidate path:

- Random Forest
- Histogram Gradient Boosting

Two additional models forecast the hidden path capacity state at the flow's estimated service start:

- Random Forest classifier
- Histogram Gradient Boosting classifier

The capacity forecast policies turn class probabilities into an expected capacity multiplier, then combine it with estimated queue wait and flow size to score each path. They use the recent observed capacity history and its age, since congestion states persist but the scheduler only receives delayed observations. The forecast metrics also include a persistence baseline that predicts the latest observed capacity state. Training rows include all buffer-eligible candidate paths, not only the path chosen by randomized routing. Validation holds out complete training scenarios, and final testing uses separate workload regimes and seeds.

## Run in VS Code on macOS

Open this `network_scheduler` folder in VS Code, select `.venv/bin/python` as the interpreter, and choose **Run and Debug → Run traffic scheduler experiment**.

Or run from the integrated terminal:

```sh
source .venv/bin/activate
python run_project.py
```

If the project environment has not been created yet:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_project.py
```

The default run uses 2,500 flows in each of 10 training scenarios, then tests nine policies across five traffic regimes with three independent seed repetitions per regime. This creates 15 paired held-out scenarios. The run is reproducible with seed 42 and can take several minutes on a laptop.

For a smaller run while editing code:

```sh
python run_project.py --flows 300 --train-scenarios 4 --test-repetitions 2 --seed 42
```

## Outputs

The run creates `artifacts/`:

- `training_examples.csv`: randomized-routing examples and observed latency labels.
- `latency_model_random_forest.joblib` and `latency_model_hist_gradient_boosting.joblib`: fitted models.
- `model_validation.csv` and `.json`: scenario-held-out MAE and R² for each model.
- `capacity_training_examples.csv`: per-flow, per-path capacity-state labels at estimated service start.
- `capacity_model_random_forest.joblib` and `capacity_model_hist_gradient_boosting.joblib`: fitted capacity forecasters.
- `capacity_model_validation.csv` and `.json`: accuracy, balanced accuracy, macro-F1, and multiplier MAE, including the last-observation persistence baseline.
- `capacity_state_confusion.csv`: confusion counts for each congestion multiplier and model.
- `capacity_feature_importance_random_forest.csv`: feature contributions for the capacity forecaster.
- `feature_importance_random_forest.csv`: Random Forest feature importances.
- `evaluation_results.csv`: result for every policy on every test seed and workload regime.
- `capacity_forecast_test_results.csv` and `capacity_forecast_test_summary.csv`: forecast quality on the final held-out scenarios, including persistence.
- `policy_summary.csv`: mean metrics and regime-stratified bootstrap 95% confidence intervals.
- `paired_comparisons.csv`: paired AI-minus-heuristic differences with regime-stratified intervals.
- `policy_comparison.png`: latency confidence intervals and drop-rate comparison.

Negative latency differences in `paired_comparisons.csv` favor the AI scheduler. If a confidence interval includes zero, the experiment does not show a reliable difference from that baseline under these scenarios.

## Metrics

- **Mean latency:** average time from flow arrival to completion.
- **p95 latency:** latency below which 95% of admitted flows finish.
- **Throughput:** admitted bytes divided by simulated elapsed time.
- **Drop rate:** dropped flows divided by offered flows.
- **95% confidence interval:** stratified bootstrap interval that resamples repetitions within each regime, preserving the fixed mix of workload regimes. It describes variation across these synthetic runs; it is not a guarantee about real networks.

## Optional C++ baseline

Build with Apple's Command Line Tools and CMake:

```sh
cmake -S . -B build
cmake --build build
mkdir -p work
./build/network_scheduler --policy least_delay --flows 20000 --seed 42 --output work/least_delay.csv
```

The C++ baseline supports round robin, shortest queue, and least delay with fixed path capacities. Its source is `src/simulator.cpp`.

## Limits and interpretation

- The simulator models flows, not individual packets or a physical network.
- Traffic and congestion are synthetic. Results do not establish performance on production traffic or Marvell hardware.
- The oracle has information the learned and heuristic policies cannot access.
- A learned policy is not automatically better. Report its paired comparison against strong heuristics and the oracle, including uncertainty and all tested regimes.
