# Generative Auto-Bidding with Unified Modeling and Exploration

<p align="center">
  <img src="./README.assets/淘天集团logo.webp" alt="淘天集团logo" width="150">
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/static/v1?label=conference&message=SIGIR%202026&color=cd5842" alt="Conference"></a>
  <a href="#"><img src="https://img.shields.io/static/v1?label=license&message=Apache%202.0&color=8ab708" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/static/v1?label=python&message=3.8%2B&color=0e76b7" alt="Python"></a>
</p>

**GUIDE** (Generative Auto-Bidding with Unified Modeling and Exploration) is a unified framework for automated bidding in computational advertising. It jointly models environmental dynamics and historical bidding action sequences via a Decision Transformer (DT), and introduces an Inverse Dynamics Module (IDM) as a safe fallback alongside a Q-value module for directed exploration and principled action selection. Together, these three components form an integrated "Explore–Safeguard–Select" pipeline that balances exploration effectiveness and operational safety.

## Features

- **Unified Modeling:** Decision Transformer jointly predicts the next bidding action and the next environment state, capturing long-term trajectory dependencies.
- **Inverse Dynamics Module (IDM):** Infers robust, behaviorally consistent actions from DT-predicted state transitions, serving as a conservative safety fallback during high-risk exploration.
- **Q-value Module:** Twin critic network guides DT exploration via regularization constraints and adaptively selects the higher-Q action between DT and IDM at inference time.
- **Two-Stage Training:** Separate pre-training followed by joint training prevents unstable gradient propagation and accelerates convergence.
- **Production-Validated:** Deployed on Taobao with significant improvements across all key advertising metrics in large-scale online A/B testing.

## Method Overview

GUIDE addresses the fundamental tension between exploration and safety in auto-bidding by integrating three core components:

1. **Decision Transformer (Explore):** Takes historical states, actions, and return-to-go as input and jointly generates the next candidate action $\hat{a}_t$ and the predicted next state $\hat{s}_{t+1}$. Q-value regularization guides the DT to explore high-value out-of-distribution trajectories.
2. **Inverse Dynamics Module (Safeguard):** Given the current state $s_t$ and the DT-predicted next state $\hat{s}_{t+1}$, the IDM infers a plausible action $\hat{a}^{\text{idm}}_t = f_{\text{idm}}(s_t, \hat{s}_{t+1})$. By imitating the behavioral policy embedded in training data, the IDM produces stable, conservative actions that serve as a reliable fallback.
3. **Q-value Module (Select):** A twin Q-network evaluates both candidate actions and selects the one with the higher estimated Q-value, ensuring value-driven and safe decision-making at inference time.

## Benchmarks & Results

**Dataset:** [AuctionNet](https://github.com/alimama-tech/AuctionNet) (NeurIPS 2024 Advertising Bidding Competition, final-round dataset)

**Offline Evaluation on AuctionNet (Score under different budget ratios):**

| Method | 50% | 75% | 100% | 125% | 150% |
|:------:|:---:|:---:|:----:|:----:|:----:|
| IQL | 17.9 | 26.9 | 30.9 | 32.0 | 37.8 |
| BC | 15.0 | 20.3 | 26.8 | 31.6 | 36.6 |
| CQL | 16.1 | 22.4 | 27.9 | 32.1 | 37.6 |
| TD3-BC | 15.0 | 22.7 | 26.4 | 31.4 | 38.0 |
| DT | 18.4 | 24.9 | 27.6 | 35.6 | 39.4 |
| AIGB | 10.7 | 22.2 | 24.6 | 31.8 | 36.5 |
| GAS | 18.4 | 27.5 | 36.1 | 40.0 | 46.5 |
| GAVE | 19.6 | 28.3 | 37.2 | 42.7 | 47.4 |
| **GUIDE (Ours)** | **20.3** | **29.1** | **37.6** | **43.3** | **48.3** |

**Simulation Environment:**

| Method | Score |
|:------:|:-----:|
| BC | 6366 |
| IQL | 6534 |
| DT | 6920 |
| TD3-BC | 7008 |
| CQL | 7138 |
| GAS | 7454 |
| AIGB | 6248 |
| **GUIDE (Ours)** | **8343** |

**Large-scale Online A/B Test (Taobao, 8-day, ~160,000 products):**

| Metric | Ad Click | Ad Cost | Ad GMV | Ad ROI |
|:------:|:--------:|:-------:|:------:|:------:|
| Improvement | +1.40% | +1.66% | +4.10% | +3.52% |

## Installation

### Environment Preparation

```bash
conda create -n your_env_name python=3.8 -y
conda activate your_env_name
pip install -r requirements.txt
```

### Data Preparation

We use [AuctionNet](https://github.com/alimama-tech/AuctionNet) as our benchmark. Please refer to https://github.com/alimama-tech/AuctionNet for dataset download instructions.

After downloading, place the data files as follows:

```
strategy_train_env/
└── data/
    ├── trajectory/
    │   └── trajectory_data.csv       # training trajectories
    └── trafficFinal/
        ├── period-7.csv              # evaluation traffic (periods 7–27)
        ├── period-8.csv
        └── ...
```

## Usage

### Train

```bash
cd strategy_train_env
python run/train_GUIDE.py \
  --data_path data/trajectory/trajectory_data.csv \
  --model_save_dir saved_model/GUIDE \
  --step_num 16000 \
  --batch_size 128
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `data/trajectory/trajectory_data.csv` | Trajectory CSV |
| `--model_save_dir` | `saved_model/GUIDE` | Checkpoint save directory |
| `--step_num` | `16000` | Total training steps |
| `--batch_size` | `128` | Batch size |
| `--detach_steps` | `500` | Phase 1 steps (separate training before joint training) |
| `--alpha` | `1.0` | Weight of Q-value regularization loss in actor objective |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

### Evaluate

```bash
cd strategy_train_env/run

# Evaluate on a single period (default: period-7)
python run_evaluate.py --data_dir ../data/trafficFinal --period 7

# Evaluate on all periods (7–27)
python run_evaluate.py --data_dir ../data/trafficFinal --all_period
```

## Project Structure

```
strategy_train_env/
├── bidding_train_env/
│   ├── baseline/GUIDE/
│   │   ├── dt_baselines.py         # DecisionTransformer, Critic, InverseDynamicsModel
│   │   └── utils.py                # EpisodeReplayBuffer
│   ├── common/utils.py             # save_normalize_dict
│   ├── offline_eval/
│   │   ├── offline_env.py          # Auction simulation environment
│   │   └── test_dataloader.py      # Loads period-N.csv for evaluation
│   └── strategy/
│       ├── base_bidding_strategy.py
│       └── guide_bidding_strategy.py   # Wraps trained model for inference
└── run/
    ├── train_GUIDE.py              # Training entry point
    └── run_evaluate.py             # Evaluation entry point
```

## Datasets

Please refer to https://tianchi.aliyun.com/competition/entrance/532236/customize448
