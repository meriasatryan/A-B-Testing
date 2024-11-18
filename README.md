### README.md

---

# A/B Testing: Multi-Armed Bandit Experiment

This project implements **Epsilon-Greedy** and **Thompson Sampling** algorithms to evaluate and optimize the performance of four advertisement options. The experiment balances exploration and exploitation across 20,000 trials.

---

## Features
- **Epsilon-Greedy**: Decaying epsilon for dynamic exploration.
- **Thompson Sampling**: Bayesian sampling for action selection.
- **Bonus**:
  - Dynamic bandit probabilities for real-world adaptation.
  - Parallel execution for faster experimentation.
  - Advanced visualizations and enhanced metrics.

---

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `loguru`

---

## Usage
1. Clone the repository:
   ```bash
   git clone <repo_link>
   cd <repo_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the experiment:
   ```bash
   python bandit_experiment.py
   ```

---

## Output
- **Plots**: Cumulative rewards and regrets for each algorithm.
- **Logs**: Detailed metrics in `experiment.log`.
- **CSV Files**: Experiment results saved for further analysis.

---

## Credits
Developed as part of an A/B testing homework assignment.
